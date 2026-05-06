"""
live_loop_gui.py
────────────────
Minimal tkinter GUI wrapper around live_loop_player logic.

Layout
  ┌────────────────────────────────────────┐
  │  Auralization filter                   │
  │  [dropdown – *.wav in                  │
  │   auralization_filters/]   [⟳]        │
  │  [channel badge: 24 ch ✓ / ✗]         │
  │  ─────────────────────────────────     │
  │  Volume  (VOLUME_INTERACTIVE)          │
  │  [ − ]   0.0200   [ + ]               │
  │  range 0.002 – 0.040   clip ±0.05     │
  │  ─────────────────────────────────     │
  │        [ START ]   [ STOP ]           │
  │                                        │
  │  Status: …                             │
  └────────────────────────────────────────┘
"""

import threading
import glob
import os
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
import sounddevice as sd
import soundfile as sf

from partitioned_auralization import PartitionedAuralization

# ── constants – exactly as in live_loop_player.py ───────────────────────────
AUDIO_DEVICE        = "Digigram-LX-DANTE"

SAMPLE_RATE         = 48000               # Hz
BLOCK_LENGTH        = 128                 # samples

VOLUME_FILE         = 0.005
VOLUME_INTERACTIVE  = 0.02               # default  (comment in original: max 0.04)
MAX_VOLUME          = 0.05               # hard clip ±0.05 – same as original

PROCESSING_DEVICE   = "gpu"

# ── GUI-specific constants ───────────────────────────────────────────────────
FILTERS_FOLDER      = "auralization_filters"
NUM_OUTPUT_CHANNELS = 24
REQUIRED_CHANNELS   = 24

VOLUME_STEP         = 0.002              # step for +/- buttons
VOLUME_MIN          = 0.002
VOLUME_MAX          = 0.04               # as noted in original source comment


# ── helpers ──────────────────────────────────────────────────────────────────

def list_wav_files() -> list:
    """Return basenames of all .wav files inside FILTERS_FOLDER."""
    pattern = os.path.join(FILTERS_FOLDER, "*.wav")
    return sorted(os.path.basename(p) for p in glob.glob(pattern))


def check_wav_channels(basename: str) -> int:
    """Return channel count of a file in FILTERS_FOLDER."""
    path = os.path.join(FILTERS_FOLDER, basename)
    return sf.info(path).channels


# ── stream manager ───────────────────────────────────────────────────────────

class StreamManager:
    """Owns the sounddevice Stream and PartitionedAuralization instance."""

    def __init__(self):
        self._stream = None
        self._stop_event = threading.Event()
        self._part_aur = None

    def start(self, basename, status_cb, volume_ref):
        """Open and start the audio stream.

        volume_ref is a one-element list whose float value is read live inside
        the callback, so GUI +/- changes take effect without restarting.
        """
        if self._stream is not None:
            return

        self._stop_event.clear()

        path = os.path.join(FILTERS_FOLDER, basename)
        try:
            aur_filters, fs = sf.read(path)
        except Exception as exc:
            status_cb(f"Error loading filter: {exc}", error=True)
            return

        if fs != SAMPLE_RATE:
            status_cb(
                f"Sample-rate mismatch: file={fs} Hz, expected={SAMPLE_RATE} Hz",
                error=True,
            )
            return

        if aur_filters.ndim == 1:
            aur_filters = aur_filters[:, np.newaxis]

        if aur_filters.shape[1] < REQUIRED_CHANNELS:
            status_cb(
                f"Filter only has {aur_filters.shape[1]} ch – {REQUIRED_CHANNELS} required.",
                error=True,
            )
            return

        aur_filters = aur_filters[:, :NUM_OUTPUT_CHANNELS]

        # feedback-cancellation disabled (zeros), same as original
        fc_filters = np.zeros((NUM_OUTPUT_CHANNELS, BLOCK_LENGTH))

        try:
            self._part_aur = PartitionedAuralization(
                aur_filter_td=aur_filters.T,
                fc_filter_td=fc_filters,
                block_length_samples=BLOCK_LENGTH,
                device=PROCESSING_DEVICE,
            )
        except Exception as exc:
            status_cb(f"Auralization init error: {exc}", error=True)
            return

        stop_ev  = self._stop_event
        part_aur = self._part_aur

        def callback(indata, outdata, frames, _time, _status):
            outdata.fill(0)                         # clear output buffer first

            if stop_ev.is_set():
                raise sd.CallbackStop

            chunk_size = min(frames, BLOCK_LENGTH)
            mic_block  = indata[:chunk_size, 0].copy()  # shape: (chunk_size,)

            # multiply by the live VOLUME_INTERACTIVE value from the GUI
            processed = part_aur.auralize(mic_block[np.newaxis] * volume_ref[0])

            # clip to ±MAX_VOLUME = ±0.05  (exactly as in original file)
            output = np.clip(
                processed[:NUM_OUTPUT_CHANNELS, :frames].T,
                -MAX_VOLUME,
                MAX_VOLUME,
            )
            outdata[:frames, :] = output

        try:
            self._stream = sd.Stream(
                device=(AUDIO_DEVICE, AUDIO_DEVICE),
                samplerate=SAMPLE_RATE,
                blocksize=BLOCK_LENGTH,
                channels=(1, NUM_OUTPUT_CHANNELS),
                callback=callback,
                dtype="float32",
            )
            self._stream.start()
        except sd.PortAudioError as exc:
            status_cb(f"Audio error: {exc}", error=True)
            self._stream = None
            return

        status_cb("Playing…  Press STOP to end.")

    def stop(self, status_cb):
        self._stop_event.set()
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        self._part_aur = None
        status_cb("Stopped.")

    @property
    def running(self):
        return self._stream is not None and self._stream.active


# ── GUI ──────────────────────────────────────────────────────────────────────

class App(tk.Tk):
    PAD    = 28
    BG     = "#f0efed"
    CARD   = "#e4e2df"
    FG     = "#1e1e1e"
    ACCENT = "#2a6496"
    WARN   = "#7a3030"
    OK     = "#2d6a3f"
    ERR    = "#a33333"
    DIM    = "#444444"
    DIV    = "#c8c5c0"

    def __init__(self):
        super().__init__()
        self.title("Live Loop Player")
        self.resizable(True, False)
        self.configure(bg=self.BG)
        self.minsize(560, 0)

        self._mgr = StreamManager()
        # volume held in a list so the audio callback reads the live value
        self._volume_ref = [VOLUME_INTERACTIVE]

        self._build_ui()
        self._refresh_files()

    # ── layout ───────────────────────────────────────────────────────────────

    def _build_ui(self):
        P = self.PAD

        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure(
            "TCombobox",
            fieldbackground=self.CARD, background=self.CARD,
            foreground=self.FG, selectbackground=self.CARD,
            selectforeground=self.FG, bordercolor=self.DIV,
            arrowcolor=self.FG,
        )
        style.map("TCombobox", fieldbackground=[("readonly", self.CARD)])

        # ── filter section ──
        top = tk.Frame(self, bg=self.BG)
        top.pack(padx=P, pady=(P, 6), fill="x")

        tk.Label(
            top, text="Auralization filter",
            bg=self.BG, fg=self.FG, font=("TkDefaultFont", 11, "bold"),
        ).pack(anchor="w")

        combo_row = tk.Frame(top, bg=self.BG)
        combo_row.pack(fill="x", pady=(4, 0))

        self._filter_var = tk.StringVar()
        self._combo = ttk.Combobox(
            combo_row, textvariable=self._filter_var,
            state="readonly", width=44, font=("TkDefaultFont", 11),
        )
        self._combo.pack(side="left", padx=(0, 8))
        self._combo.bind("<<ComboboxSelected>>", lambda _: self._on_file_selected())

        tk.Button(
            combo_row, text="⟳", command=self._refresh_files,
            bg=self.CARD, fg=self.DIM, relief="groove",
            font=("TkDefaultFont", 12), cursor="hand2",
            activebackground=self.DIV, activeforeground=self.FG,
            bd=1, padx=8,
        ).pack(side="left")

        self._ch_label = tk.Label(
            top, text="", bg=self.BG, font=("TkDefaultFont", 10),
        )
        self._ch_label.pack(anchor="w", pady=(5, 0))

        # ── divider ──
        tk.Frame(self, bg=self.DIV, height=1).pack(fill="x", padx=P, pady=(8, 0))

        # ── volume section ──
        vol_frame = tk.Frame(self, bg=self.BG)
        vol_frame.pack(padx=P, pady=(10, 4), fill="x")

        tk.Label(
            vol_frame, text="Volume  (VOLUME_INTERACTIVE)",
            bg=self.BG, fg=self.FG, font=("TkDefaultFont", 11, "bold"),
        ).pack(anchor="w")

        vol_row = tk.Frame(vol_frame, bg=self.BG)
        vol_row.pack(anchor="w", pady=(5, 0))

        vbtn = dict(
            bg=self.CARD, fg=self.FG, relief="groove",
            font=("TkDefaultFont", 13, "bold"), cursor="hand2",
            activebackground=self.DIV, activeforeground=self.FG,
            bd=1, padx=16, pady=4,
        )
        tk.Button(vol_row, text="−", command=self._vol_down, **vbtn).pack(side="left")

        self._vol_label = tk.Label(
            vol_row, text=self._fmt_vol(),
            bg=self.BG, fg=self.FG,
            font=("TkFixedFont", 13), width=8,
        )
        self._vol_label.pack(side="left", padx=8)

        tk.Button(vol_row, text="+", command=self._vol_up, **vbtn).pack(side="left")

        tk.Label(
            vol_frame,
            text=f"range {VOLUME_MIN:.3f} – {VOLUME_MAX:.3f}   |   clip ±{MAX_VOLUME:.2f}",
            bg=self.BG, fg=self.DIM, font=("TkDefaultFont", 9),
        ).pack(anchor="w", pady=(4, 0))

        # ── divider ──
        tk.Frame(self, bg=self.DIV, height=1).pack(fill="x", padx=P, pady=(8, 0))

        # ── start / stop ──
        btn_row = tk.Frame(self, bg=self.BG)
        btn_row.pack(padx=P, pady=P)

        big = dict(
            font=("TkDefaultFont", 12, "bold"), relief="raised",
            cursor="hand2", bd=2, width=12, pady=10,
        )
        self._btn_start = tk.Button(
            btn_row, text="▶  START",
            bg="#d6e8d0", fg="#1a3a1a",
            activebackground="#bdd9b5", activeforeground="#1a3a1a",
            command=self._on_start, **big,
        )
        self._btn_start.pack(side="left", padx=(0, 16))

        self._btn_stop = tk.Button(
            btn_row, text="■  STOP",
            bg="#e8d0d0", fg="#3a1a1a",
            activebackground="#d9b5b5", activeforeground="#3a1a1a",
            command=self._on_stop, state="disabled", **big,
        )
        self._btn_stop.pack(side="left")

        # ── status ──
        self._status_var = tk.StringVar(value="Select a filter file to begin.")
        self._status_lbl = tk.Label(
            self, textvariable=self._status_var,
            bg=self.BG, fg=self.DIM,
            font=("TkDefaultFont", 10), wraplength=500, justify="left",
        )
        self._status_lbl.pack(padx=P, pady=(0, P), anchor="w")

    # ── volume controls ───────────────────────────────────────────────────────

    def _fmt_vol(self):
        return f"{self._volume_ref[0]:.4f}"

    def _vol_up(self):
        self._volume_ref[0] = round(min(self._volume_ref[0] + VOLUME_STEP, VOLUME_MAX), 6)
        self._vol_label.config(text=self._fmt_vol())

    def _vol_down(self):
        self._volume_ref[0] = round(max(self._volume_ref[0] - VOLUME_STEP, VOLUME_MIN), 6)
        self._vol_label.config(text=self._fmt_vol())

    # ── file helpers ──────────────────────────────────────────────────────────

    def _refresh_files(self):
        files = list_wav_files()
        self._combo["values"] = files
        if files:
            self._filter_var.set(files[0])
            self._on_file_selected()
        else:
            self._filter_var.set("")
            self._ch_label.config(
                text=f"No .wav files found in '{FILTERS_FOLDER}/'", fg=self.ERR,
            )
            self._btn_start.config(state="disabled")

    def _on_file_selected(self):
        basename = self._filter_var.get()
        if not basename:
            return
        try:
            n = check_wav_channels(basename)
        except Exception as exc:
            self._ch_label.config(text=f"Cannot read file: {exc}", fg=self.ERR)
            self._btn_start.config(state="disabled")
            return

        if n >= REQUIRED_CHANNELS:
            self._ch_label.config(
                text=f"✓  {n} output channels detected – ready", fg=self.OK,
            )
            self._btn_start.config(state="normal")
        else:
            self._ch_label.config(
                text=f"✗  Only {n} channels – {REQUIRED_CHANNELS} required",
                fg=self.ERR,
            )
            self._btn_start.config(state="disabled")

    # ── start / stop ──────────────────────────────────────────────────────────

    def _on_start(self):
        basename = self._filter_var.get()
        if not basename:
            messagebox.showwarning("No filter", "Please select an auralization filter file.")
            return
        self._btn_start.config(state="disabled")
        self._btn_stop.config(state="normal")
        self._combo.config(state="disabled")
        self._set_status("Starting…")

        threading.Thread(
            target=lambda: self._mgr.start(
                basename, self._thread_safe_status, self._volume_ref,
            ),
            daemon=True,
        ).start()

    def _on_stop(self):
        self._btn_stop.config(state="disabled")
        # Stop is quick (just sets an event + closes the stream handle) –
        # run it synchronously so we are certain the stream is dead before
        # re-enabling the UI.
        self._mgr.stop(self._thread_safe_status)
        self._btn_start.config(state="normal")
        self._combo.config(state="readonly")

    # ── status helpers ────────────────────────────────────────────────────────

    def _set_status(self, msg, error=False):
        self._status_var.set(msg)
        self._status_lbl.config(fg=self.ERR if error else self.DIM)

    def _thread_safe_status(self, msg, error=False):
        self.after(0, lambda: self._set_status(msg, error=error))
        if error:
            self.after(0, self._reset_buttons)

    def _reset_buttons(self):
        self._btn_start.config(state="normal")
        self._btn_stop.config(state="disabled")
        self._combo.config(state="readonly")

    def _shutdown(self):
        """Single exit path: STOP button, window-close (X), Ctrl+C, SIGTERM."""
        self._mgr.stop(lambda *_: None)   # set stop_event, close stream, free GPU
        self.quit()                        # break mainloop()
        try:
            self.destroy()
        except Exception:
            pass

    def destroy(self):
        try:
            super().destroy()
        except Exception:
            pass


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import signal
    import sys

    app = App()

    # ── Ctrl+C / SIGTERM ──────────────────────────────────────────────────────
    # Signal handlers run in the main thread but Tk may be blocked inside Tcl.
    # We schedule the shutdown via after() so it executes safely on the Tk thread.
    def _handle_signal(_signum, _frame):
        app.after(0, app._shutdown)

    signal.signal(signal.SIGINT,  _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    # ── Window close button (X) ───────────────────────────────────────────────
    app.protocol("WM_DELETE_WINDOW", app._shutdown)

    # ── Keep Python signal-checking alive inside the Tk event loop ───────────
    # By default, Python only checks for signals when the interpreter is active.
    # This tiny periodic callback ensures SIGINT is delivered promptly.
    def _signal_poll():
        app.after(200, _signal_poll)

    app.after(200, _signal_poll)

    app.mainloop()
    sys.exit(0)
