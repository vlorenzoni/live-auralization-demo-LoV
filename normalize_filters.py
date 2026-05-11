"""
normalize_filters.py
────────────────────
Level-match all multitrack auralization filter .wav files in a folder
so that they produce a similar perceived loudness when played at the
same VOLUME_INTERACTIVE setting.

Reference:  the "nassau" file (or any file you specify with --reference)
Method:     RMS matching across all channels and all samples
Output:     a subfolder  <input_folder>/normalized/
            each file is written at the same sample rate and bit depth,
            scaled so its broadband RMS equals the reference RMS.

Typical two-pass workflow
─────────────────────────
  # Pass 1 – normalise all files → auralization_filters/normalized/
  python normalize_filters.py

  # Pass 2 – fix tail only on the church file → auralization_filters/normalized_tail_fixed/
  python normalize_filters.py \
      --folder auralization_filters/normalized \
      --reference filter_nassau_X7.wav \
      --fix-tail \
      --files filters_church_MM1.wav

  # Result: all other filters live in normalized/, the repaired church lives in
  #         normalized_tail_fixed/ — move it there manually to keep one clean folder.

Options
───────
  --folder          Folder containing the .wav files (default: auralization_filters)
  --reference       Filename of the reference filter (must be inside --folder)
  --dry-run         Print the gain values without writing any files
  --subtype         Output bit depth: PCM_16, PCM_24, PCM_32, FLOAT (default: FLOAT)

  --files           Space-separated list of filenames to process (basenames inside
                    --folder). If omitted, every .wav in the folder is processed.
                    Example: --files filters_church_MM1.wav

  --output-folder   Explicit output path. Defaults to <folder>/normalized/, or
                    <folder>/normalized_tail_fixed/ when --fix-tail + --files are
                    both set (so the two passes never overwrite each other).

  --fix-tail        Apply a per-channel adaptive cosine fade-out at the point
                    where each channel's reverb tail drops into the noise floor.
                    This fixes the "sudden drop" artefact caused by low-SNR RIR
                    measurements (e.g. a church filter recorded at low gain).
                    Safe on all files: clean filters with plenty of margin have
                    their fade point placed well before the file end, so only
                    the inaudible noise tail is shaped — the reverb is untouched.

  --snr-threshold   How many dB above the noise floor the signal must be before
                    the fade begins (default: 20 dB).
                    Raise to start the fade later (keep more reverb, more noise).
                    Lower to start earlier (cleaner, shorter apparent tail).

  --fade-duration   Length of the cosine fade-out window in seconds (default: 0.2).
                    The fade goes from full gain to silence over this period.

How the noise floor is estimated
─────────────────────────────────
The RMS of the last 5 % of each channel is used.  For a correctly captured
RIR this region contains only the recording system's noise floor.  The
fade point is then the last sample where the smoothed energy envelope is
still snr-threshold dB above that floor.
"""

import argparse
import os
import sys

import numpy as np
import soundfile as sf


# ── helpers – level ───────────────────────────────────────────────────────────

def rms(data: np.ndarray) -> float:
    """Broadband RMS across all channels and all samples."""
    return float(np.sqrt(np.mean(data ** 2)))


def rms_db(data: np.ndarray) -> float:
    return 20.0 * np.log10(rms(data) + 1e-12)


def peak_db(data: np.ndarray) -> float:
    return 20.0 * np.log10(float(np.max(np.abs(data))) + 1e-12)


def gain_to_match(source_rms: float, target_rms: float) -> float:
    """Linear gain that scales source_rms up/down to target_rms."""
    if source_rms < 1e-12:
        raise ValueError("Source RMS is effectively zero – cannot compute gain.")
    return target_rms / source_rms


# ── helpers – tail repair ─────────────────────────────────────────────────────

def _noise_floor_db(channel: np.ndarray, fs: int) -> float:
    """
    Estimate the noise floor as the RMS of the last 5 % of the channel.
    For a well-captured RIR this region contains only recorder noise.
    Returns the level in dBFS.
    """
    tail_start = max(0, int(0.95 * len(channel)))
    tail = channel[tail_start:]
    return 20.0 * np.log10(np.sqrt(np.mean(tail ** 2)) + 1e-12)


def _find_fade_start(channel: np.ndarray, fs: int,
                     noise_floor_db: float,
                     snr_threshold_db: float,
                     smooth_ms: float = 10.0) -> int:
    """
    Return the index of the last sample where the smoothed energy envelope
    is still snr_threshold_db above the noise floor.

    The fade window starts here: everything before is untouched,
    everything after decays to silence over fade_duration seconds.
    """
    win = max(1, int(smooth_ms * 1e-3 * fs))
    env = np.convolve(channel ** 2, np.ones(win) / win, mode='same')
    env_db = 10.0 * np.log10(env + 1e-12)

    threshold = noise_floor_db + snr_threshold_db
    above = np.where(env_db >= threshold)[0]

    if len(above) == 0:
        return 0          # entire channel already at noise floor
    return int(above[-1])


def _cosine_fade(n_total: int, fade_start: int, fade_len: int) -> np.ndarray:
    """
    Gain envelope:
      1.0  for [0 … fade_start)
      cosine ramp 1→0  for [fade_start … fade_start+fade_len)
      0.0  for [fade_start+fade_len … n_total)
    """
    env = np.ones(n_total)
    fade_end = min(fade_start + fade_len, n_total)
    actual = fade_end - fade_start
    if actual > 0:
        ramp = 0.5 * (1.0 + np.cos(np.pi * np.arange(actual) / actual))
        env[fade_start:fade_end] = ramp
    env[fade_end:] = 0.0
    return env


def fix_tail(data: np.ndarray, fs: int,
             snr_threshold_db: float = 20.0,
             fade_duration_s: float  = 0.2):
    """
    Apply a per-channel adaptive cosine fade-out where the reverb tail
    drops into the noise floor.

    Parameters
    ----------
    data             : (n_samples, n_channels) float64 array – post-gain
    fs               : sample rate
    snr_threshold_db : fade starts when SNR drops below this value
    fade_duration_s  : length of the cosine fade window

    Returns
    -------
    fixed  : np.ndarray, same shape as data
    report : list of dicts, one per channel, with diagnostic fields
    """
    fixed    = data.copy()
    fade_len = max(1, int(fade_duration_s * fs))
    n, n_ch  = data.shape
    report   = []

    for ch in range(n_ch):
        channel  = data[:, ch]
        nf_db    = _noise_floor_db(channel, fs)
        fs_idx   = _find_fade_start(channel, fs, nf_db, snr_threshold_db)
        envelope = _cosine_fade(n, fs_idx, fade_len)
        fixed[:, ch] = channel * envelope
        report.append({
            "ch":             ch,
            "noise_floor_db": nf_db,
            "fade_start_s":   fs_idx / fs,
            "fade_end_s":     min(fs_idx + fade_len, n) / fs,
        })

    return fixed, report


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--folder",        default="auralization_filters",
                        help="Folder with the .wav files")
    parser.add_argument("--reference",     default="filter_nassau_X7.wav",
                        help="Reference filename (must be in --folder)")
    parser.add_argument("--dry-run",       action="store_true",
                        help="Print gains without writing files")
    parser.add_argument("--subtype",       default="FLOAT",
                        choices=["PCM_16", "PCM_24", "PCM_32", "FLOAT"],
                        help="Output sample format (default: FLOAT)")
    parser.add_argument("--fix-tail",      action="store_true",
                        help="Apply per-channel adaptive tail fade-out")
    parser.add_argument("--snr-threshold", type=float, default=20.0,
                        help="dB above noise floor where fade starts (default: 20)")
    parser.add_argument("--fade-duration", type=float, default=0.2,
                        help="Cosine fade length in seconds (default: 0.2)")
    parser.add_argument("--files",         nargs="+", default=None,
                        metavar="FILENAME",
                        help="Process only these filenames (basenames inside --folder). "
                             "If omitted, all .wav files in the folder are processed.")
    parser.add_argument("--output-folder", default=None,
                        metavar="PATH",
                        help="Where to write output files. Defaults to "
                             "<folder>/normalized/ for a plain normalisation pass, or "
                             "<folder>/normalized_tail_fixed/ when --fix-tail and "
                             "--files are both set, so the two passes land in different "
                             "folders automatically.")
    args = parser.parse_args()

    folder   = args.folder
    ref_name = args.reference
    ref_path = os.path.join(folder, ref_name)

    # Auto-select output folder: keep the two passes separate by default
    if args.output_folder:
        out_folder = args.output_folder
    elif args.fix_tail and args.files:
        out_folder = os.path.join(folder, "normalized_tail_fixed")
    else:
        out_folder = os.path.join(folder, "normalized")

    # ── load reference ────────────────────────────────────────────────────────
    if not os.path.isfile(ref_path):
        print(f"ERROR: reference file not found: {ref_path}", file=sys.stderr)
        return 1

    ref_data, ref_fs = sf.read(ref_path, dtype="float64")
    ref_rms_val  = rms(ref_data)

    print(f"Reference : {ref_name}")
    print(f"  sample rate : {ref_fs} Hz")
    print(f"  channels    : {ref_data.shape[1] if ref_data.ndim > 1 else 1}")
    print(f"  RMS         : {ref_rms_val:.6f}  ({rms_db(ref_data):.2f} dBFS)")
    print(f"  Peak        : {peak_db(ref_data):.2f} dBFS")
    if args.fix_tail:
        print(f"  Tail fix    : ON  "
              f"(SNR threshold {args.snr_threshold:.0f} dB, "
              f"fade {args.fade_duration:.2f} s)")
    print()

    # ── collect wav files ─────────────────────────────────────────────────────
    if args.files:
        # validate every explicitly requested file exists
        missing = [f for f in args.files
                   if not os.path.isfile(os.path.join(folder, f))]
        if missing:
            for m in missing:
                print(f"ERROR: file not found in '{folder}': {m}", file=sys.stderr)
            return 1
        wav_files = sorted(args.files)
    else:
        wav_files = sorted(
            f for f in os.listdir(folder)
            if f.lower().endswith(".wav") and os.path.isfile(os.path.join(folder, f))
        )

    if not wav_files:
        print(f"No .wav files found in '{folder}'.", file=sys.stderr)
        return 1

    if not args.dry_run:
        os.makedirs(out_folder, exist_ok=True)

    # ── header ────────────────────────────────────────────────────────────────
    W = 46
    header = (f"{'File':<{W}}  {'RMS dBFS':>9}  {'Peak dBFS':>10}  "
              f"{'Gain dB':>8}  {'Clip?':>5}")
    if args.fix_tail:
        header += f"  {'Fade start (s)':>20}  {'Fade end (s)':>18}"
    print(header)
    print("─" * len(header))

    # ── process each file ─────────────────────────────────────────────────────
    for fname in wav_files:
        src_path = os.path.join(folder, fname)
        data, fs = sf.read(src_path, dtype="float64")

        src_rms_val  = rms(data)
        src_rms_db_  = rms_db(data)
        src_peak_db_ = peak_db(data)

        if src_rms_val < 1e-12:
            print(f"{fname:<{W}}  {'(silence – skipped)'}")
            continue

        # 1. gain normalisation
        g        = gain_to_match(src_rms_val, ref_rms_val)
        g_db     = 20.0 * np.log10(g)
        scaled   = data * g
        out_peak = float(np.max(np.abs(scaled)))
        clips    = "YES⚠" if out_peak > 1.0 else "no"
        if out_peak > 1.0:
            clips += f" (+{20*np.log10(out_peak):.1f} dB over)"

        tail_col = ""
        if args.fix_tail:
            # 2. tail fade — applied AFTER gain so noise floor estimate is correct
            scaled, report = fix_tail(
                scaled, fs,
                snr_threshold_db=args.snr_threshold,
                fade_duration_s =args.fade_duration,
            )
            starts = [r["fade_start_s"] for r in report]
            ends   = [r["fade_end_s"]   for r in report]
            tail_col = (f"  {min(starts):.2f} – {max(starts):.2f} s"
                        f"  {min(ends):.2f} – {max(ends):.2f} s")

        name_trunc = fname if len(fname) <= W else fname[:W-1] + "…"
        print(f"{name_trunc:<{W}}  {src_rms_db_:>9.2f}  {src_peak_db_:>10.2f}  "
              f"{g_db:>+8.2f}  {clips:>5}{tail_col}")

        if not args.dry_run:
            scaled = np.clip(scaled, -1.0, 1.0)
            sf.write(os.path.join(out_folder, fname), scaled, fs, subtype=args.subtype)

    print()
    if args.dry_run:
        print("Dry-run mode – no files written.")
    else:
        print(f"Output written to: {out_folder}/")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
