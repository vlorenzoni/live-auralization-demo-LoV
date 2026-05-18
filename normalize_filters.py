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

  --extend-tail     Synthesise a physically plausible reverb tail beyond the point
                    where the SNR collapses into the noise floor. Replaces the noisy
                    late tail (and any hard cut) with shaped exponential-decay noise
                    at the correct per-channel decay rate, crossfaded at the SNR knee.
                    Output file is padded to 2 × estimated RT60 in length.
                    Use this for low-SNR measurements of highly reverberant spaces
                    (e.g. large churches) where the recording is too short or too
                    quiet to capture the full natural decay.
                    Cannot be combined with --fix-tail (they are mutually exclusive).

  --extend-xfade    Crossfade duration in seconds at the join between real and
                    synthesised tail (default: 0.3 s).

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


# ── helpers – tail extension ──────────────────────────────────────────────────

def extend_tail(data: np.ndarray, fs: int,
                snr_threshold_db: float = 20.0,
                xfade_duration_s: float = 0.3):
    """
    Synthesise a physically plausible reverb tail for low-SNR RIR measurements.

    For each channel independently:
      1. Find the SNR knee — the last sample where the smoothed envelope is
         snr_threshold_db above the noise floor (same logic as fix_tail).
      2. Fit an exponential decay slope to the EDC in the clean region
         (−5 dB to min(−20 dB, knee level)) using OLS regression.
      3. Synthesise shaped Gaussian noise with the same exponential decay
         envelope, level-matched to the signal RMS at the knee.
      4. Crossfade the original signal out and the synthetic tail in over
         xfade_duration_s seconds centred on the knee.
      5. Pad the output to 2 × estimated RT60 in length.

    The output is a physically plausible late reverb tail, not a recording.
    It will have the correct decay rate and spectral character but no room
    modal structure in the late portion. For auralization this is inaudible —
    the late diffuse field of a large room is perceptually indistinguishable
    from shaped noise.

    Parameters
    ----------
    data             : (n_samples, n_channels) float64 array – post-gain
    fs               : sample rate
    snr_threshold_db : fade/crossfade starts at SNR knee (default 20 dB)
    xfade_duration_s : crossfade length in seconds (default 0.3 s)

    Returns
    -------
    extended : np.ndarray  shape (n_out, n_channels) — longer than input
    report   : list of dicts, one per channel
    """
    n, n_ch  = data.shape
    xfade_len = max(1, int(xfade_duration_s * fs))
    smooth_win = max(1, int(0.01 * fs))

    # first pass: find per-channel RT60 and target length
    rt60s = []
    knees = []
    for ch in range(n_ch):
        channel = data[:, ch]
        nf_db   = _noise_floor_db(channel, fs)
        fs_idx  = _find_fade_start(channel, fs, nf_db, snr_threshold_db)
        knees.append(fs_idx)

        # direct sound = peak index
        direct  = int(np.argmax(np.abs(channel)))
        rir     = channel[direct:]
        knee_rir = max(0, fs_idx - direct)

        # fit decay slope on EDC from -5 to min(-20, knee)
        sch  = np.cumsum(rir[::-1]**2)[::-1]
        sch  = np.maximum(sch, 1e-20 * sch[0])
        edc  = 10.0 * np.log10(sch / sch[0])
        t    = np.arange(len(edc)) / fs

        i5   = int(np.argmin(np.abs(edc + 5)))
        i20  = int(np.argmin(np.abs(edc + min(20.0, abs(edc[knee_rir]) if knee_rir > 0 else 20.0))))
        if i20 > i5 + fs // 10:
            slope, _ = np.polyfit(t[i5:i20], edc[i5:i20], 1)
            rt60 = -60.0 / slope if slope < 0 else 3.0
        else:
            rt60 = 3.0  # fallback
        rt60s.append(rt60)

    # output length: 2 × max RT60 across channels, at least original length
    max_rt60     = max(rt60s)
    target_s     = max(max_rt60 * 2.0, n / fs + 0.5)
    target_n     = int(target_s * fs)

    extended = np.zeros((target_n, n_ch))
    report   = []

    for ch in range(n_ch):
        channel  = data[:, ch]
        knee     = knees[ch]
        rt60     = rt60s[ch]
        nf_db    = _noise_floor_db(channel, fs)

        direct   = int(np.argmax(np.abs(channel)))
        rir      = channel[direct:]
        knee_rir = max(0, knee - direct)

        # decay slope
        sch  = np.cumsum(rir[::-1]**2)[::-1]
        sch  = np.maximum(sch, 1e-20 * sch[0])
        edc  = 10.0 * np.log10(sch / sch[0])
        t    = np.arange(len(edc)) / fs
        i5   = int(np.argmin(np.abs(edc + 5)))
        i20  = int(np.argmin(np.abs(edc + min(20.0, abs(edc[knee_rir]) if knee_rir > 0 else 20.0))))
        if i20 > i5 + fs // 10:
            slope, _ = np.polyfit(t[i5:i20], edc[i5:i20], 1)
        else:
            slope = -60.0 / rt60

        # extension length
        ext_start = knee
        ext_len   = target_n - ext_start

        # exponential envelope for extension
        t_ext     = np.arange(ext_len) / fs
        env_lin   = 10.0 ** (slope * t_ext / 20.0)

        # level-match to RMS at knee
        win_rms  = max(1, int(0.05 * fs))
        seg      = rir[max(0, knee_rir - win_rms) : max(1, knee_rir)]
        rms_knee = float(np.sqrt(np.mean(seg ** 2))) if len(seg) > 0 else 1e-6
        env_lin  *= rms_knee

        # shaped noise — envelope already encodes correct level, just apply it
        rng    = np.random.default_rng(seed=ch)
        noise  = rng.standard_normal(ext_len)
        # normalise noise to unit RMS before applying envelope
        noise  /= (float(np.sqrt(np.mean(noise ** 2))) + 1e-12)
        shaped = noise * env_lin

        # crossfade envelopes
        fade_out = np.ones(n)
        fe = min(knee + xfade_len, n)
        fade_out[knee:fe] = np.linspace(1.0, 0.0, fe - knee)
        fade_out[fe:]     = 0.0

        fade_in = np.zeros(ext_len)
        fi = min(xfade_len, ext_len)
        fade_in[:fi] = np.linspace(0.0, 1.0, fi)
        fade_in[fi:] = 1.0

        # assemble
        extended[:n, ch]                     = channel * fade_out
        extended[ext_start:ext_start+ext_len, ch] += shaped * fade_in

        report.append({
            "ch":           ch,
            "knee_s":       knee / fs,
            "rt60_s":       rt60,
            "slope_db_s":   slope,
            "target_len_s": target_n / fs,
        })

    return extended, report


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
    parser.add_argument("--extend-tail",   action="store_true",
                        help="Synthesise a plausible reverb tail beyond the SNR knee")
    parser.add_argument("--extend-xfade",  type=float, default=0.3,
                        help="Crossfade duration in seconds at the join (default: 0.3)")
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

    if getattr(args, 'fix_tail', False) and getattr(args, 'extend_tail', False):
        print("ERROR: --fix-tail and --extend-tail are mutually exclusive.", file=sys.stderr)
        return 1

    folder   = args.folder
    ref_name = args.reference
    ref_path = os.path.join(folder, ref_name)

    # Auto-select output folder: keep the two passes separate by default
    if args.output_folder:
        out_folder = args.output_folder
    elif args.fix_tail and args.files:
        out_folder = os.path.join(folder, "normalized_tail_fixed")
    elif getattr(args, 'extend_tail', False) and args.files:
        out_folder = os.path.join(folder, "normalized_tail_extended")
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
    if getattr(args, 'extend_tail', False):
        print(f"  Tail extend : ON  "
              f"(SNR threshold {args.snr_threshold:.0f} dB, "
              f"xfade {args.extend_xfade:.2f} s)")
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
    if getattr(args, 'extend_tail', False):
        header += f"  {'Knee / RT60 / OutLen':>40}"
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

        # ensure 2D array (n_samples, n_channels)
        if data.ndim == 1:
            data = data[:, np.newaxis]

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
            # 2a. tail fade — applied AFTER gain so noise floor estimate is correct
            scaled, report = fix_tail(
                scaled, fs,
                snr_threshold_db=args.snr_threshold,
                fade_duration_s =args.fade_duration,
            )
            starts = [r["fade_start_s"] for r in report]
            ends   = [r["fade_end_s"]   for r in report]
            tail_col = (f"  {min(starts):.2f} – {max(starts):.2f} s"
                        f"  {min(ends):.2f} – {max(ends):.2f} s")

        elif getattr(args, 'extend_tail', False):
            # 2b. tail extension — synthesise plausible tail beyond SNR knee
            scaled, report = extend_tail(
                scaled, fs,
                snr_threshold_db=args.snr_threshold,
                xfade_duration_s=args.extend_xfade,
            )
            knees  = [r["knee_s"]       for r in report]
            rt60s  = [r["rt60_s"]       for r in report]
            tgt    = report[0]["target_len_s"]
            tail_col = (f"  knee {min(knees):.2f}–{max(knees):.2f}s  "
                        f"RT60 {min(rt60s):.2f}–{max(rt60s):.2f}s  "
                        f"→ {tgt:.2f}s")

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


def normalize_filter_to_reference(source_path: str, reference_path: str,
                                   output_path: str) -> float:
    """
    Scale source so its broadband RMS matches the reference.
    Applies one single scalar to all channels — inter-channel
    relationships are fully preserved.

    Returns the gain applied in dB.
    """
    ref,  ref_fs  = sf.read(reference_path,  dtype="float64")
    src,  src_fs  = sf.read(source_path,     dtype="float64")

    assert ref_fs == src_fs, f"Sample rate mismatch: {ref_fs} vs {src_fs}"

    ref_rms = np.sqrt(np.mean(ref ** 2))
    src_rms = np.sqrt(np.mean(src ** 2))
    gain    = ref_rms / src_rms

    scaled  = np.clip(src * gain, -1.0, 1.0)
    sf.write(output_path, scaled, src_fs, subtype="FLOAT")

    gain_db = 20 * np.log10(gain)
    print(f"Source RMS:    {20*np.log10(src_rms):.2f} dBFS")
    print(f"Reference RMS: {20*np.log10(ref_rms):.2f} dBFS")
    print(f"Gain applied:  {gain_db:+.2f} dB")
    print(f"Written to:    {output_path}")
    return gain_db









if __name__ == "__main__":
    raise SystemExit(main())

    # # Pass 1 — normalise all filters to nassau level
    # python normalize_filters.py \
    #     --folder auralization_filters \
    #     --reference filter_nassau_X7.wav
    
    #     # Pass 2 — extend tail on the church/low-SNR filter only
    # python normalize_filters.py \
    #     --folder auralization_filters/normalized \
    #     --reference filter_nassau_X7.wav \
    #     --extend-tail \
    #     --files your_church_filter.wav
    
    # python normalize_filters.py \
    # --folder auralization_filters \
    # --reference filter_nassau_X7.wav \
    # --extend-tail \
    # --files RIR_mic_A_20250408_234754.wav \
    # --output-folder auralization_filters/extended