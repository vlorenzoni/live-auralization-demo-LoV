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

Usage
─────
    python normalize_filters.py                              # uses defaults
    python normalize_filters.py --folder auralization_filters --reference filter_nassau_X7.wav
    python normalize_filters.py --folder auralization_filters --reference filter_nassau_X7.wav --dry-run

Options
───────
  --folder      Folder containing the .wav files (default: auralization_filters)
  --reference   Filename of the reference filter (must be inside --folder)
  --dry-run     Print the gain values without writing any files
  --subtype     Output bit depth: PCM_16, PCM_24, PCM_32, FLOAT (default: FLOAT)
"""

import argparse
import os
import sys

import numpy as np
import soundfile as sf


# ── helpers ──────────────────────────────────────────────────────────────────

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


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--folder",    default="auralization_filters",
                        help="Folder with the .wav files")
    parser.add_argument("--reference", default="filter_nassau_X7.wav",
                        help="Reference filename (must be in --folder)")
    parser.add_argument("--dry-run",   action="store_true",
                        help="Print gains without writing files")
    parser.add_argument("--subtype",   default="FLOAT",
                        choices=["PCM_16", "PCM_24", "PCM_32", "FLOAT"],
                        help="Output sample format (default: FLOAT)")
    args = parser.parse_args()

    folder    = args.folder
    ref_name  = args.reference
    ref_path  = os.path.join(folder, ref_name)
    out_folder = os.path.join(folder, "normalized")

    # ── load reference ──
    if not os.path.isfile(ref_path):
        print(f"ERROR: reference file not found: {ref_path}", file=sys.stderr)
        return 1

    ref_data, ref_fs = sf.read(ref_path, dtype="float64")
    ref_rms  = rms(ref_data)
    ref_rms_db  = rms_db(ref_data)
    ref_peak_db = peak_db(ref_data)

    print(f"Reference: {ref_name}")
    print(f"  sample rate : {ref_fs} Hz")
    print(f"  channels    : {ref_data.shape[1] if ref_data.ndim > 1 else 1}")
    print(f"  RMS         : {ref_rms:.6f}  ({ref_rms_db:.2f} dBFS)")
    print(f"  Peak        : {ref_peak_db:.2f} dBFS")
    print()

    # ── collect all wav files ──
    wav_files = sorted(
        f for f in os.listdir(folder)
        if f.lower().endswith(".wav") and os.path.isfile(os.path.join(folder, f))
    )

    if not wav_files:
        print(f"No .wav files found in '{folder}'.", file=sys.stderr)
        return 1

    if not args.dry_run:
        os.makedirs(out_folder, exist_ok=True)

    # ── process each file ──
    print(f"{'File':<40}  {'RMS dBFS':>9}  {'Peak dBFS':>10}  {'Gain dB':>8}  {'Clip?':>6}")
    print("─" * 80)

    for fname in wav_files:
        src_path = os.path.join(folder, fname)
        data, fs = sf.read(src_path, dtype="float64")

        src_rms    = rms(data)
        src_rms_db_ = rms_db(data)
        src_peak_db = peak_db(data)

        if src_rms < 1e-12:
            print(f"{fname:<40}  {'(silence – skipped)':>37}")
            continue

        g          = gain_to_match(src_rms, ref_rms)
        g_db       = 20.0 * np.log10(g)
        scaled     = data * g
        out_peak   = float(np.max(np.abs(scaled)))
        clips      = "YES ⚠" if out_peak > 1.0 else "no"

        # Warn loudly if we'd clip
        clip_note = ""
        if out_peak > 1.0:
            clip_note = f"  → clipped to ±1.0 (headroom lost: {20*np.log10(out_peak):.1f} dB)"

        print(f"{fname:<40}  {src_rms_db_:>9.2f}  {src_peak_db:>10.2f}  {g_db:>+8.2f}  {clips:>6}{clip_note}")

        if not args.dry_run:
            # Soft-clip to ±1.0 to avoid wraparound artefacts (already rare
            # for auralization / HRIRs, which have low peak levels).
            scaled = np.clip(scaled, -1.0, 1.0)
            out_path = os.path.join(out_folder, fname)
            sf.write(out_path, scaled, fs, subtype=args.subtype)

    print()
    if args.dry_run:
        print("Dry-run mode – no files written.")
    else:
        print(f"Normalised files written to: {out_folder}/")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())