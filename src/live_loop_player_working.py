import sys
import threading
import signal

import numpy as np
import sounddevice as sd
import soundfile as sf

import time

from partitioned_auralization import PartitionedAuralization


AUDIO_DEVICE = "Digigram-LX-DANTE"

SAMPLE_RATE = 48000                 # Hz - sample rate for audio playback
BLOCK_LENGTH = 128                  # samples

VOLUME_FILE = 0.005
VOLUME_INTERACTIVE = 0.02 #  max 0.04
MAX_VOLUME = 0.05

AUR_FILTERS_FILENAME = "filter_nassau_X7.wav"    # desired-room / HRTF filter
FB_FILTERS_FILENAME = "rir_sp1.npy"     # measured loudspeaker-to-mic RIR

# GPU processing device for PartitionedAuralization ("gpu" or "cpu")
PROCESSING_DEVICE = "gpu"


def main() -> int:
    """Main function for live loop player."""
    
    # Load the auralization filters
    aur_filters, fs_filters = sf.read(AUR_FILTERS_FILENAME)
    assert fs_filters == SAMPLE_RATE, "Sample rate does not match with auralization filter"

    num_input_channels = 1  # Only a single input channel is supported.
    num_output_channels = 24  # 24 loudspeakers in the LoV
    mic_index = 0
   
    # load the feedback cancelation filters (measured RIRs)
    # measured_fc_filters = np.load(FB_FILTERS_FILENAME)[np.newaxis, :]  # shape: (1, N)
    measured_fc_filters = np.zeros((num_output_channels, BLOCK_LENGTH))  # Disable feedback cancelation for now.

    aur_filters = aur_filters[:, :num_output_channels]

    # Live auralization example
    part_aur = PartitionedAuralization(
        aur_filter_td=aur_filters.T,         # auralization filters
        fc_filter_td=measured_fc_filters,    # feedback cancelation filters
        block_length_samples=BLOCK_LENGTH,
        device=PROCESSING_DEVICE
    )

    print("Ready to go!")
    time.sleep(3)

    stop_event = threading.Event()

    def callback(indata: np.ndarray, outdata: np.ndarray, frames: int, _time, _status) -> None:
        chunk_size = min(frames, BLOCK_LENGTH)   # may be < frames on last block

        if stop_event.is_set():
            stop_event.set()
            raise sd.CallbackStop
        
        outdata.fill(0) # Clear the output buffer before writing new audio data
 
        mic_block = indata[:chunk_size, mic_index].copy()  # shape: (frames, 1)

        input_block = mic_block.T
        processed = part_aur.auralize(input_block * VOLUME_INTERACTIVE)

        output = np.clip(processed[:num_output_channels, :frames].T , -MAX_VOLUME, MAX_VOLUME)

        outdata[:frames, :] = output


    def handle_stop(_signum, _frame) -> None:
        stop_event.set()

    signal.signal(signal.SIGINT, handle_stop)
    signal.signal(signal.SIGTERM, handle_stop)

    print("Playing...")
    print("Press Ctrl+C to stop.")

    try:
        with sd.Stream(
            device=(AUDIO_DEVICE,AUDIO_DEVICE),
            samplerate=SAMPLE_RATE,
            blocksize=BLOCK_LENGTH,
            channels=(num_input_channels, num_output_channels),
            callback=callback,
            dtype="float32",
        ):
            stop_event.wait()
    except sd.PortAudioError as error:
        print(f"Audio output error: {error}", file=sys.stderr)
        return 1

    return 0

if __name__ == "__main__":
    raise SystemExit(main())