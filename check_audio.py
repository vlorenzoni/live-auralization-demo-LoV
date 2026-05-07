import os

import soundfile as sf
import matplotlib.pyplot as plt

# get all names of the files in the folder

auralization_files = [
    f for f in os.listdir("auralization_filters/") if f.endswith(".wav")
]

folder_path = "auralization_filters/"


for file in auralization_files:
    file_path = os.path.join(folder_path, file)
    data, samplerate = sf.read(file_path)
    print(f"File: {file}, Sample Rate: {samplerate}, Duration: {len(data)/samplerate:.2f} seconds")
    print(f"File: {file}, Size of data: {len(data)} samples")   
    plt.figure(figsize=(10, 4))
    plt.plot(data)
    plt.title(f"Waveform of {file}")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    print(f"File: {file}, Sample Rate: {samplerate}, Duration: {len(data)/samplerate:.2f} seconds")
    











plt.show()
