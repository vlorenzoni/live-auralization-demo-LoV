import os
from scipy.io import wavfile
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

# get all names of the files in the folder

folder_path = "auralization_filters/normalized"


auralization_files = [
    f for f in os.listdir(folder_path) if f.endswith(".wav")
]



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



# sr_nassau, data_nassau = wavfile.read(os.path.join(folder_path, 'filter_nassau_X7.wav'))
# sr_church, data_church = wavfile.read(os.path.join(folder_path, 'filters_church_MM1_calibrated.wav'))

# # Calculate the scaling factor
# # (Based on the ratio of the global RMS values)
# rms_nassau = np.sqrt(np.mean(data_nassau.astype(float)**2))
# rms_church = np.sqrt(np.mean(data_church.astype(float)**2))
# scaling_factor = rms_nassau / rms_church

# # Apply the full gain factor to the quieter file
# data_church_balanced = (data_church.astype(float) * scaling_factor).astype(data_church.dtype)

# #plot balanced church
# plt.figure(figsize=(10, 4))
# plt.plot(data_church_balanced)
# plt.title("Balanced Church Filter")
# plt.xlabel("Sample Index")
# plt.ylabel("Amplitude")     

# # Save the result
# wavfile.write('filters_church_MM1_balanced.wav', sr_church, data_church_balanced)


plt.show()
