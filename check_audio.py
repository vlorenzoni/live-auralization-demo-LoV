import os
from scipy.io import wavfile
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

# get all names of the files in the folder


folder_path = "auralization_filters"
file= "normalized/filters_Brou_big_room_pos_L3_MR3.wav" #"filters_church_MM1.wav"

file_path = os.path.join(folder_path, file)
data, samplerate = sf.read(file_path)
print(f"File: {file}, Sample Rate: {samplerate}, Duration: {len(data)/samplerate:.2f} seconds")
print(f"File: {file}, Size of data: {len(data)} samples")
plt.figure(figsize=(10, 4))
plt.plot(data)
plt.title(f"Waveform of {file}")
plt.xlabel("Sample Index")      
plt.ylabel("Amplitude")


# folder_path = "auralization_filters/normalized"

# file_path = os.path.join(folder_path, file)
# data, samplerate = sf.read(file_path)
# print(f"File: {file}, Sample Rate: {samplerate}, Duration: {len(data)/samplerate:.2f} seconds")
# print(f"File: {file}, Size of data: {len(data)} samples")
# plt.figure(figsize=(10, 4))
# plt.plot(data)
# plt.title(f"Waveform of {file}  (normalized)")
# plt.xlabel("Sample Index")      
# plt.ylabel("Amplitude")



# folder_path = "auralization_filters/normalized_old"

# file_path = os.path.join(folder_path, file)
# data, samplerate = sf.read(file_path)
# print(f"File: {file}, Sample Rate: {samplerate}, Duration: {len(data)/samplerate:.2f} seconds")
# print(f"File: {file}, Size of data: {len(data)} samples")
# plt.figure(figsize=(10, 4))
# plt.plot(data)
# plt.title(f"Waveform of {file}  (normalized - old)")
# plt.xlabel("Sample Index")      
# plt.ylabel("Amplitude")



# folder_path = "auralization_filters/normalized/normalized_tail_fixed"

# file_path = os.path.join(folder_path, file)
# data, samplerate = sf.read(file_path)
# print(f"File: {file}, Sample Rate: {samplerate}, Duration: {len(data)/samplerate:.2f} seconds")
# print(f"File: {file}, Size of data: {len(data)} samples")
# plt.figure(figsize=(10, 4))
# plt.plot(data)
# plt.title(f"Waveform of {file}  (normalized - tail fixed)")
# plt.xlabel("Sample Index")      
# plt.ylabel("Amplitude")







# auralization_files = [
#     f for f in os.listdir(folder_path) if f.endswith(".wav")
# ]



# for file in auralization_files:
#     file_path = os.path.join(folder_path, file)
#     data, samplerate = sf.read(file_path)
#     print(f"File: {file}, Sample Rate: {samplerate}, Duration: {len(data)/samplerate:.2f} seconds")
#     print(f"File: {file}, Size of data: {len(data)} samples")   
#     plt.figure(figsize=(10, 4))
#     plt.plot(data)
#     plt.title(f"Waveform of {file}")
#     plt.xlabel("Sample Index")
#     plt.ylabel("Amplitude")
#     print(f"File: {file}, Sample Rate: {samplerate}, Duration: {len(data)/samplerate:.2f} seconds")


mono, freq = sf.read('RIR_mic_E_L3_M3.wav')



multi_channel_data = np.tile(mono, (24, 1)).T

# 4. Save the new 24-channel file
sf.write('filter_NO-SDM_big_room_mic_E_L3_M3.wav', multi_channel_data, samplerate)

print(f"Created 24-channel file with shape: {multi_channel_data.shape}")

plt.figure()
plt.plot(mono)
plt.title("mono")


plt.figure()
plt.plot(multi_channel_data)
plt.title("multichannel")

plt.show()

norm