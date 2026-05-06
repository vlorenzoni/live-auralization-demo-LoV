import soundfile as sf
import matplotlib.pyplot as plt


#read the audio file
data, samplerate = sf.read('filter_nassau_X7.wav')
print(f"Audio data shape: {data.shape}")
print(f"Sample rate: {samplerate}")

print("Audio data type:", data.dtype)
#print max and min values to check if they are in the expected range
print(f"Max value: {data.max()}")
print(f"Min value: {data.min()}")

#plotting the first 1000 samples to visualize the audio data
plt.figure(figsize=(12, 6)) 
# plot all samples of the audio data
plt.plot(data[::])  # 
plt.xlabel("Sample")




#read other audio file
data2, samplerate2 = sf.read('filters_SDM_Brou_small_room_pos_LS1_MS1.mat.wav')
print(f"Audio data shape: {data2.shape}")
print(f"Sample rate: {samplerate2}")

print("Audio data type:", data2.dtype)
#print max and min values to check if they are in the expected range
print(f"Max value: {data2.max()}")
print(f"Min value: {data2.min()}")

#plotting the first 1000 samples to visualize the audio data
plt.figure(figsize=(12, 6))
plt.plot(data2[::])  # plot the first 3 seconds of audio
plt.xlabel("Sample")


#read other audio file
data3, samplerate3 = sf.read('filters_Brou_small_room_pos_LS1_MS1.wav')
print(f"Audio data shape: {data3.shape}")
print(f"Sample rate: {samplerate3}")

print("Audio data type:", data3.dtype)
#print max and min values to check if they are in the expected range
print(f"Max value: {data3.max()}")
print(f"Min value: {data3.min()}")

#plotting the first 1000 samples to visualize the audio data
plt.figure(figsize=(12, 6))
plt.plot(data3[::])  # plot the first 3 seconds of audio
plt.xlabel("Sample")


plt.show()













plt.show()
