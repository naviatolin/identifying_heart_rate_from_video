from matplotlib import pyplot as plt
import pickle as pkl
import scipy.signal as signal
from scipy.signal import find_peaks_cwt, argrelmax
import scipy.fftpack as fftpack
import cv2
import numpy as np

# https://bbrc.in/use-of-color-channels-to-extract-heart-beat-rate-remotely-from-videos/
# used this as a reference

with open('test_avi.pickle', 'rb') as handle:
    data = pkl.load(handle)

plt.subplot(2,2,1)
plt.plot(data[0], color='b')
plt.subplot(2,2,2)
plt.plot(data[1], color='g')
plt.subplot(2,2,3)
plt.plot(data[2], color='r')
plt.show()

#butterworth bandpass filter
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    omega = 0.5 * fs
    low = lowcut / omega
    high = highcut / omega
    b, a = signal.butter(order, [low, high], btype='band')
    y = signal.lfilter(b, a, data, axis=0)
    return y

cap = cv2.VideoCapture('/Users/pranaviboyalakuntla/Documents/Stanford/W_23/EE 269/identifying_heart_rate_from_video/dataset/s1/vid_s1_T2.mov')

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
filtered = {}
for idx, color in enumerate(['blue', 'green', 'red']):
    filtered[color] = butter_bandpass_filter(data[idx], 0.6, 3.0, fps)

print("FPS: ", fps)

plt.subplot(2,2,1)
plt.plot(filtered['blue'], color='b')
plt.subplot(2,2,2)
plt.plot(filtered['green'], color='g')
plt.subplot(2,2,3)
plt.plot(filtered['red'], color='r')
plt.show()

# from scipy.ndimage.filters import gaussian_filter1d

# dataFiltered = gaussian_filter1d(filtered['red'], sigma=5)
tMax = argrelmax(filtered['red'])[0]
# plt.plot(filtered['red'], label = 'filtered')
# plt.plot(tMax, filtered['red'][tMax], 'o', mfc= 'none', label = 'max')
# plt.show()
print(len(tMax))

from numpy import genfromtxt
my_data = genfromtxt('/Users/pranaviboyalakuntla/Documents/Stanford/W_23/EE 269/identifying_heart_rate_from_video/ubfc-ppg dataset/subject1/ground_truth.txt')
 
tMax2 = argrelmax(my_data)[0]
plt.plot(my_data, label = 'filtered')
plt.plot(tMax, my_data[tMax], '*', mfc= 'none', label = 'max')
plt.plot(tMax2, my_data[tMax2], 'o', mfc= 'none', label = 'max')
plt.show()
print(len(tMax2),'2nd')

# peaks = find_peaks_cwt(filtered['red'],  np.arange(0,frame_count))
# print(peaks)

