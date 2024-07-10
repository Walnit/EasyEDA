import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal

df = pd.read_csv("file_1.csv")[1:]

duration = 180
og_sample_rate = len(df)/duration
new_sample_rate = 4
num_samples = int(new_sample_rate * duration)

eda_resampled = signal.resample(df.EDA, num_samples)

#Butterworth lowpass filter
cutoff_frequency = 1.5
nyquist = og_sample_rate / 2
norm_cutoff = cutoff_frequency / nyquist

b, a = signal.butter(4, norm_cutoff, btype='low', analog=False)

eda_filtered = signal.filtfilt(b, a, eda_resampled)

plt.plot(eda_filtered)
plt.show()



