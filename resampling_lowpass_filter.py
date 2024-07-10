import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy import stats
from cvxEDA import cvxEDA

df = pd.read_csv("file_4.csv")[1:]

duration = 600
og_sample_rate = len(df)/duration
new_sample_rate = 25
num_samples = int(new_sample_rate * duration)

eda_resampled = signal.resample(df.EDA, num_samples)
eda_z_score = stats.zscore(eda_resampled)

eda_resampled = eda_resampled[abs(eda_z_score) < 3]

#Butterworth lowpass filter

cutoff_frequency = 1.5
nyquist = new_sample_rate/2
norm_cutoff = cutoff_frequency / nyquist

b, a = signal.butter(16, norm_cutoff, btype='low', analog=False)


eda_filtered = signal.filtfilt(b, a, eda_resampled)

[r, p, t, l, d, e, obj] = cvxEDA(stats.zscore(eda_filtered), 1./new_sample_rate)

# plt.legend(['eda_resampled', 'eda_filtered', 'eda_n', 'r', 'p', 't'])
# plt.plot(eda_filtered)
# plt.plot(eda_n)
# plt.show()
plt.plot(r)
plt.show()
plt.plot(p)
plt.show()
plt.plot(t)
plt.show()
# #



