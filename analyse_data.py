import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy import stats
from cvxEDA import cvxEDA
from tvsymp import apply_band_pass_filter, calculate_tvsymp_index

import re
import sys

df = pd.read_csv(sys.argv[1])[1:]
duration = 0

with open(sys.argv[1]) as f:
    _ = f.readline() # ignore schema
    try:
        to_match = f.readline()
        matched = re.search(r"BEGIN (.+?) FOR ([0-9]+?) SECONDS", to_match)
        print("Analysing", matched.group(1))
        duration = int(matched.group(2))
    except:
        print("Could not find section label. Continuing...")

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

r_peak_count = 0

for i in range(len(r)):
    if((i!= 0) and (i!=len(r)-1)):
        if((r[i-1] < r[i]) and (r[i+1] < r[i])):
            r_peak_count += 1

print("Total NSSCR peaks", r_peak_count)
print("NSSCR per minute", r_peak_count/(duration/60))
print("SCR Std Deviation", np.std(r))

print("Mean SCL", t.mean())

tvsymp_signal = calculate_tvsymp_index(signal.resample(eda_filtered, 2*duration))

# plt.legend(['eda_resampled', 'eda_filtered', 'eda_n', 'r', 'p', 't'])
# plt.plot(eda_filtered)
# plt.plot(eda_n)
# plt.show()
plt.title("NSSCR")
plt.plot(r)
plt.show()
plt.title("SCL")
plt.plot(t)
plt.show()
plt.title("TVSymp")
plt.plot(tvsymp_signal)
plt.show()

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
ax1.set_title("NSSCR")
ax1.plot(r)
ax2.set_title("SCL")
ax2.plot(t)
ax3.set_title("TVSymp")
ax3.plot(tvsymp_signal)
plt.show()
