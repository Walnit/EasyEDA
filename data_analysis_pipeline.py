import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy import stats
from cvxEDA import cvxEDA
from tvsymp import apply_band_pass_filter, calculate_tvsymp_index

import re
import sys

################################
# Phase 1: Importing raw data
# Format: CSV File, with the columns Time, EDA, Heart Rate, Temperature
# The first row of the CSV data (after the title headers) should have the duration of the session in seconds in the Time column
# For example, 
# ```
# Time,EDA,Heart Rate,Temperature
# 600,,,
# 1721806894825084,19938.0,22925.0,5745.0
# ```
# The name of the file should be passed as an argument to the program.
################################

if len(sys.argv) != 2:
    print("Usage: [filename].py <path to CSV file>")
    sys.exit(1)

df = pd.read_csv(sys.argv[1])
duration = df.iloc[0, 0]
df = df.iloc[1:]

print("Loaded dataframe! Duration of experiment:", duration)

################################
# Phase 2: Transforming data
# This should transform the data (more specifically, EDA data) into the appropriate units.
# Conversion formula for EDA/mS: 1/((1-(2*((eda (raw)* 2.048) / 32767)/3.3))*825000)*1000000
# Conversion formula for PPG/mV: ((PPG (raw) * 2048) /32767)
# Conversion formula for Temp/deg C: temperature (raw)*2048/(32767)/10
################################

df.loc[:, "EDA"] = 1 / ((1 - (2 * ((df.loc[:, "EDA"] * 2.048) / 32767) / 3.3)) * 825000) * 1000000
df.loc[:, "Heart Rate"] = (df.loc[:, "Heart Rate"] * 2048) / 32767
df.loc[:, "Temperature"] = df.loc[:, "Temperature"]* 2048 / 32767 / 10

################################
# Phase 3: Cleaning data
# This section should prepare the data for analysis.
# Firstly, we resample the data to 25 Hz.
# Then, remove outliers by removing data points that are more than 3 standard deviations away from the mean (or, have a z-score >3)
# Lastly, apply a butterworth lowpass filter with a cutoff frequency of 1.5 Hz, order 16.
################################

new_sample_rate = 25
num_samples = int(new_sample_rate * duration)
eda_resampled = signal.resample(df.EDA, num_samples)
eda_z_score = stats.zscore(eda_resampled)
eda_removed_outliers = eda_resampled[abs(eda_z_score) < 3]

cutoff_frequency = 1.5
nyquist = new_sample_rate/2
norm_cutoff = cutoff_frequency / nyquist
b, a = signal.butter(16, norm_cutoff, btype='low', analog=False)
eda_filtered = signal.filtfilt(b, a, eda_removed_outliers)

################################
# Phase 4: Calculation of NSSCR
# We use the works of Taylor et al. (2015) to calculate the NSSCR.
# We use an offset of 1, a start window of 4 seconds, an end window of 4 seconds, and a threshold of 0.05.
################################

from peak_detection import findPeaks
peaks = findPeaks(
    pd.DataFrame(
        {
            "EDA" : eda_removed_outliers, 
            "filtered_eda" : eda_filtered
        },
        index=pd.date_range(
            start=0, 
            periods=len(eda_filtered), 
            freq= '40ms' # 25Hz
        )
    ), 
    1*25, 4, 4, 0.05, 25
)[0].sum()

print("Peaks Per Minute:", peaks / (duration/60))

################################
# Phase 5: Calculations via cvxEDA
# We use the works of Taylor et al. (2015) to break the EDA signal to its components. 
# Afterwards, other analysis is performed.
################################

"""
r: phasic component
p: sparse SMNA driver of phasic component
t: tonic component
l: coefficients of tonic spline
d: offset and slope of the linear drift term
e: model residuals
obj: value of objective function being minimized (eq 15 of paper)
"""
[r, p, t, l, d, e, obj] = cvxEDA(stats.zscore(eda_filtered), 1./new_sample_rate)

print("Mean SCR", p.mean())
print("SCR Std Deviation", np.std(p))

print("Mean SCL", t.mean())
print("SCR Std Deviation", np.std(t))

################################
# Phase 6: Calculation of TVSymp
################################

tvsymp_signal = calculate_tvsymp_index(signal.resample(eda_filtered, 2*duration))

print("Mean TVSympt", tvsymp_signal.mean())
print("TVSymp Std Dev", np.std(tvsymp_signal))

