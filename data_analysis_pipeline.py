import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy import stats
from cvxEDA import cvxEDA
from tvsymp import apply_band_pass_filter, calculate_tvsymp_index
from EDASympn import calculate_edasymp_index

import re
import sys

################################
# Phase 1: Importing raw data
# Format: CSV File, with the columns Time, EDA, Heart Rate, Temperature
# The first row of the CSV data (after the title headers) should have the duration of the session in seconds in the Time column
# For example, 
# ```
# Time,EDA,PPG,Temp
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
df.loc[:, "PPG"] = (df.loc[:, "PPG"] * 2048) / 32767
df.loc[:, "Temp"] = df.loc[:, "Temp"]* 2048 / 32767 / 10

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
# Phase 4: Removal of Artifacts
# We use the works of Llanes-Jurado et al. (2023) to remove artifacts using an LSTM model.
################################

from artifact_remover import automatic_EDA_correct, EDABE_LSTM_1DCNN_Model

df_correction = pd.DataFrame(
    {
        "Time": pd.date_range(start=0, periods=eda_filtered.size, freq=f"{1/new_sample_rate}s"),
        "EDA": eda_filtered,
    }
)

df_result, dict_metrics = automatic_EDA_correct(df_correction, EDABE_LSTM_1DCNN_Model, 
                                                freq_signal=new_sample_rate, th_t_postprocess=2.5,
                                                eda_signal="EDA", time_column="Time")

print("No. of artifacts corrected: ", dict_metrics["number_of_artifacts"])
eda_corrected = df_result["signal_automatic"].to_numpy()

################################
# Phase 5: Calculation of NSSCR
# We use the works of Taylor et al. (2015) to calculate the NSSCR.
# We use an offset of 1, a start window of 4 seconds, an end window of 4 seconds, and a threshold of 0.05.
################################

from peak_detection import findPeaks

peaks = findPeaks(
    pd.DataFrame(
        {
            "EDA" : eda_corrected,
            "filtered_eda" : eda_corrected
        },
        index=pd.date_range(
            start=0, 
            periods=len(eda_corrected), 
            freq= '40ms' # 25H
        )
    ), 
    2, 3.5, 10, 0.05, 25
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
[r, p, t, l, d, e, obj] = cvxEDA(stats.zscore(eda_corrected), 1./new_sample_rate)

print("Mean SCR", p.mean())

print("Mean SCL", t.mean())

################################
# Phase 6: Calculation of TVSymp
################################

tvsymp_signal = calculate_tvsymp_index(signal.resample(eda_corrected, 2*duration))

print("Mean TVSympt", tvsymp_signal.mean())

################################
# Phase 7: Calculation of EDASymp and EDASymp normalized (EDASympn)
# We reimplemented the functions from the works of Posada-Quintero, H.F., Florian, J.P., Orjuela-Cañón, A.D. et al. Power Spectral Density Analysis of Electrodermal Activity for Sympathetic Function Assessment. Ann Biomed Eng 44, 3124–3135 (2016). https://doi.org/10.1007/s10439-016-1606-6
# We also utilised some of the works of NeuroKit2
# Function header: def calculate_edasymp_index(eda, samp_rate, nperseg = 128, freq_band = [0.045, 0.25]):

# psd is power spectral density (np array)
# edasymp is spectral dynamics in the frequency range of 0.045–.25 Hz -> which is basically integrated psd values within the frequency range (float)
# edasympn is normalized edasymp via max psd value
################################

[edasymp, edasympn, psd] = calculate_edasymp_index(eda_corrected, 25)

# print("edasymp:", edasymp)
# print("psd", psd)
print("edasympn:", edasympn)
