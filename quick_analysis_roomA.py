import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from scipy import signal
from scipy import stats
from cvxEDA import cvxEDA
from tvsymp import apply_band_pass_filter, calculate_tvsymp_index
from peak_detection import findPeaks
from artifact_remover import automatic_EDA_correct, EDABE_LSTM_1DCNN_Model
from EDASympn import calculate_edasymp_index
import re
import sys
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)



# with open(sys.argv[1]) as f:
#     _ = f.readline() # ignore schema
#     try:
#         to_match = f.readline()
#         matched = re.search(r"BEGIN (.+?) FOR ([0-9]+?) SECONDS", to_match)
#         print("Analysing", matched.group(1))
#         duration = int(matched.group(2))
#     except:
#         print("Could not find section label. Continuing...")
def give_excel_data(df, duration):
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

    # eda_corrected = eda_filtered
    return give_eda_stats(eda_corrected, duration, new_sample_rate)
    
    #For data that doesn't work with LSTM (For some reason - skip lstm)
def give_eda_stats(eda_corrected, duration, new_sample_rate):
    peaks = findPeaks(
            pd.DataFrame(
                {
                    "EDA" : eda_corrected,
                    "filtered_eda" : eda_corrected
                    },
                index=pd.date_range(
                    start=0, 
                    periods=len(eda_corrected), 
                    freq= '40ms' # 25Hz
                    )
                ), 
            2, 4, 10, 0.05, 25
            )[0].sum()
    print("Peaks Per Minute:", peaks / (duration/60))
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

    print(eda_corrected)
    print(type(eda_corrected))

    down_eda = signal.decimate(eda_corrected, 12)

    [edasymp, edasympn, psd] = calculate_edasymp_index(down_eda, 2)

    # print("psd", psd)
    # print("edasymp:", edasymp)
    print("edasymp_n:", edasympn)
    
    return f'{peaks / (duration/60)}\t{p.mean()}\t{t.mean()}\t{tvsymp_signal.mean()}\t{edasympn}'



dataSessions = [1,2,4,5]
schedule = [0, 120, 420, 0, 120, 420] if (int(sys.argv[3])) else [0, 180, 480, 0, 180, 480]
print(schedule)
idno = sys.argv[2].split(",")
# print(idno)
bigDir = f'Turner_Data/Room_A/exp{sys.argv[1]}/'
paster = ''
colus = ["Millis","PPG","EDA","Temp"]
for id in idno:
    for i in dataSessions:
        print(bigDir + f'{i}_id-a{int(id)}.log')
        df = pd.read_csv(bigDir + f'{i}_id-a{int(id)}.log', delimiter=" ", names=colus) 
        duration = int(schedule[i])
        print(duration)
        paster += give_excel_data(df, duration)
        paster += '\t'

    paster += '\n'

print(paster)

# plt.legend(['eda_resampled', 'eda_filtered', 'eda_n', 'r', 'p', 't'])
# plt.plot(eda_filtered)
# plt.plot(eda_n)
# plt.show()

# plt.title("NSSCR")
# plt.plot(r)
# plt.show()
# plt.title("SCL")
# plt.plot(t)
# plt.show()
# plt.title("TVSymp")
# plt.plot(tvsymp_signal)
# plt.show()
#
# fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
# ax1.set_title("NSSCR")
# ax1.plot(r)
# ax2.set_title("SCL")
# ax2.plot(t)
# ax3.set_title("TVSymp")
# ax3.plot(tvsymp_signal)
# plt.show()
