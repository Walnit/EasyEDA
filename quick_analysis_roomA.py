import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
from scipy import signal
from scipy import stats
from cvxEDA import cvxEDA
from tvsymp import apply_band_pass_filter, calculate_tvsymp_index
from peak_detection import findPeaks

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
    
    return f'{peaks / (duration/60)}\t{p.mean()}\t{t.mean()}\t{tvsymp_signal.mean()}'


dataSessions = [1,2,4,5]
schedule = [0,120, 420,0, 120, 420] if (int(sys.argv[3])) else [0, 180, 480, 0, 180, 480]
print(schedule)
idno = sys.argv[2].split(",")
bigDir = f'Turner_Data/Room_A/exp{sys.argv[1]}/'
paster = ''
colus = ["Millis","Heart Rate","EDA","Temperature"]
for id in idno:
    for i in dataSessions:
        print(bigDir + f'{i}_id-a{int(id)}.log')
        df = pd.read_csv(bigDir + f'{i}_id-a{int(id)}.log', delimiter=" ", names=colus)[1:] 
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
