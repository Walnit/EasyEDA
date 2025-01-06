import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy import stats
from cvxEDA import cvxEDA
from tvsymp import apply_band_pass_filter, calculate_tvsymp_index
from EDASympn import calculate_edasymp_index
from window_segmentation import get_segmented_list

import re
import sys

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from artifact_remover import automatic_EDA_correct, EDABE_LSTM_1DCNN_Model
from peak_detection import findPeaks

files_to_analyze = (
    # (
    #     "Turner_Data/Room_B/Batch_1/1_usb1.csv",
    #     "Turner_Data/Room_B/Batch_1/2_usb1.csv",
    #     "Turner_Data/Room_B/Batch_1/4_usb1.csv",
    #     "Turner_Data/Room_B/Batch_1/5_usb1.csv",
    # ),
    # (
    #     "Turner_Data/Room_B/Batch_1/1_usb4.csv",
    #     "Turner_Data/Room_B/Batch_1/2_usb4.csv",
    #     "Turner_Data/Room_B/Batch_1/4_usb4.csv",
    #     "Turner_Data/Room_B/Batch_1/5_usb4.csv",
    # ),
    # (
    #     "Turner_Data/Room_B/Batch_2/0_usb2.csv",
    #     "Turner_Data/Room_B/Batch_2/1_usb2.csv",
    #     "Turner_Data/Room_B/Batch_2/3_usb2.csv",
    #     "Turner_Data/Room_B/Batch_2/4_usb2.csv",
    # ),
    # (
    #     "Turner_Data/Room_B/Batch_2/0_usb4.csv",
    #     "Turner_Data/Room_B/Batch_2/1_usb4.csv",
    #     "Turner_Data/Room_B/Batch_2/3_usb4.csv",
    #     "Turner_Data/Room_B/Batch_2/4_usb4.csv",
    # ),
    # (
    #     "Turner_Data/Room_B/Batch_3/0_usb0.csv",
    #     "Turner_Data/Room_B/Batch_3/1_usb0.csv",
    #     "Turner_Data/Room_B/Batch_3/3_usb0.csv",
    #     "Turner_Data/Room_B/Batch_3/4_usb0.csv",
    # ),
    (
        "Turner_Data/Room_B/Batch_3/0_usb1.csv",
        "Turner_Data/Room_B/Batch_3/1_usb1.csv",
        "Turner_Data/Room_B/Batch_3/3_usb1.csv",
        "Turner_Data/Room_B/Batch_3/4_usb1.csv",
    ),
    # (
    #     "Turner_Data/Room_B/Batch_3/0_usb2.csv",
    #     "Turner_Data/Room_B/Batch_3/1_usb2.csv",
    #     "Turner_Data/Room_B/Batch_3/3_usb2.csv",
    #     "Turner_Data/Room_B/Batch_3/4_usb2.csv",
    # ),
    # (
    #     "Turner_Data/Room_B/Batch_3/0_usb3.csv",
    #     "Turner_Data/Room_B/Batch_3/1_usb3.csv",
    #     "Turner_Data/Room_B/Batch_3/3_usb3.csv",
    #     "Turner_Data/Room_B/Batch_3/4_usb3.csv",
    # ),
    # (
    #     "Turner_Data/Room_B/Batch_3/0_usb4.csv",
    #     "Turner_Data/Room_B/Batch_3/1_usb4.csv",
    #     "Turner_Data/Room_B/Batch_3/3_usb4.csv",
    #     "Turner_Data/Room_B/Batch_3/4_usb4.csv",
    # ),
    # (
    #     "Turner_Data/Room_B/Batch_3/0_usb5.csv",
    #     "Turner_Data/Room_B/Batch_3/1_usb5.csv",
    #     "Turner_Data/Room_B/Batch_3/3_usb5.csv",
    #     "Turner_Data/Room_B/Batch_3/4_usb5.csv",
    # ),
    # (
    #     "data/0_Hexcast/File1_baseline.csv",
    #     "data/0_Hexcast/File2_gpt.csv",
    #     "data/0_Hexcast/File4_text.csv",
    # ),
    # (
    #     "data/1_Hiphop/File1_baseline.csv",
    #     "data/1_Hiphop/File2_text.csv",
    #     "data/1_Hiphop/File5_gpt.csv",
    # ),
    # (
    #     "data/2_Hiphop/File1_baseline.csv",
    #     "data/2_Hiphop/File2_gpt.csv",
    #     "data/2_Hiphop/File5_text.csv",
    # ),
    # (
    #     "data/3_Math/File1_baseline.csv",
    #     "data/3_Math/File2_text.csv",
    #     "data/3_Math/File5_gpt.csv",
    # ),
    # (
    #     "data/4_Hiphop/File1_baseline.csv",
    #     "data/4_Hiphop/File2_gpt.csv",
    #     "data/4_Hiphop/File5_text.csv",
    # ),
    # (
    #     "data/5_Hiphop/File1_baseline.csv",
    #     "data/5_Hiphop/File2_text.csv",
    #     "data/5_Hiphop/File5_gpt.csv",
    # ),
    # (
    #     "data/6_Math/File1_baseline.csv",
    #     "data/6_Math/File2_gpt.csv",
    #     "data/6_Math/File5_text.csv",
    # ),
)
participant_results = []

##################################
# We refactor the analysis function into multiple functions for ease of code.
##################################

def transform_EDA(df, duration):
    ################################
    # Transforming data
    # This should transform the data (more specifically, EDA data) into the appropriate units.
    # Conversion formula for EDA/mS: 1/((1-(2*((eda (raw)* 2.048) / 32767)/3.3))*825000)*1000000
    # Conversion formula for PPG/mV: ((PPG (raw) * 2048) /32767)
    # Conversion formula for Temp/deg C: temperature (raw)*2048/(32767)/10
    ################################

    df.loc[:, "EDA"] = 1 / ((1 - (2 * ((df.loc[:, "EDA"] * 2.048) / 32767) / 3.3)) * 825000) * 1000000
    df.loc[:, "PPG"] = (df.loc[:, "PPG"] * 2048) / 32767
    df.loc[:, "Temp"] = df.loc[:, "Temp"]* 2048 / 32767 / 10

    ################################
    # Cleaning data
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
    # Removal of Artifacts (UNUSED)
    # We use the works of Llanes-Jurado et al. (2023) to remove artifacts using an LSTM model.
    ################################

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

    # # WORKAROUND FOR COMMENTING THIS PART OUT
    # eda_corrected = eda_filtered

    return eda_corrected, new_sample_rate

def give_list_of_segmented_processed_EDA(df, duration, window_dur, overlap_dur):
    ################################
    # Transforming data
    # This should transform the data (more specifically, EDA data) into the appropriate units.
    # Conversion formula for EDA/mS: 1/((1-(2*((eda (raw)* 2.048) / 32767)/3.3))*825000)*1000000
    # Conversion formula for PPG/mV: ((PPG (raw) * 2048) /32767)
    # Conversion formula for Temp/deg C: temperature (raw)*2048/(32767)/10
    ################################

    df.loc[:, "EDA"] = 1 / ((1 - (2 * ((df.loc[:, "EDA"] * 2.048) / 32767) / 3.3)) * 825000) * 1000000
    df.loc[:, "PPG"] = (df.loc[:, "PPG"] * 2048) / 32767
    df.loc[:, "Temp"] = df.loc[:, "Temp"]* 2048 / 32767 / 10

    ################################
    # Cleaning data
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
    # Removal of Artifacts 
    # We use the works of Llanes-Jurado et al. (2023) to remove artifacts using an LSTM model.
    ################################

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

    # WORKAROUND FOR COMMENTING THIS PART OUT
    # eda_corrected = eda_filtered
    return get_segmented_list(pd.DataFrame({"EDA":eda_corrected}).EDA, duration, window_dur, overlap_dur), new_sample_rate


def give_eda_stats(eda_corrected, duration, new_sample_rate, stage_results):
    ################################
    # Calculation of NSSCR
    # We use the works of Taylor et al. (2015) to calculate the NSSCR.
    # We use an offset of 1, a start window of 4 seconds, an end window of 4 seconds, and a threshold of 0.05.
    ################################

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
            2, 4, 10, 0.00, 25
            )[0].sum()
    print("Peaks Per Minute:", peaks / (duration/60))
    stage_results.append(peaks / (duration/60))
    ################################
    # Calculations via cvxEDA
    # We use the works of Greco et al. (2015) to break the EDA signal to its components. 
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

    # print("Mean SCR", p.mean())
    stage_results.append(p.mean())

    # print("Mean SCL", t.mean())
    stage_results.append(t.mean())

    ################################
    # Calculation of TVSymp
    ################################

    tvsymp_signal = calculate_tvsymp_index(signal.resample(eda_corrected, 2*duration))

    # print("Mean TVSympt", tvsymp_signal.mean())
    stage_results.append(tvsymp_signal.mean())

    ################################
    # Calculation of EDASymp and EDASymp normalized (EDASympn)
    # We reimplemented the functions from the works of Posada-Quintero, H.F., Florian, J.P., Orjuela-Cañón, A.D. et al. Power Spectral Density Analysis of Electrodermal Activity for Sympathetic Function Assessment. Ann Biomed Eng 44, 3124–3135 (2016). https://doi.org/10.1007/s10439-016-1606-6
    # We also utilised some of the works of NeuroKit2
    # Function header: def calculate_edasymp_index(eda, samp_rate, nperseg = 128, freq_band = [0.045, 0.25]):

    # psd is power spectral density (np array)
    # edasymp is spectral dynamics in the frequency range of 0.045–.25 Hz -> which is basically integrated psd values within the frequency range (float)
    # edasympn is normalized edasymp via max psd value
    ################################

    #Downsampling corrected eda data from 25hz to around 2hz (2.008)
    down_eda = signal.decimate(eda_corrected, 12)

    [edasymp, edasympn, psd] = calculate_edasymp_index(down_eda, 2)

    # print("Edasymp_n: ", edasympn)
    stage_results.append(edasympn)


for participant in files_to_analyze:
    baseline_length = -1
    stage_results = []
    for file in participant:
        ################################
        # Importing raw data
        # Format: CSV File, with the columns Time, EDA, Heart Rate, Temperature
        # The first row of the CSV data (after the title headers) should have the duration of the session in seconds in the Time column
        # For example, 
        # ```
        # Time,EDA,PPG,Temp
        # 600,,,
        # 1721806894825084,19938.0,22925.0,5745.0
        # ```
        ################################

        print("Analyzing ", file)

        df = pd.read_csv(file)
        duration = df.iloc[0, 0]
        df = df.iloc[1:]

        if baseline_length == -1: baseline_length = duration # Assuming the first item in the list is always the baseline

        print("Loaded dataframe! Duration of experiment:", duration)

        # # Windowing Parameters
        window_duration = baseline_length
        overlap_duration = baseline_length/2

        # eda_data, sample_rate = transform_EDA(df, duration)
        # give_eda_stats(eda_data, duration, sample_rate, stage_results)

        # Segmentation
        listo, samp = give_list_of_segmented_processed_EDA(df, duration, window_duration, overlap_duration)


        for l in listo:
            give_eda_stats(l, window_duration, samp, stage_results)

    participant_results.append(stage_results)

print(participant_results)

result_df = pd.DataFrame(columns=list(range(len(participant_results[0]))))
for row in participant_results:
    result_df = pd.concat([result_df, pd.DataFrame([row], columns=result_df.columns)], ignore_index=True)

result_df.to_csv("results.csv")