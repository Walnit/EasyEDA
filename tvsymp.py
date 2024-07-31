import numpy as np
from scipy import signal
from scipy.signal import coherence, hilbert, medfilt, resample, butter, sosfilt, periodogram

# Define spectral frequency parameters
SPECTRAL_FREQ_RADIUS = 0.04
SPECTRAL_FREQ = [0.04, 0.12, 0.2, 0.28, 0.36, 0.44, 0.52, 0.6, 0.68, 0.76, 0.84, 0.92]
SPECTRAL_FREQ_NORMALIZED = np.array(SPECTRAL_FREQ) / 2

# Define filter coefficients
FILTER_COEFF_B = [0.000390127271227124, 0.000405459060155711, 0.000420493677022170, 0.000434787534021812, 0.000447420013756411, 0.000456990976553924, 0.000461640486657997, 0.000459091256384998, 0.000446713577198476, 0.000421611760493414, 0.000380730370430932, 0.000320977818426799, 0.000239364225554114, 0.000133149865900841,- 2.87028701691325e-19,- 0.000161858490128205,- 0.000353483529614734,- 0.000575072353133937,- 0.000825829096021346,- 0.00110385035002714,- 0.00140603327645369,- 0.00172801044603661,- 0.00206411501916451,- 0.00240737920357959,- 0.00274956814482856,- 0.00308125053593767,- 0.00339190629834381,- 0.00367007070955053,- 0.00390351335946665,- 0.00407944933309568,- 0.00418477906864838,- 0.00420635245332026,- 0.00413125191887007,- 0.00394708860891930,- 0.00364230513028529,- 0.00320647798934759,- 0.00263061256556294,- 0.00190742339794200,- 0.00103159266242848, 1.08166872624911e-18, 0.00118808268636128, 0.00253083722080551, 0.00402379090177735, 0.00565975805643438, 0.00742883440961241, 0.00931844613714829, 0.0113134542668271, 0.0133963138432051, 0.0155472860150618, 0.0177446999668954, 0.0199652604291916, 0.0221843953957702, 0.0243766376782397, 0.0265160330629930, 0.0285765671276791, 0.0305326022404599, 0.0323593159212027, 0.0340331315991056, 0.0355321328612793, 0.0368364525516934, 0.0379286285447783, 0.0387939186730359, 0.0394205681187585, 0.0398000235674744, 0.0399270895422646, 0.0398000235674744, 0.0394205681187585, 0.0387939186730359, 0.0379286285447783, 0.0368364525516934, 0.0355321328612793, 0.0340331315991056, 0.0323593159212027, 0.0305326022404599, 0.0285765671276791, 0.0265160330629930, 0.0243766376782397, 0.0221843953957702, 0.0199652604291916, 0.0177446999668954, 0.0155472860150618, 0.0133963138432051, 0.0113134542668271, 0.00931844613714829, 0.00742883440961241, 0.00565975805643438, 0.00402379090177735, 0.00253083722080551, 0.00118808268636128, 1.08166872624911e-18,- 0.00103159266242848,- 0.00190742339794200,- 0.00263061256556294,- 0.00320647798934759,- 0.00364230513028529,- 0.00394708860891930,- 0.00413125191887007,- 0.00420635245332026,- 0.00418477906864838,- 0.00407944933309568,- 0.00390351335946665,- 0.00367007070955053,- 0.00339190629834381,- 0.00308125053593767,- 0.00274956814482856,- 0.00240737920357959,- 0.00206411501916451,- 0.00172801044603661,- 0.00140603327645369,- 0.00110385035002714,- 0.000825829096021346,- 0.000575072353133937,- 0.000353483529614734,- 0.000161858490128205,- 2.87028701691325e-19, 0.000133149865900841, 0.000239364225554114, 0.000320977818426799, 0.000380730370430932, 0.000421611760493414, 0.000446713577198476, 0.000459091256384998, 0.000461640486657997, 0.000456990976553924, 0.000447420013756411, 0.000434787534021812, 0.000420493677022170, 0.000405459060155711, 0.000390127271227124]  # Filter coefficients for the lfilter function
FILTER_COEFF_B2 = b2 =[0.9597822300872402845328679177328012883663177490234375, -3.83912892034896113813147167093120515346527099609375,   5.7586933805234412631079976563341915607452392578125,  -3.83912892034896113813147167093120515346527099609375,   0.9597822300872402845328679177328012883663177490234375]
FILTER_COEFF_A2 = [1.000000000000000,  -3.9179078653919905406155521632172167301177978515625,   5.7570763791180770141409084317274391651153564453125,  -3.7603495076945367969756262027658522129058837890625,   0.921181929191239756704590035951696336269378662109375]

def apply_band_pass_filter(signal_data, modulation_freq):
    """
    Applies a band-pass filter to the input signal_data using the modulation frequency modulation_freq.
    """
    cutoff_freq = 0.001  # Cutoff frequency as a fraction of the sampling rate
    transition_band = 0.08  # Transition band, as a fraction of the sampling rate

    filter_length = int(np.ceil((4 / transition_band)))
    filter_length = 129
    if not filter_length % 2: 
        filter_length += 1  # Ensure filter_length is odd
    filter_indices = np.arange(filter_length)

    # Compute sinc filter and apply Blackman window
    sinc_filter = np.sinc(2 * cutoff_freq * (filter_indices - (filter_length - 1) / 2))
    blackman_window = np.blackman(filter_length)
    sinc_filter = sinc_filter * blackman_window
    sinc_filter = sinc_filter / np.sum(sinc_filter)

    # Modify filter coefficients
    max_filter_coeff_b = max(FILTER_COEFF_B)
    modified_filter_coeff = signal.windows.flattop(129, sym=True) / np.sinc(2 * cutoff_freq * (filter_indices - (filter_length - 1) / 2))
    modulation_factor = (max_filter_coeff_b - FILTER_COEFF_B[0]) / max(modified_filter_coeff)
    modified_filter_coeff = modified_filter_coeff * modulation_factor + FILTER_COEFF_B[0]

    signal_length = len(signal_data)
    phase_accum = 0
    modulated_signal_cos = np.empty(signal_length)
    modulated_signal_sin = np.empty(signal_length)

    # Modulate the input signal_data
    for i in range(signal_length):
        phase_accum += modulation_freq[i]
        modulated_signal_cos[i] = (signal_data[i] * 2 * np.cos(-2 * np.pi * phase_accum))
        modulated_signal_sin[i] = (signal_data[i] * 2 * np.sin(-2 * np.pi * phase_accum))

    # Pad the signal
    padding_size = round((len(FILTER_COEFF_B) - 1) / 2)
    padded_cos_signal = np.concatenate((np.zeros(padding_size), modulated_signal_cos, np.zeros(padding_size)), axis=None)
    padded_sin_signal = np.concatenate((np.zeros(padding_size), modulated_signal_sin, np.zeros(padding_size)), axis=None)

    # Apply the filter to the modulated signals
    filtered_cos_signal = signal.lfilter(FILTER_COEFF_B, 1, padded_cos_signal)
    filtered_sin_signal = signal.lfilter(FILTER_COEFF_B, 1, padded_sin_signal)

    filtered_cos_signal = filtered_cos_signal[padding_size: signal_length + padding_size]
    filtered_sin_signal = filtered_sin_signal[padding_size: signal_length + padding_size]

    phase = np.zeros(signal_length)
    demodulated_signal = np.zeros(signal_length)
    amplitude_envelope = np.sqrt(np.square(filtered_cos_signal) + np.square(filtered_sin_signal))

    # Compute phase and demodulated signal
    for i in range(signal_length):
        if filtered_cos_signal[i] == 0:
            phase[i] = np.pi / 2
        elif filtered_cos_signal[i] > 0:
            phase[i] = np.arctan(filtered_sin_signal[i] / filtered_cos_signal[i])
        else:
            phase[i] = np.arctan(filtered_sin_signal[i] / filtered_cos_signal[i]) + np.pi

    for j in range(signal_length):
        demodulated_signal[j] = amplitude_envelope[j] * np.cos(2 * np.pi * modulation_freq[-1] * (j + 1) + phase[j])
    return demodulated_signal

def calculate_tvsymp_index(signal_data):
    """
    Calculate the TVSymp index for the input signal_data.
    """
    signal_data = signal.filtfilt(FILTER_COEFF_B2, FILTER_COEFF_A2, signal_data, padtype="odd")  # Apply a band-pass filter

    constant_array = np.ones((len(signal_data), 1))
    demodulated_signal_1 = apply_band_pass_filter(signal_data, SPECTRAL_FREQ_NORMALIZED[0] * constant_array)
    demodulated_signal_2 = apply_band_pass_filter(signal_data, SPECTRAL_FREQ_NORMALIZED[1] * constant_array)
    demodulated_signal_3 = apply_band_pass_filter(signal_data, SPECTRAL_FREQ_NORMALIZED[2] * constant_array)
    demodulated_signal_4 = apply_band_pass_filter(signal_data, SPECTRAL_FREQ_NORMALIZED[3] * constant_array)
    demodulated_signal_5 = apply_band_pass_filter(signal_data, SPECTRAL_FREQ_NORMALIZED[4] * constant_array)
    demodulated_signal_6 = apply_band_pass_filter(signal_data, SPECTRAL_FREQ_NORMALIZED[5] * constant_array)
    demodulated_signal_7 = apply_band_pass_filter(signal_data, SPECTRAL_FREQ_NORMALIZED[6] * constant_array)

    # Sum the spectral components and normalize
    combined_signal = demodulated_signal_2 + demodulated_signal_3
    combined_signal = combined_signal / np.std(combined_signal)

    # Compute the Hilbert transform and return the absolute value
    tvsymp_index = np.abs(hilbert(combined_signal))
    return tvsymp_index

# signal_data is assumed to be sampled at 2 Hz and has undergone noise filtering
