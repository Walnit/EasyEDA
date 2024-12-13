"""
Implementation of EDASympn following:
Posada-Quintero HF, Florian JP, Orjuela-Cañón AD, Aljama-Corrales T, Charleston-Villalobos S, Chon KH. Power Spectral Density Analysis of Electrodermal Activity for Sympathetic Function Assessment. Ann Biomed Eng. 2016 Oct;44(10):3124-3135. doi: 10.1007/s10439-016-1606-6. Epub 2016 Apr 8. PMID: 27059225.
"""

import numpy as np
import pandas as pd
from scipy.signal import welch, get_window



def calculate_edasymp_index(eda, samp_rate, nperseg = 128, freq_band = [0.045, 0.25]):
    noverlap = nperseg / 2

    freq, pow = welch(eda, fs=samp_rate, window="blackman", nperseg=nperseg, noverlap=noverlap)
    psd = pd.DataFrame({"Frequency": freq, "Power": pow})
    print(psd)

    #calculate EDASymp
    where = (psd["Frequency"] >= freq_band[0]) & (psd["Frequency"] < freq_band[1])
    eda_symp = np.trapz(y=psd["Power"][where], x=psd["Frequency"][where])

    #normalised psd to get EDASympn
    psd["Power"] /= np.max(psd["Power"])
    where = (psd["Frequency"] >= freq_band[0]) & (psd["Frequency"] < freq_band[1])
    eda_sympn = np.trapz(y=psd["Power"][where], x=psd["Frequency"][where])

    return eda_symp, eda_sympn, psd
