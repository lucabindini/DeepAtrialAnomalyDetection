import numpy as np
from scipy.signal import find_peaks


def duration_openep(bipolar):
    # Calculate EGM energy within the window of concern
    energy = np.square(bipolar)

    # Find threshold values within this window
    energy_threshold = energy.max()*0.1

    i_qrs = np.where(energy > energy_threshold)

    return len(i_qrs[0])

def num_peaks_openep(bipolar, height=0.1):
    return len(find_peaks(np.square(bipolar), height=height*np.square(bipolar).max())[0])

def uni_slope_annotation_time(unipolar):
    return round(np.gradient(unipolar)[int(len(unipolar)/2)],3)

def bip_slope_annotation_time(bipolar):
    return round(np.min(np.gradient(bipolar)),3)

def integral_EGM(bipolar): 
    return round(np.abs(bipolar).sum()/(np.max(bipolar) - np.min(bipolar)),3)