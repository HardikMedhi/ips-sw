import numpy as np
from scipy import signal

import data_process as dpr
from filterbank import Filterbank

def compute_power_spectrum(time_series: np.ndarray, sampling_rate: float, chunk_duration:float =30):
    """
    Compute power spectrum using Welch's method.
    
    Parameters:
    -----------
    time_series : ndarray
        Input time series
    sampling_rate : float
        Sampling rate in Hz
    chunk_duration : float
        Duration of each chunk in seconds (default: 30)
        
    Returns:
    --------
    freqs : ndarray
        Frequency array
    psd : ndarray
        Power spectral density
    """
    nperseg = int(chunk_duration * sampling_rate)
    noverlap = nperseg // 2

    freqs, psd = signal.welch(
        time_series,
        fs=sampling_rate,
        scaling='density',
        nperseg=nperseg,
        noverlap=noverlap,

    )

    return freqs, psd

def uniform_statistical_averaging(freqs: np.ndarray, psd: np.ndarray, k:float=4):
    """
    Implements the binning procedure from the text.
    Returns a reduced number of points with constant fractional error.
    
    Parameters:
    -----------
    freqs : ndarray
        Frequency array
    psd : ndarray
        Power spectral density array
    k : float
        Binning parameter (default: 4)
        
    Returns:
    --------
    out_freqs : ndarray
        Binned frequency array
    out_psd : ndarray
        Binned PSD array
    N_vals : ndarray
        Number of points in each bin
    """
    out_freqs = []
    out_psd = []
    N_vals = []
    
    current_idx = 0 
    n_total = len(psd)
    
    while current_idx < n_total:
        f_current = freqs[current_idx]

        # Relation: N_avg = k * nu (eq 4.12)
        # Constraint: Minimum of 4 points
        N = int(np.round(k * f_current))
        N = max(N, 4)

        # Define the bin range
        end_idx = current_idx + N
        if end_idx > n_total:
            break  # Stop if we don't have enough points left for a full bin

        # Average the frequencies and the PSD values in this bin
        out_freqs.append(freqs[current_idx:end_idx].mean())
        out_psd.append(psd[current_idx:end_idx].mean())
        N_vals.append(N)

        # JUMP to the start of the next bin (Decimation)
        current_idx = end_idx

    return np.array(out_freqs), np.array(out_psd), np.array(N_vals)

def get_ps(fb_obj: Filterbank, 
           f1:float=None, f2:float=None, 
           t1:float=None, t2:float=None,
           bpnorm:bool=False,
           nodetrend:bool=False, nodespike:bool=False):
    matrix = fb_obj.matrix

    if f1 is not None or f2 is not None:
        matrix, _ = dpr.get_sub_matrix_freq(matrix, f1, f2, fb_obj.freq_channels)
    if t1 is not None or t2 is not None:
        matrix, _ = dpr.get_sub_matrix_time(matrix, t1, t2, fb_obj.time_samples)

    if bpnorm:
        matrix = dpr.bp_norm_matrix(matrix)

    sampling_rate = 1 / fb_obj.header.tsamp
    time_profile = np.nanmean(matrix, axis=1)

    if not nodetrend:
        window_size = 10 #s
        time_profile = dpr.remove_running_median(time_profile, sampling_rate, window_size)

    if not nodespike:
        kernel, threshold = 3, 6
        time_profile = dpr.despike(time_profile, kernel, threshold)

    freqs, psd = compute_power_spectrum(time_profile, sampling_rate)

    return freqs, psd