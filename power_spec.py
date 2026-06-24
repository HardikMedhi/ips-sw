import numpy as np
from scipy import signal

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

def uniform_statistical_averaging(freqs: np.ndarray, psd: np.ndarray, k:float =4):
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