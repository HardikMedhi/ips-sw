import re
import numpy as np
from pathlib import Path
from scipy import signal

import ips_sw.utils.file_utils as fut

def get_time_freqs(data: np.ndarray, tsampl: float, nchan: int, chan_bw: float, freq_start: float) -> tuple[np.array, np.array]:
    """Build the time and frequency axes for a reshaped dynamic spectrum."""
    nsamples = data.shape[0]
    # Time increases by the sample interval; frequency steps by channel width.
    time_samples = np.arange(nsamples) * tsampl
    freq_channels = freq_start + np.arange(nchan) * chan_bw

    return time_samples, freq_channels


def get_data_matrix(file_path: Path) -> np.ndarray:
    """Read a supported file and reshape its data into samples x channels."""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Detect file type and read it through the shared file utilities.
    file_type = fut.get_file_type(file_path)    
    header, data = fut.read_filbank(file_path) if file_type == 'filterbank' else fut.read_fits(file_path)

    nchan = header.nchans
    reshaped_data = data.reshape(-1, nchan)
    return reshaped_data

def get_sub_matrix_freq(matrix: np.ndarray, f1:float, f2:float, freq_channels: np.ndarray) -> tuple[np.ndarray, np.array]:
    """Extract a frequency sub-band from the full matrix and channel axis."""
    if f1 is None and f2 is None:
        print("No f1 and f2 provided!")
        return matrix, freq_channels

    # Use the data's native frequency range as the reference bounds.
    f_start = freq_channels[0]
    f_end = freq_channels[-1]

    # Clamp the requested frequencies to the available band.
    freq1 = min(f1, f_start) if f1 is not None else f_start
    freq2 = max(f2, f_end) if f2 is not None else f_end

    # Keep only the channels inside the selected band.
    freq_mask = (freq_channels <= freq1) & (freq_channels >= freq2)
    sub_matrix = matrix[:, freq_mask]
    sub_freq_channels = freq_channels[freq_mask]
    
    print(f"  Frequency range: {freq1:.2f} - {freq2:.2f} MHz")
    print(f"  Filtered data shape: {sub_matrix.shape} (samples × channels)")

    return sub_matrix, sub_freq_channels

def get_sub_matrix_time(matrix: np.ndarray, t1:float, t2:float, time_samples: np.ndarray) -> tuple[np.ndarray, np.array]:
    """Extract a time sub-band from the full matrix and channel axis."""
    if t1 is None and t2 is None:
        print("No t1 and t2 provided!")
        return matrix, time_samples

    # Use the data's native time range as the reference bounds.
    t_start = time_samples[0]
    t_end = time_samples[-1]

    # Clamp the requested times to the available band.
    time1 = max(t1, t_start) if t1 is not None else t_start
    time2 = min(t2, t_end) if t2 is not None else t_end

    # Keep only the samples inside the selected band.
    time_mask = (time1 <= time_samples) & (time_samples <= time2)
    sub_matrix = matrix[time_mask, :]
    sub_time_samples = time_samples[time_mask]
    
    print(f"  Time range: {time1:.2f} - {time2:.2f} s")
    print(f"  Filtered data shape: {sub_matrix.shape} (samples × channels)")

    return sub_matrix, sub_time_samples

def get_median_bandpass(matrix: np.ndarray) -> np.array:
    """Return the per-channel median across time for bandpass correction."""

    # Median across time (axis=0) gives one bandpass value per channel.
    bandpass = np.median(matrix, axis=0)
    return bandpass

def bp_norm_matrix(matrix:np.ndarray, bandpass:np.ndarray=None) -> np.ndarray:
    """Normalize each channel by dividing by its bandpass.

    Zero bandpass values are converted to NaN so the corresponding
    normalized samples also become NaN.
    """

    # Use the median bandpass unless the caller supplies one explicitly.
    bandpass = get_median_bandpass(matrix) if bandpass is None else bandpass

    # Replace zero bandpass values with NaN so divisions yield NaN.
    bandpass_safe = bandpass.astype(float).copy()
    bandpass_safe[bandpass_safe == 0] = np.nan

    # Divide channel values by their bandpass estimate.
    bpnorm_matrix = matrix / bandpass_safe

    return bpnorm_matrix

def remove_running_median(time_series: np.ndarray, sampling_rate: float, window_duration:float=10):
    """
    Remove running median from time series.
    
    Parameters:
    -----------
    time_series : ndarray
        Input time series
    sampling_rate : float
        Sampling rate in Hz
    window_duration : float
        Duration of running median window in seconds (default: 10)
        
    Returns:
    --------
    detrended : ndarray
        Time series with running median removed
    """
    window_size = int(window_duration * sampling_rate)
    # Ensure odd window size for medfilt
    if window_size % 2 == 0:
        window_size += 1
    
    running_median = signal.medfilt(time_series, kernel_size=window_size)
    detrended = time_series - running_median
    
    return detrended

def despike(time_series:np.ndarray, kernel_size:float=3, threshold:float=6):
    """
    Remove narrow spikes using a MAD filter and a given threshold.
    This is used to remove 1PPS narrow pulses from the offsource region
    to get better RMS noise estimates.
    
    Parameters:
    -----------
    time_series : ndarray
        Input time series
    kernel_size : int
        Size of the median filter kernel (default: 3)
    threshold : float
        Threshold for spike detection in units of MAD (default: 3.5)
        
    Returns:
    --------
    data : ndarray
        Despicked time series
    """
    data = time_series.copy()  # Don't modify input array
    
    med_filtered = signal.medfilt(data, kernel_size=kernel_size)
    diff = data - med_filtered
    mad = np.median(np.abs(diff)) / 0.6744897501960817
    mask = np.abs(diff) > (threshold * mad)
    data[mask] = med_filtered[mask]
    
    return data

def get_coords_colnames(col_names:list) -> tuple[str, str]:
    ra_pattern = re.compile(r'^ra(?:[_ -]?j2000)?$', re.IGNORECASE)
    dec_pattern = re.compile(r'^dec(?:[_ -]?j2000)?$', re.IGNORECASE)
    
    ra_col = next((col for col in col_names if ra_pattern.match(col)), None)
    dec_col = next((col for col in col_names if dec_pattern.match(col)), None)
    
    if ra_col is None or dec_col is None:
        raise ValueError(
            f"Could not find required RA/Dec columns in the catalog.\n"
            f"Available columns: {list(col_names)}\n"
            f"Expected RA pattern: {ra_pattern.pattern}\n"
            f"Expected Dec pattern: {dec_pattern.pattern}"
        )
    
    return ra_col, dec_col