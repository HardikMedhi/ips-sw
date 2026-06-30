import numpy as np
from scipy.integrate import trapezoid

from filterbank import Filterbank
import data_process as dpr

def get_m_ts(onsrc:np.ndarray, offsrc:np.ndarray, defln:float=None):
    """
    Calculate scintillation index from time series.
    
    Parameters:
    -----------
    onsrc : ndarray
        On-source time series
    offsrc : ndarray or list of ndarray
        Off-source time series (or list of time series to be averaged)
        
    Returns:
    --------
    m : float
        Scintillation index from time series
    """
    # 1. Basic Stats for ON
    m_on = np.mean(onsrc)
    v_on = np.var(onsrc)
    
    # 2. Basic Stats for OFF (handle multiple time series)
    m_off = np.mean(offsrc)
    v_off = np.var(offsrc)
    
    # 3. Scintillation Index Calculation
    defln = (m_on - m_off) if defln is None else defln
    vardif = (v_on - v_off)
    
    m = np.sqrt(vardif) / defln
    return m

def get_m_ps(psd:np.ndarray, freqs:np.ndarray, fc:float,
               onsrc_ts:np.ndarray=None, offsrc_ts:np.ndarray=None, defln:float=None):
    """
    Calculate scintillation index from power spectrum.
    
    Parameters:
    -----------
    P_arr : ndarray
        Scintillation power spectrum
    f_arr : ndarray
        Frequency array
    fc : float
        Crossover frequency
    onsrc : ndarray
        On-source time series
    offsrc : ndarray or list of ndarray
        Off-source time series (or list of time series to be averaged)
        
    Returns:
    --------
    m : float
        Scintillation index from power spectrum
    """
    if (onsrc_ts is None or offsrc_ts is None) and defln is None:
        print("No time series or deflection value provided.")
        return np.nan

    mask = (freqs >= 0) & (freqs <= fc)

    # Extract the relevant data
    f_integration = freqs[mask]
    P_integration = psd[mask]

    # Perform numerical integration using trapezoidal rule
    integrated_power = trapezoid(P_integration, f_integration)

    onsrc_mu = np.mean(onsrc_ts)
    offsrc_mu = np.mean(offsrc_ts)
    mean_intensity = onsrc_mu - offsrc_mu if defln is None else defln

    m = np.sqrt(integrated_power) / mean_intensity

    return m

def get_m_vals(onsrc_fb: Filterbank, offsrc_fb: Filterbank,
               scint_psd: np.ndarray, scint_freqs: np.ndarray,
               crossover_freq:float):
    
    onsrc_median_bp = dpr.get_median_bandpass(onsrc_fb.matrix)
    offsrc_median_bp = dpr.get_median_bandpass(offsrc_fb.matrix)
    defln = onsrc_median_bp - offsrc_median_bp

    # m from PS
    m_ps = get_m_ps(scint_psd, scint_freqs, crossover_freq, defln=defln)

    # m from TS
    onsrc_ts_bpnorm = np.mean(dpr.bp_norm_matrix(onsrc_fb.matrix), axis=1)
    offsrc_ts_bpnorm = np.mean(dpr.bp_norm_matrix(offsrc_fb.matrix), axis=1)

    m_ts = get_m_ts(onsrc_ts_bpnorm, offsrc_ts_bpnorm, defln=defln)

    return m_ps, m_ts

