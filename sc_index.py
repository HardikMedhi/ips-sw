import numpy as np
from scipy.integrate import trapezoid

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
               onsrc_ts:np.ndarray, offsrc_ts:np.ndarray, defln:float=None):
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