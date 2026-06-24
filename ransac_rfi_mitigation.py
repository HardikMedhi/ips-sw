import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_laplace
from sklearn.linear_model import RANSACRegressor, LinearRegression

# ==========================================
# 1. Core Algorithm Components
# ==========================================

def apply_log_filter(spectrum: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """
    Step 2: Convolves the spectrum with a 1D Laplacian of Gaussian (LoG) filter.
    This removes the complex standing-wave baseline and highlights sharp RFI spikes 
    based on their rate of change.
    """
    # Note: gaussian_laplace computes the 2nd derivative. We invert it so spikes are positive.
    return -gaussian_laplace(spectrum, sigma=sigma)

def detect_outliers_ransac(filtered_spectrum: np.ndarray, x_indices: np.ndarray, residual_threshold: float = 3.0) -> np.ndarray:
    """
    Step 3: Fits a robust line model to the LoG-filtered spectrum using RANSAC.
    Identifies 'inner data' (clean noise floor) and 'outer data' (RFI spikes).
    """
    # RANSAC requires 2D arrays for features (X)
    X = x_indices.reshape(-1, 1)
    
    # Initialize RANSAC with a simple linear model
    ransac = RANSACRegressor(
        estimator=LinearRegression(),
        residual_threshold=residual_threshold,
        random_state=42
    )
    
    # Fit the model to the filtered data
    ransac.fit(X, filtered_spectrum)
    
    # The boolean mask of outliers (True = RFI spike)
    outlier_mask = ~ransac.inlier_mask_
    return outlier_mask

def interpolate_spectrum(original_spectrum: np.ndarray, x_indices: np.ndarray, outlier_mask: np.ndarray) -> np.ndarray:
    """
    Step 4: Performs average interpolation for the 'outer data' (RFI) 
    using the surrounding valid 'inner data'.
    """
    inlier_mask = ~outlier_mask
    
    # Edge cases: no outliers or all outliers
    if not np.any(outlier_mask) or not np.any(inlier_mask):
        return original_spectrum.copy()
        
    # Interpolate the rejected points using valid neighbors
    interpolated_values = np.interp(
        x_indices[outlier_mask], 
        x_indices[inlier_mask], 
        original_spectrum[inlier_mask]
    )
    
    cleaned_spectrum = original_spectrum.copy()
    cleaned_spectrum[outlier_mask] = interpolated_values
    return cleaned_spectrum

# ==========================================
# 2. Main Iterative Pipeline
# ==========================================

def iterative_rfi_mitigation(spectrum: np.ndarray, frequencies: np.ndarray, 
                             log_sigma: float = 1.5, ransac_thresh: float = 5.0, 
                             max_iter: int = 10, tol: float = 1e-3):
    """
    Orchestrates the FAST RFI mitigation pipeline iteratively until RMS convergence.
    """
    current_spectrum = spectrum.copy()
    global_outlier_mask = np.zeros(len(spectrum), dtype=bool)
    prev_rms = np.std(current_spectrum)
    
    for iteration in range(1, max_iter + 1):
        # 1. Apply LoG Filter to current iteration's spectrum
        log_spectrum = apply_log_filter(current_spectrum, sigma=log_sigma)
        
        # 2. Detect anomalies using RANSAC
        current_outliers = detect_outliers_ransac(log_spectrum, frequencies, residual_threshold=ransac_thresh)
        
        # Accumulate the detected outliers
        global_outlier_mask |= current_outliers
        
        # 3. Interpolate the original spectrum using the accumulated valid data
        current_spectrum = interpolate_spectrum(spectrum, frequencies, global_outlier_mask)
        
        # 4. Check for RMS convergence (Step 5 of the paper)
        current_rms = np.std(current_spectrum)
        rms_diff = abs(prev_rms - current_rms)
        
        if rms_diff < tol:
            print(f"Converged after {iteration} iterations (RMS diff: {rms_diff:.5f}).")
            break
            
        prev_rms = current_rms
        
    return current_spectrum, global_outlier_mask, log_spectrum

# ==========================================
# 3. Plotting & Visualization
# ==========================================

def plot_rfi_mitigation(frequencies, original, cleaned, log_filtered, outlier_mask):
    """Generates a visualization mimicking Figure 2 in the paper, plus an explanatory plot."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # --- Top Plot: Original vs Cleaned Spectrum (Recreating Fig 2a) ---
    ax1.plot(frequencies, original, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='Raw Data (with RFI)')
    ax1.plot(frequencies, cleaned, color='black', linewidth=1.5, label='Cleaned Data (Interpolated)')
    
    # Highlight the specific points that were flagged as RFI
    ax1.scatter(frequencies[outlier_mask], original[outlier_mask], color='red', s=10, zorder=5, label='Detected RFI Spikes')
    
    ax1.set_title("FAST UWB Spectrum: RFI Mitigation (Recreating Fig 2a)")
    ax1.set_ylabel("Power (arbitrary units)")
    ax1.legend()
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    # --- Bottom Plot: The LoG Filter Mechanism ---
    ax2.plot(frequencies, log_filtered, color='teal', linewidth=1, label='LoG Filtered (Rate of Change)')
    ax2.axhline(0, color='black', linewidth=1)
    
    ax2.set_title("Behind the Scenes: Laplacian of Gaussian (LoG) Domain")
    ax2.set_xlabel("Frequency (MHz)")
    ax2.set_ylabel("Rate of Change Magnitude")
    ax2.legend()
    ax2.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.show()

# ==========================================
# 4. Mock Data Generation & Execution
# ==========================================

def simulate_fast_spectrum(n_points=1000):
    """Generates a mock FAST spectrum with a wavy baseline, noise, and RFI spikes."""
    np.random.seed(42)
    frequencies = np.linspace(270, 800, n_points)
    
    # Create a complex, wavy baseline simulating standing waves
    baseline = 50 + 15 * np.sin(2 * np.pi * frequencies / 120) + 10 * np.sin(2 * np.pi * frequencies / 60)
    
    # Add Gaussian white noise (the Signal of Interest / natural emission)
    noise = np.random.normal(0, 1.5, n_points)
    spectrum = baseline + noise
    
    # Inject narrow-band RFI spikes randomly
    rfi_indices = np.random.choice(n_points, size=25, replace=False)
    spectrum[rfi_indices] += np.random.uniform(40, 120, size=len(rfi_indices))
    
    # Inject a few slightly wider RFI spikes (spilling over 2-3 channels)
    wide_rfi = np.random.choice(n_points-2, size=5, replace=False)
    for idx in wide_rfi:
        spectrum[idx:idx+3] += np.random.uniform(30, 80)
        
    return frequencies, spectrum

if __name__ == "__main__":
    # 1. Generate the mock FAST wideband spectrum
    freqs, raw_spectrum = simulate_fast_spectrum(n_points=1500)
    
    # 2. Run the Iterative LoG + RANSAC mitigation pipeline
    print("Starting FAST iterative RFI mitigation...")
    cleaned_spec, rfi_mask, log_spec = iterative_rfi_mitigation(
        spectrum=raw_spectrum, 
        frequencies=freqs,
        log_sigma=1.5,         # Tune based on expected width of RFI
        ransac_thresh=3.0,     # RANSAC residual tolerance in the LoG domain
        max_iter=15, 
        tol=0.01
    )
    
    print(f"Total Channels Flagged as RFI: {np.sum(rfi_mask)} out of {len(raw_spectrum)}")
    
    # 3. Visualize the results
    plot_rfi_mitigation(freqs, raw_spectrum, cleaned_spec, log_spec, rfi_mask)