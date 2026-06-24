import numpy as np
from scipy.stats import norm, linregress
import matplotlib.pyplot as plt
from filterbank import Filterbank
from pathlib import Path

# ==========================================
# 1. Core Algorithm Components
# ==========================================

def get_initial_parameters(x: np.ndarray, sample_ratio: float) -> tuple:
    """Randomly subsamples the data to estimate initial Gaussian parameters."""
    N = len(x)
    s = int(sample_ratio * N)
    S_indices = np.random.choice(N, size=s, replace=False)
    S = x[S_indices]
    
    mu_0 = np.mean(S)
    sigma_0 = np.std(S, ddof=1)
    
    return mu_0, sigma_0

def get_preliminary_inliers(x: np.ndarray, mu: float, sigma: float, z_thr: float) -> np.ndarray:
    """Identifies preliminary inliers based on a standard Z-score threshold."""
    if sigma == 0:
        return np.array([], dtype=int)
        
    z = (x - mu) / sigma
    return np.where(np.abs(z) <= z_thr)[0]

def refine_inliers_via_qq(x: np.ndarray, inlier_indices: np.ndarray, q: float) -> np.ndarray:
    """Filters preliminary inliers using a Q-Q plot Gaussian consistency test."""
    if len(inlier_indices) < 2:
        return np.array([], dtype=int)

    # Extract and sort the preliminary inliers
    inlier_values = x[inlier_indices]
    sort_order = np.argsort(inlier_values)
    y = inlier_values[sort_order]
    original_sorted_indices = inlier_indices[sort_order]
    
    # Calculate standard normal quantiles (z_q)
    ranks = np.arange(1, len(y) + 1)
    z_q = norm.ppf((ranks - 0.5) / len(y))
    
    # Linear fit: y_i = a + b * z_q
    b, a = np.polyfit(z_q, y, 1)
    
    # Calculate residuals and their robust scale (MAD)
    r = y - (a + b * z_q)
    median_r = np.median(r)
    mad_r = 1.4826 * np.median(np.abs(r - median_r))
    
    if mad_r == 0:
        return np.array([], dtype=int)
        
    # Apply 3-sigma rule on residuals
    valid_mask = np.abs(r) <= (q * mad_r)
    return original_sorted_indices[valid_mask]

# ==========================================
# 2. Main RANSAC Routine
# ==========================================

def gaussian_ransac_rfi_detection(x: np.ndarray, num_iterations=2500, sample_ratio=0.15, z_thr=3.0, q=3.0, k=3.0):
    """
    Main loop for the Gaussian-RANSAC RFI detection algorithm.
    """
    x = np.asarray(x)
    best_inlier_count = 0
    best_inlier_indices = np.array([], dtype=int)

    for _ in range(num_iterations):
        # 1. Subsample and estimate
        mu_0, sigma_0 = get_initial_parameters(x, sample_ratio)
        
        # 2. Find preliminary inliers
        I_1_indices = get_preliminary_inliers(x, mu_0, sigma_0, z_thr)
        
        # 3. Refine via Gaussian consistency check
        I_2_indices = refine_inliers_via_qq(x, I_1_indices, q)
        
        # 4. Update the best model
        if len(I_2_indices) > best_inlier_count:
            best_inlier_count = len(I_2_indices)
            best_inlier_indices = I_2_indices

    # 5. Final statistical evaluation and masking
    if best_inlier_count == 0:
        return np.zeros(len(x), dtype=bool), np.mean(x), np.std(x, ddof=1)
        
    final_inliers = x[best_inlier_indices]
    mu_hat = np.mean(final_inliers)
    sigma_hat = np.std(final_inliers, ddof=1)
    
    rfi_mask = np.abs(x - mu_hat) > (k * sigma_hat)
    
    return rfi_mask, mu_hat, sigma_hat

# ==========================================
# 3. Plotting & Visualization
# ==========================================

def calculate_qq_data(data: np.ndarray):
    """Helper function to calculate theoretical and actual quantiles for plotting."""
    y = np.sort(data)
    ranks = np.arange(1, len(y) + 1)
    z_q = norm.ppf((ranks - 0.5) / len(y))
    
    # Normalize y to standard normal scale for direct Q-Q comparison
    y_norm = (y - np.mean(y)) / np.std(y, ddof=1)
    return z_q, y_norm

def plot_results(x_original: np.ndarray, rfi_mask: np.ndarray):
    """Generates the Time-Series and Q-Q plots."""
    x_cleaned = x_original[~rfi_mask]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # --- Plot 1: Time Series Comparison ---
    ax1 = axes[0]
    ax1.plot(x_original, color='blue', alpha=0.6, label='Original Data (with RFI)', linewidth=1)
    
    # To plot the cleaned data seamlessly without dropping x-indices, we mask the RFI values with NaN
    x_clean_plot = x_original.copy()
    x_clean_plot[rfi_mask] = np.nan
    ax1.plot(x_clean_plot, color='red', alpha=0.9, label='Proposed (Cleaned)', linewidth=1)
    
    ax1.set_title("Time-Domain Voltage Data")
    ax1.set_xlabel("Time Window Index")
    ax1.set_ylabel("Voltage (mV)")
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.5)

    # --- Plot 2: Q-Q Plot (Original Data) ---
    ax2 = axes[1]
    z_q_orig, y_norm_orig = calculate_qq_data(x_original)
    slope_orig, intercept_orig, r_value_orig, _, _ = linregress(z_q_orig, y_norm_orig)
    
    ax2.scatter(z_q_orig, y_norm_orig, color='gray', alpha=0.3, s=10)
    ax2.plot(z_q_orig, intercept_orig + slope_orig * z_q_orig, color='black', alpha=0.5, 
             label=f"y = {slope_orig:.5f}x\n$R^2$ = {r_value_orig**2:.5f}")
    ax2.plot([-3, 3], [-3, 3], color='lightgray', linestyle='--') # Perfect Gaussian reference line
    
    ax2.set_title("Q-Q Plot: Original Data")
    ax2.set_xlabel("Standard Normal Quantiles")
    ax2.set_ylabel("Sample Quantiles")
    ax2.legend(loc="lower right")
    ax2.set_xlim(-3.5, 3.5)
    ax2.set_ylim(-3.5, 3.5)
    ax2.grid(True, linestyle='--', alpha=0.5)

    # --- Plot 3: Q-Q Plot (Cleaned Data) ---
    ax3 = axes[2]
    z_q_clean, y_norm_clean = calculate_qq_data(x_cleaned)
    slope_clean, intercept_clean, r_value_clean, _, _ = linregress(z_q_clean, y_norm_clean)
    
    ax3.scatter(z_q_clean, y_norm_clean, color='teal', alpha=0.3, s=10)
    ax3.plot(z_q_clean, intercept_clean + slope_clean * z_q_clean, color='darkcyan', alpha=0.8,
             label=f"y = {slope_clean:.5f}x\n$R^2$ = {r_value_clean**2:.5f}")
    ax3.plot([-3, 3], [-3, 3], color='lightgray', linestyle='--')
    
    ax3.set_title("Q-Q Plot: Proposed Method")
    ax3.set_xlabel("Standard Normal Quantiles")
    ax3.set_ylabel("Sample Quantiles")
    ax3.legend(loc="lower right")
    ax3.set_xlim(-3.5, 3.5)
    ax3.set_ylim(-3.5, 3.5)
    ax3.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

# ==========================================
# 4. Example / Test Execution
# ==========================================

if __name__ == "__main__":
    # Simulate a mock radio dataset
    # np.random.seed(42)
    # N = 1300  
    
    # # Generate Gaussian background noise
    # true_mu = 1446.0
    # true_sigma = 5.0
    # x_mock = np.random.normal(loc=true_mu, scale=true_sigma, size=N)
    
    # # Inject aggressive, non-Gaussian RFI spikes (both positive and negative)
    # rfi_indices_pos = np.random.choice(N, size=int(0.06 * N), replace=False)
    # x_mock[rfi_indices_pos] += np.random.uniform(15.0, 35.0, size=len(rfi_indices_pos))
    
    # rfi_indices_neg = np.random.choice(N, size=int(0.03 * N), replace=False)
    # x_mock[rfi_indices_neg] -= np.random.uniform(15.0, 25.0, size=len(rfi_indices_neg))

    fil_path = Path("/data/PhD/thesis/data/ips/3C161_61213.477083_ort.fil")
    fb_obj = Filterbank(fil_path)
    matrix = fb_obj.matrix
    time_data = np.nanmean(matrix, axis=1)
    N = len(time_data)

    print(f"Running Gaussian-RANSAC on {N} samples...")
    # Execute the algorithm
    rfi_mask, mu_est, sigma_est = gaussian_ransac_rfi_detection(time_data, num_iterations=2500)
    
    # print(f"Original Data Mean: {np.mean(x_mock):.2f}")
    # print(f"Recovered Mean:     {mu_est:.2f} (True: {true_mu})")
    # print(f"Recovered Sigma:    {sigma_est:.2f} (True: {true_sigma})")
    print(f"Flagged RFI Points: {np.sum(rfi_mask)} out of {N}")

    # Generate the plots
    plot_results(time_data, rfi_mask)