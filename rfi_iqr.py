import numpy as np

def iqr(matrix):
    """
    Applies the outlier detection and channel filtering algorithm.
    
    Args:
        data_matrix (np.ndarray): Matrix of shape (observations, channels)
    Returns:
        tuple: (cleaned_data_matrix, stats_dict)
    """
    n_obs, n_channels = matrix.shape
    
    # 1. Calculate the 90th percentile of each observation frequency channel
    p90_per_channel = np.percentile(matrix, 90, axis=0)

    # 2. Find outlier percentiles using IQR method (k=1.2) across the channels
    q1_p90 = np.percentile(p90_per_channel, 25)
    q3_p90 = np.percentile(p90_per_channel, 75)
    iqr_p90 = q3_p90 - q1_p90
    
    lower_bound = q1_p90 - 1.2 * iqr_p90
    upper_bound = q3_p90 + 1.2 * iqr_p90
    
    # Identify which channels have outlier 90th percentiles
    is_outlier_channel = (p90_per_channel < lower_bound) | (p90_per_channel > upper_bound)
    
    # Tracking stats
    removed_channels = 0
    replaced_in_marginal = 0
    replaced_in_stable = 0
    
    channels_to_delete = []
    output_data = np.copy(matrix).astype(float)

    for ch in range(n_channels):
        channel_data = matrix[:, ch]
        q1, q2, q3 = np.percentile(channel_data, [25, 50, 75])
        iqr = q3 - q1

        if is_outlier_channel[ch]:
            # 3. Ratio test for outlier channels
            r = q3 / q2 if q2 != 0 else float('inf')
            
            if r >= 5:
                channels_to_delete.append(ch)
                removed_channels += 1
            else:
                # Apply 1.5x IQR filter
                ch_lower = q1 - 1.5 * iqr
                ch_upper = q3 + 1.5 * iqr
                mask = (channel_data < ch_lower) | (channel_data > ch_upper)
                replaced_in_marginal += np.sum(mask)
                output_data[mask, ch] = q2
        
        else:
            # 4. Standard filter for non-outlier channels
            # Apply 3x IQR filter
            ch_lower = q1 - 3.0 * iqr
            ch_upper = q3 + 3.0 * iqr
            mask = (channel_data < ch_lower) | (channel_data > ch_upper)
            replaced_in_stable += np.sum(mask)
            output_data[mask, ch] = q2

    # Final channel removal
    clean_matrix = np.delete(output_data, channels_to_delete, axis=1)

    return clean_matrix