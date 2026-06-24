import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import file_utils as fut
import time_utils as tut
import data_process as dpr
import power_spec
from filterbank import Filterbank

def visualize_ps(fb_obj: Filterbank, is_db:bool=True,
                 f1:float=None, f2:float=None,
                 bpnorm:bool=False, save_folder_path:Path=None,
                 uni_stat_avg:float=None,
                ):
    # Start from the full matrix, then optionally crop the frequency range.
    matrix = fb_obj.matrix 
    freq_chans = fb_obj.freq_channels

    if f1 is not None or f2 is not None:
        matrix, sub_freq_chan = dpr.get_sub_matrix(matrix, f1, f2, fb_obj.freq_channels)
        freq_chans = sub_freq_chan

    filename_suffix = ''
    if bpnorm:
        # Apply bandpass normalization before plotting when requested.
        matrix = dpr.bp_norm_matrix(matrix)
        filename_suffix = '_bpnorm'

    time_profile = np.nanmean(matrix, axis=1)
    freq_profile = np.nanmean(matrix, axis=0)

    sampling_rate = 1 / fb_obj.header.tsamp
    freqs, psd = power_spec.compute_power_spectrum(time_profile, sampling_rate)

    if uni_stat_avg is not None:
        freqs, psd, _ = power_spec.uniform_statistical_averaging(freqs, psd, uni_stat_avg)

    # Create figure with subplots
    # Main spectrum in middle, frequency series on right, time series at the bottom
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3,
                          height_ratios=[3, 1], width_ratios=[0.15, 3, 1],
                          hspace=0.2, wspace=0.25)
    
    # Main dynamic spectrum
    ax_main = fig.add_subplot(gs[0, 1])
    if is_db:
        ax_main.plot(freqs, 10*np.log10(psd), linewidth=2, marker='o', markersize=4,
                     color='green', linestyle='-', alpha=0.8)
        ax_main.set_xscale('log')
    else:
        ax_main.plot(freqs, psd, linewidth=2, marker='o', markersize=4,
                     color='green', linestyle='-', alpha=0.8)
    
    ax_main.set_xlabel("Frequency", fontsize=11)
    ax_main.set_ylabel("Power", fontsize=11)
    mjd_dt = tut.mjd_to_datetime(fb_obj.mjd).strftime('%Y-%m-%d %H:%M:%S')
    ax_main.set_title(f"{fb_obj.source_name}\n{mjd_dt}", 
                      fontsize=12, fontweight='bold')
    ax_main.grid(True, alpha=0.3)
    
    # Time series (below main plot)
    ax_time = fig.add_subplot(gs[1, 1])
    ax_time.plot(fb_obj.time_samples, time_profile, color='black', linewidth=1)
    ax_time.set_xlabel("Time (s)", fontsize=11)
    ax_time.set_ylabel("Mean Intensity", fontsize=10)
    ax_time.grid(True, alpha=0.3)
    
    # Frequency series (right of main plot)
    ax_freq = fig.add_subplot(gs[0, 2])
    ax_freq.plot(freq_profile, freq_chans, color='black', linewidth=1)
    ax_freq.set_xlabel("Mean Intensity", fontsize=10)
    #ax_freq.set_ylabel("Frequency (MHz)", fontsize=11)
    ax_freq.set_ylim(freq_chans[0], freq_chans[-1])  # Match direction with main plot
    ax_freq.grid(True, alpha=0.3)

    if save_folder_path is not None:
        folder_path = Path(save_folder_path)
        #folder_path = save_folder_path / source_name
        folder_path.mkdir(exist_ok=True)
        
        # Construct the output filename, preserving any requested suffix.
        filename = f"{fb_obj.source_name}_{fb_obj.mjd}_{freq_chans[0]:.2f}_{freq_chans[-1]:.2f}_dyn_spec{filename_suffix}.jpeg"
        file_path = folder_path / filename
        
        # Delegate overwrite handling to the shared helper.
        file_path = fut.handle_file_existence(file_path)
        
        fig.savefig(file_path, bbox_inches='tight', dpi=150)
        plt.close(fig)

        print(f"Plot saved to {file_path}")
        del fig
        return None
    
    print("Displaying plot.")
    plt.show()
    return fig, ax_main
    

def get_args():
    """Parse CLI arguments for dynamic spectrum plotting."""
    parser = argparse.ArgumentParser(
        description='Plot power spectrum from a filterbank (.fil) or FITS file.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python plot_ps.py /path/to/data.fil
  python plot_ps.py /path/to/data.fits
  python plot_ps.py /path/to/data.fil --save output_plots/
        """
    )
    
    parser.add_argument('file_path', type=str,
                       help='Path to the filterbank (.fil) or FITS (.fits) file')
    parser.add_argument('--save', type=str, default=None,
                       help='Folder to save the plot (if not provided, plot will be displayed)')
    parser.add_argument('--f1', type=float, default=None,
                       help='Higher frequency in MHz (default: filterbank start frequency)')
    parser.add_argument('--f2', type=float, default=None,
                       help='Lower frequency in MHz (default: filterbank end frequency)')
    parser.add_argument('--bpnorm', action='store_true',
                       help='Also produce a bandpass-normalized dynamic spectrum plot')
    parser.add_argument('--nodb', action='store_false',
                       help='Plot the power spectrum in decibel scale')
    parser.add_argument('--uni-stat-avg', type=float, default=None, dest='uni_stat_avg',
                       help='Perform uniform statistical averaging')
    
    args = parser.parse_args()

    # Normalize parsed values into pathlib objects for downstream code.
    file_path = Path(args.file_path)
    nodb = args.nodb
    save_folder_path = Path(args.save) if args.save is not None else None
    f1, f2 = args.f1, args.f2
    bpnorm = args.bpnorm
    uni_stat_avg = args.uni_stat_avg

    return file_path, nodb, f1, f2, bpnorm, save_folder_path, uni_stat_avg

if __name__ == "__main__":
    file_path, nodb, f1, f2, bpnorm, save_folder_path, uni_stat_avg = get_args()
    fb_obj = Filterbank(file_path)
    
    visualize_ps(fb_obj, nodb, f1, f2, bpnorm, save_folder_path, uni_stat_avg)