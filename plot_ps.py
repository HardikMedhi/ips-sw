import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import file_utils as fut
import time_utils as tut
import data_process as dpr
import power_spec
from filterbank import Filterbank

def visualize_ps(psd: np.ndarray, freqs: np.ndarray,
                source_name:str, mjd:str,
                f1:float, f2:float,
                elong:float,
                nodb:bool=False, save_folder_path:Path=None
                ):

    fig, ax = plt.subplots(figsize=(14, 7))
    if not nodb:
        ax.plot(freqs, 10*np.log10(psd), linewidth=2, marker='o', markersize=4,
                     color='green', linestyle='-', alpha=0.8)
        ax.set_xscale('log')
    else:
        ax.plot(freqs, psd, linewidth=2, marker='o', markersize=4,
                     color='green', linestyle='-', alpha=0.8)
    
    ax.set_xlabel("Frequency (Hz)", fontsize=11)
    ax.set_ylabel("Power (dB)", fontsize=11)

    mjd_dt = tut.mjd_to_datetime(mjd).strftime('%Y-%m-%d %H:%M:%S')
    title = f"{source_name} (ϵ = {elong:.2f}°)\n{mjd_dt}"
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')

    if save_folder_path is not None:
        folder_path = Path(save_folder_path)
        #folder_path = save_folder_path / source_name
        folder_path.mkdir(exist_ok=True)
        
        # Construct the output filename, preserving any requested suffix.
        filename = f"{source_name}_{mjd}_{f1:.2f}_{f2:.2f}_power_spec.jpeg"
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
    return fig, ax
    

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
                       help='Use a bandpass-normalized data')
    parser.add_argument('--uni-stat-avg', type=float, default=None, dest='uni_stat_avg',
                       help='Perform uniform statistical averaging')
    parser.add_argument('--nodb', action='store_true',
                       help="Don't plot the power spectra in decibel scale")
    parser.add_argument('--nodetrend', action='store_true',
                        help="Don't detrend the timeseries")
    parser.add_argument('--nodespike', action='store_true',
                        help="Don't despike the timeseries")
    
    args = parser.parse_args()

    # Normalize parsed values into pathlib objects for downstream code.
    file_path = Path(args.file_path)
    nodb = args.nodb
    nodetrend = args.nodetrend
    nodespike = args.nodespike
    save_folder_path = Path(args.save) if args.save is not None else None
    f1, f2 = args.f1, args.f2
    bpnorm = args.bpnorm
    uni_stat_avg = args.uni_stat_avg

    return file_path, nodb, nodetrend, nodespike, f1, f2, bpnorm, save_folder_path, uni_stat_avg

def get_ps(fb_obj: Filterbank, 
           f1:float=None, f2:float=None, 
           bpnorm:bool=False,
           nodetrend:bool=False, nodespike:bool=False):
    
    matrix = fb_obj.matrix

    if f1 is not None or f2 is not None:
        matrix, _ = dpr.get_sub_matrix_freq(matrix, f1, f2, fb_obj.freq_channels)

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

    freqs, psd = power_spec.compute_power_spectrum(time_profile, sampling_rate)

    return freqs, psd

if __name__ == "__main__":
    file_path, nodb, nodetrend, nodespike, f1, f2, bpnorm, save_folder_path, uni_stat_avg = get_args()
    fb_obj = Filterbank(file_path)

    freqs, psd = get_ps(fb_obj, f1, f2, bpnorm, nodetrend, nodespike)
    if uni_stat_avg is not None:
        freqs, psd, _ = power_spec.uniform_statistical_averaging(freqs, psd, uni_stat_avg)

    _, freq_chans = dpr.get_sub_matrix_freq(fb_obj.matrix, f1, f2, fb_obj.freq_channels)
    
    visualize_ps(
        psd, freqs, 
        fb_obj.source_name, fb_obj.mjd,
        freq_chans[0], freq_chans[-1],
        fb_obj.elong,
        nodb, save_folder_path
    )