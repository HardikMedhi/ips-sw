import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import file_utils as fut
import time_utils as tut
import data_process as dpr
from filterbank import Filterbank

def main():
    file_path, f1, f2, t1, t2, bpnorm, save_folder_path = get_args()
    fb_obj = Filterbank(file_path)
    
    visualize_ds(fb_obj, f1, f2, t1, t2, bpnorm, save_folder_path)

def get_args():
    """Parse CLI arguments for dynamic spectrum plotting."""
    parser = argparse.ArgumentParser(
        description='Plot dynamic spectrum from a filterbank (.fil) or FITS file.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python plot_ds.py /path/to/data.fil
  python plot_ds.py /path/to/data.fits
  python plot_ds.py /path/to/data.fil --save output_plots/
        """
    )
    
    parser.add_argument('file_path', type=str,
                       help='Path to the filterbank (.fil) or FITS (.fits) file')
    parser.add_argument('--save', type=str, default=None,
                       help='Folder to save the plot (if not provided, plot will be displayed)')
    parser.add_argument('--f1', type=float, default=None,
                       help='Higher frequency in MHz')
    parser.add_argument('--f2', type=float, default=None,
                       help='Lower frequency in MHz')
    parser.add_argument('--t1', type=float, default=None,
                       help='Lower time value in seconds')
    parser.add_argument('--t2', type=float, default=None,
                       help='Higher time value in seconds')
    parser.add_argument('--bpnorm', action='store_true',
                       help='Also produce a bandpass-normalized dynamic spectrum plot')
    
    args = parser.parse_args()

    # Normalize parsed values into pathlib objects for downstream code.
    file_path = Path(args.file_path)
    save_folder_path = Path(args.save) if args.save is not None else None
    f1, f2 = args.f1, args.f2
    t1, t2 = args.t1, args.t2
    bpnorm = args.bpnorm

    return file_path, f1, f2, t1, t2, bpnorm, save_folder_path

def visualize_ds(fb_obj: Filterbank, 
                 f1:float=None, f2:float=None,
                 t1:float=None, t2:float=None,
                 bpnorm:bool=False, save_folder_path:Path=None):
    """Plot a dynamic spectrum and its mean time/frequency profiles.

    If a save folder is provided, the figure is written to disk. Otherwise it
    is shown interactively.
    """

    # Start from the full matrix, then optionally crop the frequency range.
    matrix = fb_obj.matrix 
    freq_chans = fb_obj.freq_channels
    time_samples = fb_obj.time_samples

    if f1 is not None or f2 is not None:
        matrix, sub_freq_chan = dpr.get_sub_matrix_freq(matrix, f1, f2, fb_obj.freq_channels)
        freq_chans = sub_freq_chan

    if t1 is not None or t2 is not None:
        matrix, sub_time_samples = dpr.get_sub_matrix_time(matrix, t1, t2, fb_obj.time_samples)
        time_samples = sub_time_samples

    filename_suffix = ''
    if bpnorm:
        # Apply bandpass normalization before plotting when requested.
        matrix = dpr.bp_norm_matrix(matrix)
        filename_suffix = '_bpnorm'

    time_profile = np.nanmean(matrix, axis=1)
    freq_profile = np.nanmean(matrix, axis=0)
    
    # Create figure with subplots
    # Colorbar on left, main spectrum in middle, frequency series on right
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3,
                          height_ratios=[3, 1], width_ratios=[0.15, 3, 1],
                          hspace=0.05, wspace=0.25)
    
    # Colorbar (left of main plot)
    ax_cbar = fig.add_subplot(gs[0, 0])
    
    # Main dynamic spectrum
    ax_main = fig.add_subplot(gs[0, 1])
    im = ax_main.imshow(
        matrix.T,
        aspect='auto',
        origin='lower',
        cmap='inferno',
        vmin=np.nanpercentile(matrix, 5),
        vmax=np.nanpercentile(matrix, 95),
        extent=[fb_obj.time_samples[0], fb_obj.time_samples[-1],
                freq_chans[0], freq_chans[-1]]
    )
    ax_main.set_ylabel("Frequency (MHz)", fontsize=11)
    mjd_dt = tut.mjd_to_datetime(fb_obj.mjd).strftime('%Y-%m-%d %H:%M:%S')
    ax_main.set_title(f"{fb_obj.source_name}\n{mjd_dt}", 
                      fontsize=12, fontweight='bold')
    ax_main.tick_params(labelbottom=False)
    
    # Colorbar with tick labels on the left
    cbar = plt.colorbar(im, cax=ax_cbar, label="Intensity")
    ax_cbar.yaxis.set_ticks_position('left')
    ax_cbar.yaxis.set_label_position('left')
    
    # Time series (below main plot)
    ax_time = fig.add_subplot(gs[1, 1], sharex=ax_main)
    ax_time.plot(time_samples, time_profile, color='black', linewidth=1)
    ax_time.set_xlabel("Time (s)", fontsize=11)
    ax_time.set_ylabel("Mean Intensity", fontsize=10)
    ax_time.set_xlim(time_samples[0], time_samples[-1])
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

if __name__ == "__main__":
    main()
    

    


    
