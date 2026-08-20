import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import ips_sw.utils.file_utils as fut
import ips_sw.utils.time_utils as tut
import ips_sw.utils.data_utils as dut
import ips_sw.power_spectra.power_spec as power_spec
from ips_sw.classes.filterbank import Filterbank

from importlib.resources import files
style_path = files("ips_sw").joinpath("matplotlib_styles/style_paper.mplstyle")
plt.style.use(style_path)

def main(args:tuple) -> tuple:
    file_path, nodb, nodetrend, nodespike, f1, f2, bpnorm, save_folder_path, uni_stat_avg = args
    fb_obj = Filterbank(file_path)
    
    freqs, psd = power_spec.get_ps(fb_obj, f1=f1, f2=f2, bpnorm=bpnorm, nodetrend=nodetrend, nodespike=nodespike)
    if uni_stat_avg is not None:
        freqs, psd, _ = power_spec.uniform_statistical_averaging(freqs, psd, uni_stat_avg)
    
    _, freq_chans = dut.get_sub_matrix_freq(fb_obj.matrix, f1, f2, fb_obj.freq_channels)
    
    fig, ax = visualize_ps(
        psd, freqs, 
        fb_obj.source_name, fb_obj.mjd,
        freq_chans[0], freq_chans[-1],
        fb_obj.elong,
        nodb, save_folder_path
    )

    return fig, ax

def get_args() -> tuple[Path, bool, bool, bool, float, float, bool, Path, float]:
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

def visualize_ps(psd: np.ndarray, freqs: np.ndarray,
                source_name:str, mjd:str,
                f1:float, f2:float,
                elong:float,
                nodb:bool=False, save_folder_path:Path=None
                ) -> tuple:

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
        print(f"Plot saved to {file_path}")

    return fig, ax
    
if __name__ == "__main__":
    args = get_args()
    fig, ax = main(args)
    plt.show()