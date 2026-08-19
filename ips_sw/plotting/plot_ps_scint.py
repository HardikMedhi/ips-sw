import gc
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import ips_sw.utils.file_utils as fut
import ips_sw.utils.time_utils as tut
import ips_sw.sc_index as sc
import ips_sw.power_spectra.power_spec
from ips_sw.classes.filterbank import Filterbank

def visualize_scint_ps(psd_arr: np.ndarray, freqs: np.ndarray, 
                       f1:float, f2:float, t1:float, t2:float, crossover_freq:float,
                       m_ps:float, m_ts:float,
                       onsrc_name:str, offsrc_name:str, mjd:str, elong:float,
                       nodb:bool=False, save_folder_path:Path=None):
    
    scint_psd, onsrc_psd, offsrc_psd = psd_arr

    fig, ax = plt.subplots(figsize=(14, 7))

    if not nodb:
        ax.plot(freqs, 10*np.log10(scint_psd), linewidth=2, marker='^', markersize=4, 
            label='Scintillation', color='blue', linestyle='--', alpha=0.8)
        ax.plot(freqs, 10*np.log10(onsrc_psd), linewidth=2, marker='o', markersize=4, 
                label='On-source', color='green', linestyle='-', alpha=0.8)
        ax.plot(freqs, 10*np.log10(offsrc_psd), linewidth=2, marker='s', markersize=4, 
                label='Off-source', color='red', linestyle='-', alpha=0.8)

        ax.set_xscale('log')
    else:
        ax.plot(freqs, scint_psd, linewidth=2, marker='^', markersize=4, 
            label='Scintillation', color='blue', linestyle='--', alpha=0.8)
        ax.plot(freqs, onsrc_psd, linewidth=2, marker='o', markersize=4, 
                label='On-source', color='green', linestyle='-', alpha=0.8)
        ax.plot(freqs, offsrc_psd, linewidth=2, marker='s', markersize=4, 
                label='Off-source', color='red', linestyle='-', alpha=0.8)
        
    ax.set_xlabel("Frequency (Hz)", fontsize=11)
    ax.set_ylabel("Power (dB)", fontsize=11)

    ax.axvline(x=crossover_freq, color='black', linestyle=':', linewidth=1.5, 
                   label=f'Crossover freq: {crossover_freq:.2f} Hz', alpha=0.6)

    mjd_dt = tut.mjd_to_datetime(mjd).strftime('%Y-%m-%d %H:%M:%S')
    title = f"{onsrc_name} - {offsrc_name}\n{mjd_dt} (ϵ = {elong:.2f}°)\n{f2:.2f} - {f1:.2f} MHz | {t1:.2f} - {t2:.2f} s"
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')

    # Display m_ps and m_ts in a separate boxed text (not part of legend)
    # stats_text = f"m (PS): {m_ps:.4e}\n m (TS): {m_ts:.4e}"
    # ax.text(0.18, 0.02, stats_text, transform=ax.transAxes,
    #     fontsize=12, verticalalignment='bottom', horizontalalignment='right',
    #     bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8, edgecolor='black'))


    if save_folder_path is not None:
        folder_path = Path(save_folder_path)
        #folder_path = save_folder_path / source_name
        folder_path.mkdir(exist_ok=True)
        
        # Construct the output filename, preserving any requested suffix.
        filename = f"{onsrc_name}_{offsrc_name}_{mjd}_{f1:.2f}_{f2:.2f}_scint_power_spec.jpeg"
        file_path = folder_path / filename
        
        # Delegate overwrite handling to the shared helper.
        #file_path = fut.handle_file_existence(file_path)
        
        fig.savefig(file_path, bbox_inches='tight', dpi=150)
        plt.close(fig)

        print(f"Plot saved to {file_path}")
        del fig
        return None
    
    print("Displaying plot.")
    plt.show()


def get_args():
    """Parse CLI arguments for dynamic spectrum plotting."""
    parser = argparse.ArgumentParser(
        description='Plot power spectra from multiple filterbank (.fil) or FITS file' \
                    'whose filepaths are specified in plot_ps_multi.yaml',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument('onsrc_filepath', type=str,
                       help='Path to the on-source filterbank file.')
    parser.add_argument('--offsrc', type=str, action='extend', nargs='+',
                    help='Path (s) to the off-source filterbank file(s).')
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

    onsrc_filepath = Path(args.onsrc_filepath)
    offsrc_filepaths = args.offsrc
    nodb = args.nodb
    nodetrend = args.nodetrend
    nodespike = args.nodespike
    save_folder_path = Path(args.save) if args.save is not None else None
    f1, f2 = args.f1, args.f2
    t1, t2 = args.t1, args.t2
    bpnorm = args.bpnorm
    uni_stat_avg = args.uni_stat_avg

    return onsrc_filepath, offsrc_filepaths, nodb, nodetrend, nodespike, f1, f2, t1, t2, bpnorm, save_folder_path, uni_stat_avg

if __name__ == "__main__":
    onsrc_filepath, offsrc_filepaths, nodb, nodetrend, nodespike, f1, f2, t1, t2, bpnorm, save_folder_path, uni_stat_avg = get_args()

    onsrc_fb = Filterbank(onsrc_filepath)   

    for offsrc in offsrc_filepaths:
        offsrc_fb = Filterbank(Path(offsrc))

        #TODO: Very inefficient to repeat the on source PS calculation! Figure out a solution!!!
        freqs, onsrc_psd = power_spec.get_ps(onsrc_fb, f1, f2, t1, t2, bpnorm, nodetrend, nodespike)
        freqs, offsrc_psd = power_spec.get_ps(offsrc_fb, f1, f2, t1, t2, bpnorm, nodetrend, nodespike)
        scint_psd = onsrc_psd - offsrc_psd

        if uni_stat_avg is not None:
            freqs_avg, scint_psd, _ = power_spec.uniform_statistical_averaging(freqs, scint_psd, uni_stat_avg)
            _, onsrc_psd, _ = power_spec.uniform_statistical_averaging(freqs, onsrc_psd, uni_stat_avg)
            _, offsrc_psd, _ = power_spec.uniform_statistical_averaging(freqs, offsrc_psd, uni_stat_avg)

            freqs = freqs_avg

        f1 = f1 if f1 is not None else onsrc_fb.freq_channels[0]
        f2 = f2 if f2 is not None else onsrc_fb.freq_channels[-1]

        t1 = t1 if t1 is not None else onsrc_fb.time_samples[0]
        t2 = t2 if t2 is not None else onsrc_fb.time_samples[-1]

        # Get crossover frequency
        crossover_freq = freqs[np.argmin(np.abs(scint_psd - offsrc_psd))]

        onsrc_ts = np.nanmean(onsrc_fb.matrix, axis=1)
        offsrc_ts = np.nanmean(offsrc_fb.matrix, axis=1)

        m_ps, m_ts = 0, 0

        # m_ps, m_ts = sc.get_m_vals(
        #     onsrc_fb, offsrc_fb,
        #     scint_psd, freqs,
        #     crossover_freq
        # )

        visualize_scint_ps(
            np.array([scint_psd, onsrc_psd, offsrc_psd]), freqs, 
            f1, f2, t1, t2, crossover_freq,
            m_ps, m_ts,
            onsrc_fb.source_name, offsrc_fb.source_name, onsrc_fb.mjd, onsrc_fb.elong,
            nodb, save_folder_path
        )

        del offsrc_fb, offsrc_psd, offsrc_ts, scint_psd, freqs
        gc.collect()




    