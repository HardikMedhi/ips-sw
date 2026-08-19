import yaml
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import ips_sw.utils.get_tel_info as get_tel_info

from importlib.resources import files
style_path = files("ips_sw").joinpath("matplotlib_styles/style_paper.mplstyle")
plt.style.use(style_path)

PATCH_LEN_RA = 3.3 # degree
PATCH_LEN_DEC = 3.3 # degree

def main(args) -> tuple:
    pointing_dec, path_save_folder, offsets = args
    tel_info = get_tel_info.main(["ort"])

    sensitivity_pattern = get_sensitivity_pattern(tel_info, pointing_dec)
    fig, ax = plot_beam_pattern(sensitivity_pattern, pointing_dec, path_save_folder)
    return fig, ax

def get_sensitivity_pattern(tel_info:dict, pointing_dec:float) -> np.ndarray:
    # Create coordinate grid (degrees)
    # We look at a 5x5 degree patch around the pointing center
    x = np.linspace(-PATCH_LEN_RA/2, PATCH_LEN_RA/2, 400) # RA offset
    y = np.linspace(-PATCH_LEN_DEC/2, PATCH_LEN_DEC/2, 400) # Dec offset
    X, Y = np.meshgrid(x, y)

    # Projection effect factor
    cos_dec = np.cos(np.radians(pointing_dec))

    l_ew = tel_info['len_ew_total']
    l_ns_mod = tel_info['len_ns_mod']
    lam = tel_info['lam']

    # EW Term (X axis)
    arg_ew = (l_ew * np.sin(np.radians(X) * cos_dec)) / lam
    # NS Term (Y axis) - using single module length
    arg_ns_mod = (l_ns_mod * np.sin(np.radians(Y))) / (lam * cos_dec)
    primary_beam = (np.sinc(arg_ew)**2) * (np.sinc(arg_ns_mod)**2)

    return primary_beam

def plot_beam_pattern(sensitivity_pattern:np.ndarray, pointing_dec:float, path_save_folder:Path) -> tuple:
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    im = ax.imshow(
        sensitivity_pattern, 
        extent=[-PATCH_LEN_RA/2, PATCH_LEN_RA/2, -PATCH_LEN_DEC/2, PATCH_LEN_DEC/2], 
        origin='lower', 
        cmap='inferno', 
    )
    plt.colorbar(im, label='Primary Beam Sensitivity')

    ax.set_title("ORT Primary Beam Sensitivity "+r"$(\text{sinc}^2)$"+f"\nPointing Dec = {pointing_dec}°")
    ax.set_xlabel('RA Offset (Degrees)')
    ax.set_ylabel('Dec Offset (Degrees)')
    ax.grid(True, alpha=0.4)

    if path_save_folder is not None:
        filename = f"ort_primary_beam_{pointing_dec}d.jpeg"
        filepath = path_save_folder / filename
        if filepath.exists():
            print(f"File {filepath} already exists. Overwriting..")
        fig.savefig(filepath, bbox_inches="tight")
        print(f"Plot saved to {filepath}")
    
    return fig, ax

def get_args() -> tuple[float, Path, np.ndarray]:
    parser = argparse.ArgumentParser(
        description="Program to plot ORT's beam pattern with sinc^2 sensitivity"
    )

    parser.add_argument(
        "pointing_dec", type=float,
        help="Pointing declination in degrees."
    )
    parser.add_argument(
        "--save", type=str, default=None,
        help="(Optional) Path to the folder where the image needs to be saved."
    )
    #TODO: Implement this functionality!!!
    parser.add_argument(
        "--offsets", type=float, action="extend", nargs="+", default=[],
        help="(Optional) List of offsets for plotting synthesized beams with 22 modules."
    )

    args = parser.parse_args()
    pointing_dec = args.pointing_dec
    path_save_folder = Path(args.save) if args.save is not None else None
    offsets = np.array(args.offsets) if len(args.offsets) != 0 else np.array([])

    return pointing_dec, path_save_folder, offsets

if __name__ == "__main__":
    args = get_args()
    fig, ax = main(args)
    plt.show()