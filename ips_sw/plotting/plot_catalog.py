import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from astropy.coordinates import SkyCoord
from astropy import units as U
from astropy.table import Table
import warnings
warnings.filterwarnings("ignore")

import ips_sw.utils.data_utils as dut

from importlib.resources import files
style_path = files("ips_sw").joinpath("matplotlib_styles/style_paper.mplstyle")
plt.style.use(style_path)

def main(args:tuple) -> tuple:
    cat_paths, cat_names, path_save_folder = args

    cat_dict = {
        n:get_cat_coords(p)
        for p, n in zip(cat_paths, cat_names)
    }

    fig, ax = plot_catalog(cat_dict, path_save_folder)
    return fig, ax

def get_args() -> tuple:
    parser = argparse.ArgumentParser(
        description="Program to plot source catalogues."
    )

    parser.add_argument("--cat-paths", "-p", nargs="+", required=True,
                    help="Filepaths to the catalogues (one or more).")
    parser.add_argument("--cat-names", "-n", nargs="+", required=True,
                        help="Catalogue names in the same order (one or more).")
    parser.add_argument("--save", "-s", type=str, default=None,
                        help="Optional path to save the plot.")

    args = parser.parse_args()
    cat_paths = [Path(p) for p in args.cat_paths]
    cat_names = args.cat_names
    path_save_folder = Path(args.save) if args.save is not None else None

    if len(cat_names) != len(cat_paths):
        raise ValueError(f"Number of filepaths and catalogue names need to be same!\
                         Got {len(cat_paths)} filepaths and {len(cat_names)} catalogue names.")

    return cat_paths, cat_names, path_save_folder

def get_cat_coords(filepath_cat:Path) -> SkyCoord:
    try:
        data = Table.read(filepath_cat)
    except Exception as e:
        raise ValueError(f"Unable to read catalog {filepath_cat}: {e}")

    colname_ra, colname_dec = dut.get_coords_colnames(data.columns)
    coords = SkyCoord(ra=data[colname_ra], dec=data[colname_dec], unit=(U.hour, U.degree), frame='icrs')
    return coords

def plot_catalog(cat_dict:dict, path_save_folder:Path) -> tuple:
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='mollweide')

    colors = ['blue', 'red', 'green', 'orange', 'black', 'yellow', 'brown', 'pink', 'purple']
    if len(cat_dict) > len(colors):
        colors = plt.cm.jet(np.linspace(0, 1, len(cat_dict)))

    for i, (name, coords) in enumerate(cat_dict.items()):
        ra_rad = coords.ra.wrap_at(180*U.degree).radian
        dec_rad = coords.dec.radian

        ax.scatter(ra_rad, dec_rad, color=colors[i], alpha=0.7, s=10, label=name)

    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_title("Sky Distribution of IPS Sources")
    ax.set_xlabel("Right Ascension")
    ax.set_ylabel("Declination")
    ax.legend(loc="lower right")

    # Fix the RA labels so they read 0 to 360 instead of -180 to 180
    ax.set_xticklabels(['14h','16h','18h','20h','22h','0h','2h','4h','6h','8h','10h'])

    if path_save_folder is not None:
        cat_names = list(cat_dict.keys())
        filename = "_".join(cat_names) + "_cat.jpeg"
        filepath = path_save_folder / filename.lower()
        if filepath.exists():
            print(f"File {filepath} exists. Overwriting..")
        fig.savefig(filepath, bbox_inches="tight")
        print(f"File saved to {filepath}")

    return fig, ax

if __name__ == "__main__":
    args = get_args()
    fig, ax = main(args)
    plt.show()