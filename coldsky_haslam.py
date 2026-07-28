import argparse
import numpy as np
import healpy as hp
import astropy.units as u
from astropy.coordinates import SkyCoord
from pathlib import Path
import pandas as pd
import yaml

import data_process as dp

def main(args:list):
    map_filepath, srclist_path, usera, telescope, fc_lowlim, fc_highlim, date_precess = args

    if fc_lowlim is None or fc_highlim is None:
        tel_info_filepath = Path("/data/thesis/code2/tel_info.yaml")
        with open(tel_info_filepath, "r") as file:
            tel_info = yaml.safe_load(file)
        
        if fc_lowlim is None:
            fc_lowlim = 0 if usera else tel_info[telescope.lower()]['dec_low'] # degree
        if fc_highlim is None:
            fc_lowlim = 360 if usera else tel_info[telescope.lower()]['dec_high']  # degree

    haslam_map = hp.read_map(map_filepath)


    
def get_args():
    parser = argparse.ArgumentParser(
        description="Get coordinates of cold-sky regions from the Haslam 408 MHz Map."
    )

    parser.add_argument("map_filepath", type=str,
                        help="Filepath to the Haslam Map")
    parser.add_argument("--srclist", type=str, default=None,
                        help="(Optional) Filepath to a source list with coordinates. " \
                        "The cold-sky regions will have the same RA (or Dec if --usedec flag is True) as the sources.")
    # TODO: Make this program agnostic of the free coordinate!!!
    # parser.add_argument("--usera", action="store_true",
    #                     help="(Optional) Get cold-sky regions with the same RA as the sources given by --srclist." \
    #                     "Dec is the free coordinate then. By default, RA is the free coordinate and Dec is constrained to the source list.")
    parser.add_argument("--tel", type=str, default="ort",
                        help="Choice of telescope location between ORT and GMRT. Default is ORT.")
    parser.add_argument("--low", type=float, default=None,
                        help="(Optional) Lower limit of the free coordinate in degrees (see --srclist and --usedec)")
    parser.add_argument("--high", type=float, default=None,
                        help="(Optional) Higher limit of the free coordinate in degrees (see --srclist and --usedec)")
    parser.add_argument("--precess", type="str", default=None,
                        help="(Optional) Precess the coordinates to the given date in YYYYMMDD format.")
    
    args = parser.parse_args()
    map_filepath = Path(args.map_filepath)
    srclist_path = Path(args.srclist) if args.srclist is not None else None
    usera = args.usera
    telescope = args.tel
    fc_lowlim, fc_highlim = args.low, args.high # fc - free coordinate
    date_precess = args.precess

    return map_filepath, srclist_path, usera, telescope, fc_lowlim, fc_highlim, date_precess

def get_tel_hpbw(tel:str, pointing_dec:float=0):
    if tel.lower() == 'ort':
        hpbw_ra = 2.0 # degrees
        hpbw_dec = (6.0 / 60.0) / np.cos(np.rad2deg(pointing_dec)) # degrees
        return hpbw_ra, hpbw_dec
    else:
        print(f"Telescope {tel} not recognized. Code is yet to be updated for it.")
        return

def calculate_sinc_weights(d_ra_sky: np.ndarray, d_dec: np.ndarray, hpbw_ra: float, hpbw_dec: float):
    """
    Calculates the 2D sinc-squared weights for a given set of angular offsets.
    """
    # Scaling factor to ensure the weight is exactly 0.5 at HPBW/2
    sinc_scale = 0.8859 
    
    w_ra = np.sinc(sinc_scale * (d_ra_sky / hpbw_ra))**2
    w_dec = np.sinc(sinc_scale * (d_dec / hpbw_dec))**2
    
    return w_ra * w_dec
    
def get_icrs_coordinates_from_map(nside: int):
    """
    Pre-calculates the ICRS (RA, Dec) coordinates for every pixel in a HEALPix map.
    If scanning multiple declinations, call this once and pass the result to the scanner.
    """
    npix = hp.npix(nside)
    lon_gal, lat_gal = hp.pix2ang(nside, np.arange(npix), lonlat=True)
    
    coords_gal = SkyCoord(l=lon_gal*u.deg, b=lat_gal*u.deg, frame='galactic')
    coords_icrs = coords_gal.transform_to('icrs')
    
    return coords_icrs.ra.degree, coords_icrs.dec.degree







if __name__ == "__main__":
    args = get_args()
    main(args)


# def get_coldsky(map:np.ndarray):
#     min_temp_idx = np.argmin(map)
#     min_temp = map[min_temp_idx]

#     nside = hp.get_nside(map)
#     lon, lat = hp.pix2ang(nside, min_temp_idx, lonlat=True)

#     coords = SkyCoord(ra=lon, dec=lat, unit=(u.degree, u.degree), frame='galactic').transform_to('icrs')
#     return coords, min_temp

# def get_coldsky_for_srclist(map:np.ndarray, src_df:pd.DataFrame, fc_lowlim:float, fc_highlim:float, usedec:bool=False):
#     nside = hp.get_nside(map)
#     colname_ra, colname_dec = dp.get_coords_colnames(src_df)
#     coords = src_df[[colname_ra, colname_dec]]

#     if usedec:
#         fc_arr = np.linspace(fc_lowlim, fc_highlim, 10000) # ra
#         const_arr = coords[colname_dec] # dec

#         ra_grid, dec_grid = np.meshgrid(fc_arr, const_arr, indexing='xy')
#     else:
#         fc_arr = np.linspace(fc_lowlim, fc_highlim, 10000) # dec
#         const_arr = coords[colname_ra] # ra

#         ra_grid, dec_grid = np.meshgrid(const_arr, fc_arr, indexing='ij')

#     grid = SkyCoord(ra=ra_grid*u.degree,
#                     dec=dec_grid*u.degree,
#                     frame='icrs').transform_to('galactic')
    
#     pixels = hp.ang2pix(nside, grid.l.deg, grid.b.deg, lonlat=True)
#     temps = map[pixels]