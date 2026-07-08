import argparse
import numpy as np
import healpy as hp
from astropy.io import fits
from astropy.coordinates import SkyCoord
from pathlib import Path

PATH_HASALAM_MAP = "/data/PhD/thesis/data/catalogs/haslam408_dsds_Remazeilles2014.fits"
MIN_DEC = -60
MAX_DEC = 60

def read_haslam_map():
    """
    Read the Haslam 408 MHz map from FITS file.
    
    Parameters
    ----------
    fits_path : str
        Path to the FITS file containing the Haslam map
        
    Returns
    -------
    temperature : ndarray
        Temperature map data from the TEMPERATURE column
    nside : int
        HEALPix nside parameter
    """
    hdul = fits.open(PATH_HASALAM_MAP)
    haslam_map = hdul[1].data['TEMPERATURE']
    header = hdul[1].header
    hdul.close()
    
    # print(f"Haslam map shape: {haslam_map.shape}")
    # print(f"Header info:")
    for key in ['NSIDE', 'ORDERING', 'EXTNAME']:
        if key in header:
            print(f"  {key}: {header[key]}")
    
    # Handle 2D maps - flatten if necessary
    if haslam_map.ndim == 2:
        print(f"Map is 2D with shape {haslam_map.shape}")
        # For a 2D healpix map, typically shape is (npix,) or rarely (freq, npix)
        # Flatten to 1D
        haslam_map = haslam_map.flatten()
    elif haslam_map.ndim > 2:
        print(f"Map is {haslam_map.ndim}D, flattening...")
        haslam_map = haslam_map.flatten()
    
    print(f"Final Haslam map shape: {haslam_map.shape}")
    print(f"Map size: {len(haslam_map)} pixels")
    
    # Calculate NSIDE from map size
    npix = len(haslam_map)
    nside = hp.npix2nside(int(npix))
    print(f"Calculated NSIDE: {nside}\n")
    
    return haslam_map, nside

if __name__ == "__main__":
    read_haslam_map()