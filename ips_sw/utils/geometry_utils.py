import numpy as np
from astropy.coordinates import SkyCoord, get_sun, GCRS
from astropy.time import Time
from astropy import units as u
from sunpy.coordinates import get_earth    

def get_solar_elongation(mjd:str, src_coords:SkyCoord) -> float:
    """
    Calculates signed solar elongation. 
    Positive = East of Sun (Prograde), Negative = West of Sun (Retrograde).
    """
    if src_coords is None:
        print("Cannot calculate solar elongation because no source coordinates are given.")
        return None

    t = Time(mjd, format='mjd', scale='utc')
    
    # If coords aren't passed, look up by name    
    src_coords_current = src_coords.transform_to(GCRS(obstime=t))
    sun_coords = get_sun(t)

    elong = sun_coords.separation(src_coords_current).deg

    # Compute RA difference in degrees
    dra = (src_coords_current.ra.deg - sun_coords.ra.deg + 360) % 360

    if dra < 180:
        side = +1   # source east of Sun
    else:
        side = -1   # source west of Sun

    return elong * side

def get_solar_elongation_arr(mjds: str | float | list[str | float] | np.ndarray, src_coords: SkyCoord) -> np.ndarray[float]:
    """Get solar elongations for many MJDs and sources.

    Args:
        mjds: One or more MJD values as a scalar, list, or 1D array.
        src_coords: A single SkyCoord object containing all source coordinates.

    Raises:
        TypeError: If src_coords is not a SkyCoord object.
        ValueError: If src_coords is empty or mjds is not 1D-like.

    Returns:
        np.ndarray: A 2D array of signed elongations with shape (n_mjds, n_sources).
        Positive values are east of the Sun; negative values are west of the Sun.
    """
    if not isinstance(src_coords, SkyCoord):
        raise TypeError("src_coords must be a SkyCoord object containing all source coordinates.")

    if len(src_coords) == 0:
        raise ValueError("Cannot calculate solar elongation because no source coordinates are given.")

    mjd_arr = np.atleast_1d(mjds)
    if mjd_arr.ndim != 1:
        raise ValueError("mjds must be a scalar or a 1D array-like of MJD values.")

    time_arr = Time(mjd_arr, format='mjd', scale='utc')
    sun_pos_arr = get_sun(time_arr)

    sun_pos_arr = sun_pos_arr[:, np.newaxis]
    src_coords = src_coords[np.newaxis, :]

    seps = sun_pos_arr.separation(src_coords).deg
    dra = (src_coords.ra.deg - sun_pos_arr.ra.deg + 360) % 360
    signs = np.where(dra < 180, 1, -1)

    elongs = signs * seps
    return elongs


def get_p_point_dist(obs_date: Time, elongation:np.array) -> tuple[np.ndarray, np.ndarray]:
    # Ensure elongation is a numpy array for vector operations
    elongation = np.asarray(elongation)
    elongation_rad = np.deg2rad(np.abs(elongation))
    
    # Extract the raw float array immediately to avoid Astropy Quantity clashes with np.where
    d_sun_au = get_earth(obs_date).radius.to_value(u.AU)

    # Use np.where instead of if/else to handle arrays
    valid_mask = elongation <= 90
    
    p_point_sun_au = np.where(valid_mask, d_sun_au * np.sin(elongation_rad), np.nan)
    p_point_earth_au = np.where(valid_mask, d_sun_au * np.cos(elongation_rad), np.nan)

    return p_point_sun_au, p_point_earth_au

def get_R_turnover(freq:float) -> float:
    return 10**3.3 * freq ** (-0.7) #Solar Radii