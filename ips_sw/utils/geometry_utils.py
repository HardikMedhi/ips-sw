import numpy as np
from astropy.coordinates import SkyCoord, get_sun, GCRS
from astropy.time import Time
from astropy import units as u
from sunpy.coordinates import get_earth

def get_solar_elongation(mjd:str, src_coords:SkyCoord):
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

def get_p_point_dist(obs_date: Time, elongation):
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

def get_R_turnover(freq:float):
    return 10**3.3 * freq ** (-0.7) #Solar Radii