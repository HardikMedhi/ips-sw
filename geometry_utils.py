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

def get_p_point_dist(obs_date: Time, elongation: float):
    elongation_rad = np.deg2rad(np.abs(elongation))
    d_sun_au = get_earth(obs_date).radius.to(u.AU)

    p_point_sun_au = d_sun_au * np.sin(elongation_rad) if elongation <= 90 else np.nan
    p_point_earth_au = d_sun_au * np.cos(elongation_rad) if elongation <= 90 else np.nan

    return p_point_sun_au.value, p_point_earth_au.value