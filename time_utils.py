import astropy.units as u
from astropy.time import Time, TimezoneInfo

def mjd_to_datetime(mjd: str):
    """Converts MJD to a datetime object."""
    t = Time(mjd, format='mjd', scale='utc')
    ist_offset = TimezoneInfo(utc_offset=5.5*u.hour)
    
    return t.to_datetime(timezone=ist_offset)