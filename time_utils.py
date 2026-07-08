import astropy.units as u
from astropy.time import Time, TimezoneInfo
from datetime import datetime, timedelta

def mjd_to_datetime(mjd: str):
    """Converts MJD to a datetime object."""
    t = Time(mjd, format='mjd', scale='utc')
    ist_offset = TimezoneInfo(utc_offset=5.5*u.hour)
    
    return t.to_datetime(timezone=ist_offset)

def datetime_to_mjd(date_value, utc_offset:u=None):
    """Converts a date string or datetime object to MJD.

    If `date_value` includes a time component, `utc_offset` should be the
    local offset from UTC in hours (for example, 5.5 for IST).
    """
    if isinstance(date_value, str):
        if len(date_value) == 8 and date_value.isdigit():
            dt = datetime.strptime(date_value, "%Y%m%d")
        else:
            dt = Time(date_value).to_datetime()
    elif isinstance(date_value, datetime):
        dt = date_value
    else:
        raise TypeError("date_value must be a string or datetime object")

    if utc_offset is not None:
        dt = dt - timedelta(hours=utc_offset)

    return Time(dt, scale='utc').mjd

