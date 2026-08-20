import astropy.units as u
from astropy.time import Time, TimezoneInfo
from datetime import datetime, timedelta, timezone

def mjd_to_datetime(mjd: str) -> datetime:
    """Converts MJD to a datetime object with IST offset."""
    t = Time(mjd, format='mjd', scale='utc')
    ist_offset = TimezoneInfo(utc_offset=5.5*u.hour)
    
    return t.to_datetime(timezone=ist_offset)

def datetime_to_mjd(date_value:str|datetime, utc_offset:u=None) -> float:
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

def get_dt_obj(dt_str: str) -> Time:
    """Returns a Time object with IST offset

    Args:
        dt_str (str): date string in YYYYMMDD format

    Returns:
        Time: astropy.Time object
    """
    if len(dt_str) > 8:
        date = f"{dt_str[:4]}-{dt_str[4:6]}-{dt_str[6:8]}"
        time = f"{dt_str[9:11]}:{dt_str[11:]}"
        full_str = f"{date} {time}"
        full_format = "%Y-%m-%d %H:%M"
    else:
        date = f"{dt_str[:4]}-{dt_str[4:6]}-{dt_str[6:8]}"
        full_str = f"{date}"
        full_format = "%Y-%m-%d"
    
    # Create a datetime object and set it to IST (UTC+5:30)
    ist_tz = timezone(timedelta(hours=5, minutes=30))
    dt_obj = datetime.strptime(full_str, full_format)
    dt_obj = dt_obj.replace(tzinfo=ist_tz)
    
    # Astropy Time will now correctly interpret it and convert internally
    dt = Time(dt_obj)

    return dt
