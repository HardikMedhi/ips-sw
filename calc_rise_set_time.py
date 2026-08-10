import yaml
import argparse
from pathlib import Path
import datetime
from astropy import units as u
from astropy.coordinates import SkyCoord, EarthLocation
from astroplan import Observer
from astropy.time import Time
from pytz import timezone

def main(args):
    ra, dec, date, tel = args
    time_rise_ist, time_set_ist = calc_rise_set_time(ra, dec, date, tel)
    return time_rise_ist, time_set_ist

def get_args():
    parser = argparse.ArgumentParser(
        description="Program to calculate the IST rise and set time of a given J2000 coordinate.",
        prefix_chars="@"
    )

    parser.add_argument(
        "ra", type=str,
        help="RA (J2000) Coordinate in HMS format."
    )
    parser.add_argument(
        "dec", type=str,
        help="Dec (J2000) Coordinate in DMS format."
    )
    parser.add_argument(
        "date", type=str,
        help="Date around which the rise and set times are to be calculated. In YYYY-MM-DD format."
    )
    parser.add_argument(
        "@tel", type=str, default="ort",
        help="Telescope choice between the ORT and the GMRT. Default is the ORT."
    )
    args = parser.parse_args()

    ra = args.ra
    dec = args.dec
    date = args.date
    tel = args.tel.lower()
    
    return ra, dec, date, tel

def calc_rise_set_time(ra: str, dec: str, date_str: str, tel: str):
    # 1. Parse coordinates safely
    obj_coord = SkyCoord(ra=ra, dec=dec, unit=(u.hour, u.degree), frame='icrs')
    
    # 2. Setup your observer
    tel_loc = get_tel_loc(tel)
    observer = Observer(location=tel_loc)
    
    # 3. Create a reliable IST Local Midnight anchor
    tz_ist = timezone("Asia/Kolkata")
    # Expects date_str in 'YYYY-MM-DD' format
    yr, mo, dy = map(int, date_str.split('-'))
    
    local_midnight = datetime.datetime(yr, mo, dy, 0, 0, 0)
    local_midnight_aware = tz_ist.localize(local_midnight)
    
    # Automatically converts IST midnight to the exact matching UTC time scale
    t_now = Time(local_midnight_aware)

    # Find the very next rise after local midnight
    time_rise_utc = observer.target_rise_time(t_now, obj_coord, which='next')
    time_rise_ist = time_rise_utc.to_datetime(timezone=tz_ist)
    
    # Base the set time search from the RISE time to keep them paired as a single event
    time_set_utc = observer.target_set_time(time_rise_utc, obj_coord, which='next')
    time_set_ist = time_set_utc.to_datetime(timezone=tz_ist)
    
    return time_rise_ist, time_set_ist

def get_tel_loc(telescope:str):
    tel_info_filepath = Path("/data/PhD/thesis/code2/tel_info.yaml")
    with open(tel_info_filepath, "r") as file:
        tel_info = yaml.safe_load(file)

    if telescope.lower() == "ort":
        loc = EarthLocation.from_geodetic(
            lon=tel_info['ort']['lon']*u.degree,
            lat=tel_info['ort']['lat']*u.degree,
            height=tel_info['ort']['height']*u.m
        )
    elif telescope.lower() == "gmrt":
        loc = EarthLocation.from_geodetic(
            lon=tel_info['gmrt']['lon']*u.degree,
            lat=tel_info['gmrt']['lat']*u.degree,
            height=tel_info['gmrt']['height']*u.m
        )
    else:
        print(f"Telescope choices are between ORT and GMRT only, not {telescope}.")
        loc = None
    
    return loc

if __name__ == "__main__":
    args = get_args()
    time_rise_ist, time_set_ist = main(args)
    print(f"Rise - {time_rise_ist.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"Set - {time_set_ist.strftime('%Y-%m-%d %H:%M:%S %Z')}")