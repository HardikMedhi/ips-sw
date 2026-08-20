import argparse
import numpy as np
import yaml
from pathlib import Path
from astropy.time import Time
from astropy import units as u
from astropy.coordinates import EarthLocation
from astroplan import Observer
from importlib.resources import files

from ips_sw.utils import time_utils as tut

def main(args:tuple):
    dt_start_str, dt_end_str, tel = args

    dt_start = tut.get_dt_obj(dt_start_str)
    dt_end = tut.get_dt_obj(dt_end_str)

    tel_info = get_tel_info(tel)
    observer = get_observer(tel_info)
    
    ra_start, ra_end = get_visible_ras(observer, dt_start, dt_end, tel_info)
    if ra_start <= ra_end:
        print(f"\nObservable RA Range: From {ra_start:.2f}h to {ra_end:.2f}h")
    else:
        print(f"\nObservable RA Range: From {ra_start:.2f}h through 0h/24h to {ra_end:.2f}h")

def get_args():
    parser = argparse.ArgumentParser(
        description="Program to obtain the visible RA range, given the date and time of observation."
    )

    parser.add_argument(
        "dt_start", type=str,
        help="Start datetime of observation in YYYYMMDDTHHMM format (24 hours) in IST"
    )
    parser.add_argument(
            "dt_end", type=str,
            help="End datetime of observation in YYYYMMDDTHHMM format (24 hours) in IST"
        )
    parser.add_argument(
        "--tel", type=str, default="ort",
        help="Telescope choice between ORT/GMRT. Default is ORT."
    )
    args = parser.parse_args()

    dt_start_str = args.dt_start
    dt_end_str = args.dt_end
    tel = args.tel.lower()

    return dt_start_str, dt_end_str, tel

def get_tel_info(tel:str):
    filepath_tel_info = files("ips_sw").joinpath("yaml_info/tel_info.yaml")
    with open(filepath_tel_info, "r") as f:
        tel_info = yaml.safe_load(f)
    return tel_info[tel]

def get_observer(tel_info:dict):
    location = EarthLocation(
            lat=tel_info['lat']*u.degree, 
            lon=tel_info['lon']*u.degree,
            height=tel_info['height']*u.m
        )
    observer = Observer(location=location)
    return observer

def get_visible_ras(observer:Observer, dt_start:Time, dt_end:Time, tel_info:dict):
    lst_start = observer.local_sidereal_time(dt_start).hourangle
    lst_end = observer.local_sidereal_time(dt_end).hourangle

    obs_lat = observer.latitude.radian
    alt_lim = np.deg2rad(obs_lat)
    ha_lim = tel_info['ha_lim']

    grid_decs = np.radians(np.linspace(tel_info['dec_low'], tel_info['dec_high'], 500))
    num = np.sin(alt_lim) - np.sin(obs_lat) * np.sin(grid_decs)
    denom = np.cos(obs_lat) * np.cos(grid_decs)

    valid = (num / denom <= 1) & (num / denom >= -1)
    max_ha_at_alt = np.max(np.arccos(num[valid] / denom[valid]))
    max_ha_hours = np.degrees(max_ha_at_alt) / 15.0

    h_eff = min(ha_lim, max_ha_hours)

    ra_start = (lst_start - h_eff) % 24
    ra_end = (lst_end + h_eff) % 24

    return ra_start, ra_end # hours

if __name__ == "__main__":
    args = get_args()
    main(args)
    