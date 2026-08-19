import argparse
import numpy as np
from astropy.coordinates import SkyCoord, FK5
from astropy.time import Time
import astropy.units as u

def main(args):
    ra, dec = args

    precessed_coords, present_time = precess_coords(ra, dec)
    precess_coords_hmsdms = precessed_coords.to_string(style='hmsdms')

    ra_precess = precess_coords_hmsdms.split(" ")[0]
    dec_precess = precess_coords_hmsdms.split(" ")[1]

    ra_precess = ra_precess.split(".")[0] + "." + ra_precess.split(".")[1][:2]
    dec_precess = dec_precess.split(".")[0] + "." + dec_precess.split(".")[1][:2]

    ra_precess = ra_precess.replace("h", ":")
    ra_precess = ra_precess.replace("m", ":")

    dec_precess = dec_precess.replace("d", ":")
    dec_precess = dec_precess.replace("m", ":")

    return ra_precess, dec_precess, present_time

def get_args():
    parser = argparse.ArgumentParser(
        description="Program to parse J2000 coordinates to the present year.",
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
    args = parser.parse_args()

    ra = args.ra
    dec = args.dec

    return ra, dec

def precess_coords(ra:str, dec:str):
    present_time = f"J{Time.now().jyear:.3f}"

    coord = SkyCoord(ra=ra, dec=dec, unit=(u.hour, u.degree), frame='icrs')
    coord_fk5 = coord.transform_to('fk5')
    coord_fk5_now = coord_fk5.transform_to(FK5(equinox=present_time))

    return coord_fk5_now, present_time

if __name__ == "__main__":
    args = get_args()
    ra_precess, dec_precess, present_time = main(args)
    print(f"Precessed to {int(float(present_time[1:]))} (RA Dec): {ra_precess} {dec_precess}")