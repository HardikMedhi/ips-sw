import astropy.units as u
from astropy.coordinates import EarthLocation

ORT = EarthLocation.from_geodetic(lon=76.66*u.deg, lat=11.38*u.deg, height=2240*u.m)
GMRT = EarthLocation.from_geodetic(lon=74.0497*u.deg, lat=19.0965*u.deg, height=650*u.m)