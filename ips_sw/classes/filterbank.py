from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
from pathlib import Path
from astropy.coordinates import SkyCoord

import ips_sw.utils.file_utils as fut
import ips_sw.utils.data_utils as dut
import ips_sw.utils.time_utils as tut
import ips_sw.utils.geometry_utils as gut

if TYPE_CHECKING:
    from your import your
    from datetime import datetime

class Filterbank:
    """Load a filterbank or FITS file and expose commonly used metadata."""

    def __init__(self, file_path: Path):
        """Read the input file and prepare derived data products."""
        self._file_path = file_path
        self._read_file()

        # Derive reusable metadata and precomputed arrays once during init.
        self._source_name = fut.get_source_name(self._file_path)
        self._mjd = self._get_mjd()
        self._matrix = dut.get_data_matrix(self._file_path)
        self._time_samples, self._freq_channels = dut.get_time_freqs(self._matrix, self._header.tsamp,
                                                                     self._header.nchans, self._header.foff,
                                                                     self._header.fch1)
                                                                     
        self._time_profile = np.nanmean(self._matrix, axis=1)
        self._freq_profile = np.nanmean(self._matrix, axis=0)

        self._source_coords = self._get_source_coords()
        self._elong = gut.get_solar_elongation(self._mjd, self._source_coords)

    def _read_file(self):
        """Read file contents using the appropriate loader for the file type."""
        if fut.get_file_type(self._file_path) == 'filterbank':
            self._header, self._data = fut.read_filbank(self._file_path)
        else:
            self._header, self._data = fut.read_fits(self._file_path)

    def _get_mjd(self) -> str:
        """Format the start time as an MJD string with fixed precision."""
        mjd = str(round(self._header.tstart, 6))

        if len(mjd) < 12:
            mjd += '0' * (12 - len(mjd))

        return mjd
    
    def _get_source_coords(self) -> (SkyCoord | None):
        """Get the ICRS source coordinates from SkyCoord"""
        source_name = self._source_name
        parts = source_name.split("_")
        if 'cal' in source_name.lower():
            source_name = parts[1]
        elif 'cal' not in source_name.lower() and 'off' in source_name.lower():
            source_name = parts[0]

        try:
            coords = SkyCoord.from_name(source_name)
        except:
            print(f"Coordinates for {self._source_name} could not be fetched.")
            return None
        return coords

    @property
    def file_path(self) -> str:
        """Return the source file path."""
        return self._file_path

    @property
    def header(self) -> your.header:
        """Return the parsed file header."""
        return self._header
    
    @property
    def data(self) -> np.ndarray:
        """Return the raw data array read from disk."""
        return self._data
    
    @property
    def source_name(self) -> str:
        """Return the derived source name."""
        return self._source_name
    
    @property
    def source_coords(self) -> SkyCoord:
        """Return the source's coordinates"""
        return self._source_coords
    
    @source_coords.setter
    def source_coords(self, coords:SkyCoord):
        """Set the source coordinates"""
        self._source_coords = coords
    
    @property
    def mjd(self) -> str:
        """Return the formatted MJD string."""
        return self._mjd
    
    @property
    def elong(self) -> float:
        """Return the solar elongation"""
        return self._elong
    
    @property
    def datetime(self) -> datetime:
        """Return the formatted datetime string."""
        return tut.mjd_to_datetime(self._mjd)
    
    @property
    def matrix(self) -> np.ndarray:
        """Return the processed data matrix."""
        return self._matrix
    
    @property
    def time_profile(self) -> np.ndarray:
        return self._time_profile
    
    @property
    def freq_profile(self) -> np.ndarray:
        return self._freq_profile

    @property
    def time_samples(self) -> np.ndarray:
        """Return the time axis values for the matrix."""
        return self._time_samples
    
    @property
    def freq_channels(self) -> np.ndarray:
        """Return the frequency axis values for the matrix."""
        return self._freq_channels


    #TODO: Add a power spectrum property!!