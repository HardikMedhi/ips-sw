import your
from astropy.io import fits
from pathlib import Path

def get_source_name(file_path: Path):
    """Extract the source name from a filename stem."""
    file_path = Path(file_path)
    stem_parts = file_path.stem.split("_")
    source_name = stem_parts[0] if stem_parts[0].lower() != 'cal' else '_'.join(stem_parts[:2])
    if 'off' in file_path.stem.lower():
         source_name += '_off'
    return source_name

def get_file_type(file_path: Path):
    """Return the supported file type based on the file extension."""
    file_extension_dict = {
        ".fits" : "fits",
        ".fit" : "fits",
        ".fil" : "filterbank",
        ".txt" : "text",
        ".csv" : "csv"
    }

    file_path = Path(file_path)
    file_ext = file_path.suffix.lower()
    if file_ext in list(file_extension_dict.keys()):
        return file_extension_dict[file_ext]
    
    raise ValueError(f"File format unrecognized: {file_ext}")

def read_filbank(fb_path: str):
    """Read a filterbank file and return its header and full data array."""
    fb_path = str(fb_path)
    fb_obj = your.Your(fb_path)
    header = fb_obj.your_header

    nsamp = header.nspectra
    data = fb_obj.get_data(nstart=0, nsamp=nsamp)

    return header, data

def read_fits(fits_path: Path):
    """Read a FITS file and adapt its metadata to the expected header interface."""
    with fits.open(fits_path) as hdul:
        # Prefer the primary HDU, then fall back to the first extension HDU.
        header_dict = dict(hdul[0].header)
        data = hdul[0].data
        
        if data is None and len(hdul) > 1:
            # Some files store the data in the first extension instead.
            data = hdul[1].data
            header_dict = dict(hdul[1].header)
    
    # Build a lightweight header-like object to match the filterbank path.
    class FitsHeader:
        pass
    
    header = FitsHeader()
    
    # Extract or infer the fields used by the rest of the pipeline.
    header.nchans = header_dict.get('NAXIS1', header_dict.get('NCHAN', data.shape[-1] if data is not None else 1))
    header.tsamp = header_dict.get('TSAMP', header_dict.get('CDELT2', 1.0))
    header.fch1 = header_dict.get('FCH1', header_dict.get('CRVAL1', 0.0))
    header.foff = header_dict.get('FOFF', header_dict.get('CDELT1', 1.0))
    header.tstart = header_dict.get('TSTART', header_dict.get('MJD-OBS', 0.0))
    header.basename = header_dict.get('BASENAME', Path(fits_path).stem)
    
    return header, data

def get_type_filepaths(parent_folder: Path, extension: str):
    """Return all files under a folder that match the given extension."""
    parent_folder = Path(parent_folder)
    return [path for path in parent_folder.rglob(f"*.{extension}")]

def handle_file_existence(filepath: Path):
    """Prompt for a filename suffix when a target output file already exists."""
    filepath = Path(filepath)
    if filepath.exists():
        print(f"File already exists: {filepath}")
        user_input = input("Do you want to append a word to the filename? (yes/no): ").strip().lower()
        
        if user_input in ['yes', 'y']:
            additional_word = input("Enter the word to append: ").strip()
            if additional_word:
                # Insert the suffix before the extension.
                filepath = filepath.parent / Path(filepath.stem + f"_{additional_word}" + filepath.suffix)
    
    return filepath