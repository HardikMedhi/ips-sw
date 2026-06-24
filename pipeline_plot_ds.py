"""
pipeline_plot_ds.py

Daily pipeline for generating dynamic spectrum plots from IPS filterbank data.

Workflow:
    1. Fetch a list of new .fil files from the remote server via SSH.
    2. Skip files whose plots already exist (idempotency check).
    3. Download each new file via rsync.
    4. Generate the raw and bandpass-normalised dynamic spectra in parallel.
    5. Delete the raw .fil file to free up disk space.
    6. Organise all plots into date-based subdirectories.
"""

import subprocess
import logging
import gc
import multiprocessing
import sys
import os
from contextlib import contextmanager
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# Suppress any print statements emitted at import time by custom modules
with open(os.devnull, 'w') as _devnull:
    _stdout = sys.stdout
    sys.stdout = _devnull
    import thesis.code2.plot_ds as dsp
    from filterbank import Filterbank
    import data_process as dpr
    import organize_plots_date as opd
    sys.stdout = _stdout

# ==========================================
# Configuration Variables
# ==========================================
REMOTE_USER = "pulsar1"
REMOTE_HOST = "pulsar1"
REMOTE_DIR = "/data/ips"
LOCAL_DATA_DIR = Path("/data/IPS/")
PLOT_DIR = Path("/home/hardikmedhi/PhD/plots/ds")
START_DATE = '2026-04-01'
LOG_FILE = Path("/home/hardikmedhi/PhD/logs/pipeline_plot_ds.log")

# ==========================================
# Logging Setup
# ==========================================
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.FileHandler(LOG_FILE)]
)

# ==========================================
# Utilities
# ==========================================

@contextmanager
def suppress_stdout():
    """Context manager that redirects stdout to /dev/null.

    Used to silence print statements emitted by custom modules
    during data loading, processing, and plotting calls.
    stdout is always restored, even if an exception is raised.
    """
    with open(os.devnull, 'w') as _devnull:
        _stdout = sys.stdout
        sys.stdout = _devnull
        try:
            yield
        finally:
            sys.stdout = _stdout

# ==========================================
# Remote File Discovery
# ==========================================

def get_recent_remote_files() -> list:
    """Fetch filterbank files newer than START_DATE from the remote server via SSH.

    Uses `find -newermt` on the remote host to list all files under REMOTE_DIR
    that were created or modified after START_DATE.

    Returns:
        list[str]: Relative file paths (e.g. 'source/3C459_61176.236111_ort.fil'),
                   or an empty list if the SSH command fails.
    """
    find_cmd = f"cd {REMOTE_DIR} && find . -type f -newermt {START_DATE}"
    ssh_cmd = ["ssh", f"{REMOTE_USER}@{REMOTE_HOST}", find_cmd]

    try:
        result = subprocess.run(ssh_cmd, capture_output=True, text=True, check=True)
        # Strip the leading './' that `find` prepends to each path
        files = [line.lstrip("./") for line in result.stdout.strip().split("\n") if line]
        return files
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to fetch remote file list: {e.stderr}")
        return []

# ==========================================
# Idempotency Check
# ==========================================

def check_plots_exist(filename: str) -> bool:
    """Check whether both output plots already exist for a given filterbank file.

    Looks for the raw dynamic spectrum and the bandpass-normalised dynamic
    spectrum JPEGs in PLOT_DIR. If both are present, the file can be skipped.

    Args:
        filename (str): The filterbank filename, e.g. '3C459_61176.236111_ort.fil'.

    Returns:
        bool: True if both plots exist, False otherwise.
    """
    # Drop the '_ort.fil' suffix to get the shared base prefix used in plot filenames
    base_prefix = filename.replace("_ort.fil", "")

    dyn_spec_pattern = f"{base_prefix}_334.50_318.56_dyn_spec.jpeg"
    bpnorm_pattern = f"{base_prefix}_334.50_318.56_dyn_spec_bpnorm.jpeg"

    dyn_spec_matches = list(PLOT_DIR.glob(dyn_spec_pattern))
    bpnorm_matches = list(PLOT_DIR.glob(bpnorm_pattern))

    return len(dyn_spec_matches) > 0 and len(bpnorm_matches) > 0

# ==========================================
# Data Transfer
# ==========================================

def download_data(clean_filepath: str) -> bool:
    """Download a single filterbank file from the remote server via rsync.

    Args:
        clean_filepath (str): Relative path to the file on the remote server,
                              e.g. 'source/3C459_61176.236111_ort.fil'.

    Returns:
        bool: True if the download succeeded, False otherwise.
    """
    remote_target = f"{REMOTE_USER}@{REMOTE_HOST}:{REMOTE_DIR}/{clean_filepath}"
    rsync_cmd = ["rsync", "-avz", "--progress", remote_target, str(LOCAL_DATA_DIR) + "/"]

    try:
        logging.info(f"Downloading: {clean_filepath}...")
        subprocess.run(rsync_cmd, check=True, stdout=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Rsync failed for {clean_filepath}: {e}")
        return False

def cleanup_data(local_file: Path):
    """Delete a local filterbank file after its plots have been generated.

    Frees up disk space by removing the raw .fil file. The file is retained
    if deletion fails, so it can be inspected manually.

    Args:
        local_file (Path): Path to the local .fil file to delete.
    """
    try:
        local_file.unlink(missing_ok=True)
        logging.info(f"Cleaned up raw data: {local_file.name}")
    except OSError as e:
        logging.error(f"Failed to delete {local_file.name}: {e}")

# ==========================================
# Plot Generation
# ==========================================

def _run_visualize_ds(args: tuple):
    """Multiprocessing worker that calls dsp.visualize_ds and cleans up memory.

    Designed to be dispatched via multiprocessing.Pool.map. Each worker
    handles one visualize_ds call, then explicitly triggers garbage collection
    to release the matrix memory held in the subprocess.

    Args:
        args (tuple): Positional arguments to forward to dsp.visualize_ds.
    """
    try:
        with suppress_stdout():
            dsp.visualize_ds(*args)
    finally:
        gc.collect()

def make_plots(file_path: Path, save_folder_path: Path):
    """Load a filterbank file and generate both dynamic spectrum plots in parallel.

    Reads the filterbank data, computes the bandpass-normalised matrix, then
    dispatches two dsp.visualize_ds calls concurrently — one for the raw dynamic
    spectrum and one for the bandpass-normalised version.

    Args:
        file_path (Path): Path to the local .fil filterbank file.
        save_folder_path (Path): Directory where the output JPEGs will be saved.
    """
    with suppress_stdout():
        fb_obj = Filterbank(file_path)

    # Pack arguments for each visualize_ds call
    args_ds = (
        fb_obj,
        None, None,
        False, save_folder_path
    )

    args_bpnorm = (
        fb_obj,
        None, None,
        True, save_folder_path
    )

    logging.info("Making dynamic spectrum and BP-normalised dynamic spectrum...")
    with multiprocessing.Pool(processes=2) as pool:
        pool.map(_run_visualize_ds, [args_ds, args_bpnorm])

    gc.collect()

# ==========================================
# Pipeline Steps
# ==========================================

def setup_directories():
    """Create the local data and plot output directories if they don't exist."""
    LOCAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

def filter_new_files(remote_files: list) -> list:
    """Identify which remote files still need to be processed.

    Iterates over the candidate files and excludes any whose output plots
    are already present on disk. Logs each skipped file.

    Args:
        remote_files (list[str]): Candidate relative file paths from the remote server.

    Returns:
        list[str]: Subset of remote_files that require downloading and plotting.
    """
    files_to_process = []
    for clean_filepath in remote_files:
        filename = clean_filepath.split("/")[-1]
        if check_plots_exist(filename):
            logging.info(f"Skipping {filename}: both plots already exist.")
        else:
            files_to_process.append(clean_filepath)
    return files_to_process

def process_file(clean_filepath: str):
    """Run the full download → plot → cleanup pipeline for a single filterbank file.

    Downloads the file, generates the dynamic spectrum plots, then deletes the
    raw .fil file. If plot generation fails, the raw file is retained for debugging.

    Args:
        clean_filepath (str): Relative path to the file on the remote server.
    """
    filename = clean_filepath.split("/")[-1]
    local_file = LOCAL_DATA_DIR / filename

    logging.info("-" * 40)
    logging.info(f"Processing: {filename}")

    if not download_data(clean_filepath):
        logging.error(f"Skipping {filename}: download failed.")
        return

    try:
        logging.info(f"Generating plots for {filename}...")
        make_plots(local_file, PLOT_DIR)
        logging.info(f"Successfully generated plots for {filename}.")
        cleanup_data(local_file)
    except Exception as e:
        # Broad catch for errors from NumPy, Astropy, or the custom plotting modules
        logging.error(f"Plot generation failed for {filename}: {e}")
        cleanup_data(local_file)

# ==========================================
# Main Execution Flow
# ==========================================

def main():
    """Entry point for the daily IPS dynamic spectrum pipeline.

    Orchestrates the full run: directory setup, remote file discovery,
    idempotency filtering, per-file processing, and final plot organisation.
    """
    logging.info("Starting daily pipeline scan...")

    setup_directories()

    remote_files = get_recent_remote_files()
    if not remote_files:
        logging.info("No new files found on the remote server.")
        return

    files_to_process = filter_new_files(remote_files)

    for clean_filepath in tqdm(files_to_process, desc="Processing files", unit="file", disable=not sys.stdout.isatty()):
        process_file(clean_filepath)

    logging.info("Organising plots by date...")
    with suppress_stdout():
        opd.organize()

    logging.info("Pipeline execution complete.")

if __name__ == "__main__":
    main()