
import subprocess
import logging
import gc
import multiprocessing
import sys
import os
import pickle
from contextlib import contextmanager
from pathlib import Path
from tqdm import tqdm
import itertools as it
import warnings
warnings.filterwarnings("ignore")

# Suppress any print statements emitted at import time by custom modules
with open(os.devnull, 'w') as _devnull:
    _stdout = sys.stdout
    sys.stdout = _devnull
    import plot_ds as dsp
    from filterbank import Filterbank
    import data_process as dpr
    import organize_dsplots_date as opd
    import time_utils as tut
    sys.stdout = _stdout

# ==========================================
# Configuration Variables
# ==========================================
REMOTE_USER = "pulsar1"
REMOTE_HOST = "pulsar1"
REMOTE_DIR = "/data/ips"
LOCAL_DATA_DIR = Path("/data/IPS")
PLOT_DIR = Path("~/PhD/plots/ps").expanduser()
START_DATE = '2026-04-01'
LOG_FILE = Path("/home/hardikmedhi/PhD/logs/pipeline_plot_ps.log")

F1 = 330 #Hz
F2 = 322 #Hz

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
    find_cmd = f"find {REMOTE_DIR} -type f -newermt {START_DATE}"
    ssh_cmd = ["ssh", f"{REMOTE_USER}@{REMOTE_HOST}", find_cmd]

    try:
        result = subprocess.run(ssh_cmd, capture_output=True, text=True, check=True)
        # Strip the leading './' that `find` prepends to each path
        files = [line for line in result.stdout.strip().split("\n") if line]
        return files
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to fetch remote file list: {e.stderr}")
        return []

# ==========================================
# Idempotency Check
# ==========================================

def get_onoff_pairs(files: list):
    pairs_dict = {}
    date_file_dict = {}

    for f in files:
        file = Path(f)
        mjd = file.name.split("_")[-2]
        dt = tut.mjd_to_datetime(mjd).strftime("%Y%m%d")

        if dt not in date_file_dict.keys():
            date_file_dict[dt] = []

        date_file_dict[dt].append(file)

    for k, v in date_file_dict.items():
        onsrcs = []
        offsrcs = []

        for f in v:
            name = Path(f).name
            if '_off_' in name.lower():
                offsrcs.append(f)
            else:
                onsrcs.append(f)

        pairs = list(it.product(onsrcs, offsrcs))
        pairs_dict[k] = pairs

    return pairs_dict

# ==========================================
# Data Transfer
# ==========================================

def download_data(filepath: str) -> bool:
    """Download a single filterbank file from the remote server via rsync.

    Args:
        clean_filepath (str): Relative path to the file on the remote server,
                              e.g. 'source/3C459_61176.236111_ort.fil'.

    Returns:
        bool: True if the download succeeded, False otherwise.
    """
    remote_target = f"{REMOTE_USER}@{REMOTE_HOST}:{filepath}"
    rsync_cmd = ["rsync", "-avz", "--progress", remote_target, str(LOCAL_DATA_DIR) + "/"]

    try:
        logging.info(f"Downloading: {filepath}...")
        subprocess.run(rsync_cmd, check=True, stdout=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Rsync failed for {filepath}: {e}")
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
# Pipeline Steps
# ==========================================

def setup_directories():
    """Create the local data and plot output directories if they don't exist."""
    LOCAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

def get_files_process(pairs_dict: dict) -> dict:
    pairs_dict_flags = pairs_dict.copy()

    for k, v in pairs_dict.items():
        date_folder_path = PLOT_DIR / k
        if not date_folder_path.exists() or list(date_folder_path.iterdir()) == 0:
            date_folder_path.mkdir(exist_ok=True)
            for pair in v:
                list(pair).append(True)
            pairs_dict_flags[k] = v
            continue

        for pair in v:
            pair = list(pair)
            onsrc_path = Path(pair[0])
            offsrc_path = Path(pair[1])

            mjd = onsrc_path.name.split("_")[-2]

            onsrc_name = "_".join(onsrc_path.name.split("_")[0:-2])
            offsrc_name = "_".join(offsrc_path.name.split("_")[0:-2])

            plot_filename = f"{onsrc_name}_{offsrc_name}_{mjd}_{F1:.2f}_{F2:.2f}_scint_power_spec.jpeg"
            plot_filepath = date_folder_path / plot_filename

            if plot_filepath.exists():
                logging.info(f"{plot_filepath} exists. Skipping.")
                pair.append(False)
            else:
               # logging.info(f"{plot_filepath} doesn't exist. Proceeding.")
                pair.append(True)

    files_to_process = {}
    for k,v in pairs_dict_flags.items():
        files_to_process[k] = [
            [pair[0], pair[1]]
            for pair in v
            if pair[-1]
        ]

    return files_to_process   

def process_pair(date:str, pair: list):
    local_onsrc_path = LOCAL_DATA_DIR / pair[0].name
    local_offsrc_path = LOCAL_DATA_DIR / pair[1].name

    if not Path(local_onsrc_path).exists():
        download_data(pair[0])

    if not Path(local_offsrc_path).exists():
        download_data(pair[1])

    cmd = f"python3 plot_ps_scint.py --f1 {F1} --f2 {F2} --uni-stat-avg 4 {local_onsrc_path} --offsrc {local_offsrc_path} --save {PLOT_DIR / date}"
    try:
        subprocess.run(cmd.split(" "), capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to process {local_onsrc_path.name} and {local_offsrc_path.name}: {e.stderr}\n{cmd}")
    finally:
        return [local_onsrc_path, local_offsrc_path]

def process_date_files(item: tuple):
    date, pairs = item
    file_paths = []

    for pair in pairs:
        paths = process_pair(date, pair)
        file_paths.append(paths)

    return file_paths

def get_total_num_pairs(pairs_dict: dict):
    total = 0
    for v in pairs_dict.values():
        total += len(v)

    return total

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
    
    logging.info("Getting on-off pairs")
    pairs_dict = get_onoff_pairs(remote_files)

    logging.info("Getting the pairs to be processed.")
    files_to_process = get_files_process(pairs_dict)

    if len(files_to_process.items()) == 0:
        logging.info("No pairs to process.")
        return

    logging.info(f"Processing the pairs.\n{get_total_num_pairs(files_to_process)} out of {get_total_num_pairs(pairs_dict)}")
    file_paths = []

    # processes = 5#max(1, min(len(files_to_process), max(1, multiprocessing.cpu_count() - 5)))
    # with multiprocessing.Pool(processes=processes) as pool:
    #     for paths in tqdm(
    #         pool.imap(process_date_files, files_to_process.items()),
    #         total=len(files_to_process),
    #         desc="Processing dates",
    #     ):
    #         file_paths.extend(paths)

    # logging.info("Cleaning data.")
    # file_paths = sum(file_paths, [])
    # for f in file_paths:
    #     cleanup_data(f)

    # logging.info("Pipeline execution complete.")

if __name__ == "__main__":
    main()