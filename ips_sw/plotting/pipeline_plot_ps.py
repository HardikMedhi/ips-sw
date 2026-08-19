
import subprocess
import logging
import multiprocessing
import sys
import os
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
    import ips_sw.utils.time_utils as tut
    sys.stdout = _stdout

# ==========================================
# Configuration Variables
# ==========================================
REMOTE_USER = "pulsar1"
REMOTE_HOST = "192.168.200.111"
REMOTE_DIR = "/data/ips"
LOCAL_DATA_DIR = Path("/data/IPS")
PLOT_DIR = Path("~/PhD/plots/ps").expanduser()
START_DATE = '2026-04-01'
LOG_FILE = Path("/home/hardikmedhi/PhD/logs/pipeline_plot_ps.log")

F1 = 330 #Hz
F2 = 322 #Hz

USE_MULTIPROCESSING = True
NUM_PROCESS = 5

RSYNC_TIMEOUT_S = 200
PLOT_TIMEOUT_S = 200
DATE_TASK_TIMEOUT_S = 250
POOL_PROCESSES = 3

# ==========================================
# Logging Setup
# ==========================================
from logging.handlers import RotatingFileHandler

LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        RotatingFileHandler(
            LOG_FILE, 
            maxBytes=10 * 1024 * 1024,  # Limits each log file to 10 MB
            backupCount=5               # Keeps only the 5 most recent backup files
        )
    ]
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

        pairs = [list(l) for l in list(it.product(onsrcs, offsrcs))]
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
        subprocess.run(
            rsync_cmd,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=RSYNC_TIMEOUT_S,
        )
        return True
    except subprocess.TimeoutExpired:
        logging.error(f"Rsync timed out for {filepath} after {RSYNC_TIMEOUT_S}s")
        return False
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
        logging.info(f"Cleaned up raw data: {local_file}")
    except OSError as e:
        logging.error(f"Failed to delete {local_file}: {e}")

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
        date_folder_path.mkdir(exist_ok=True)

        for pair in v:
            onsrc_path = Path(pair[0])
            offsrc_path = Path(pair[1])

            mjd = onsrc_path.name.split("_")[-2]

            onsrc_name = "_".join(onsrc_path.name.split("_")[0:-2])
            offsrc_name = "_".join(offsrc_path.name.split("_")[0:-2])

            plot_filename = f"{onsrc_name}_{offsrc_name}_{mjd}_{F1:.2f}_{F2:.2f}_scint_power_spec.jpeg"
            plot_filepath = date_folder_path / plot_filename

            pair.append(plot_filepath.exists())

    files_to_process = {}
    for k,v in pairs_dict_flags.items():
        files_to_process[k] = [
            [pair[0], pair[1]]
            for pair in v
            if not pair[-1]
        ]

    return files_to_process   

def process_pair(date:str, pair: list):
    local_onsrc_path = LOCAL_DATA_DIR / pair[0].name
    local_offsrc_path = LOCAL_DATA_DIR / pair[1].name

    if not Path(local_onsrc_path).exists():
        if not download_data(pair[0]):
            logging.error(f"Skipping pair due to failed on-source download: {pair[0]}")
            return [local_onsrc_path, local_offsrc_path]

    if not Path(local_offsrc_path).exists():
        if not download_data(pair[1]):
            logging.error(f"Skipping pair due to failed off-source download: {pair[1]}")
            return [local_onsrc_path, local_offsrc_path]

    if not local_onsrc_path.exists() or not local_offsrc_path.exists():
        logging.error(
            f"Skipping pair because local files are missing: {local_onsrc_path}, {local_offsrc_path}"
        )
        return [local_onsrc_path, local_offsrc_path]

    cmd = [
        "python3",
        "/home/hardikmedhi/PhD/ips-sw/plot_ps_scint.py", 
        "--f1",
        str(F1),
        "--f2",
        str(F2),
        "--uni-stat-avg",
        "4",
        str(local_onsrc_path),
        "--offsrc",
        str(local_offsrc_path),
        "--save",
        str(PLOT_DIR / date),
    ]

    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=PLOT_TIMEOUT_S)
        logging.info(cmd)
    except subprocess.TimeoutExpired:
        logging.error(
            f"Timeout while processing {local_onsrc_path.name} and {local_offsrc_path.name} after {PLOT_TIMEOUT_S}s"
        )
    except subprocess.CalledProcessError as e:
        logging.error(
            f"Failed to process {local_onsrc_path.name} and {local_offsrc_path.name}: {e.stderr}\n{' '.join(cmd)}"
        )
    else:
        logging.info("Plot successfully saved!")

    return [local_onsrc_path, local_offsrc_path]

def process_date_files(item: tuple):
    date, pairs = item
    file_paths = []

    for pair in pairs:
        try:
            paths = process_pair(date, pair)
            file_paths.append(paths)
        except Exception:
            logging.exception(
                f"Unexpected error while processing pair for date {date}: {pair}"
            )

    return file_paths

def get_total_num_pairs(pairs_dict: dict):
    total = 0
    for v in pairs_dict.values():
        total += len(v)

    return total

# ==========================================
# Main Execution Flow
# ==========================================

def main_mp():
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

    num_pairs_to_process = get_total_num_pairs(files_to_process)
    if num_pairs_to_process == 0:
        logging.info("No pairs to process.")
        return

    logging.info(f"Processing the pairs.\n{num_pairs_to_process} out of {get_total_num_pairs(pairs_dict)}")
    print(f"Processing the pairs.\n{num_pairs_to_process} out of {get_total_num_pairs(pairs_dict)}")

    file_paths = []

    try:
        with multiprocessing.Pool(processes=NUM_PROCESS) as pool:
            async_results = [
                pool.apply_async(process_date_files, (item,)) 
                for item in files_to_process.items()
            ]

            for result in tqdm(async_results, total=len(async_results), desc="Processing dates"):
                try:
                    paths = result.get(timeout=DATE_TASK_TIMEOUT_S)
                    file_paths.extend(paths)
                except multiprocessing.TimeoutError:
                    logging.error(f"Task exceeded {DATE_TASK_TIMEOUT_S}s timeout. Terminating remaining.")
                    pool.terminate()  # Instantly kills remaining workers
                    break             # Bypasses the need for a 'timed_out' flag
                except Exception:
                    logging.exception("Unhandled worker error during multiprocessing execution.")
            
            # Cleanly shut down the pool if we didn't terminate early
            pool.close()
            pool.join()

    except Exception:
        logging.exception("Unhandled error during multiprocessing execution.")
    finally:
        logging.info("Cleaning data.")
        file_paths = list(it.chain.from_iterable(file_paths))
        for f in file_paths:
            cleanup_data(f)

    logging.info("Pipeline execution complete.")

def main_sequential():
    """Entry point for the daily IPS dynamic spectrum pipeline (Sequential).

    Executes the exact same orchestration as main(), but processes all 
    files sequentially in a single process. Ideal for debugging or 
    running on highly resource-constrained environments.
    """
    logging.info("\nStarting daily pipeline scan (Sequential Mode)...")

    setup_directories()

    remote_files = get_recent_remote_files()
    if not remote_files:
        logging.info("No new files found on the remote server.")
        return
    
    logging.info("Getting on-off pairs")
    pairs_dict = get_onoff_pairs(remote_files)

    logging.info("Getting the pairs to be processed.")
    files_to_process = get_files_process(pairs_dict)
    
    num_pairs_to_process = get_total_num_pairs(files_to_process)
     
    if num_pairs_to_process == 0:
        logging.info("No pairs to process.")
        return

    logging.info(f"Processing the pairs.\n{num_pairs_to_process} out of {get_total_num_pairs(pairs_dict)}")
    print(f"Processing the pairs.\n{num_pairs_to_process} out of {get_total_num_pairs(pairs_dict)}")

    file_paths = []

    try:
        # Loop directly over the items without a multiprocessing pool
        for item in tqdm(files_to_process.items(), total=len(files_to_process), desc="Processing dates"):
            try:
                paths = process_date_files(item)
                file_paths.extend(paths)
            except Exception:
                logging.exception(f"Unhandled error during sequential execution for date: {item[0]}")
                
    except Exception:
        logging.exception("Unhandled error during overall sequential execution.")
    finally:
        logging.info("Cleaning data.")
        file_paths = list(it.chain.from_iterable(file_paths))
        for f in file_paths:
            cleanup_data(f)

    logging.info("Sequential pipeline execution complete.\n")

if __name__ == "__main__":   
    if USE_MULTIPROCESSING:
        print(f"Using {NUM_PROCESS} processes.")
        main_mp()
    else:
        main_sequential()