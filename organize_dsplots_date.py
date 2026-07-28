from pathlib import Path
import pickle

import time_utils as tut

PLOTS_DIR = Path("~/PhD/plots/ds").expanduser()
#PLOTS_DIR = Path("/data/PhD/thesis/plots/ds/")

def get_datefile_dict():
    file_paths = [p for p in PLOTS_DIR.iterdir() if p.is_file()]
    date_file_dict = {}

    for f in file_paths:
        stem_parts = f.stem.split("_")

        mjd = stem_parts[-5] if 'bpnorm' not in stem_parts else stem_parts[-6]
        mjd = mjd.split(".")[0]

        date = tut.mjd_to_datetime(mjd).strftime('%Y%m%d')

        if date not in date_file_dict.keys():
            date_file_dict[date] = []

        date_file_dict[date].append(f)

    date_file_dict = {k:date_file_dict[k] for k in sorted(date_file_dict)}

    # pickle_filename = PLOTS_DIR / "date_file_dict.pkl"
    # with open(pickle_filename, "wb") as file:
    #     pickle.dump(date_file_dict, file)

    return date_file_dict

def make_date_folder(date: str):
    folder_path = PLOTS_DIR / date
    folder_path.mkdir(exist_ok=True)

    return folder_path

def move_file(source_file_path: Path, dest_folder_path: Path):
    file_name = source_file_path.name
    dest_file_path = dest_folder_path / file_name
    source_file_path.rename(dest_file_path)

def organize():
    date_file_dict = get_datefile_dict()
    
    for k, v in date_file_dict.items():
        dest_folder_path = make_date_folder(k)
        for source_file_path in v:
            move_file(source_file_path, dest_folder_path)

if __name__ == "__main__":
    organize()

