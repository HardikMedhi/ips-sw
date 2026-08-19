import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from astropy.coordinates import SkyCoord
from astropy import units as u

import sys
sys.path.append("/data/PhD/thesis/ships")

import source_highlight as ships
import ips_sw.utils.calc_rise_set_time as rst
import ips_sw.utils.precess_coords_now as prec

FILEPATH_CALS = Path("/data/PhD/thesis/data/catalogs/ort/cals.csv")
FILEPATH_OFFSRCS = Path("/data/PhD/thesis/data/catalogs/ort/ort_offsrcs_potential.csv")
DIR_OUTPUT = Path("/data/PhD/thesis/observation_plans")

def main(args:tuple):
    cat_filepath, date_start, date_end, max_num_src, precess, telescope, elong_low, elong_high, cal_names = args

    outputcsv_filepath = ships.main([
        cat_filepath, date_start, date_end, False, False, telescope, elong_low, elong_high
    ])

    output_df = pd.read_csv(outputcsv_filepath)
    selected_sources = bin_and_select_sources(output_df, max_num_src)
    filtered_output_df = output_df[output_df['source_name'].isin(selected_sources)]

    unique_coords = filtered_output_df.drop_duplicates(subset='source_name')
    ras = unique_coords['ra_j2000'].to_numpy()
    decs = unique_coords['dec_j2000'].to_numpy()
    coords = SkyCoord(ra=ras, dec=decs, unit=(u.hour, u.degree), frame='icrs')
    ra_min, ra_max = coords.ra.degree.min(), coords.ra.degree.max() # degree

    if cal_names.size != 0:
        cal_names = ["cal_" + cal for cal in cal_names if "cal_" not in cal]

    cals_vis = get_visible_calibrators(ra_min, ra_max, cal_names)
    # offsrcs_vis = get_visible_offsrcs(ra_min, ra_max)

    # final_df = get_combined_df(filtered_output_df, cals_vis, offsrcs_vis)
    # final_df = insert_rise_set_times(final_df, date_start)
    # final_df = precess_coords(final_df)

    # final_df = final_df[[
    #     'source_name', 'ra_j2000', 'dec_j2000', 'ra_precess', 'dec_precess', 'rise_time_ist', 'set_time_ist', 'source_type', 'flux_jy'
    # ]]

    # save_csv(final_df, date_start, date_end)

def get_args():
    parser = argparse.ArgumentParser(description="Monthly Source List Maker")

    parser.add_argument("cat_filepath", type=str,
                        help="Catalog path (FITS/Text)")
    parser.add_argument("date_start", type=str,
                        help="Start YYYYMMDD")
    parser.add_argument("date_end", type=str,
                        help="End YYYYMMDD")
    parser.add_argument("max_num_src", type=int,
                        help="Maximum number of sources")
    parser.add_argument("--precess", action='store_true',
                        help="Precess the coordinates to the present year")
    parser.add_argument("--tel", type=str, default="ort",
                        help="Choice of telescope location between ORT and GMRT. Default is ORT.")
    parser.add_argument("--elonglow", type=float, default=10,
                        help="Elongation lower limit in degrees. Default value is 10.")
    parser.add_argument("--cals", type=str, action='extend', nargs='+', default=[],
                        help="List of calibrator names to be considered." \
                        "If not given, then the brightest calibrator from the catalogue will be considered.")
    
    parser.add_argument("--elonghigh", type=float, default=90,
                        help="Elongation higher limit in degrees. Default value is 90.")
    args = parser.parse_args()

    cat_filepath = Path(args.cat_filepath)
    date_start = args.date_start
    date_end = args.date_end
    max_num_src = args.max_num_src
    precess = args.precess
    telescope = args.tel
    elong_low = args.elonglow
    elong_high = args.elonghigh
    cal_names = np.array(args.cals) if len(args.cals) != 0 else np.array([])

    return cat_filepath, date_start, date_end, max_num_src, precess, telescope, elong_low, elong_high, cal_names

def bin_and_select_sources(df:pd.DataFrame, max_num_srcs:int):
    sources_unique = df['source_name'].unique()
    bin_edges = np.arange(-90, 90, 5)

    #Get number of bins for each source
    sources_elong_bins_num = {}
    for source in sources_unique:
        elongs = df[df['source_name'] == source].elongation.to_numpy()
        bin_indices = np.digitize(elongs, bin_edges)
        sources_elong_bins_num[source] = np.unique(bin_indices).size

    sources_elong_bins_num = pd.DataFrame(
        list(sources_elong_bins_num.items()),
        columns=['source_name', 'num_bins']
    ).sort_values(by='num_bins', ascending=False)

    #Get top max_num_srcs with the highest number of bins
    top_num_rows = sources_elong_bins_num.iloc[:max_num_srcs]
    selected_sources = top_num_rows['source_name']
    return selected_sources.to_numpy()

#TODO Fix this function!!
def get_visible_calibrators(ra_min:float, ra_max:float, cal_names:np.array):
    cals = pd.read_csv(FILEPATH_CALS)
    if cal_names is not None:
        mask_in_cat = cals['source_name'].isin(cal_names)
        ras = cals[mask_in_cat]['ra_j2000'].to_list()
        decs = cals[mask_in_cat]['dec_j2000'].to_list()

        if mask_in_cat.sum() != len(cal_names):
            mask_not_in_cat = ~np.isin(cal_names, cals['source_name'].to_numpy())
            not_in_cat = cal_names[mask_not_in_cat]
            print(f"Skipping the following calibrators as they are not present in the official calibrator list:")
            print("\n".join(not_in_cat))
            print("The spelling/case could be the culprit!")

    if len(ras) != 0:
        cals_coords = SkyCoord(ra=ras, dec=decs, unit=(u.hour, u.degree), frame='icrs')
    else:
        print("\nNo calibrators from the official calibrator list was specified.")
        print("Choosing the 2 brightest calibrators from the official list.\n")
        cals_coords = SkyCoord(ra=cals['ra_j2000'], dec=cals['dec_j2000'], unit=(u.hour, u.degree), frame='icrs')

    ra_mask = (ra_min <= cals_coords.ra.degree) & (cals_coords.ra.degree <= ra_max)
    cals_coords_vis = cals[ra_mask]
    cals_vis = cals_coords_vis.sort_values(by='flux_jy').reset_index(drop=True).iloc[:2]

    print(cals_vis)

    return cals_vis

def get_visible_offsrcs(ra_min:float, ra_max:float):
    offsrcs = pd.read_csv(FILEPATH_OFFSRCS)

    offsrcs_coords = SkyCoord(ra=offsrcs['ra_j2000'], dec=offsrcs['dec_j2000'], unit=(u.hour, u.degree), frame='icrs')
    
    ra_mask = (ra_min <= offsrcs_coords.ra.degree) & (offsrcs_coords.ra.degree <= ra_max)
    offsrcs_coords_vis = offsrcs[ra_mask]
    offsrcs_vis = offsrcs_coords_vis.reset_index(drop=True).iloc[:4]

    return offsrcs_vis

def get_combined_df(target:pd.DataFrame, cals:pd.DataFrame, offsrcs:pd.DataFrame):
    # 1. Deduplicate targets and select matching columns
    targets_unique = (
        target
        .drop_duplicates(subset='source_name')
        [['source_name', 'ra_j2000', 'dec_j2000', 'flux_jy']]
        .copy()
    )
    targets_unique['source_type'] = 'target'

    # 2. Add source_type to calibrators and off-sources
    cals = cals.copy()
    cals['source_type'] = 'calibrator'

    offsrcs = offsrcs.copy()
    offsrcs['source_type'] = 'off_source'

    # 3. Combine all three dataframes together
    cols_to_keep = ['source_type', 'source_name', 'ra_j2000', 'dec_j2000', 'flux_jy']

    combined_df = pd.concat(
        [
            targets_unique[cols_to_keep], 
            cals[['source_type', 'source_name', 'ra_j2000', 'dec_j2000', 'flux_jy']], 
            offsrcs[['source_type', 'source_name', 'ra_j2000', 'dec_j2000', 'flux_jy']]
        ],
        ignore_index=True
    )

    return combined_df

def insert_rise_set_times(df:pd.DataFrame, ref_date:str):
    time_rise_ist_arr = []
    time_set_ist_arr = []

    ref_date = f"{ref_date[:4]}-{ref_date[4:6]}-{ref_date[6:]}"

    for row in df.itertuples():
        ra, dec = row.ra_j2000, row.dec_j2000
        time_rise_ist, time_set_ist = rst.main([ra, dec, ref_date, 'ort'])

        time_rise_ist_arr.append(f"{time_rise_ist.strftime('%H:%M:%S')}")
        time_set_ist_arr.append(f"{time_set_ist.strftime('%H:%M:%S')}")

    temp = df.copy()
    temp['rise_time_ist'] = time_rise_ist_arr
    temp['set_time_ist'] = time_set_ist_arr

    return temp

def precess_coords(df:pd.DataFrame):
    ra_precess_arr = []
    dec_precess_arr = []

    for row in df.itertuples():
        ra, dec = row.ra_j2000, row.dec_j2000
        ra_precess, dec_precess, _ = prec.main([ra, dec])

        ra_precess_arr.append(ra_precess)
        dec_precess_arr.append(dec_precess)

    temp = df.copy()
    temp['ra_precess'] = ra_precess_arr
    temp['dec_precess'] = dec_precess_arr

    return temp

def save_csv(df:pd.DataFrame, date_start:str, date_end:str):
    DIR_OUTPUT.mkdir(parents=True, exist_ok=True)
    filename = f"obs_{date_start}_{date_end}.csv"
    filepath = DIR_OUTPUT / filename
    if filepath.exists():
        print(f"File {filepath} already exists!")
        return filepath

    df.to_csv(filepath, index=False)
    print(f"Observation plan successfully saved to {filepath}!")
    return filepath

if __name__ == "__main__":
    args = get_args()
    main(args)
    