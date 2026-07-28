import argparse
import numpy as np
import pandas as pd
from pathlib import Path

import sys
sys.path.add("/data/PhD/thesis/ships")

import source_highlight as ships

def main(args:tuple):
    cat_filepath, date_start, date_end, max_num_src, precess, telescope, elong_low, elong_high = args

    outputcsv_filepath = ships.main([
        cat_filepath, date_start, date_end, precess, telescope, elong_low, elong_high
    ])

    output_df = pd.read_csv(outputcsv_filepath)
    selected_sources = bin_and_select_sources(output_df, max_num_src)



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
                        help="Precess the coordinates to start date")
    parser.add_argument("--tel", type=str, default="ort",
                        help="Choice of telescope location between ORT and GMRT. Default is ORT.")
    parser.add_argument("--elonglow", type=float, default=10,
                        help="Elongation lower limit in degrees. Default value is 10.")
    
    parser.add_argument("--elonghigh", type=float, default=90,
                        help="Elongation higher limit in degrees. Default value is 90.")
    args = parser.parse_args()

    cat_filepath = Path(parser.cat_filepath)
    date_start, date_end = parser.date_start, parser.date_end
    max_num_src = parser.max_num_src
    precess = parser.precess
    telescope = args.tel
    elong_low = args.elonglow
    elong_high = args.elonghigh

    return cat_filepath, date_start, date_end, max_num_src, precess, telescope, elong_low, elong_high

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
    return selected_sources

if __name__ == "__main__":
    args = get_args()
    main(args)
    