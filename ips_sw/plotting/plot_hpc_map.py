import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import pandas as pd
from pathlib import Path
from astropy import units as u
from astropy.coordinates import SkyCoord, GCRS
from astropy.time import Time
from sunpy.coordinates import frames, get_earth

import warnings
warnings.filterwarnings("ignore")

import ips_sw.utils.geometry_utils as gut

from importlib.resources import files
style_path = files("ips_sw").joinpath("matplotlib_styles/style_paper.mplstyle")
plt.style.use(style_path)

def main():
    #set_pyplot_rcparams()

    filepath_sources_csv, ref_date, sideview, topview, save_filepath, frequency, p_pt, skip_days = get_args()

    if save_filepath is not None:
        save_filepath = save_filepath.parent / (save_filepath.stem + f"{frequency:.0f}" + save_filepath.suffix)

    sources_df = pd.read_csv(filepath_sources_csv, sep=",", header=0)
    sources_df = insert_p_point_dist(sources_df)

    if ref_date is None:
        fig, ax = manage_plot_tracks(sources_df, save_filepath, p_pt, skip_days)
        plt.show()
    else:
        if sideview is False and topview is False:
            manage_plot_sources(sources_df, ref_date, save_filepath, p_pt)
        elif sideview is True and topview is False:
            manage_plot_side_view(sources_df, ref_date, save_filepath, p_pt)
        elif sideview is False and topview is True:
            manage_plot_top_view(sources_df, ref_date, save_filepath, p_pt)
        else:
            manage_plot_side_view(sources_df, ref_date, save_filepath, p_pt)
            manage_plot_top_view(sources_df, ref_date, save_filepath, p_pt)

# def set_pyplot_rcparams():
#     plt.rcParams['axes.titlesize'] = 16
#     plt.rcParams['axes.labelsize'] = 14
#     plt.rcParams['axes.titleweight'] = 'bold'
#     plt.rcParams['xtick.labelsize'] = 12
#     plt.rcParams['ytick.labelsize'] = 12
#     plt.rcParams['legend.fontsize'] = 12
            
def get_args():
    parser = argparse.ArgumentParser(
        description="Plot a Helioprojective Coordinate Map for a given list of sources" \
                    "and a reference date.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument('filepath_sources_csv', type=str,
                        help="Path to the csv file with date, source names, RA/Dec (HMS/DMS) coordinates, elongation")
    parser.add_argument('--refdate', type=str, default=None,
                        help="Reference date for plotting just the sources. Format: YYYY-MM-DD." \
                        "If you give this value, the tracks will NOT be plotted.")
    parser.add_argument('--sideview', action='store_true',
                        help="Plot a side view. This requires a reference date!")
    parser.add_argument('--topview', action='store_true',
                        help="Plot a top view. This requires a reference date!")
    parser.add_argument('--save-filepath', type=str, dest='save_filepath', default=None,
                        help="Save filepath.")
    parser.add_argument("--freq", type=float, default=326.5,
                        help="Frequency of observation in MHz. Default is 326.5 MHz.")
    parser.add_argument("--ppt", action='store_true',
                        help="Plot the p-points instead.")
    parser.add_argument("--skipdays", type=int, default=1,
                        help="Number of consecutive days to skip while plotting tracks.")
    
    args = parser.parse_args()

    filepath_sources_csv = Path(args.filepath_sources_csv)
    ref_date = args.refdate
    sideview = args.sideview
    topview = args.topview
    save_filepath = Path(args.save_filepath) if args.save_filepath is not None else None
    frequency = args.freq
    p_pt = args.ppt
    skip_days = args.skipdays

    return filepath_sources_csv, ref_date, sideview, topview, save_filepath, frequency, p_pt, skip_days

def insert_p_point_dist(sources_df: pd.DataFrame):
    sources_df_copy = sources_df.copy()

    # Convert the entire date column to an Astropy Time array
    # .tolist() is often the safest way to ingest Pandas datetime series into Astropy
    obs_dates = Time(sources_df_copy['date'].tolist())
    elongations = sources_df_copy['elongation'].to_numpy()

    # Pass the arrays directly into our newly vectorized function
    p_sun, p_earth = gut.get_p_point_dist(obs_dates, elongations)

    # Assign the resulting 1D arrays back to the dataframe
    sources_df_copy['p_point_sun_au'] = p_sun
    sources_df_copy['p_point_earth_au'] = p_earth

    return sources_df_copy

def get_hpc_coords(coords_icrs:SkyCoord, ref_date:Time):
    observer = get_earth(ref_date)    

    hpc_frame = frames.Helioprojective(observer=observer, obstime=ref_date)
    coords_gcrs = coords_icrs.transform_to(GCRS(obstime=ref_date))
    coords_hpc = coords_gcrs.transform_to(hpc_frame)

    tx_deg = coords_hpc.Tx.to(u.degree)
    ty_deg = coords_hpc.Ty.to(u.degree)

    d_sun_au = observer.radius.to(u.AU)

    x_au = d_sun_au * np.tan(np.deg2rad(tx_deg))
    y_au = d_sun_au * np.tan(np.deg2rad(ty_deg))

    return tx_deg.value, ty_deg.value, x_au.value, y_au.value

def manage_plot_sources(sources_df:pd.DataFrame, ref_date:str, save_filepath:str):
    ref_date = Time(ref_date)
    unique_sources_df = sources_df.drop_duplicates(subset=['source_name']).reset_index()

    coords_icrs = SkyCoord(
        ra=unique_sources_df.ra,
        dec=unique_sources_df.dec,
        unit=(u.hour, u.deg),
        distance=1.e9*u.pc,
        frame='icrs'
    )

    tx_deg, ty_deg, x_au, y_au = get_hpc_coords(coords_icrs, ref_date)

    plot_sources(tx_deg, ty_deg, x_au, y_au, unique_sources_df.p_point_sun_au.to_numpy(), ref_date)
    
def plot_sources(tx_deg:np.ndarray, ty_deg:np.ndarray,
                x_au:np.ndarray, y_au:np.ndarray,
                p_point_au:np.ndarray, ref_date:Time
                ):
    
    fig, ax = plt.subplots(figsize=(8, 8))

    size_factor = 50
    marker_sizes = size_factor / p_point_au

    scatter_plot = ax.scatter(
        tx_deg, ty_deg,
        s=marker_sizes,
        c='blue',
        alpha=0.7,
        edgecolors='none'
    )

    handles, labels = scatter_plot.legend_elements(
        prop="sizes", 
        num=5,                  # Number of reference sizes to show
        fmt="{x:.1f}",          # Format labels to 1 decimal place
        func=lambda s: size_factor / s,
        color='blue',
        alpha=0.7
    )
    
    # 4. Position the Legend on the right
    # bbox_to_anchor places the anchor outside the axes framework.
    size_legend = ax.legend(
        handles, labels, 
        title="P-Point Distance\nfrom the Sun\n(AU)",
        loc="best", 
        bbox_to_anchor=(0.2, 1),
        frameon=True,
        labelspacing=0.1
    )

    ax.add_artist(size_legend)

    d_sun_au = get_earth(ref_date).radius.to(u.AU).value

    sun_radius_deg = min(tx_deg.min(), ty_deg.min()) / 6

    sun_disk = plt.Circle((0, 0), sun_radius_deg, color='orange', alpha=0.6)
    ax.add_patch(sun_disk)

    # Add concentric circles
    min_radius = min(0.5, min(x_au.min(), y_au.min()))
    max_radius = np.ceil(max(x_au.max(), y_au.max()))
    radii_au = np.arange(0.5, max(x_au.max(), y_au.max())+0.5, 0.5)
    for r_au in radii_au:
        # Convert physical AU distance to angular degrees for plotting
        r_deg = np.rad2deg(np.arctan(r_au / d_sun_au))
        
        circle = plt.Circle((0, 0), r_deg, color='green', fill=False, ls=':', alpha=0.6)
        ax.add_patch(circle)
        
        # Label the whole-number AU circles (and 0.5) to avoid clutter
        if r_au.is_integer() or r_au == 0.5:
            ax.text(0, r_deg, f' {r_au} AU', color='green', alpha=0.8, 
                    ha='left', va='bottom', fontsize=8)

    ax.axhline(0, color='gray', lw=0.5, ls='--')
    ax.axvline(0, color='gray', lw=0.5, ls='--')

    limit = max(np.abs(tx_deg).max(), np.abs(ty_deg).max()) + 10
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect('equal')

    ax.set_xlabel(r'$\theta_X$ (deg)')
    ax.set_ylabel(r'$\theta_Y$ (deg)')
    ax.set_title(f'Distribution of IPS Sources\n{ref_date.iso.split(" ")[0]}')
    ax.legend()

    plt.grid(True, alpha=0.3)
    plt.show()

def manage_plot_tracks(sources_df: pd.DataFrame, save_filepath: str, p_pt: bool, skip_days: int):
    # 1. Sort the dataframe to ensure chronological tracks before downsampling
    sources_df = sources_df.sort_values(by=['source_name', 'date'])
    
    # 2. Group by source_name natively
    grouped_sources = sources_df.groupby('source_name')

    sub_df_geom_info = {}
    
    for src_name, df_sub in grouped_sources:
        # 3. Downsample early
        df_sub = df_sub.iloc[::skip_days]
        
        coords_icrs = SkyCoord(
            ra=df_sub['ra'].iloc[0],
            dec=df_sub['dec'].iloc[0],
            unit=(u.hour, u.deg),
            distance=1e9*u.pc,
            frame='icrs'
        )

        # 4. Convert the dates into a single Astropy Time array
        dates = Time(df_sub['date'].tolist())
        
        # 5. Call the natively vectorized function (returns 4 1D arrays)
        tx, ty, x, y = get_hpc_coords(coords_icrs, dates)

        # 6. Stack the arrays natively
        geom_info = np.column_stack((
            tx, ty, x, y, 
            df_sub['p_point_sun_au'].to_numpy(), 
            df_sub['p_point_earth_au'].to_numpy()
        ))
        
        sub_df_geom_info[src_name] = geom_info

    min_date = sources_df['date'].min()
    max_date = sources_df['date'].max()
    plot_title = f'Source Tracks of {len(sub_df_geom_info)} Sources\n{min_date} - {max_date}'

    fig, ax = plot_tracks(sub_df_geom_info, plot_title, save_filepath, p_pt)
    return fig, ax

def plot_tracks(geom_info_dict: dict, plot_title: str, save_filepath: str, p_pt: bool):
    fig, ax = plt.subplots(figsize=(10, 10))
    colors = plt.cm.jet(np.linspace(0, 1, len(geom_info_dict)))

    base_area = 5 
    
    # Initialize lists to collect data 
    all_x, all_y, all_sizes, all_colors = [], [], [], []
    segments = [] # List to hold the coordinate pairs for the lines

    # tx, ty, x, y, p_sun, p_earth
    for i, (k, v) in enumerate(geom_info_dict.items()):
        # if k not in ['0732+33', '0418+236']:
        #     continue
        if p_pt:
            angles = np.deg2rad(v[:, :2])
            proj_vals = np.sin(2 * angles) / 2
            x_vals, y_vals = proj_vals[:, 0], proj_vals[:, 1]
        else:
            x_vals, y_vals = v[:, 0], v[:, 1]

        p_earth = v[:, -1]
        
        # Area scaling and clipping
        marker_sizes = base_area / (p_earth ** 2)
        marker_sizes = np.clip(marker_sizes, a_min=2, a_max=60)

        # Append to scatter lists
        all_x.append(x_vals)
        all_y.append(y_vals)
        all_sizes.append(marker_sizes)
        
        num_points = len(x_vals)
        all_colors.append(np.tile(colors[i], (num_points, 1)))

        # Create the line segment for this specific track and append it
        track_points = np.column_stack([x_vals, y_vals])
        segments.append(track_points)

        label_x = x_vals[-1]
        label_y = y_vals[-1]
        
        ax.text(
            label_x, label_y, 
            s=k,                # The source name
            color=colors[i],    # Match the text color to the track color
            fontsize=10,         # Keep it extremely small for breathability
            alpha=0.8,          # Slight transparency so it doesn't block data
            ha='left',          # Align to the right of the point
            va='center',        # Center vertically
            zorder=5            # Ensure text stays on top of everything
        )

    # --- Draw Lines ---
    # LineCollection renders all track paths in one optimized pass
    lc = LineCollection(segments, colors=colors, alpha=0.2, linewidths=0.5, linestyles='-')
    ax.add_collection(lc)

    # --- Draw Scatter Points ---
    flat_x = np.concatenate(all_x)
    flat_y = np.concatenate(all_y)
    flat_sizes = np.concatenate(all_sizes)
    flat_colors = np.concatenate(all_colors)

    if p_pt:
        scatter_plot = ax.scatter(
            flat_x, flat_y,
            s=flat_sizes,
            c=flat_colors,
            alpha=0.4,
            edgecolors='white',
            linewidths=0.3,
            zorder=3 # Ensures points render on top of the lines
        )

        # --- Legend and Formatting ---
        handles, labels = scatter_plot.legend_elements(
                prop="sizes",
                num=5,
                fmt="{x:.1f}",
                func=lambda s: np.sqrt(base_area / s),
                color='black',
                alpha=0.7
        )
        
        size_legend = ax.legend(
            handles, labels,
            title="P-Point Distance\nfrom the Earth\n(AU)",
            bbox_to_anchor=(0.1, 0.2),
            frameon=True,
            labelspacing=0.1
        )
        ax.add_artist(size_legend)
    else:
        ax.scatter(
            flat_x, flat_y,
            s=25,
            c=flat_colors,
            alpha=0.4,
            edgecolors='white',
            linewidths=0.3,
            zorder=3 # Ensures points render on top of the lines
        )
    

    sun_radius = 10
    ax.plot(0, 0, marker="*", markersize=sun_radius, color='orange', alpha=0.8, zorder=4, markeredgecolor='black', markeredgewidth=1)

    ax.axhline(0, color='gray', lw=0.5, ls='--')
    ax.axvline(0, color='gray', lw=0.5, ls='--')

    ax.set_xlabel("X (AU)" if p_pt else r'$\theta_X$ (deg)')
    ax.set_ylabel("Y (AU)" if p_pt else r'$\theta_Y$ (deg)')
    ax.set_title(plot_title)
    plt.grid(True, alpha=0.3)

    if save_filepath is not None:
        fig.savefig(save_filepath, bbox_inches="tight", dpi=300)
        print(f"Plot saved to {save_filepath}.")

    #plt.show()

    return fig, ax


def manage_plot_side_view(sources_df:pd.DataFrame, ref_date:str, save_filepath:str):
    ref_date = Time(ref_date)
    unique_sources_df = sources_df.drop_duplicates(subset=['source_name']).reset_index()

    coords_icrs = SkyCoord(
        ra=unique_sources_df.ra,
        dec=unique_sources_df.dec,
        unit=(u.hour, u.deg),
        distance=1.e9*u.pc,
        frame='icrs'
    )

    _, ty_deg, _, _ = get_hpc_coords(coords_icrs, ref_date)

    plot_side_view(ty_deg, ref_date)

def plot_side_view(ty_deg:np.ndarray, ref_date:Time):
    fig, ax = plt.subplots(figsize=(10, 6))

    # 1. Get exact Earth-Sun distance for the observation date
    d_sun_au = get_earth(ref_date).radius.to(u.AU).value

    # 2. Draw the Sun and Earth (Sizes exaggerated for visibility)
    sun = plt.Circle((0, 0), 0.08, color='orange', label='Sun')
    earth = plt.Circle((d_sun_au, 0), 0.04, color='blue', label='Earth')
    ax.add_patch(sun)
    ax.add_patch(earth)

    # 3. Draw the direct Sun-Earth line
    ax.plot([-1.5, d_sun_au], [0, 0], color='black', lw=1, ls='--', alpha=0.5, label='Sun-Earth Line')

    # 4. Generate the Lines of Sight and P-points
    z_ray = np.array([d_sun_au, -1.5])

    for ty in ty_deg:
        ty_rad = np.deg2rad(ty)
        
        # Plot the Line of Sight
        dist_from_earth = d_sun_au - z_ray
        y_ray = dist_from_earth * np.tan(ty_rad)
        ax.plot(z_ray, y_ray, color='dodgerblue', alpha=0.6, lw=1.5, label='Line of Sight')

        # Calculate exact Cartesian coordinates for the p-point
        z_pp = d_sun_au * (np.sin(ty_rad)**2)
        y_pp = d_sun_au * np.sin(ty_rad) * np.cos(ty_rad)
        
        # Overlay the p-point
        ax.plot(z_pp, y_pp, marker='o', color='red', markersize=6, 
                linestyle='None', label='P-point')

    # 5. Format the Plot
    ax.set_aspect('equal')
    ax.set_xlim(-1.5, d_sun_au + 0.2)
    ax.set_ylim(-1.5, 1.5)
    
    ax.set_xlabel('Distance along Sun-Earth Line (AU)')
    ax.set_ylabel('Transverse Distance Y [North-South] (AU)')
    ax.set_title(f'90-Degree Profile View of IPS Lines of Sight & P-points\n{ref_date.iso.split(" ")[0]}')

    plt.grid(True, alpha=0.3)
    plt.show()

def manage_plot_top_view(sources_df:pd.DataFrame, ref_date:str, save_filepath:str):
    ref_date = Time(ref_date)
    unique_sources_df = sources_df.drop_duplicates(subset=['source_name']).reset_index()

    coords_icrs = SkyCoord(
        ra=unique_sources_df.ra,
        dec=unique_sources_df.dec,
        unit=(u.hour, u.deg),
        distance=1.e9*u.pc,
        frame='icrs'
    )

    tx_deg, _, _, _ = get_hpc_coords(coords_icrs, ref_date)

    plot_top_view(tx_deg, ref_date)

def plot_top_view(tx_deg:np.ndarray, ref_date:Time):
    fig, ax = plt.subplots(figsize=(10, 6))

    # 1. Get exact Earth-Sun distance for the observation date
    d_sun_au = get_earth(ref_date).radius.to(u.AU).value

    # 2. Draw the Sun and Earth (Sizes exaggerated for visibility)
    sun = plt.Circle((0, 0), 0.08, color='orange', label='Sun')
    earth = plt.Circle((0, -d_sun_au), 0.04, color='blue', label='Earth')
    ax.add_patch(sun)
    ax.add_patch(earth)

    # 3. Draw the direct Sun-Earth line
    ax.plot([0, 0], [d_sun_au, -1.5], color='black', lw=1, ls='--', alpha=0.5, label='Sun-Earth Line')

    # 4. Generate the Lines of Sight and P-points
    y_ray = np.array([-d_sun_au, 1.5])

    for tx in tx_deg:
        tx_rad = np.deg2rad(tx)
        
        # Plot the Line of Sight
        dist_from_earth = y_ray + d_sun_au
        x_ray = dist_from_earth * np.tan(tx_rad)
        ax.plot(x_ray, y_ray, color='dodgerblue', alpha=0.6, lw=1.5, label='Line of Sight')

        # Calculate exact Cartesian coordinates for the p-point
        x_pp = d_sun_au * np.sin(tx_rad) * np.cos(tx_rad)
        y_pp = -d_sun_au * (np.sin(tx_rad)**2)
        
        # Overlay the p-point
        ax.plot(x_pp, y_pp, marker='o', color='red', markersize=6, 
                linestyle='None', label='P-point')

    # 5. Format the Plot
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-d_sun_au - 0.2, 1.5)
    ax.set_aspect('equal')
    
    ax.set_xlabel('Distance along Sun-Earth Line (AU)')
    ax.set_ylabel('Transverse Distance Y [North-South] (AU)')
    ax.set_title(f'90-Degree Profile View of IPS Lines of Sight & P-points\n{ref_date.iso.split(" ")[0]}')

    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    main()