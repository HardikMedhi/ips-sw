import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord, GCRS
from astropy.time import Time
from sunpy.coordinates import frames, get_earth

import warnings
warnings.filterwarnings("ignore")

import geometry_utils as gut

def main():
    filepath_sources_csv, ref_date, sideview, topview = get_args()

    sources_df = pd.read_csv(filepath_sources_csv, sep=",", header=0)
    sources_df = insert_p_point_dist(sources_df)

    if ref_date is None:
        manage_plot_tracks(sources_df)
    else:
        if sideview is False and topview is False:
            manage_plot_sources(sources_df, ref_date)
        elif sideview is True and topview is False:
            manage_plot_side_view(sources_df, ref_date)
        elif sideview is False and topview is True:
            manage_plot_top_view(sources_df, ref_date)
        else:
            manage_plot_side_view(sources_df, ref_date)
            manage_plot_top_view(sources_df, ref_date)

    
            
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
    
    args = parser.parse_args()

    filepath_sources_csv = args.filepath_sources_csv
    ref_date = args.refdate
    sideview = args.sideview
    topview = args.topview

    return filepath_sources_csv, ref_date, sideview, topview

def insert_p_point_dist(sources_df:pd.DataFrame):
    sources_df_copy = sources_df.copy()

    p_point_sun_au = np.array([gut.get_p_point_dist(row.date, row.elongation) for row in sources_df_copy[['date', 'elongation']].itertuples()])

    sources_df_copy['p_point_sun_au'] = p_point_sun_au[:, 0]
    sources_df_copy['p_point_earth_au'] = p_point_sun_au[:, 1]

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

def manage_plot_sources(sources_df:pd.DataFrame, ref_date:str):
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

def manage_plot_tracks(sources_df:pd.DataFrame):
    unique_sources_df = sources_df.drop_duplicates(subset=['source_name']).reset_index()

    unique_sources_names = unique_sources_df.source_name
    sub_df_dict = {
        name:sources_df[sources_df.source_name == name]
        for name in unique_sources_names
    }

    sub_df_hpc_coords = {}
    for src_name, df in sub_df_dict.items():
        coords_icrs = SkyCoord(
            ra=df.ra.iloc[0],
            dec=df.dec.iloc[0],
            unit=(u.hour, u.deg),
            distance=1e9*u.pc,
            frame='icrs'
        )

        hpc_coords = [
            #tx, ty, x, y
            get_hpc_coords(coords_icrs, date) 
            for date in df.date
        ]
        
        sub_df_hpc_coords[src_name] = np.array(hpc_coords)

    plot_tracks(sub_df_hpc_coords, sources_df.date.min(), sources_df.date.max())

def plot_tracks(hpc_coords_dict:dict, min_date:str, max_date:str):
    fig, ax = plt.subplots(figsize=(10, 10))
    colors = plt.cm.jet(np.linspace(0, 1, len(hpc_coords_dict)))

    for i, (k, v) in enumerate(hpc_coords_dict.items()):
        ax.plot(v[:, 0], v[:, 1], marker='o', markersize=8, color=colors[i], linestyle='-', linewidth=2, alpha=0.8, label=k)

    d_sun_au = get_earth(min_date).radius.to(u.AU).value

    #TODO
    sun_radius_deg = 0.5 #min(tx_deg.min(), ty_deg.min()) / 6

    sun_disk = plt.Circle((0, 0), sun_radius_deg, color='orange', alpha=0.6)
    ax.add_patch(sun_disk)

    # Add concentric circles
    #TODO
    # radii_au = np.arange(0.5, 2, 0.5)#np.arange(0.5, max(x_au.max(), y_au.max())+0.5, 0.5)
    # for r_au in radii_au:
    #     # Convert physical AU distance to angular degrees for plotting
    #     r_deg = np.rad2deg(np.arctan(r_au / d_sun_au))
        
    #     circle = plt.Circle((0, 0), r_deg, color='green', fill=False, ls=':', alpha=0.6)
    #     ax.add_patch(circle)
        
    #     # Label the whole-number AU circles (and 0.5) to avoid clutter
    #     if r_au.is_integer() or r_au == 0.5:
    #         ax.text(0, r_deg, f' {r_au} AU', color='green', alpha=0.8, 
    #                 ha='left', va='bottom', fontsize=8)

    ax.axhline(0, color='gray', lw=0.5, ls='--')
    ax.axvline(0, color='gray', lw=0.5, ls='--')

    ax.set_xlabel(r'$\theta_X$ (deg)')
    ax.set_ylabel(r'$\theta_Y$ (deg)')
    ax.set_title(f'Distribution of IPS Sources\n{min_date} - {max_date}')
    ax.legend()

    ax.set_xlim(-100, 100)
    ax.set_ylim(-100, 100)
    #ax.set_aspect('equal')

    plt.grid(True, alpha=0.3)
    plt.show()

def manage_plot_side_view(sources_df:pd.DataFrame, ref_date:str):
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

def manage_plot_top_view(sources_df:pd.DataFrame, ref_date:str):
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








# def plot_sources(
#         x_au:np.ndarray, y_au:np.ndarray,
#         p_point_au:np.ndarray,
#         ref_date:Time
# ):
    
#     fig, ax = plt.subplots(figsize=(15, 15))

#     size_factor = 30
#     marker_sizes = size_factor / p_point_au

#     scatter_plot = ax.scatter(
#         x_au, y_au,
#         s=marker_sizes,
#         c='blue',
#         alpha=0.7,
#         edgecolors='none'
#     )

#     handles, labels = scatter_plot.legend_elements(
#         prop="sizes", 
#         num=5,                  # Number of reference sizes to show
#         fmt="{x:.1f}",          # Format labels to 1 decimal place
#         func=lambda s: size_factor / s,
#         color='blue',
#         alpha=0.7
#     )
    
#     # 4. Position the Legend on the right
#     # bbox_to_anchor places the anchor outside the axes framework.
#     size_legend = ax.legend(
#         handles, labels, 
#         title="P-Point Distance\nfrom the Sun\n(AU)",
#         loc="best", 
#         bbox_to_anchor=(0.2, 1),
#         frameon=True,
#         labelspacing=0.1
#     )

#     ax.add_artist(size_legend)

#     d_sun_au = get_earth(ref_date).radius.to(u.AU).value

#     sun_radius_au = min(x_au.min(), y_au.min()) / 6

#     sun_disk = plt.Circle((0, 0), sun_radius_au, color='orange', alpha=0.6)
#     ax.add_patch(sun_disk)

#     # Add concentric circles
#     elong_range = np.arange(10, 90, 10)
#     for elong in elong_range:
#         d_sun_au = get_earth(ref_date).radius.to(u.AU).value
#         radius = d_sun_au * np.tan(np.deg2rad(elong))
        
#         circle = plt.Circle((0, 0), radius, color='green', fill=False, ls=':', alpha=0.6)
#         ax.add_patch(circle)
        
#         # Label the whole-number AU circles (and 0.5) to avoid clutter
#         # if r_au.is_integer() or r_au == 0.5:
#         ax.text(0, radius, f'{elong}'+r"$^{\circ}$", color='green', alpha=0.8, 
#                 ha='left', va='bottom', fontsize=8)

#     ax.axhline(0, color='gray', lw=0.5, ls='--')
#     ax.axvline(0, color='gray', lw=0.5, ls='--')

#     limit = max(np.abs(x_au).max(), np.abs(y_au).max()) + 0.5
#     ax.set_xlim(-limit, limit)
#     ax.set_ylim(-limit, limit)

#     ax.set_aspect('equal')
#     ax.set_xlabel('Projected X (AU)')
#     ax.set_ylabel('Projected Y (AU)')
#     ax.set_title(f'Distribution of IPS Sources\n{ref_date.iso.split(" ")[0]}')
#     ax.legend()
#     ax.grid(True, alpha=0.3)

#     plt.show()