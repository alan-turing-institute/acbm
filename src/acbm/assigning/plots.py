import math
import os
from pathlib import Path
from typing import Optional

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import patches as mpatches
from shapely.geometry import LineString


def plot_workzone_assignment_line(
    assignment_results: pd.DataFrame,
    n: int,
    selection_type: str = "random",
    sort_by: str = "assigned",
    save_dir: str | Path | None = None,
    display: bool = False,
):
    """
    Plot the demand_actual and demand_assigned values for n origin_zones in subplots with two plots per row.
    Home zones can be selected randomly or based on the top actual demand.

    Parameters
    ----------
    assignment_results : DataFrame
        DataFrame containing the actual and assigned demand values.
    n : int
        Number of origin_zones to plot.
    selection_type : str
        Method of selecting origin_zones. Options: 'random', 'top'
        'random': Select n origin_zones randomly,
        'top': Select n home zones with the highest actual demand leaving them.
    sort_by : str
        Column to sort the origin_zones by when selecting the top n. Options: 'actual', 'assigned'
        'actual': Sort by the actual demand, 'assigned': Sort by the assigned

    save_dir: str
        Output directory for saving plots.

    display: bool
        Whether to display plots by calling `plt.show()`.

    Returns
    -------
    A matplotlib plot.
    """
    nrows = np.ceil(n / 2).astype(int)
    fig, axes = plt.subplots(nrows, 2, figsize=(20, 6 * nrows))

    axes = axes.flatten() if n > 2 else np.array([axes]).flatten()

    selected_zones = []
    if selection_type == "random":
        selected_zones = assignment_results["origin_zone"].sample(n).values
    elif selection_type == "top":
        # sort
        top_zones = (
            assignment_results.groupby("origin_zone")[f"demand_{sort_by}"]
            .sum()
            .nlargest(n)
            .index
        )
        selected_zones = top_zones.values

    for i, origin_zone in enumerate(selected_zones):
        origin_zone_df = assignment_results[
            assignment_results["origin_zone"] == origin_zone
        ]

        ax = axes[i]
        ax.plot(
            origin_zone_df["assigned_zone"],
            origin_zone_df["pct_of_total_demand_actual"],
            "b-",
            label="Actual (% of Total)",
        )
        ax.plot(
            origin_zone_df["assigned_zone"],
            origin_zone_df["pct_of_total_demand_assigned"],
            "b--",
            label="Assigned (% of Total)",
        )
        ax.plot(
            origin_zone_df["assigned_zone"],
            origin_zone_df["pct_of_o_total_actual"],
            "r-",
            label="Actual (% of Origin Total)",
        )
        ax.plot(
            origin_zone_df["assigned_zone"],
            origin_zone_df["pct_of_o_total_assigned"],
            "r--",
            label="Assigned (% of Origin Total)",
        )
        ax.plot(
            origin_zone_df["assigned_zone"],
            origin_zone_df["pct_of_d_total_actual"],
            "g-",
            label="Actual (% of Dest Total)",
        )
        ax.plot(
            origin_zone_df["assigned_zone"],
            origin_zone_df["pct_of_d_total_assigned"],
            "g--",
            label="Assigned (% of Dest Total)",
        )
        ax.set_xlabel("Destination Zone")
        ax.set_ylabel("Demand (%)")
        ax.set_title(
            f"Difference in Actual and Assigned Demand for Origin Zone {origin_zone}"
        )
        ax.legend()
        ax.tick_params(axis="x", rotation=60)

    plt.tight_layout()

    # Save the plot if save_dir is specified
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plot_path = os.path.join(
            save_dir, f"workzone_assignment_line_{n}_{selection_type}_origins.png"
        )
        plt.savefig(plot_path)
        print(f"Plot saved to {plot_path}")

    if display:
        plt.show()


def plot_workzone_assignment_heatmap(
    assignment_results: pd.DataFrame,
    n: int,
    selection_type: str = "random",
    sort_by: str = "assigned",
    save_dir: str | Path | None = None,
    display: bool = False,
):
    """
    Create three heatmaps side by side showing the aggregated difference between actual and assigned demand percentages
    for the same n origin_zones across all categories (Global, Origin_Sum, Destination_Sum). The origin_zones are
    either randomly selected or the top n zones with the highest actual demand, consistent across all categories.

    Parameters
    ----------
    assignment_results : DataFrame
        DataFrame containing the actual and assigned demand values.
    n : int
        Number of unique origin_zones to include.
    selection_type : str
        Type of selection for origin_zones. Options: 'random', 'top'.
    sort_by : str
        Column to sort the origin_zones by when selecting the top n. Options: 'actual', 'assigned'
        'actual': Sort by the actual demand, 'assigned': Sort by the assigned

    Returns
    -------
    A matplotlib + seaborn plot.

    """
    categories = ["Global", "Origin", "Destination"]
    fig, axes = plt.subplots(1, len(categories), figsize=(18, 6), sharey=True)

    # Select zones based on selection_type
    if selection_type == "random":
        unique_zones = assignment_results["origin_zone"].unique()
        n = min(n, len(unique_zones))
        selected_zones = np.random.choice(unique_zones, size=n, replace=False)
    elif selection_type == "top":
        # Sort
        top_zones_df = (
            assignment_results.sort_values(by=f"demand_{sort_by}", ascending=False)
            .drop_duplicates("origin_zone")
            .head(n)
        )
        selected_zones = top_zones_df["origin_zone"].values

    for i, category in enumerate(categories):
        prefix_map = {
            "Global": "pct_of_total_demand",
            "Origin": "pct_of_o_total",
            "Destination": "pct_of_d_total",
        }
        prefix = prefix_map[category]

        filtered_df = assignment_results[
            assignment_results["origin_zone"].isin(selected_zones)
        ].copy()
        filtered_df["difference"] = (
            filtered_df[f"{prefix}_actual"] - filtered_df[f"{prefix}_assigned"]
        )
        heatmap_data = filtered_df.pivot_table(
            index="assigned_zone",
            columns="origin_zone",
            values="difference",
            aggfunc=np.mean,
        )

        ax = sns.heatmap(
            heatmap_data, cmap="viridis", ax=axes[i], cbar=i == len(categories) - 1
        )
        axes[i].set_title(f"Demand Difference: % of {category} Total")
        axes[i].set_xlabel("Home Zone")
        if i == 0:
            axes[i].set_ylabel("Assigned Zone")
        else:
            axes[i].set_ylabel("")

        if i == len(categories) - 1:
            # Create a colorbar with a vertical title
            cbar = ax.collections[0].colorbar
            cbar.ax.set_ylabel(
                "Demand Difference: Actual (%) - Assigned (%)",
                rotation=270,
                labelpad=15,
            )

    plt.tight_layout()

    # Save the plot if save_dir is specified
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plot_path = os.path.join(
            save_dir, f"workzone_assignment_heatmap_{n}_{selection_type}_origins.png"
        )
        plt.savefig(plot_path)
        print(f"Plot saved to {plot_path}")

    if display:
        plt.show()


def plot_desire_lines(
    activities: pd.DataFrame,
    activity_type_col: str,
    activity_type: str,
    bin_size: int,
    boundaries: gpd.GeoDataFrame,
    crs: str,
    sample_size: Optional[int] = None,
    save_dir: str | Path | None = None,
) -> None:
    """
    Plots activity chains for a given activity type, bin size, geographical boundaries, and an optional sample size.

    Parameters:
    activities: pd.DataFrame
        A DataFrame containing the activities data. Geometry is a LineString.
    activity_type_col: str
        The column name containing the activity type.
    activity_type: str
        The type of activity to plot.
    bin_size: int
        The size of the bins for the histogram. (in meters)
    boundaries: gpd.GeoDataFrame
        A GeoDataFrame containing the geographical boundaries for the plot.
    crs: str
        The coordinate reference system (CRS) of the activities data.
    sample_size: int, optional
        The size of the sample to plot. If None, all data is plotted.

    Returns:
        None
    """

    activity_chains_plot = activities.copy()

    activity_chains_plot = activity_chains_plot[
        activity_chains_plot[activity_type_col] == activity_type
    ]

    # filter to only include rows where activity_geom is not NA
    activity_chains_plot = activity_chains_plot[
        activity_chains_plot["end_location_geometry"].notna()
        & activity_chains_plot["start_location_geometry"].notna()
    ]

    activity_chains_plot["line_geometry"] = activity_chains_plot.apply(
        lambda row: LineString(
            [row["start_location_geometry"], row["end_location_geometry"]]
        ),
        axis=1,
    )

    # Convert to GeoDataFrame and set the geometry column to 'line_geometry'
    activity_chains_plot = gpd.GeoDataFrame(
        activity_chains_plot, geometry="line_geometry", crs=crs
    )

    # convert crs to metric
    activity_chains_plot = activity_chains_plot.to_crs(crs=crs)
    # calculate the length of the line_geometry in meters
    activity_chains_plot["length"] = activity_chains_plot["line_geometry"].length

    # If a sample size is specified, sample the activities
    if sample_size is not None and sample_size < len(activity_chains_plot):
        activity_chains_plot = activity_chains_plot.sample(n=sample_size)

    # Mode legend
    modes = activity_chains_plot["mode"].unique()  # Collect all unique modes
    colormap = plt.colormaps.get_cmap("Dark2")  # Generate a colormap
    mode_colors = {
        mode: colormap(i) for i, mode in enumerate(modes)
    }  # Map modes to colors
    legend_patches = [
        mpatches.Patch(color=mode_colors[mode], label=mode) for mode in modes
    ]  # Create legend handles

    # Calculate the number of bins based on the maximum value of 'length'
    num_bins = math.ceil(activity_chains_plot["length"].max() / bin_size)
    # Calculate the bin edges
    bins = np.arange(num_bins + 1) * bin_size
    # Create a new column 'length_band' by cutting 'length' into distance bands
    activity_chains_plot["length_band"] = pd.cut(
        activity_chains_plot["length"], bins, include_lowest=True
    )
    # Get unique bands and sort them
    bands = activity_chains_plot["length_band"].unique()
    bands = sorted(bands, key=lambda x: x.left)
    # Calculate the total number of trips
    total_trips = len(activity_chains_plot)

    # Calculate the number of rows and columns for the subplots
    nrows = math.ceil(len(bands) / 3)
    ncols = 3
    # Create a grid of subplots
    fig, axs = plt.subplots(nrows, ncols, figsize=(20, 6 * nrows))
    # Flatten axs for easy iteration
    axs = axs.flatten()

    for ax, band in zip(axs, bands):
        # Get the subset for this band
        subset_band = activity_chains_plot[activity_chains_plot["length_band"] == band]

        # Calculate the percentage of trips in this band
        percentage = len(subset_band) / total_trips * 100

        # Plot the boundaries
        boundaries.plot(ax=ax, color="lightgrey")

        # Plot the subset with correct colors
        for mode in modes:
            # check if mode is in subset_band, and plot if it is
            if mode in subset_band["mode"].unique():
                subset_mode = subset_band[subset_band["mode"] == mode]
                subset_mode.plot(ax=ax, color=mode_colors[mode], label=mode)

        # Set the title
        ax.set_title(
            f"{activity_type},\ndistance band: {band},\nNo. of trips: {len(subset_band)} ({percentage:.2f}%)"
        )

    # Remove any unused subplots
    for i in range(len(bands), nrows * ncols):
        fig.delaxes(axs[i])

    # Place the legend at the bottom
    plt.figlegend(
        handles=legend_patches, loc="lower center", ncol=5, title="Transportation Modes"
    )  # Adjust 'ncol' as needed

    plt.tight_layout()
    plt.subplots_adjust(
        bottom=0.03, wspace=0.1, hspace=0.2
    )  # Adjust space to show the legend properly

    # Save the plot if save_dir is specified
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plot_path = os.path.join(
            save_dir,
            f"{activity_type}_assignment_desire_lines_{sample_size}_ activities.png",
        )
        plt.savefig(plot_path)
        print(f"Plot saved to {plot_path}")


def plot_scatter_actual_reported(
    activities: pd.DataFrame,
    x_col: str,
    y_col: str,
    x_label: str,
    y_label: str,
    title_prefix: str,
    activity_type: str,
    activity_type_col: str,
    crs: str,
    save_dir: str | Path | None = None,
    display: bool = False,
    y_scale: float = 1 / 1000,
):
    """
    Plots scatter plots with trend lines for different modes in activity chains.

    Parameters:
    -----------
    - activity_chains_plot: DataFrame containing the activity chains data.
    - x_col: Column name for the x-axis values.
    - y_col: Column name for the y-axis values.
    - x_label: Label for the x-axis.
    - y_label: Label for the y-axis.
    - title_prefix: Prefix for the plot titles.
    - activity_type: Type of activity to plot.
    - activity_type_col: Column name for the activity type.
    - crs: Coordinate reference system (CRS) of the activities data.
    - save_dir: Directory to save the plots. If None, plots are not saved.
    - display: Whether to display plots by calling `plt.show()`.
    """

    activity_chains_plot = activities.copy()
    # only include rows where the activity type is the one we are interested in
    activity_chains_plot = activity_chains_plot[
        activity_chains_plot[activity_type_col] == activity_type
    ]
    # filter to only include rows where activity_geom is not NA
    activity_chains_plot = activity_chains_plot[
        activity_chains_plot["end_location_geometry"].notna()
        & activity_chains_plot["start_location_geometry"].notna()
    ]
    activity_chains_plot["line_geometry"] = activity_chains_plot.apply(
        lambda row: LineString(
            [row["start_location_geometry"], row["end_location_geometry"]]
        ),
        axis=1,
    )
    # Set the geometry column to 'line_geometry'
    activity_chains_plot = activity_chains_plot.set_geometry("line_geometry", crs=crs)

    # convert crs to metric
    activity_chains_plot = activity_chains_plot.to_crs(crs=crs)
    # calculate the length of the line_geometry in meters
    activity_chains_plot["length"] = activity_chains_plot["line_geometry"].length

    # Calculate the number of rows and columns for the subplots. It is a function of the number of modes
    nrows = math.ceil(len(activity_chains_plot["mode"].unique()) / 2)
    ncols = 2

    # Create a grid of subplots
    fig, axs = plt.subplots(nrows, ncols, figsize=(20, 6 * nrows))

    # Flatten axs for easy iteration
    axs = axs.flatten()

    # Create a scatter plot for each mode
    for i, mode in enumerate(activity_chains_plot["mode"].unique()):
        # Get the subset for this mode
        subset_mode = activity_chains_plot[activity_chains_plot["mode"] == mode]

        # Plot the scatter plot
        ax = axs[i]
        ax.scatter(
            subset_mode[x_col], subset_mode[y_col] * y_scale, alpha=0.1, lw=0
        )  # Use a single color for all plots

        # Calculate and plot the trend line
        z = np.polyfit(subset_mode[x_col], subset_mode[y_col] * y_scale, 1)
        p = np.poly1d(z)
        ax.plot(subset_mode[x_col], p(subset_mode[x_col]), "r--")

        ax.set_title(f"{title_prefix} for mode: {mode}")
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

    # Adjust layout
    plt.tight_layout()

    # Save the plot if save_dir is specified
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plot_path = os.path.join(
            save_dir, f"{activity_type}_assignment_scatter_{x_col}_{y_col}.png"
        )
        plt.savefig(plot_path)
        print(f"Plot saved to {plot_path}")

    # Display the plot
    if display:
        plt.show()
