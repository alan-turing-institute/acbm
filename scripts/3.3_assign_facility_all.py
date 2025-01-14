import geopandas as gpd
import pandas as pd
from libpysal.weights import Queen

from acbm.assigning.plots import plot_desire_lines, plot_scatter_actual_reported
from acbm.assigning.select_facility import map_activity_locations, select_facility
from acbm.cli import acbm_cli
from acbm.config import load_and_setup_config
from acbm.utils import get_travel_times


@acbm_cli
def main(config_file):
    config = load_and_setup_config(config_file)
    logger = config.get_logger("assigning_facility_locations", __file__)

    # --- Load data: activity chains
    logger.info("Loading activity chains")

    activity_chains = pd.read_csv(config.output_path / "legs.csv")
    activity_chains = activity_chains.drop(columns=["Unnamed: 0", "freq"])

    # --- Preprocess: Split activity chains by activity purpose
    logger.info("Splitting activity chains by activity purpose")

    activity_chains_home = activity_chains[
        activity_chains["destination activity"] == "home"
    ]
    activity_chains_work = activity_chains[
        activity_chains["destination activity"] == "work"
    ]
    activity_chains_edu = activity_chains[
        activity_chains["destination activity"] == "education"
    ]
    # secondary activities
    activities_to_exclude = ["home", "work", "education"]
    activity_chains_other = activity_chains[
        ~activity_chains["destination activity"].isin(activities_to_exclude)
    ]

    # --- Load data: POI locations
    logger.info("Loading facility data")

    osm_data_gdf = gpd.read_parquet(config.osm_path)

    # --- Load data: Boundaries
    logger.info("Loading study area boundaries")
    boundaries = config.get_study_area_boundaries()
    logger.info(f"Study area boundaries loaded and reprojected to {config.output_crs}")

    # --- Prepprocess: add zone column to POI data
    logger.info("Adding zone column to POI data")

    # ensure that osm_data_gdf and boundaries are in the same crs
    osm_data_gdf = osm_data_gdf.to_crs(boundaries.crs)

    osm_data_gdf = gpd.sjoin(
        osm_data_gdf,
        boundaries[[config.zone_id, "geometry"]],
        how="inner",
        predicate="within",
    )

    # --- Analysis: SELECTING FACILITIES
    logger.info("Selecting facilities")

    # Get neighboring zones
    logger.info("1. Calculating neighboring zones")

    # get neighbors
    zone_neighbors = Queen.from_dataframe(
        boundaries, idVariable=config.zone_id
    ).neighbors

    # - HOME LOCATIONS
    logger.info("2. Selecting HOME locations")

    # Keep one row per household and select only household and OA21CD columns
    activity_chains_home_hh = activity_chains_home.drop_duplicates(subset=["hid"])
    activity_chains_home_hh = activity_chains_home_hh[
        ["hid", "destination activity", "dzone"]
    ]

    activity_locations_home = select_facility(
        df=activity_chains_home_hh,
        unique_id_col="hid",
        facilities_gdf=osm_data_gdf,
        row_destination_zone_col="dzone",
        row_activity_type_col="destination activity",
        gdf_facility_zone_col=config.zone_id,
        gdf_facility_type_col="activities",
        gdf_sample_col="floor_area",
        neighboring_zones=zone_neighbors,
    )

    # Map the activity_id and activity_geometry to the activity_chains_home_df DataFrame
    activity_chains_home = map_activity_locations(
        activity_chains_df=activity_chains_home,
        activity_locations_dict=activity_locations_home,
        id_col="hid",
    )

    # - WORK LOCATIONS
    logger.info("3. Selecting WORK locations")

    activity_locations_work = select_facility(
        df=activity_chains_work,
        unique_id_col="pid",
        facilities_gdf=osm_data_gdf,
        row_destination_zone_col="dzone",
        row_activity_type_col="destination activity",
        gdf_facility_zone_col=config.zone_id,
        gdf_facility_type_col="activities",
        gdf_sample_col="floor_area",
        neighboring_zones=zone_neighbors,
    )

    # Map the activity_id and activity_geometry to the activity_chains_df DataFrame
    activity_chains_work = map_activity_locations(
        activity_chains_df=activity_chains_work,
        activity_locations_dict=activity_locations_work,
        id_col="pid",
    )

    logger.info(f"Shape of activity chains work: {activity_chains_work.shape}")

    # - EDUCATION LOCATIONS
    logger.info("4. Selecting EDUCATION locations")

    logger.info("a. Adding eduction type as fallback")
    # load in activity chains
    spc_with_nts = pd.read_parquet(
        config.spc_with_nts_trips_filepath,
        columns=["id", "education_type", "seq", "TripTotalTime", "TripDisIncSW"],
    )
    # we get one row per id
    spc_with_nts_edu = spc_with_nts[["id", "education_type"]].drop_duplicates(
        subset="id"
    )
    # merge the education type with the activity chains
    activity_chains_edu = activity_chains_edu.merge(
        spc_with_nts_edu, left_on="pid", right_on="id", how="left"
    ).drop(columns=["id"])

    logger.info("b. Selecting education locations")

    # apply the function to a row in activity_chains_ex
    activity_locations_edu = select_facility(
        df=activity_chains_edu,
        unique_id_col="pid",
        facilities_gdf=osm_data_gdf,
        row_destination_zone_col="dzone",
        row_activity_type_col="education_type",
        gdf_facility_zone_col=config.zone_id,
        gdf_facility_type_col="activities",
        gdf_sample_col="floor_area",
        neighboring_zones=zone_neighbors,
        fallback_type="education",
    )

    logger.info(f"Shape of activity chains edu: {activity_chains_edu.shape}")

    # Map the activity_id and activity_geometry to the activity_chains_home_df DataFrame
    activity_chains_edu = map_activity_locations(
        activity_chains_df=activity_chains_edu,
        activity_locations_dict=activity_locations_edu,
        id_col="pid",
    )

    # - SECONDARY LOCATIONS
    logger.info("5. Selecting SECONDARY locations")

    logger.info("a. creating unique_id column")
    # pid and hid are not unique id columns, as there can be many different secondary
    # activities done by the same person.
    # We create a unique identifier that can be mapped back to the original data.

    # Unique id column: Concatenate pid, seq
    activity_chains_other["act_id"] = (
        activity_chains_other["pid"].astype(str)
        + "_"
        + activity_chains_other["seq"].astype(str)
    )

    logger.info("b. Selecting secondary locations")

    # apply the function to a row in activity_chains_ex
    activity_locations_other = select_facility(
        df=activity_chains_other,
        unique_id_col="act_id",
        facilities_gdf=osm_data_gdf,
        row_destination_zone_col="dzone",
        row_activity_type_col="purp",
        gdf_facility_zone_col=config.zone_id,
        gdf_facility_type_col="activities",
        gdf_sample_col="floor_area",
        neighboring_zones=zone_neighbors,
        fallback_to_random=True,
    )

    # Map the activity_id and activity_geometry to the activity_chains_home_df DataFrame
    activity_chains_other = map_activity_locations(
        activity_chains_df=activity_chains_other,
        activity_locations_dict=activity_locations_other,
        id_col="act_id",
    )

    # --- Analysis: Merging data
    logger.info("Merging all activity chains")

    activity_chains_all = pd.concat(
        [
            activity_chains_home,
            activity_chains_work,
            activity_chains_edu,
            activity_chains_other,
        ]
    )

    activity_chains_all = activity_chains_all.sort_values(by=["hid", "pid", "seq"])

    # --- Analysis: Create start_location_id and start_location_geometry column
    logger.info("Creating start_location_id and start_location_geometry columns")

    # Create start_location_id and start_location_geometry by shifting end_location_id and end_location_geometry within each 'pid'
    activity_chains_all["start_location_id"] = activity_chains_all.groupby("pid")[
        "end_location_id"
    ].shift(1)
    activity_chains_all["start_location_geometry"] = activity_chains_all.groupby("pid")[
        "end_location_geometry"
    ].shift(1)

    logger.info("Fill rows where seq = 1 with home location")

    mask = activity_chains_all["seq"] == 1
    # Aggregate duplicates by taking the first occurrence
    activity_chains_home_agg = activity_chains_home.groupby("hid").first().reset_index()
    # Map home location data to the start_location_id and start_location_geometry columns
    activity_chains_all.loc[mask, "start_location_id"] = activity_chains_all.loc[
        mask, "hid"
    ].map(activity_chains_home_agg.set_index("hid")["end_location_id"])
    activity_chains_all.loc[mask, "start_location_geometry"] = activity_chains_all.loc[
        mask, "hid"
    ].map(activity_chains_home_agg.set_index("hid")["end_location_geometry"])

    # --- Save data

    # Keep necessary columns

    # select only the columns we need
    activity_chains_all = activity_chains_all[
        [
            "pid",
            "hid",
            "ozone",
            "dzone",
            "purp",
            "origin activity",
            "destination activity",
            "mode",
            "seq",
            "tst",
            "tet",
            "duration",
            "start_location_id",
            "start_location_geometry",
            "end_location_id",
            "end_location_geometry",
        ]
    ]

    # save as parquet: note to serialize the geometries, need to convert
    # non-missing values to e.g. wkt
    geom_cols = ["start_location_geometry", "end_location_geometry"]
    for col in geom_cols:
        activity_chains_all.loc[:, col + "_wkt"] = activity_chains_all[col].map(
            lambda point: point if pd.isna(point) else point.wkt
        )
    activity_chains_all.drop(columns=geom_cols).to_parquet(
        config.output_path / "legs_with_locations.parquet"
    )

    # --- Plots

    logger.info("Creating plots")

    # merge actual times from the NTS
    activity_chains_all = activity_chains_all.merge(
        spc_with_nts[["id", "seq", "TripTotalTime", "TripDisIncSW"]],
        left_on=["pid", "seq"],
        right_on=["id", "seq"],
        how="left",
    ).drop(columns=["id"])

    # Get unique activity types from the 'purp' column
    unique_activity_types = activity_chains_all["purp"].unique()

    # Plot 1: Euclidian travel distance vs reported (NTS) travel DISTANCE
    logger.info("Plotting Euclidian travel distance vs reported (NTS) travel DISTANCE")

    # Iterate over each unique activity type and create a plot
    for activity_type in unique_activity_types:
        plot_scatter_actual_reported(
            activities=activity_chains_all,
            activity_type=activity_type,
            activity_type_col="destination activity",
            x_col="TripDisIncSW",
            y_col="length",
            x_label="Reported Travel Distance (km)",
            y_label="Actual Distance - Estimated (km)",
            detour_factor=config.feasible_assignment.detour_factor,
            decay_rate=config.feasible_assignment.decay_rate,
            x_axis_max=50,
            crs=f"EPSG:{config.output_crs}",
            title_prefix=f"Scatter plot of TripDisIncSW vs. Length for {activity_type}",
            save_dir=config.output_path / "plots/assigning/",
        )

    # Plot 2: Euclidian travel distance vs reported (NTS) travel TIME
    logger.info("Plotting Euclidian travel distance vs reported (NTS) travel TIME")

    # # convert duration to numeric
    # activity_chains_all["duration"] = pd.to_timedelta(activity_chains_all["duration"], errors="coerce")
    # activity_chains_all['duration'] = activity_chains_all['duration'].apply(lambda x: x + pd.Timedelta(days=1) if x.days < 0 else x)
    # activity_chains_all["duration"] = activity_chains_all["duration"].dt.total_seconds() / 60
    # activity_chains_all["duration"] = activity_chains_all["duration"].astype(int)

    # Iterate over each unique activity type and create a plot
    for activity_type in unique_activity_types:
        plot_scatter_actual_reported(
            activities=activity_chains_all,
            activity_type=activity_type,
            activity_type_col="destination activity",
            x_col="TripTotalTime",
            y_col="length",
            x_label="Reported Travel Time (min)",
            y_label="Actual Distance - Estimated (km)",
            detour_factor=config.feasible_assignment.detour_factor,
            decay_rate=config.feasible_assignment.decay_rate,
            x_axis_max=180,
            crs=f"EPSG:{config.output_crs}",
            title_prefix="Scatter plot of TripTotalTime vs. Length",
            save_dir=config.output_path / "plots/assigning/",
        )

    # Add travel times
    tte = get_travel_times(config)
    activity_chains_all = activity_chains_all.merge(
        tte[[tte.columns[0], tte.columns[1], "mode", "time"]],
        left_on=["ozone", "dzone", "mode"],
        right_on=[tte.columns[0], tte.columns[1], "mode"],
        how="left",
    )
    # Iterate over each unique activity type and create a plot
    for activity_type in unique_activity_types:
        plot_scatter_actual_reported(
            activities=activity_chains_all,
            activity_type=activity_type,
            activity_type_col="destination activity",
            x_col="TripTotalTime",
            y_col="time",
            x_label="Reported Travel TIme (min)",
            y_label="Modelled time (min)",
            detour_factor=config.feasible_assignment.detour_factor,
            decay_rate=config.feasible_assignment.decay_rate,
            x_axis_max=180,
            crs=f"EPSG:{config.output_crs}",
            title_prefix="Scatter plot of TripTotalTime vs. Modelled time",
            save_dir=config.output_path / "plots/assigning/",
            y_scale=1.0,
        )
    # ....

    # Plot 3: Desire lines between start and end locations
    logger.info("Plotting desire lines between start and end locations")

    for activity_type in unique_activity_types:
        plot_desire_lines(
            activities=activity_chains_all,
            activity_type_col="destination activity",
            activity_type=activity_type,
            bin_size=5000,
            boundaries=boundaries,
            sample_size=1000,
            crs=f"EPSG:{config.output_crs}",
            save_dir=config.output_path / "plots/assigning/",
        )


if __name__ == "__main__":
    main()
