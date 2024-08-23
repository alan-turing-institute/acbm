from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd

from acbm.logger_config import assigning_primary_locations_logger as logger


def select_zone(
    row: pd.Series,
    possible_zones: dict,
    activities_per_zone: pd.DataFrame,
    weighting: str = "none",
    zone_id_col: str = "OA21CD",
) -> str:
    """
    Select a zone for an activity. For each activity, we have a list of possible zones
    stored in the possible_zones dictionary. For each activity, the function will filter the
    activities_per_zone_df to the zones that (a) are in possible_zones for that activity key,
    and (b) have an activity that matches the activity in row["education_type"]. For example, if
    the activity is "education_university", the function will filter the activities_per_zone_df to
    only include zones with a university.
    The funciton relaxes this constraint if no zones match the conditions

    the next step is to sample a zone from the shortlisted zones. This is done based on the total
    floor area of available facilities in the zone (that match the activity purpose). This is from
    activities_per_zone_df["floor_area"]. Again, different options exist if floor area cannot be used

    Parameters
    ----------
    row: pd.Series
        the row that we are applying the funciton to
    possible_zones: dict
        a dictionary with keys as the index of the "row" and values as a nested dictionary
        key: Origin Zone Id, values: List of possible destination zones
    activities_per_zone: pd.DataFrame
        a dataframe with columns: OA21CD | counts | floor_area | activity
        activity is a category (e.g. work, education_school) from osm data labeling (through osmox)
        counts and floor_area are totals for a specific activity and zone
    weighting: str
        options are: "floor_area", "counts", "none"
        once we have a list of feasible zones, do we want to do weighted sampling based on
        "floor_area" or "counts" or just random sampling "none"
    zone_id: str
        The column name of the zone id in activities_per_zone. The ids should also match the nested key in
        possible_zones {key: {KEY: value}}

    Returns
    -------
    str
        The zone_id of the selected zone.
    """

    # Check if the input is in the list of allowed values
    allowed_weightings = ["floor_area", "counts", "none"]
    if weighting not in allowed_weightings:
        error_message = f"Invalid value for weighting: {weighting}. Allowed values are {allowed_weightings}."
        raise ValueError(error_message)

    # get the values from possible_zones_school that match the index of the row
    # use try/except as some activities might have no possible zones
    try:
        activity_i_options = list(possible_zones[row.name].values())
        if not activity_i_options:  # Check if the list is empty
            logger.info(f"Activity {row.name}: No zones available")
            return "NA"
        # log the number of options for the specific index
        logger.debug(
            f"Activity {row.name}: Initial number of options for activity = {len(activity_i_options[0])}"
        )

        # Attempt 1: filter activities_per_zone_df to only include possible_zones
        options = activities_per_zone[
            (activities_per_zone["activity"] == row["education_type"])
            & (activities_per_zone[zone_id_col].isin(activity_i_options[0]))
        ]
        logger.debug(
            f"Activity {row.name}: Number of options after filtering by education type: {len(options)}"
        )

        # Attempt 2: if no options meet the conditions, relax the constraint by considering all education types
        # not just the education type that maps onto the person's age
        if options.empty:
            # print("No options available. Relaxing constraint")
            options = activities_per_zone[
                (activities_per_zone["activity"].str.contains("education"))
                & (activities_per_zone[zone_id_col].isin(activity_i_options[0]))
            ]
            logger.debug(
                f"Activity {row.name}: Number of options after first relaxation: {len(options)}"
            )

        # Attempt 3: if options is still empty, relax the constraint further by considering all possible zones
        # regardless of activities
        if options.empty:
            logger.info(
                f"Activity {row.name}: No zones with required facility type. Selecting from all possible zones"
            )
            options = activities_per_zone[
                activities_per_zone[zone_id_col].isin(activity_i_options[0])
            ]
            logger.debug(
                f"Activity {row.name}: Number of options after second relaxation: {len(options)}"
            )

        # Attempt 4: if options is still empty (there were no options in possible_zones), return NA
        if options.empty:
            logger.info(f"Activity {row.name}: No options available. Returning NA")
            return "NA"

        # Sample based on "weighting" argument
        if weighting == "floor_area":
            # check the sum of floor_area is not zero
            if options["floor_area"].sum() != 0:
                logger.debug(f"Activity {row.name}: sampling based on floor area")
                selected_zone = options.sample(1, weights="floor_area")[
                    zone_id_col
                ].values[0]
            elif options["counts"].sum() != 0:
                logger.debug(
                    f"Activity {row.name}: No floor area data. sampling based on counts"
                )
                selected_zone = options.sample(1, weights="counts")[zone_id_col].values[
                    0
                ]
            else:
                logger.debug(
                    f"Activity {row.name}: No floor area or count data. sampling randomly"
                )
                selected_zone = options.sample(1)[zone_id_col].values[0]
        elif weighting == "counts":
            if options["counts"].sum() != 0:
                logger.debug(f"Activity {row.name}: sampling based on counts")
                selected_zone = options.sample(1, weights="counts")[zone_id_col].values[
                    0
                ]
            else:
                logger.debug(f"Activity {row.name}: No count data. sampling randomly")
                selected_zone = options.sample(1)[zone_id_col].values[0]
        else:
            logger.debug(f"Activity {row.name}: sampling randomly")
            selected_zone = options.sample(1)[zone_id_col].values[0]

        return selected_zone

    except KeyError:
        logger.info(f"KeyError: Key {row.name} in possible_zones has no values")
        return "NA"


def select_facility(
    row: pd.Series,
    facilities_gdf: gpd.GeoDataFrame,
    row_destination_zone_col: str,
    gdf_facility_zone_col: str,
    row_activity_type_col: str,
    gdf_facility_type_col: str,
    fallback_type: Optional[str] = None,
    fallback_to_random: bool = False,
    neighboring_zones: Optional[dict] = None,
    gdf_sample_col: Optional[str] = None,
) -> pd.Series:
    """
    Select a suitable facility based on the activity type and a specific zone from a GeoDataFrame.
    Optionally:
     - looks in neighboring zones when there is no suitable facility in the initial zone
     - add a fallback type to search for a more general type of facility when no specific facilities are found
       (e.g. 'education' instead of 'education_university')
     - sample based on a specific column in the GeoDataFrame (e..g. floor_area)

    Parameters
    ----------
    selection_row : pandas.Series
        A row from the DataFrame indicating the selection criteria, including the destination zone and activity type.
    facilities_gdf : geopandas.GeoDataFrame
        GeoDataFrame containing facilities to sample from.
    row_destination_zone_col : str
        The column name in `selection_row` that indicates the destination zone.
    gdf_facility_zone_col : str
        The column name in `facilities_gdf` that indicates the facility zone.
    row_activity_type_col : str
        The column in `selection_row` indicating the type of activity (e.g., 'education', 'work').
    gdf_facility_type_col : str
        The column in `facilities_gdf` to filter facilities by type based on the activity type.
    fallback_type : Optional[str]
        A more general type of facility to fallback to if no specific facilities are found. By default None.
    fallback_to_random : bool
        If True, sample from all facilities in the zone if no specific facilities are found. By default False.
    neighboring_zones : Optional[dict]
        A dictionary mapping zones to their neighboring zones for fallback searches, by default None.
    gdf_sample_col : Optional[str]
        The column to sample from, by default None. The only feasible input is "floor_area". If "floor_area" is specified,
        uses this column's values as weights for sampling.

    Returns
    -------
    pd.Series
        Series containing the id and geometry of the chosen facility. Returns NaN if no suitable facility is found.
    """
    # ----- Step 1. Find valid facilities in the destination zone

    # Extract the destination zone from the input row
    destination_zone = row[row_destination_zone_col]
    if pd.isna(destination_zone):
        logger.info(f"Activity {row.name}: Destination zone is NA")
        return pd.Series([np.nan, np.nan])

    # Filter facilities within the specified destination zone
    facilities_in_zone = facilities_gdf[
        facilities_gdf[gdf_facility_zone_col] == destination_zone
    ]
    # Attempt to find facilities matching the specific facility type
    facilities_valid = facilities_in_zone[
        facilities_in_zone[gdf_facility_type_col].apply(
            lambda x: row[row_activity_type_col] in x
        )
    ]
    logger.info(
        f"Activity {row.name}: Found {len(facilities_valid)} matching facilities in zone {destination_zone}"
    )

    # If no specific facilities found in the initial zone, and neighboring zones are provided, search in neighboring zones
    if facilities_valid.empty and neighboring_zones:
        logger.info(
            f"Activity {row.name}: No {row[row_activity_type_col]} facilities in {destination_zone}. Expanding search to neighboring zones"
        )
        neighbors = neighboring_zones.get(destination_zone, [])
        facilities_in_neighboring_zones = facilities_gdf[
            facilities_gdf[gdf_facility_zone_col].isin(neighbors)
        ]
        facilities_valid = facilities_in_neighboring_zones[
            facilities_in_neighboring_zones[gdf_facility_type_col].apply(
                lambda x: row[row_activity_type_col] in x
            )
        ]
        logger.info(
            f"Activity {row.name}: Found {len(facilities_valid)} matching facilities in neighboring zones"
        )

    # If no specific facilities found and a fallback type is provided, attempt to find facilities matching the fallback type
    if facilities_valid.empty and fallback_type:
        logger.info(
            f"Activity {row.name}: No {row[row_activity_type_col]} facilities in zone {destination_zone} or neighboring zones, trying with {fallback_type}"
        )
        # This should consider both the initial zone and neighboring zones if the previous step expanded the search
        facilities_valid = facilities_in_zone[
            facilities_in_zone[gdf_facility_type_col].apply(
                lambda x: fallback_type in x
            )
        ]
        logger.info(
            f"Activity {row.name}: Found {len(facilities_valid)} matching facilities with type: {fallback_type}"
        )

    # if no specific facilities found and fallback_to_random is True, take all facilities in the zone
    if facilities_valid.empty and fallback_to_random:
        logger.info(
            f"Activity {row.name}: No facilities in zone {destination_zone} with {gdf_facility_type_col} '{fallback_type or row[row_activity_type_col]}'. Sampling from all facilities in the zone"
        )
        facilities_valid = facilities_in_zone

    # If no facilities found after all attempts, log the failure and return NaN
    if facilities_valid.empty:
        logger.info(
            f"Activity {row.name}: No facilities in zone {destination_zone} with {gdf_facility_type_col} '{fallback_type or row[row_activity_type_col]}'"
        )
        return pd.Series([np.nan, np.nan])

    # ----- Step 2. Sample a facility from the valid facilities

    # If "floor_area" is specified for sampling
    if (
        gdf_sample_col == "floor_area"
        and "floor_area" in facilities_valid.columns
        and facilities_valid["floor_area"].sum() != 0
    ):
        # Ensure floor_area is numeric
        facilities_valid["floor_area"] = pd.to_numeric(
            facilities_valid["floor_area"], errors="coerce"
        )
        facilities_valid = facilities_valid.dropna(subset=["floor_area"])
        facility = facilities_valid.sample(1, weights=facilities_valid["floor_area"])
        logger.info(f"Activity {row.name}: Sampled facility based on floor area)")
    else:
        # Otherwise, randomly sample one facility from the valid facilities
        facility = facilities_valid.sample(1)
        logger.info(f"Activity {row.name}: Sampled facility randomly")

    # Return the id and geometry of the selected facility
    return pd.Series([facility["id"].values[0], facility["geometry"].values[0]])


def fill_missing_zones(
    activity: pd.Series,
    travel_times_est: dict,
    activities_per_zone: pd.DataFrame,
    activity_col: str,
    use_mode: bool = False,
):
    """
    This function fills in missing zones in the activity chains. It uses a travel time matrix based on euclidian distance instead of
    the computed travel time matrix which is a sparse matrix.

    Parameters
    ----------
    activity: pd.Series
        A row from the activity chains dataframe
    travel_times_est: dict
        A dictionary with keys as tuples ({id_col}_from, {id_col}_to) and values as dictionaries containing time estimates.
        It is the output of zones_to_time_matrix()
    activities_per_zone: pd.DataFrame
        A dataframe with the number of activities and floorspace for each zone. The columns are 'OA21CD', 'counts', 'floor_area', 'activity'
        where 'activity' is the activity type as defined in the osmox config file
    activity_col: str
        The column name for the activity type
    use_mode: bool
        If True, the function will use the mode of transportation to estimate the travel time: time_{mode}.
        If False, it will use the average travel time: time_average.
        Default is False.

    Returns
    -------
    str
        The zone that has the estimated time closest to the given time. The zone also has an activity that matches the activity_col value.

    """
    activity_purpose = activity[activity_col]
    from_zone = activity["OA21CD"]
    to_zones = activities_per_zone[activities_per_zone["activity"] == activity_purpose][
        "OA21CD"
    ].tolist()

    logger.debug(
        f"Activity {activity.TripID} | person: {activity.id}: Number of possible destination zones: {len(to_zones)}"
    )

    time = activity["TripTotalTime"]

    mode = activity["mode"] if use_mode else None

    return _get_zones_using_time_estimate(
        estimated_times=travel_times_est,
        from_zone=from_zone,
        to_zones=to_zones,
        time=time,
        mode=mode,
    )


def _get_zones_using_time_estimate(
    estimated_times: dict,
    from_zone: str,
    to_zones: list,
    time: int,
    mode: Optional[str] = None,
) -> str:
    """
    This function returns the zone that has the estimated time closest to the given time. It is meant to be used inside fill_missing_zones()

    Parameters:
    ----------
    estimated_times: dict
        A dictionary with keys as tuples ({id_col}_from, {id_col}_to) and values as dictionaries containing time estimates. It is the output of zones_to_time_matrix()
    id_col: str
        The column name for the zone id.
    from_zone: str
        The zone of the previous activity
    to_zones (list): A list of destination zones.
    time:
        The target time to compare the estimated times with.
    mode: str, optional
        The mode of transportation. It should be one of ['car', 'pt', 'walk', 'cycle']. If not provided, the function uses 'time_average'.

    Returns
    -------
    str
        The zone that has the estimated time closest to the given time.
    """

    acceptable_modes = ["car", "pt", "walk", "cycle"]

    if mode is not None and mode not in acceptable_modes:
        error_message = f"Invalid mode: {mode}. Mode must be one of {acceptable_modes}."
        raise ValueError(error_message)

    # Convert to_zones to a set for faster lookup
    to_zones_set = set(to_zones)

    # Filter the entries where {id_col}_from matches the specific zone and {id_col}_to is in the specific list of zones
    filtered_dict = {
        k: v
        for k, v in estimated_times.items()
        if k[0] == from_zone and k[1] in to_zones_set
    }
    # get to_zone where time_average is closest to "time"
    if mode is not None:
        closest_to_zone = min(
            filtered_dict.items(), key=lambda item: abs(item[1][f"time_{mode}"] - time)
        )
    else:
        closest_to_zone = min(
            filtered_dict.items(), key=lambda item: abs(item[1]["time_average"] - time)
        )

    return closest_to_zone[0][1]
