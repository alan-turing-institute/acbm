import logging
from typing import Optional

import geopandas as gpd
import pandas as pd
import pandera as pa
from pandarallel import pandarallel
from pandera import Check, Column, DataFrameSchema
from pandera.errors import SchemaErrors

from acbm.assigning.utils import (
    _map_day_to_wkday_binary,
    _map_time_to_day_part,
    zones_to_time_matrix,
)
from acbm.config import Config

pandarallel.initialize(progress_bar=True)


logger = logging.getLogger("assigning_primary_feasible")

# --- Schemas for validation

activity_chains_schema = DataFrameSchema(
    {
        "mode": Column(str),
        # "TravDay": Column(pa.Float, Check.isin([1, 2, 3, 4, 5, 6, 7]), nullable=True),
        "tst": Column(pa.Float, Check.less_than_or_equal_to(1440), nullable=True),
        "TripTotalTime": Column(pa.Float, nullable=True),
        # TODO: add more columns ...
    },
    strict=False,
)

activities_per_zone_schema = DataFrameSchema(
    {
        "counts": Column(pa.Int),
        "floor_area": Column(pa.Float),
        "activity": Column(str),
    },
    strict=False,
)

boundaries_schema = DataFrameSchema(
    {
        "geometry": Column("geometry"),
    },
    strict=False,
)

travel_times_schema = DataFrameSchema(
    {
        "mode": Column(str),
        # "weekday": Column(pa.Float, Check.isin([0, 1]), nullable=True), # Does not exist if we make our own estimate
        # "time_of_day": Column(str, nullable=True),
        "time": Column(float),
    },
    strict=False,
)

input_schemas = {
    "activity_chains": activity_chains_schema,
    "activities_per_zone": activities_per_zone_schema,
    "boundaries": boundaries_schema,
    "travel_times": travel_times_schema,
}


def get_possible_zones(
    activity_chains: pd.DataFrame,
    activities_per_zone: pd.DataFrame,
    activity_col: str,
    key_col: str,
    boundaries: gpd.GeoDataFrame,
    zone_id: str,
    travel_times: Optional[pd.DataFrame] = None,
    filter_by_activity: bool = False,
    time_tolerance: float = 0.2,
    detour_factor: float = 1.56,
    decay_rate: float = 0.0001,
) -> dict:
    """
    Get possible zones for all activity chains in the dataset. This function loops over the travel_times dataframe and filters by mode, time of day and weekday/weekend.
    At each loop it applies the _get_possible_zones function to each row in the activity_chains dataframe.
    The travel_times dataframe is big, so doing some initial filtering before applying _get_possible_zones makes the process faster as less filtering is done for each
    row when running _get_possible_zones.

    Parameters
    ----------
    activity_chains: pd.DataFrame
        A dataframe with activity chains
    travel_times: Optional[pd.DataFrame]
        A dataframe with travel times between zones. If not provided, it will be created using zones_to_time_matrix.
    activities_per_zone: pd.DataFrame
        A dataframe with the number of activities and floorspace for each zone. The columns are 'OA21CD', 'counts', 'floor_area', 'activity'
        where 'activity' is the activity type as defined in the osmox config file
    key_col: str
        The column in activity_chains that will be used as a key in the dictionary
    boundaries: gpd.GeoDataFrame
        A GeoDataFrame with the boundaries of the zones. Used to create the travel_times dataframe if not provided
    zone_id: str
        The column name of the zone id in the activity_chains dataframe
    filter_by_activity: bool
        If True, we will return a results that only includes destination zones that have an activity that matches the activity purpose
    time_tolerance: int
        The time tolerance is used to filter the travel_times dataframe to only include travel times within a certain range of the
        activity chain's travel time (which is stored in "TripTotalTime"). Allowable travel_times are those that fall in the range of:
        travel_time_reported * (1 - time_tolerance) <= travel_time_reported <= travel_time_reported * (1 + time_tolerance)
        Default = 0.2
    detour_factor: float
        The detour factor used to calculate travel distances between zones from euclidian distances.
        Default = 1.56
    decay_rate: float
        The decay rate used to calculate travel times between zones from travel distances. Detours make up a smaller portion of
        longer distance trips. Default = 0.0001


    Returns
    -------
    dict
        A dictionary of dictionaries. Each dictionary is for one of the rows in activity chains
        with the origin zone as the key and a list of possible destination zones as the value. Eg:
        {
        164: {'E00059011': ['E00056917','E00056922', 'E00056923']},
        165: {'E00059012': ['E00056918','E00056952', 'E00056923']}
        }
    """

    # Validate inputs lazily
    try:
        activity_chains = input_schemas["activity_chains"].validate(
            activity_chains, lazy=True
        )
        activities_per_zone = input_schemas["activities_per_zone"].validate(
            activities_per_zone, lazy=True
        )
        boundaries = input_schemas["boundaries"].validate(boundaries, lazy=True)
        travel_times = input_schemas["travel_times"].validate(travel_times, lazy=True)

    except SchemaErrors as e:
        logger.error("Validation failed with errors:")
        logger.error(e.failure_cases)  # prints all the validation errors at once
        return None

    if travel_times is None:
        logger.info("Travel time matrix not provided: Creating travel times estimates")
        travel_times = zones_to_time_matrix(
            zones=boundaries,
            id_col=zone_id,
            time_units="m",
            detour_factor=detour_factor,
            decay_rate=decay_rate,
        )

    list_of_modes = activity_chains["mode"].unique()
    logger.info(f"Unique modes found in the dataset are: {list_of_modes}")

    # use map_day_to_wkday_binary to identify if activity is on a weekday or weekend
    activity_chains["weekday"] = activity_chains["TravDay"].apply(
        _map_day_to_wkday_binary
    )
    # day types identifier (weekday/weekend)
    day_types = activity_chains["weekday"].unique()
    # use map_time_to_day_part to add a column to activity
    activity_chains["time_of_day"] = activity_chains["tst"].apply(_map_time_to_day_part)
    # get unique time_of_day values
    list_of_times_of_day = activity_chains["time_of_day"].unique()

    # Initialize a list to collect results
    results_list = []

    # loop over the list of modes
    for mode in list_of_modes:
        logger.info(f"Processing mode: {mode}")
        # filter the travel_times dataframe to only include rows with the current mode
        travel_times_filtered_mode = travel_times[travel_times["mode"] == mode]

        # if the mode is public transport, we need to filter the travel_times data based on time_of_day and weekday/weekend
        # this only applies if we have the time_of_day column in the travel_times dataframe (not the case if we've estimated
        # travel times).

        # if "weekday" does not exist, we skip. Our travel_time_estimates (from zones_to_time_matrix())
        # don't have weekday information

        if (
            mode == "pt"
            and "time_of_day" in travel_times.columns
            and "weekday" in travel_times.columns
        ):
            for time_of_day in list_of_times_of_day:
                logger.info(f"Processing time of day: {time_of_day} | mode: {mode}")
                for day_type in day_types:
                    logger.info(
                        f"Processing time of day: {time_of_day} | weekday: {day_type} | mode: {mode}"
                    )
                    # filter the travel_times dataframe to only include rows with the current time_of_day and weekday
                    travel_times_filtered_mode_time_day = travel_times_filtered_mode[
                        (travel_times_filtered_mode["weekday"] == day_type)
                        & (travel_times_filtered_mode["time_of_day"] == time_of_day)
                    ]
                    unique_modes = travel_times_filtered_mode_time_day["mode"].unique()
                    logger.info(f"unique modes after filtering are: {unique_modes}")

                    # filter the activity chains to the current mode, time_of_day and weekday
                    activity_chains_filtered = activity_chains[
                        (activity_chains["mode"] == mode)
                        & (activity_chains["time_of_day"] == time_of_day)
                        & (activity_chains["weekday"] == day_type)
                    ]

                    if (
                        not travel_times_filtered_mode_time_day.empty
                        and not activity_chains_filtered.empty
                    ):
                        # apply get_possible_zones to each row in activity_chains_filtered
                        # pandarallel.initialize(progress_bar=True)
                        possible_zones = activity_chains_filtered.parallel_apply(
                            lambda row, tt=travel_times_filtered_mode_time_day: {
                                row[key_col]: _get_possible_zones(
                                    activity=row,
                                    travel_times=tt,
                                    activities_per_zone=activities_per_zone,
                                    filter_by_activity=filter_by_activity,
                                    activity_col=activity_col,
                                    zone_id=zone_id,
                                    time_tolerance=time_tolerance,
                                )
                            },
                            axis=1,
                        )

                        results_list.extend(possible_zones)

        # for all other modes, we don't care about time of day and weekday/weekend
        else:
            travel_times_filtered_mode_time_day = travel_times_filtered_mode
            # if mode = car_passenger, we compare to the travel time for car (we don't
            # have travel times for car_passenger)
            if mode in ("car_passenger", "taxi"):
                activity_chains_filtered = activity_chains[
                    (activity_chains["mode"] == "car")
                ]
            else:
                activity_chains_filtered = activity_chains[
                    (activity_chains["mode"] == mode)
                ]

            if (
                not travel_times_filtered_mode_time_day.empty
                and not activity_chains_filtered.empty
            ):
                # apply _get_possible_zones to each row in activity_chains_filtered
                # pandarallel.initialize(progress_bar=True)
                possible_zones = activity_chains_filtered.parallel_apply(
                    lambda row, tt=travel_times_filtered_mode_time_day: {
                        row[key_col]: _get_possible_zones(
                            activity=row,
                            travel_times=tt,
                            activities_per_zone=activities_per_zone,
                            filter_by_activity=filter_by_activity,
                            activity_col=activity_col,
                            zone_id=zone_id,
                            time_tolerance=time_tolerance,
                        )
                    },
                    axis=1,
                )

                results_list.extend(possible_zones)

    # Combine all dictionaries in the list into a single dictionary
    results = {}
    for result in results_list:
        for key, value in result.items():
            results[key] = value

    return results


def _get_possible_zones(
    activity: pd.Series,
    travel_times: pd.DataFrame,
    activities_per_zone: pd.DataFrame,
    filter_by_activity: bool,
    activity_col: str,
    zone_id: str,
    time_tolerance: float = 0.2,
) -> dict:
    """
    Get possible zones for a given activity chain

    Parameters
    ----------
    activity: pd.Series
        A row from the activity chains dataframe. It should contain the following columns: 'tst', 'TripTotalTime', 'mode', 'OA21CD'
    travel_times: pd.DataFrame
        A dataframe with travel times between zones
    activities_per_zone: pd.DataFrame
        A dataframe with the number of activities and floorspace for each zone. The columns are 'OA21CD', 'counts', 'floor_area', 'activity'
        where 'activity' is the activity type as defined in the osmox config file
    filter_by_activity: bool
        If True, we will return a results that only includes destination zones that have an activity that matches the activity purpose
    time_tolerance: int
        The time tolerance is used to filter the travel_times dataframe to only include travel times within a certain range of the
        activity chain's travel time (which is stored in "TripTotalTime"). Allowable travel_times are those that fall in the range of:
        travel_time_reported * (1 - time_tolerance) <= travel_time_reported <= travel_time_reported * (1 + time_tolerance)
        Default = 0.2

    Returns
    -------
    dict
        A dictionary with the origin zone as the key and a list of possible destination zones as the value
    """
    # get the travel time
    travel_time = activity["TripTotalTime"]
    # get the origin zone
    origin_zone = activity[zone_id]
    # get the activity purpose
    activity_purpose = activity[activity_col]

    # filter the travel_times dataframe by trip_origin and activity_purpose
    travel_times_filtered_origin_mode = travel_times[
        travel_times[Config.origin_zone_id(zone_id)] == origin_zone
    ]
    # do we include only zones that have an activity that matches the activity purpose?
    if filter_by_activity:
        filtered_activities_per_zone = activities_per_zone[
            # activities_per_zone["activity"].str.split("_").str[0] == activity_purpose
            activities_per_zone["activity"] == activity_purpose
        ]
        logger.debug(
            f"Activity {activity.id}: Number of zones with activity {activity_purpose}: \
            {len(filtered_activities_per_zone)}"
        )

        # keep only the zones that have the activity purpose
        travel_times_filtered_origin_mode = travel_times_filtered_origin_mode[
            travel_times_filtered_origin_mode[Config.destination_zone_id(zone_id)].isin(
                filtered_activities_per_zone[zone_id]
            )
        ]

    # how many zones are reachable?
    logger.debug(
        f"Activity {activity.id}: Number of zones with activity {activity_purpose} \
        that are reachable using reported mode: {len(travel_times_filtered_origin_mode)}"
    )

    # filter by reported trip time
    travel_times_filtered_time = travel_times_filtered_origin_mode[
        (
            travel_times_filtered_origin_mode["time"]
            >= travel_time - time_tolerance * travel_time
        )
        & (
            travel_times_filtered_origin_mode["time"]
            <= travel_time + time_tolerance * travel_time
        )
    ]

    logger.debug(
        f"Activity {activity.id}: Number of zones with activity {activity_purpose} within threshold of reported time {travel_time}: \
            {len(travel_times_filtered_time)}"
    )

    # if travel_times_filtered_time returns an empty df, select the row with the closest time to the reported time
    if travel_times_filtered_time.empty:
        logger.debug(
            f"Activity {activity.id}: NO zones match activity {activity_purpose} within threshold of reported time {travel_time}: \
            Relaxing tolerance and getting matching zone that is closest to reported travel time"
        )
        travel_times_filtered_time = travel_times_filtered_origin_mode.iloc[
            (travel_times_filtered_origin_mode["time"] - travel_time)
            .abs()
            .argsort()[:1]
        ]

    # create dictionary with key = origin_zone and values = list of travel_times_filtered.OA21CD_to
    return (
        travel_times_filtered_time.groupby(Config.origin_zone_id(zone_id))[
            Config.destination_zone_id(zone_id)
        ]
        .apply(list)
        .to_dict()
    )
