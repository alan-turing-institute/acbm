import geopandas as gpd
import pandas as pd
from shapely import wkt

from acbm.assigning.utils import _adjust_distance


def process_sequences(
    df: pd.DataFrame,
    pid_col: str,
    seq_col: str,
    origin_activity_col: str,
    destination_activity_col: str,
    suffix: str,
) -> pd.DataFrame:
    """
    Processes a DataFrame to generate activity sequences and counts the number of
    occurrences of each sequence.


    Parameters
    ----------
    df: pd.DataFrame
        The input DataFrame containing the data.
    pid_col: str
        The name of the column representing the unique identifier for each group.
    seq_col: str
        The name of the column representing the sequence order within each group.
    origin_activity_col: str
        The name of the column representing the origin activity.
    destination_activity_col: str
        The name of the column representing the destination activity.
    suffix: str
        The suffix to be added to the count column name.

    Returns
    -------
    pd.DataFrame
        A DataFrame with the activity sequences and their counts.

        activity_sequence           count_{suffix}
        ----------------------      --------------
        home - work - visit - home              5
        home - school - home                    3
        home - work - home                     20
    """
    # Step 1: Sort the DataFrame by 'pid' and 'seq'
    sorted_df = df.sort_values(by=[pid_col, seq_col])

    # Step 2: Group by 'pid' and concatenate 'origin activity' values followed by the
    # last 'destination activity' value
    activity_sequence_df = (
        sorted_df.groupby(pid_col)
        .apply(
            lambda x: " - ".join(
                [*x[origin_activity_col], x[destination_activity_col].iloc[-1]]
            )
        )
        .reset_index()
    )

    # Rename the columns for clarity
    activity_sequence_df.columns = [pid_col, "activity_sequence"]

    # Step 3: Group by the resulting 'activity_sequence' column and count the number of
    # values in each group
    return (
        activity_sequence_df.groupby("activity_sequence")
        .size()
        .reset_index(name=f"count_{suffix}")
    )


# TODO: add crs to config, and check other scripts
def calculate_od_distances(
    df: pd.DataFrame,
    start_wkt_col: str,
    end_wkt_col: str,
    crs_epsg: int,
    projected_epsg: int = 3857,
    detour_factor: float = 1.56,
    decay_rate: float = 0.0001,
) -> pd.DataFrame:
    """
    Calculate distances between start and end geometries in a DataFrame.

    Parameters
    ----------

    df: pd.DataFrame
        DataFrame containing WKT geometry columns.
    start_wkt_col: str
        Column name for start location WKT geometries.
    end_wkt_col: str
        Column name for end location WKT geometries.
    crs_epsg: int
        EPSG code for the original CRS (default is 4326 for WGS84).
    projected_epsg: int
        EPSG code for the projected CRS (default is 3857). We need a metric crs
        to calculte distances in meters.
    detour_factor: float
        Factor to adjust the estimated distance.
    decay_rate: float
        Decay rate for the distance adjustment. Detours are a smaller proportion of
        the direct distance for longer trips.

    Returns
    -------
    pd.DataFrame
        DataFrame with an additional 'distance' column containing distances in meters.
    """
    # Convert WKT strings to shapely geometries
    df["start_geometry"] = df[start_wkt_col].apply(wkt.loads)
    df["end_geometry"] = df[end_wkt_col].apply(wkt.loads)

    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(df, geometry="start_geometry")

    # Set the original CRS
    gdf.set_crs(epsg=crs_epsg, inplace=True)

    # Create a separate GeoDataFrame for the end geometries
    end_gdf = gdf.set_geometry("end_geometry")

    # Set the original CRS for the end_gdf
    end_gdf.set_crs(epsg=crs_epsg, inplace=True)

    # Transform both GeoDataFrames to a projected CRS
    gdf = gdf.to_crs(epsg=projected_epsg)
    end_gdf = end_gdf.to_crs(epsg=projected_epsg)

    # Calculate the distance between start and end geometries (in m)
    gdf["distance"] = gdf.geometry.distance(end_gdf.geometry)

    # Estimate actual travel distance
    gdf["distance"] = gdf["distance"].apply(
        lambda d: _adjust_distance(
            d, detour_factor=detour_factor, decay_rate=decay_rate
        )
    )

    # convert distance to km
    gdf["distance"] = round(gdf["distance"] / 1000, 1)

    return gdf
