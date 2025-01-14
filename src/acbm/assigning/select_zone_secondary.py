import logging
from copy import deepcopy

import numpy as np
import pam
import pandas as pd
from pam.planner.choice_location import DiscretionaryTrips

logger = logging.getLogger("assigning_secondary_zone")


def set_home_ozone(data: pd.DataFrame, oact_col: str, ozone_col: str, hzone_col: str):
    """
    Ensure that all rows where 'oact' is 'home' have 'ozone' set to 'hzone'.

    Parameters
    ----------
    data : pandas DataFrame
        The input DataFrame.
    oact_col : str
        The column name for the activity data.
    ozone_col : str
        The column name for the ozone data to be filled.
    hzone_col : str
        The column name for the home zone data.
    """
    home_rows = data[oact_col] == "home"
    data.loc[home_rows, ozone_col] = data.loc[home_rows, hzone_col]


def shift_and_fill_column(
    data: pd.DataFrame,
    group_col: str,
    source_col: str,
    target_col: str,
    initial_value_col: str | None = None,
    oact_col: str | None = None,
    hzone_col: str | None = None,
) -> pd.DataFrame:
    """
    Fill the 'target_col' column by shifting the 'source_col' column within each group defined by 'group_col'.
    Optionally set the first row of 'target_col' in each group from another column.
    Ensure that all rows where 'oact' is 'home' have 'ozone' set to 'hzone'.

    Use case: Fill the 'ozone' column by shifting the 'dzone' column within each group defined by 'id_col'.
    Optionally set the first row of 'ozone' in each group from another column.

    Input: id | seq | oact      | dact     | ozone | dzone | hzone |
            1 |   1 | home      | work     | NaN   |     A |     X |
            1 |   2 | work      | shopping | NaN   |   NaN |     X |
            1 |   3 | shopping  | home     |  NaN  |   NaN |     X |

    Output:  id | seq | oact    | dact     | ozone | dzone | hzone |
              1 |   1 | home    | work     | X     |     A |     X |
              1 |   2 | work    | shopping | A     |   NaN |     X |
              1 |   3 | shopping| home     | C     |   X   |     X |


    Parameters
    ----------
    data : pandas DataFrame
        The input DataFrame.
    group_col : str
        The column name to group by.
    source_col : str
        The column name for the source data to be shifted.
    target_col : str
        The column name for the target data to be filled.
    initial_value_col : str, optional
        The column name to take the first row value from. Default is None.
    oact_col : str, optional
        The column name for the activity data. Default is None.
    hzone_col : str, optional
        The column name for the home zone data. Default is None.

    Returns
    -------
    pandas DataFrame
        The DataFrame with the 'target_col' column filled.
    """
    # Create a deep copy of the DataFrame to avoid modifying the original
    data_copy = deepcopy(data)

    # Set all values in the target column to NaN
    data_copy[target_col] = np.nan

    # Group by 'group_col' and apply the shift operation within each group
    data_copy[target_col] = data_copy.groupby(group_col)[source_col].shift(1)

    # Optionally set the first row of 'target_col' in each group from another column
    if initial_value_col is not None:
        first_rows = data_copy.groupby(group_col).head(1).index
        data_copy.loc[first_rows, target_col] = data_copy.loc[
            first_rows, initial_value_col
        ]

    # Ensure that all rows where 'oact' is 'home' have 'ozone' set to 'hzone'
    if oact_col is not None and hzone_col is not None:
        set_home_ozone(data_copy, oact_col, target_col, hzone_col)

    return data_copy


def create_od_matrices(
    df: pd.DataFrame,
    mode_column: str,
    value_column: str,
    zone_labels: tuple,
    fill_value: int,
    zone_from: str = "OA21CD_from",
    zone_to: str = "OA21CD_to",
) -> dict:
    """
    Create OD matrices for each mode in the DataFrame. This function is uused to create matrices for
    - travel times
    - od_probs
    to be used in discretionary activity selection

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data
    mode_column : str
        Column name containing the mode of transport
    value_column : str
        Column name containing the value to be used in the OD matrix
    fill_value : int
        Value to use when a value for a specific od pair is not available

    Returns
    -------
    dict
        A dictionary containing OD matrices for each mode.
        Key: str
            Mode of transport
        Value: np.array
            OD matrix

    """

    # Initialize dictionaries to hold OD matrices for each combination type
    modes = df[mode_column].unique()
    od_matrices = {
        mode: np.full((len(zone_labels), len(zone_labels)), fill_value)
        for mode in modes
    }

    # Create a mapping from zone labels to indices
    zone_index = {label: idx for idx, label in enumerate(zone_labels)}

    # Vectorized operation to populate OD matrices
    from_indices = df[zone_from].map(zone_index)
    to_indices = df[zone_to].map(zone_index)

    for mode in modes:
        logger.info(f"Starting mode: {mode}")
        mask = df[mode_column] == mode
        values = df[mask][value_column].fillna(fill_value)  # Fill missing values
        od_matrices[mode][from_indices[mask], to_indices[mask]] = values
        logger.info(f"Finished mode: {mode}")

    return od_matrices


def update_population_plans(
    population: pam.core.Population,
    od: pam.planner.od.ODFactory,
) -> None:
    """
    Update the plans in a population object using the DiscretionaryTrips planner

    """
    people_list = list(population.people())
    for i, plan in enumerate(population.plans()):
        try:
            planner = DiscretionaryTrips(plan=plan, od=od)
            planner.update_plan()
            logger.info(f"Updated plan for person id {people_list[i][0]}")
        except Exception as e:
            logger.error(f"An error occurred with person id {people_list[i][0]}: {e}")
