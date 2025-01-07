import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from acbm.config import Config


def calculate_rmse(predictions, targets):
    """
    Calculate the Root Mean Squared Error (RMSE) between predictions and targets,
    excluding NaN values.

    Parameters:
    - predictions (np.ndarray or pd.Series): The predicted values.
    - targets (np.ndarray or pd.Series): The actual values.

    Returns:
    - float: The RMSE value.
    """
    # Ensure inputs are numpy arrays
    predictions = np.asarray(predictions)
    targets = np.asarray(targets)

    # Create a mask for non-NaN values
    mask = ~np.isnan(predictions) & ~np.isnan(targets)

    # Filter out NaN values
    filtered_predictions = predictions[mask]
    filtered_targets = targets[mask]

    # Calculate mean squared error
    mse = mean_squared_error(filtered_predictions, filtered_targets)

    # Calculate and return RMSE
    return np.sqrt(mse)


def get_travel_times(config: Config, use_estimates: bool = False) -> pd.DataFrame:
    if config.parameters.travel_times and not use_estimates:
        return pd.read_parquet(config.travel_times_filepath)
    return pd.read_parquet(config.travel_times_estimates_filepath)


def households_with_common_travel_days(
    nts_trips: pd.DataFrame, days: list[int]
) -> list[int]:
    return (
        nts_trips.groupby(["HouseholdID", "IndividualID"])["TravDay"]
        .apply(list)
        .map(set)
        .to_frame()
        .groupby(["HouseholdID"])["TravDay"]
        .apply(
            lambda sets_of_days: set.intersection(*sets_of_days)
            if set.intersection(*sets_of_days)
            else None
        )
        .to_frame()["TravDay"]
        .apply(
            lambda common_days: [day for day in common_days if day in days]
            if common_days is not None
            else []
        )
        .apply(lambda common_days: common_days if common_days else pd.NA)
        .dropna()
        .reset_index()["HouseholdID"]
        .to_list()
    )
