from dataclasses import dataclass
from typing import Self

import numpy as np
import pandas as pd

from acbm.config import Config


@dataclass
class Population:
    individuals: pd.DataFrame
    households: pd.DataFrame
    activities: pd.DataFrame
    legs: pd.DataFrame
    legs_geo: pd.DataFrame

    @classmethod
    def read(cls, config: Config) -> Self:
        individuals = pd.read_csv(config.output_path / "people.csv")
        households = pd.read_csv(config.output_path / "households.csv")
        activities = pd.read_csv(config.output_path / "activities.csv")
        legs = pd.read_csv(config.output_path / "legs.csv")
        legs_geo = pd.read_parquet(config.output_path / "legs_with_locations.parquet")
        return Population(
            individuals=individuals,
            households=households,
            activities=activities,
            legs=legs,
            legs_geo=legs_geo,
        )

    def filter_by_pid(self) -> Self:
        """
        Filter the input DataFrames to include only include people (pids) that exist in all
        dfs

        Parameters
        ----------
        individuals: pd.DataFrame
            Individuals DataFrame.
        activities: pd.DataFrame
            Activities DataFrame.
        legs: pd.DataFrame:
            Legs DataFrame.
        legs_geo: pd.DataFrame
            Legs with geo DataFrame.
        households: pd.DataFrame
            Households DataFrame.

        Returns
        -------
        tuple
            A tuple containing the filtered DataFrames (individuals, activities, legs, legs_geo, households).
        """
        # Identify common pids
        common_pids = (
            set(self.individuals["pid"])
            .intersection(self.activities["pid"])
            .intersection(self.legs["pid"])
            .intersection(self.legs_geo["pid"])
        )

        # Filter Individual Level DataFrames
        individuals = self.individuals[self.individuals["pid"].isin(common_pids)]
        activities = self.activities[self.activities["pid"].isin(common_pids)]
        legs = self.legs[self.legs["pid"].isin(common_pids)]
        legs_geo = self.legs_geo[self.legs_geo["pid"].isin(common_pids)]

        # Filter Household Level DataFrame
        households = self.households[self.households["hid"].isin(individuals["hid"])]

        return Population(
            individuals=individuals,
            households=households,
            activities=activities,
            legs=legs,
            legs_geo=legs_geo,
        )

    def filter_no_location(self) -> Self:
        """
        Cleans the provided DataFrames by removing rows without location data. Gets all pids
        that have at least one row with missing location data, and removes all rows with
        these pids. pids are geneerated from two sources:
        - legs_geo with missing start_loc or end_loc
        - individuals with missing hzone

        Parameters
        ----------
        individuals : pd.DataFrame
            DataFrame containing individual data.
        activities : pd.DataFrame
            DataFrame containing activity data.
        legs : pd.DataFrame
            DataFrame containing legs data.
        legs_geo : pd.DataFrame
            DataFrame containing legs with geographic data.
        households : pd.DataFrame
            DataFrame containing household data.

        Returns
        -------
        tuple
            A tuple containing the cleaned DataFrames
            (individuals_cleaned, households_cleaned, activities_cleaned, legs_cleaned, legs_geo_cleaned).
        """
        # Identify rows in legs_geo where start_loc or end_loc are null
        invalid_rows_legs_geo = self.legs_geo[
            self.legs_geo["start_loc"].isnull() | self.legs_geo["end_loc"].isnull()
        ]

        # Extract the pid values associated with these rows
        invalid_pids_legs_geo = invalid_rows_legs_geo["pid"].unique()

        # Identify rows in individuals where hzone is null
        invalid_rows_individuals = self.individuals[self.individuals["hzone"].isnull()]

        # Extract the pid values associated with these rows
        invalid_pids_individuals = invalid_rows_individuals["pid"].unique()

        # Combine the invalid pid values from both sources
        invalid_pids = set(invalid_pids_legs_geo).union(set(invalid_pids_individuals))

        # Remove rows with these pids from all DataFrames
        individuals_cleaned = self.individuals[
            ~self.individuals["pid"].isin(invalid_pids)
        ]
        activities_cleaned = self.activities[~self.activities["pid"].isin(invalid_pids)]
        legs_cleaned = self.legs[~self.legs["pid"].isin(invalid_pids)]
        legs_geo_cleaned = self.legs_geo[~self.legs_geo["pid"].isin(invalid_pids)]

        # Extract remaining hid values from individuals_cleaned
        remaining_hids = individuals_cleaned["hid"].unique()

        # Filter households_cleaned to only include rows with hid values in remaining_hids
        households_cleaned = self.households[
            self.households["hid"].isin(remaining_hids)
        ]

        return Population(
            individuals=individuals_cleaned,
            households=households_cleaned,
            activities=activities_cleaned,
            legs=legs_cleaned,
            legs_geo=legs_geo_cleaned,
        )


def add_home_location_to_individuals(
    legs_geo: pd.DataFrame, individuals: pd.DataFrame
) -> pd.DataFrame:
    """
    Adds home location to individuals dataframe. Location is obtained
    from legs_geo (rows with orign activity = home)

    Parameters
    ----------
    legs_geo : pd.DataFrame
        DataFrame containing legs with geographic data.
    individuals : pd.DataFrame
        DataFrame containing individual data.

    Returns
    -------
    pd.DataFrame
        The modified individuals DataFrame with location information.
    """
    # Filter by origin activity = home
    legs_geo_home = legs_geo[legs_geo["origin activity"] == "home"]

    # Get one row for each hid group
    legs_geo_home = legs_geo_home.groupby("hid").first().reset_index()

    # Keep only the columns we need: hid and start_location
    legs_geo_home = legs_geo_home[["hid", "start_loc"]]

    # Rename start_loc to loc
    legs_geo_home.rename(columns={"start_loc": "loc"}, inplace=True)

    # Merge legs_geo_home with individuals
    individuals_geo = individuals.copy()
    individuals_geo = individuals_geo.merge(legs_geo_home, on="hid")

    # Remove rows with missing loc
    return individuals_geo[individuals_geo["loc"].notnull()]


def log_row_count(
    df: pd.DataFrame, name: str, operation: str, row_counts: list[tuple[str, str, int]]
):
    """
    Logs the row count of a DataFrame along with a specified operation and name.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame whose row count is to be logged.
    name : str
        The name associated with the DataFrame.
    operation : str
        The operation being performed on the DataFrame.
    row_counts : list
        The list to which the row count information will be appended.

    Returns
    -------
    None
    """
    row_counts.append((operation, name, len(df)))


def calculate_percentage_remaining(
    row_counts: list[tuple[str, str, int]],
) -> list[tuple[str, str, int, float]]:
    """
    Calculate the percentage of rows remaining for each DataFrame based on the
    initial counts.

    Parameters
    ----------
    row_counts : list of tuples
        List of tuples containing stage, DataFrame names,
        and their row counts.

    Returns
    -------
    list of tuples
        List of tuples containing stage, DataFrame names, and their percentage
        of rows remaining.
    """
    # Extract initial counts
    initial_counts = {
        df_name: count for stage, df_name, count in row_counts if stage == "0_initial"
    }

    # Calculate percentage remaining
    percentage_remaining = []
    for stage, df_name, count in row_counts:
        if df_name in initial_counts:
            initial_count = initial_counts[df_name]
            percentage = round((count / initial_count) * 100, 1)
            percentage_remaining.append((stage, df_name, count, percentage))

    # Sort by df_name
    percentage_remaining.sort(key=lambda x: x[1])

    return percentage_remaining


# FUNCTIONS TO ADD ATTRIBUTES TO INDIVIDUALS


def get_passengers(
    legs: pd.DataFrame, individuals: pd.DataFrame, modes: list
) -> pd.DataFrame:
    """
    Marks individuals as (car) passengers based on the mode of transportation in the legs DataFrame.

    Parameters
    ----------
    legs : pd.DataFrame
        DataFrame containing legs data with info on an activity leg. Needs a 'mode' column and a 'pid' column.
    individuals : pd.DataFrame
        DataFrame containing individual data with a 'pid' column.
    modes : list
        List of passenger modes.

    Returns
    -------
    pd.DataFrame
        Updated individuals DataFrame with an 'isPassenger' boolean column.
    """
    # Get a list of unique pids where mode matches the chosen list of modes
    passenger_pids = legs[legs["mode"].isin(modes)]["pid"].unique()

    # Add a boolean column 'isPassenger' to the individuals DataFrame
    individuals["isPassenger"] = individuals["pid"].isin(passenger_pids)

    return individuals


def get_pt_subscription(individuals: pd.DataFrame, age_threshold=60):
    """
    Marks individuals as having a public transport subscription based on their age.

    Parameters
    ----------
    individuals : pd.DataFrame
        DataFrame containing individual data with an 'age' column.
    age_threshold : int
        Age threshold for public transport subscription. (normally the pension age)

    Returns
    -------
    pd.DataFrame
        Updated individuals DataFrame with an 'hasPTSubscription' boolean column.
    """
    # Add a boolean column 'hasPTSubscription' to the individuals DataFrame
    individuals["hasPTSubscription"] = individuals["age"] >= age_threshold

    return individuals


def get_students(
    individuals: pd.DataFrame,
    activities: pd.DataFrame,
    age_base_threshold: int | None = None,
    age_upper_threshold: int | None = None,
    activity: str = "education",
) -> pd.DataFrame:
    """
    Marks individuals as students based on whether they have an education activity,
    and optionally whether they are also below certain age thresholds.

    Parameters
    ----------
    individuals : pd.DataFrame
        DataFrame containing individual data with a 'pid' column.
    activities : pd.DataFrame
        DataFrame containing activity data with a 'pid' column.
    age_base_threshold : Optional[int]
        If specified, anyone below this age is automatically a student
    age_upper_threshold : Optional[int]
        If specified, this is the age limit for people to be a student. If someone has an education
        trip but is above this threshold, they are not a student
    activity : str, optional
        Activity type to consider for being a student. Default is 'education'.

    Returns
    -------
    pd.DataFrame
        Updated individuals DataFrame with an 'isStudent' boolean column.
    """

    # Get a list of unique pids where the activity is 'education'
    education_pids = activities[activities["activity"] == activity]["pid"].unique()

    if age_base_threshold is not None:
        # Everyone below age_base_threshold should be assigned to student
        base_students = individuals[individuals["age"] < age_base_threshold][
            "pid"
        ].unique()
        # Everyone below age_upper_threshold who has an education trip should also be a student
        if age_upper_threshold is not None:
            upper_students = individuals[
                (individuals["age"] < age_upper_threshold)
                & (individuals["pid"].isin(education_pids))
            ]["pid"].unique()
            student_pids = set(base_students).union(set(upper_students))
        else:
            student_pids = set(base_students)
    elif age_upper_threshold is not None:
        # Everyone below age_upper_threshold who has an education trip should be a student
        student_pids = individuals[
            (individuals["age"] < age_upper_threshold)
            & (individuals["pid"].isin(education_pids))
        ]["pid"].unique()
    else:
        # Only people with an education trip should be students
        student_pids = education_pids

    # Add a boolean column 'isStudent' to the individuals DataFrame
    individuals["isStudent"] = individuals["pid"].isin(student_pids)

    return individuals


def get_hhlIncome(
    individuals: pd.DataFrame,
    individuals_with_salary: pd.DataFrame,
    pension_age: int = 66,
    pension: int = 13000,
) -> pd.DataFrame:
    """
    Function to calculate the household level income from the individual level income data in the SPC
    dataset. The function groups salary data by household and then merges the household level income
    data back onto the individual level data.

    The salary data is missing for many individuals in the SPC dataset. It also does not include pension
    We add the state pension if the person has reached the state pension age (66 years) and has no salary data.

    TODO: add student maintenance loan?

    Parameters
    ----------
    individuals : pd.DataFrame
        The individual level data output from acbm
    individuals_with_salary : pd.DataFrame
        The original SPC dataset with the salary_yearly column

    Returns
    -------
    pd.DataFrame
        The individual level data with the hhlIncome column added
    """
    individuals_income = individuals.copy()

    # If person is a pensioner, add pension, otherwise keep salary
    individuals_with_salary["income_modeled"] = np.where(
        (individuals_with_salary["salary_yearly"] == 0)
        | (individuals_with_salary["salary_yearly"].isna()),
        np.where(individuals_with_salary["age_years"] >= pension_age, pension, 0),
        individuals_with_salary["salary_yearly"],
    )

    # Summarize the data by household to create the hhlIncome column
    household_income = (
        individuals_with_salary.groupby("household")["income_modeled"]
        .sum()
        .reset_index()
    )
    household_income.rename(columns={"income_modeled": "hhlIncome"}, inplace=True)
    # round hhlIncome to the nearest whole number
    household_income["hhlIncome"] = household_income["hhlIncome"].round()

    # Merge the household_income data onto the individuals data
    individuals_income = pd.merge(
        individuals_income,
        household_income,
        how="left",
        left_on="hid",
        right_on="household",
    )

    individuals_income.drop(columns="household", inplace=True)

    return individuals_income
