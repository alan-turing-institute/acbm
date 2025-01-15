import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

# categorical (exact) matching - (for household level)

logger = logging.getLogger("matching")


@dataclass
class MatcherExact:
    df_pop: pd.DataFrame
    df_pop_id: str
    df_sample: pd.DataFrame
    df_sample_id: str
    matching_dict: Dict[str, List[str]]
    fixed_cols: List[str]
    optional_cols: List[str]
    n_matches: int | None = 10
    chunk_size: int = 50000
    show_progress: bool = True
    matched_dict: Dict[str, List[str]] = field(
        default_factory=lambda: defaultdict(list)
    )
    match_count: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def __post_init__(self):
        # Extract equivalent column names from dictionary
        self.fixed_pop_cols = [self.matching_dict[col][0] for col in self.fixed_cols]
        self.fixed_sample_cols = [self.matching_dict[col][1] for col in self.fixed_cols]
        self.optional_pop_cols = [
            self.matching_dict[col][0] for col in self.optional_cols
        ]
        self.optional_sample_cols = [
            self.matching_dict[col][1] for col in self.optional_cols
        ]
        self.remaining_df_pop = self.df_pop.copy()
        self.remaining_df_sample = self.df_sample.copy()

    def _match_categorical(
        self,
        df_pop: pd.DataFrame,
        df_pop_cols: List[str],
        df_pop_id: str,
        df_sample: pd.DataFrame,
        df_sample_cols: List[str],
        df_sample_id: str,
        chunk_size: int,
        show_progress: bool = True,
    ) -> Dict[str, List[str]]:
        """
        Match the rows in two DataFrames based on specified columns.
        The function matches the rows in df_pop to the rows in df_sample based
        on the columns in df_pop_cols and df_sample_cols. The matching is done
        in chunks to avoid memory issues.

        Parameters
        ----------
        df_pop: pandas DataFrame
            The DataFrame to be matched on
        df_pop_cols: list
            The columns to be used for matching in df_pop
        df_pop_id: str
            The column name that contains the unique identifier in df_pop
            It is the key in the final dictionary
        df_sample: pandas DataFrame
            The DataFrame to be matched with
        df_sample_cols: list
            The columns to be used for matching in df_sample
        df_sample_id: str
            The column name that contains the unique identifier in df_sample
            It is the value in the final dictionary
        chunk_size: int
            The number of rows to process at a time
        show_progress: bool
            Whether to print the progress of the matching to the console

        Returns
        -------
        results: dict
            A dictionary with the matched rows {df_pop_id: [df_sample_id_1, df_sample_id_2, ...]}
        """
        results = {}

        # loop over the df_pop DataFrame in chunks
        for i in range(0, df_pop.shape[0], chunk_size):
            # Adjust chunk size if remaining rows are less than chunk size
            j = min(i + chunk_size, df_pop.shape[0])
            if show_progress:
                print("matching rows ", i, "to", j, " out of ", df_pop.shape[0])

            df_pop_chunk = df_pop.iloc[i:j]
            # exact match through a join on the specified columns
            df_matched_chunk = df_pop_chunk.merge(
                df_sample, left_on=df_pop_cols, right_on=df_sample_cols, how="left"
            )
            # convert matched df to dictionary
            df_matched_dict_i = (
                df_matched_chunk.groupby(df_pop_id)[df_sample_id].apply(list).to_dict()
            )
            # Filter out NaN values from the lists
            df_matched_dict_i = {
                k: [x for x in v if pd.notna(x)] for k, v in df_matched_dict_i.items()
            }
            results.update(df_matched_dict_i)
        return results

    def iterative_match_categorical(self) -> Dict[str, List[str]]:
        """
        Perform categorical matching iteratively, relaxing constraints in each round by
        removing one optional column at a time (optional columns are ordered by importance).
        For each household in df_pop, we stop matching when if matches have exceeded
        n_matches.

        Returns
        -------
        dict
            Dictionary with matched households.
        """
        for i in range(len(self.optional_pop_cols) + 1):
            if i > 0:
                self.optional_pop_cols.pop()
                self.optional_sample_cols.pop()

            current_pop_cols = self.fixed_pop_cols + self.optional_pop_cols
            current_sample_cols = self.fixed_sample_cols + self.optional_sample_cols

            print(f"Categorical matching level {i}: {current_pop_cols}")

            current_matches = self._match_categorical(
                df_pop=self.remaining_df_pop,
                df_pop_cols=current_pop_cols,
                df_pop_id=self.df_pop_id,
                df_sample=self.remaining_df_sample,
                df_sample_cols=current_sample_cols,
                df_sample_id=self.df_sample_id,
                chunk_size=self.chunk_size,
                show_progress=self.show_progress,
            )

            for pop_id, sample_ids in current_matches.items():
                unique_sample_ids = [
                    sid for sid in sample_ids if sid not in self.matched_dict[pop_id]
                ]
                self.matched_dict[pop_id].extend(unique_sample_ids)
                self.match_count[pop_id] += len(unique_sample_ids)

            matched_ids = (
                [
                    pop_id
                    for pop_id, count in self.match_count.items()
                    if count >= self.n_matches
                ]
                if self.n_matches is not None
                else []
            )
            self.remaining_df_pop = self.remaining_df_pop[
                ~self.remaining_df_pop[self.df_pop_id].isin(matched_ids)
            ]

            if self.remaining_df_pop.empty:
                break

        return dict(self.matched_dict)


# propensity score matching - (for individual level)


def match_psm(df1: pd.DataFrame, df2: pd.DataFrame, matching_columns: list) -> dict:
    """
    Use the Propensity Score Matching (PSM) method to match the rows in two DataFrames
    The distances between columns is calculated using the NearestNeighbors algorithm

    Parameters
    ----------
    df1: pandas DataFrame
        The first DataFrame to be matched on
    df2: pandas DataFrame
        The second DataFrame to be matched with
    matching_columns: list
        The columns to be used for the matching

    Returns
    -------
    matches: dict
        A dictionary with the matched row indeces from the two DataFrames {df1: df2}
    """

    # Initialize an empty dict to store the matches
    matches = {}

    # Matching without replacement
    while not df1.empty:
        # Fit a NearestNeighbors model on the specified columns for df2
        nn = NearestNeighbors(n_neighbors=1, algorithm="ball_tree")
        nn.fit(df2[matching_columns])

        # Find the closest row in df2 for each row in df1
        distances, indices = nn.kneighbors(df1[matching_columns])

        # Get the index of the closest match in df2 for each row in df1
        closest_indices = indices.flatten()

        # Get the row in df1 with the smallest distance to its closest match in df2
        min_distance_index = np.argmin(distances)

        # Get the corresponding row in df2
        closest_df2_index = closest_indices[min_distance_index]

        # Get the row id from df1 and df2
        row_id_df1 = df1.index[min_distance_index]
        row_id_df2 = df2.index[closest_df2_index]

        # Store the match in the dictionary
        matches[row_id_df1] = row_id_df2

        # Remove the matched rows from df1 and df2
        df1 = df1.drop(df1.index[min_distance_index])
        df2 = df2.drop(df2.index[closest_df2_index])

    return matches


# TODO: parallelize the matching process. See this stackoverflow suggestion
# for iterating over dict keys https://stackoverflow.com/a/30075659
def match_individuals(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    matching_columns: list,
    df1_id: str,
    df2_id: str,
    matches_hh: dict,
    show_progress: bool = False,
) -> dict:
    """
    Apply a matching function iteratively to members of each household.
    In each iteration, filter df1 and df2 to the household ids of item i
    in matches_hh, and then apply the matching function to the filtered DataFrames.

    Parameters
    ----------
    df1: pandas DataFrame
        The first DataFrame to be matched on
    df2: pandas DataFrame
        The second DataFrame to be matched with
    matching_columns: list
        The columns to be used for the matching
    df1_id: str
        The household_id from the first DataFrame
    df2_id: str
        The household_id from the second DataFrame
    matches_hh: dict
        A dictionary with the matched household ids {df1_id: df2_id}
    show_progress: bool
        Whether to print the progress of the matching to the console

    Returns
    -------
    matches: dict
        A dictionary with the matched row indeces from the two DataFrames {df1: df2}

    """
    # Initialize an empty dic to store the matches
    matches = {}
    # Remove all unmateched households
    matches_hh = {key: value for key, value in matches_hh.items() if not pd.isna(value)}

    # loop over all groups of df1_id
    # note: for large populations looping through the groups (keys) of the
    # large dataframe (assumed to be df1) is more efficient than looping
    # over keys and subsetting on a key in each iteration.
    for i, (key, rows_df1) in enumerate(df1.groupby(df1_id), 1):
        try:
            value = matches_hh[key]
        except Exception:
            # Continue if key not in matches_hh
            continue
        rows_df2 = df2[df2[df2_id] == int(value)]

        if show_progress and i % 100 == 0:
            # Print the iteration number and the number of keys in the dict
            logger.info(f"Matching for household {i} out of: {len(matches_hh)}")

        # apply the matching
        match = match_psm(rows_df1, rows_df2, matching_columns)

        # append the results to the main dict
        matches.update(match)

    return matches


def match_remaining_individuals(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    matching_columns: list,
    remaining_ids: list[int],
    show_progress: bool = False,
) -> dict:
    """
    Apply a matching function iteratively to members of each household.
    In each iteration, filter df1 and df2 to the household ids of item i
    in matches_hh, and then apply the matching function to the filtered DataFrames.

    Parameters
    ----------
    df1: pandas DataFrame
        The first DataFrame to be matched on
    df2: pandas DataFrame
        The second DataFrame to be matched with
    matching_columns: list
        The columns to be used for the matching
    matches_hh: dict
        A dictionary with the matched household ids {df1_id: df2_id}
    show_progress: bool
        Whether to print the progress of the matching to the console

    Returns
    -------
    matches: dict
        A dictionary with the matched row indeces from the two DataFrames {df1: df2}

    """
    # Initialize an empty dic to store the matches
    matches = {}

    # loop over all groups of df1_id
    # note: for large populations looping through the groups (keys) of the
    # large dataframe (assumed to be df1) is more efficient than looping
    # over keys and subsetting on a key in each iteration.
    df1_remaining = df1.loc[df1["id"].isin(remaining_ids)]
    chunk_size = 1000
    for i, rows_df1 in df1_remaining.groupby(
        np.arange(len(df1_remaining)) // chunk_size
    ):
        rows_df2 = df2
        if show_progress:
            # Print the iteration number and the number of keys in the dict
            print(
                f"Matching remaining individuals, {i * chunk_size} out of: {len(remaining_ids)}"
            )

        # apply the matching
        match = match_psm(rows_df1, rows_df2, matching_columns)

        # append the results to the main dict
        matches.update(match)

    return matches
