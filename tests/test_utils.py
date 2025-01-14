import pandas as pd
import pytest

from acbm.utils import households_with_common_travel_days


@pytest.fixture
def nts_trips():
    return pd.DataFrame.from_dict(
        {
            "IndividualID": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
            "HouseholdID": [1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 5, 5, 5],
            "TravDay": [
                1,
                1,
                1,
                2,
                3,
                2,
                3,
                3,
                3,
                3,
                pd.NA,
                pd.NA,
                pd.NA,
                pd.NA,
                pd.NA,
                4,
            ],
        }
    )


def test_households_with_common_travel_days(nts_trips):
    assert households_with_common_travel_days(nts_trips, [1]) == [1]
    assert households_with_common_travel_days(nts_trips, [1, 2]) == [1]
    assert households_with_common_travel_days(nts_trips, [1, 3]) == [1, 3]
    assert households_with_common_travel_days(nts_trips, [1, 3, 4]) == [1, 3]
