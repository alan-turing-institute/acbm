import pickle as pkl
from pathlib import Path

import numpy as np
import pandas as pd

from acbm.assigning.utils import cols_for_assignment_all
from acbm.cli import acbm_cli
from acbm.config import load_and_setup_config
from acbm.matching import MatcherExact, match_individuals
from acbm.preprocessing import (
    count_per_group,
    nts_filter_by_region,
    nts_filter_by_year,
    num_adult_child_hh,
    transform_by_group,
    truncate_values,
)


@acbm_cli
def main(config_file):
    config = load_and_setup_config(config_file)
    logger = config.get_logger("matching", __file__)

    pd.set_option("display.max_columns", None)

    def get_interim_path(
        file_name: str,
    ) -> Path:
        return config.interim_path / "matching" / file_name

    # ## Step 1: Load in the datasets

    # ### SPC

    logger.info("Loading SPC data")

    # Read in the spc data (parquet format)
    spc = pd.read_parquet(config.spc_combined_filepath)

    logger.info("Filtering SPC data to specific columns")
    # select columns
    spc = spc[
        [
            "id",
            "household",
            "pid_hs",
            "msoa11cd",
            "oa11cd",
            "members",
            "sic1d2007",
            "sic2d2007",
            "pwkstat",
            "salary_yearly",
            "salary_hourly",
            "hid",
            "accommodation_type",
            "communal_type",
            "num_rooms",
            "central_heat",
            "tenure",
            "num_cars",
            "sex",
            "age_years",
            "ethnicity",
            "nssec8",
        ]
    ]

    logger.info("Sampling SPC data")
    # --- temporary reduction of the dataset for quick analysis

    # Identify unique households
    unique_households = spc["household"].unique()
    # Sample a subset of households, RNG seeded above with `init_rng``
    sampled_households = pd.Series(unique_households).sample(
        n=(
            config.parameters.number_of_households
            if config.parameters.number_of_households is not None
            else unique_households.shape[0]
        ),
    )
    # Filter the original DataFrame based on the sampled households
    spc = spc[spc["household"].isin(sampled_households)]

    logger.info(f"Sampled {spc.shape[0]} individuals from SPC data")

    # ### NTS
    #
    # The NTS is split up into multiple tables. We will load in the following tables:
    # - individuals
    # - households
    # - trips

    logger.info("Loading NTS data")

    # #### PSU

    logger.info("Loading NTS data: PSU table")
    path_psu = config.psu_filepath
    psu = pd.read_csv(path_psu, sep="\t")

    # #### Individuals
    logger.info("Loading NTS data: individuals table")

    nts_individuals = pd.read_csv(
        config.nts_individuals_filepath,
        sep="\t",
        usecols=[
            "IndividualID",
            "HouseholdID",
            "PSUID",
            "Age_B01ID",
            "Age_B04ID",
            "Sex_B01ID",
            "OfPenAge_B01ID",
            "HRPRelation_B01ID",
            "EdAttn1_B01ID",
            "EdAttn2_B01ID",
            "EdAttn3_B01ID",
            "OwnCycleN_B01ID",  # Owns a cycle
            "DrivLic_B02ID",  # type of driving license
            "CarAccess_B01ID",
            "IndIncome2002_B02ID",
            "IndWkGOR_B02ID",  # Region of usual place of work
            "EcoStat_B02ID",  # Working status of individual
            "EcoStat_B03ID",
            "NSSec_B03ID",  # NSSEC high level breakdown
            "SC_B01ID",  # Social class of individual
            "Stat_B01ID",  # employee or self-employed
            "WkMode_B01ID",  # Usual means of travel to work
            "WkHome_B01ID",  # Work from home
            "PossHom_B01ID",  # Is it possible to work from home?
            "OftHome_B01ID",  # How often work from home
            "TravSh_B01ID",  # Usual mode from main food shopping trip
            "SchDly_B01ID",  # Daily school journey?
            "SchTrav_B01ID",  # Usual mode of travel to school
            "SchAcc_B01ID",  # IS school trip accompanied by an adult?
            "FdShp_B01ID",  # How do you usually carry ot main food shop (go to shop, online etc)
        ],
    )

    # #### Households
    logger.info("Loading NTS data: household table")

    nts_households = pd.read_csv(
        config.nts_households_filepath,
        sep="\t",
        usecols=[
            "HouseholdID",
            "PSUID",
            "HHIncome2002_B02ID",
            "AddressType_B01ID",  # type of house
            "Ten1_B02ID",  # type of tenure
            "HHoldNumAdults",  # total no. of adults in household
            "HHoldNumChildren",  # total no. of children in household
            "HHoldNumPeople",  # total no. of people in household
            "NumLicHolders",  # total no. of driving license holders in household
            "HHoldEmploy_B01ID",  # number of employed in household
            "NumBike",  # no. of bikes
            "NumCar",  # no. of cars
            "NumVanLorry",  # no. of vans or lorries
            "NumMCycle",  # no. of motorcycles
            "WalkBus_B01ID",  # walk time from house to nearest bus stop
            "Getbus_B01ID",  # frequency of bus service
            "WalkRail_B01ID",  # walk time from house to nearest rail station
            "JTimeHosp_B01ID",  # journey time to nearest hospital
            "DVShop_B01ID",  # person no. for main food shooper in hh
            "Settlement2011EW_B03ID",  # ONS Urban/Rural: 2 categories
            "Settlement2011EW_B04ID",  # ONS Urban/Rural: 3 categories
            "HHoldOAClass2011_B03ID",  # Census 2011 OA Classification
            "HRPWorkStat_B02ID",  # HH ref person working status
            "HRPSEGWorkStat_B01ID",  #  HH ref person socio economic group for active workers
            "W0",  # Unweighted interview sample
            "W1",  # Unweighted diary sample
            "W2",  # Weighted diary sample
            "W3",  # Weighted interview sample
        ],
    )

    # #### Trips
    logger.info("Loading NTS data: trips table")

    nts_trips = pd.read_csv(
        config.nts_trips_filepath,
        sep="\t",
        usecols=[
            "TripID",
            "DayID",
            "IndividualID",
            "HouseholdID",
            "PSUID",
            "PersNo",
            "TravDay",
            "JourSeq",
            "ShortWalkTrip_B01ID",
            "NumStages",
            "MainMode_B03ID",
            "MainMode_B04ID",
            "TripPurpFrom_B01ID",
            "TripPurpTo_B01ID",
            "TripPurpose_B04ID",
            "TripStart",
            "TripEnd",
            "TripTotalTime",
            "TripTravTime",
            "TripDisIncSW",
            "TripDisExSW",
            "TripOrigGOR_B02ID",
            "TripDestGOR_B02ID",
            "W5",
            "W5xHH",
        ],
    )

    # #### Filter by year
    #
    # We will filter the NTS data to only include data from specific years. We can choose
    # only 1 year, or multiple years to increase our sample size and the likelihood of a
    # match with the spc.

    logger.info("Filtering NTS data by specified year(s)")

    years = config.parameters.nts_years

    nts_individuals = nts_filter_by_year(nts_individuals, psu, years)
    nts_households = nts_filter_by_year(nts_households, psu, years)
    nts_trips = nts_filter_by_year(nts_trips, psu, years)

    # #### Filter by geography
    #

    regions = config.parameters.nts_regions

    nts_individuals = nts_filter_by_region(nts_individuals, psu, regions)
    nts_households = nts_filter_by_region(nts_households, psu, regions)
    nts_trips = nts_filter_by_region(nts_trips, psu, regions)

    # Create dictionaries of key value pairs

    """
    guide to the dictionaries:

    _nts_hh: from NTS households table
    _nts_ind: from NTS individuals table
    _spc: from SPC

    """
    logger.info("Categorical matching: Data preparation")

    logger.info("Categorical matching: Creating dictionaries")

    # ---------- NTS

    # Create a dictionary for the HHIncome2002_B02ID column
    income_dict_nts_hh = {
        "1": "0-25k",
        "2": "25k-50k",
        "3": "50k+",
        "-8": "NA",
        # should be -10, but
        # it could be a typo in household_eul_2002-2022_ukda_data_dictionary
        "-1": "DEAD",
    }

    # Create a dictionary for the HHoldEmploy_B01ID column
    # (PT: Part time, FT: Full time)
    employment_dict_nts_hh = {
        "1": "None",
        "2": "0 FT, 1 PT",
        "3": "1 FT, 0 PT",
        "4": "0 FT, 2 PT",
        "5": "1 FT, 1 PT",
        "6": "2 FT, 0 PT",
        "7": "1 FT, 2+ PT",
        "8": "2 FT, 1+ PT",
        "9": "0 FT, 3+ PT",
        "10": "3+ FT, 0 PT",
        "11": "3+ FT, 1+ PT",
        "-8": "NA",
        "-10": "DEAD",
    }

    # Create a dictionary for the Ten1_B02ID column
    tenure_dict_nts_hh = {
        "1": "Owns / buying",
        "2": "Rents",
        "3": "Other (including rent free)",
        "-8": "NA",
        "-9": "DNA",
        "-10": "DEAD",
    }

    # ---------- SPC

    # create a dictionary for the pwkstat column
    employment_dict_spc = {
        "0": "Not applicable (age < 16)",
        "1": "Employee FT",
        "2": "Employee PT",
        "3": "Employee unspecified",
        "4": "Self-employed",
        "5": "Unemployed",
        "6": "Retired",
        "7": "Homemaker/Maternal leave",
        "8": "Student",
        "9": "Long term sickness/disability",
        "10": "Other",
    }

    # Create a dictionary for the tenure column
    tenure_dict_spc = {
        "1": "Owned: Owned outright",
        "2": "Owned: Owned with a mortgage or loan or shared ownership",
        "3": "Rented or living rent free: Total",
        "4": "Rented: Social rented",
        "5": "Rented: Private rented or living rent free",
        "-8": "NA",
        "-9": "DNA",
        "-10": "DEAD",
    }

    # Combine the dictionaries into a dictionary of dictionaries

    dict_nts = {
        "HHIncome2002_B02ID": income_dict_nts_hh,
        "HHoldEmploy_B01ID": employment_dict_nts_hh,
        "Ten1_B02ID": tenure_dict_nts_hh,
    }

    dict_spc = {"pwkstat": employment_dict_spc, "tenure": tenure_dict_spc}

    # ## Step 2: Decide on matching variables
    #
    # We need to identify the socio-demographic characteristics that we will match on. The
    # schema for the synthetic population can be found [here](https://github.com/alan-turing-institute/uatk-spc/blob/main/synthpop.proto).
    #
    # Matching between the SPC and the NTS will happen in two steps:
    #
    # 1. Match at the household level
    # 2. Match individuals within the household
    #
    # ### Household level matching
    #
    # | Variable           | Name (NTS)           | Name (SPC)      | Transformation (NTS) | Transformation (SPC) |
    # | ------------------ | -------------------- | --------------- | -------------------- | -------------------- |
    # | Household income   | `HHIncome2002_BO2ID` | `salary_yearly` | NA                   | Group by household ID and sum |
    # | Number of adults   | `HHoldNumAdults`        | `age_years`     | NA                   | Group by household ID and count |
    # | Number of children | `HHoldNumChildren`      | `age_years`     | NA                   | Group by household ID and count |
    # | Employment status  | `HHoldEmploy_B01ID`  | `pwkstat`       | NA                   | a) match to NTS categories. b) group by household ID |
    # | Car ownership      | `NumCar`             | `num_cars`      | SPC is capped at 2. We change all entries > 2 to 2 | NA  |
    #
    # Other columns to match in the future
    # | Variable           | Name (NTS)           | Name (SPC)      | Transformation (NTS) | Transformation (SPC) |
    # | ------------------ | -------------------- | --------------- | -------------------- | -------------------- |
    # | Type of tenancy    | `Ten1_B02ID`         | `tenure`        | ?? | ?? |
    # |  Urban-Rural classification of residence | `Settlement2011EW_B04ID`         | NA     | NA            | Spatial join between [layer](https://www.gov.uk/government/collections/rural-urban-classification) and SPC  |
    #
    #

    # ### 2.1 Edit SPC columns

    # #### Household Income
    logger.info("Categorical matching: Editing SPC columns (HH income)")
    #
    # Edit the spc so that we have household income as well as individual income.

    # add household income column for SPC
    spc_edited = transform_by_group(
        data=spc,
        group_col="household",
        transform_col="salary_yearly",
        new_col="salary_yearly_hh",
        transformation_type="sum",
    )

    # --- Recode column so that it matches the reported NTS values (Use income_dict_nts_hh
    # dictionary for reference)

    # Define the bins (first )
    bins = [0, 24999, 49999, np.inf]
    # Define the labels for the bins
    labels = [1, 2, 3]

    spc_edited = spc_edited.copy()

    spc_edited["salary_yearly_hh_cat"] = (
        pd.cut(
            spc_edited["salary_yearly_hh"],
            bins=bins,
            labels=labels,
            include_lowest=True,
        )
        .astype("str")
        .astype("float")
    )

    # replace NA values with -8 (to be consistent with NTS)
    spc_edited["salary_yearly_hh_cat"] = spc_edited["salary_yearly_hh_cat"].fillna(-8)

    # Convert the column to int
    spc_edited["salary_yearly_hh_cat"] = spc_edited["salary_yearly_hh_cat"].astype(
        "int"
    )

    # #### Household Composition (No. of Adults / Children)
    logger.info(
        "Categorical matching: Editing SPC columns (number of adults / children)"
    )

    # Number of adults and children in the household

    spc_edited = num_adult_child_hh(
        data=spc_edited, group_col="household", age_col="age_years"
    )

    # #### Employment Status
    logger.info("Categorical matching: Editing SPC columns (employment status)")

    # Employment status

    # check the colums values from our dictionary
    dict_spc["pwkstat"], dict_nts["HHoldEmploy_B01ID"]

    # The NTS only reports the number of Full time and Part time employees for each
    # household. For the SPC we also need to get the number of full time and part-time
    # workers for each household.
    #
    # Step 1: Create a column for Full time and a column for Part time

    # We will only use '1' and '2' for the employment status

    counts_df = count_per_group(
        df=spc_edited,
        group_col="household",
        count_col="pwkstat",
        values=[1, 2],
        value_names=["pwkstat_FT_hh", "pwkstat_PT_hh"],
    )

    counts_df.head(10)

    # Create a column that matches the NTS categories (m FT, n PT)

    # We want to match the SPC values to the NTS
    dict_nts["HHoldEmploy_B01ID"]
    """
    {
        '1': 'None',
        '2': '0 FT, 1 PT',
        '3': '1 FT, 0 PT',
        '4': '0 FT, 2 PT',
        '5': '1 FT, 1 PT',
        '6': '2 FT, 0 PT',
        '7': '1 FT, 2+ PT',
        '8': '2 FT, 1+ PT',
        '9': '0 FT, 3+ PT',
        '10': '3+ FT, 0 PT',
        '11': '3+ FT, 1+ PT',
        '-8': 'NA',
        '-10': 'DEAD'}
    """

    # 1) Match each row to the NTS

    # Define the conditions and outputs.
    # We are using the keys in dict_nts['HHoldEmploy_B01ID'] as reference
    conditions = [
        (counts_df["pwkstat_FT_hh"] == 0) & (counts_df["pwkstat_PT_hh"] == 0),
        (counts_df["pwkstat_FT_hh"] == 0) & (counts_df["pwkstat_PT_hh"] == 1),
        (counts_df["pwkstat_FT_hh"] == 1) & (counts_df["pwkstat_PT_hh"] == 0),
        (counts_df["pwkstat_FT_hh"] == 0) & (counts_df["pwkstat_PT_hh"] == 2),
        (counts_df["pwkstat_FT_hh"] == 1) & (counts_df["pwkstat_PT_hh"] == 1),
        (counts_df["pwkstat_FT_hh"] == 2) & (counts_df["pwkstat_PT_hh"] == 0),
        (counts_df["pwkstat_FT_hh"] == 1) & (counts_df["pwkstat_PT_hh"] >= 2),
        (counts_df["pwkstat_FT_hh"] == 2) & (counts_df["pwkstat_PT_hh"] >= 1),
        (counts_df["pwkstat_FT_hh"] == 0) & (counts_df["pwkstat_PT_hh"] >= 3),
        (counts_df["pwkstat_FT_hh"] >= 3) & (counts_df["pwkstat_PT_hh"] == 0),
        (counts_df["pwkstat_FT_hh"] >= 3) & (counts_df["pwkstat_PT_hh"] >= 1),
    ]

    # Define the corresponding outputs based on dict_nts['HHoldEmploy_B01ID]
    outputs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

    # Create a new column using np.select
    counts_df["pwkstat_NTS_match"] = np.select(conditions, outputs, default=-8)

    # 2) merge back onto the spc
    spc_edited = spc_edited.merge(counts_df, left_on="household", right_index=True)

    # check the output
    spc_edited[
        ["household", "pwkstat", "pwkstat_FT_hh", "pwkstat_PT_hh", "pwkstat_NTS_match"]
    ].head(10)

    # #### Urban Rural Classification
    logger.info(
        "Categorical matching: Editing SPC columns (urban / rural classification)"
    )

    # We use the 2011 rural urban classification to match the SPC to the NTS. The NTS has 2 columns that we can use to match to the SPC: `Settlement2011EW_B03ID` and `Settlement2011EW_B04ID`. The `Settlement2011EW_B03ID` column is more general (urban / rural only), while the `Settlement2011EW_B04ID` column is more specific. We stick to the more general column for now.

    # read the rural urban classification data
    rural_urban = pd.read_csv(config.rural_urban_filepath)

    # merge the rural_urban data with the spc
    spc_edited = spc_edited.merge(
        rural_urban[["OA11CD", "RUC11", "RUC11CD"]], left_on="oa11cd", right_on="OA11CD"
    )

    # create dictionary from the NTS `Settlement2011EW_B03ID` column
    Settlement2011EW_B03ID_nts_hh = {
        "1": "Urban",
        "2": "Rural",
        "3": "Scotland",
        "-8": "NA",
        "-10": "DEAD",
    }

    Settlement2011EW_B04ID_nts_hh = {
        "1": "Urban Conurbation",
        "2": "Urban City and Town",
        "3": "Rural Town and Fringe",
        "4": "Rural Village, Hamlet and Isolated Dwellings",
        "5": "Scotland",
        "-8": "NA",
        "-10": "DEAD",
    }

    census_2011_to_nts_B03ID = {
        "Urban major conurbation": "Urban",
        "Urban minor conurbation": "Urban",
        "Urban city and town": "Urban",
        "Urban city and town in a sparse setting": "Urban",
        "Rural town and fringe": "Rural",
        "Rural town and fringe in a sparse setting": "Rural",
        "Rural village": "Rural",
        "Rural village in a sparse setting": "Rural",
        "Rural hamlets and isolated dwellings": "Rural",
        "Rural hamlets and isolated dwellings in a sparse setting": "Rural",
    }

    census_2011_to_nts_B04ID = {
        "Urban major conurbation": "Urban Conurbation",
        "Urban minor conurbation": "Urban Conurbation",
        "Urban city and town": "Urban City and Town",
        "Urban city and town in a sparse setting": "Urban City and Town",
        "Rural town and fringe": "Rural Town and Fringe",
        "Rural town and fringe in a sparse setting": "Rural Town and Fringe",
        "Rural village": "Rural Village, Hamlet and Isolated Dwellings",
        "Rural village in a sparse setting": "Rural Village, Hamlet and Isolated Dwellings",
        "Rural hamlets and isolated dwellings": "Rural Village, Hamlet and Isolated Dwellings",
        "Rural hamlets and isolated dwellings in a sparse setting": "Rural Village, Hamlet and Isolated Dwellings",
    }

    # add the nts Settlement2011EW_B03ID and Settlement2011EW_B04ID columns to the spc
    spc_edited["Settlement2011EW_B03ID_spc"] = spc_edited["RUC11"].map(
        census_2011_to_nts_B03ID
    )
    spc_edited["Settlement2011EW_B04ID_spc"] = spc_edited["RUC11"].map(
        census_2011_to_nts_B04ID
    )
    spc_edited.head()

    # add the keys from nts_Settlement2011EW_B03ID and nts_Settlement2011EW_B04ID to the spc based on above mappings

    # reverse the dictionaries
    Settlement2011EW_B03ID_nts_rev = {
        v: k for k, v in Settlement2011EW_B03ID_nts_hh.items()
    }
    # map the values
    spc_edited["Settlement2011EW_B03ID_spc_CD"] = (
        spc_edited["Settlement2011EW_B03ID_spc"]
        .map(Settlement2011EW_B03ID_nts_rev)
        .astype("int")
    )

    Settlement2011EW_B04ID_nts_rev = {
        v: k for k, v in Settlement2011EW_B04ID_nts_hh.items()
    }
    spc_edited["Settlement2011EW_B04ID_spc_CD"] = (
        spc_edited["Settlement2011EW_B04ID_spc"]
        .map(Settlement2011EW_B04ID_nts_rev)
        .astype("int")
    )
    spc_edited.head()

    # ### 2.2 Edit NTS columns
    logger.info("Categorical matching: Editing NTS columns (number of pensioners")

    # #### Number of people of pension age

    nts_pensioners = count_per_group(
        df=nts_individuals,
        group_col="HouseholdID",
        count_col="OfPenAge_B01ID",
        values=[1],
        value_names=["num_pension_age_nts"],
    )

    nts_pensioners.head()

    # join onto the nts household df
    nts_households = nts_households.merge(
        nts_pensioners, left_on="HouseholdID", right_index=True, how="left"
    )

    # #### Number of cars
    logger.info("Categorical matching: Editing NTS columns (number of cars")

    # - `SPC.num_cars` only has values [0, 1, 2]. 2 is for all households with 2 or more cars
    # - `NTS.NumCar` is more detailed. It has the actual value of the number of cars. We will cap this at 2.

    # Create a new column in NTS
    nts_households.loc[:, "NumCar_SPC_match"] = nts_households["NumCar"].apply(
        truncate_values, upper=2
    )

    # #### Type of tenancy
    logger.info("Categorical matching: Editing NTS columns (tenure status)")
    # Create dictionaries to map tenure onto the spc and nts dfs

    # Dictionary showing how we want the final columns to look like
    _tenure_dict_nts_spc = {
        1: "Owned",
        2: "Rented or rent free",
        -8: "NA",
        -9: "DNA",
        -10: "DEAD",
    }

    # Matching NTS to tenure_dict_nts_spc

    # Create a new dictionary for matching
    matching_dict_nts_tenure = {1: 1, 2: 2, 3: 2}

    matching_dict_spc_tenure = {
        1: 1,  #'Owned: Owned outright' : 'Owned'
        2: 1,  #'Owned: Owned with a mortgage or loan or shared ownership', : 'Owned'
        3: 2,  #'Rented or living rent free: Total', : 'Rented or rent free'
        4: 2,  #'Rented: Social rented', : 'Rented or rent free'
        5: 2,  #'Rented: Private rented or living rent free', : 'Rented or rent free'
    }

    # map dictionaries to create comparable columns

    # Create a new column in nts_households
    nts_households["tenure_nts_for_matching"] = (
        nts_households["Ten1_B02ID"]
        .map(matching_dict_nts_tenure)  # map the values to the new dictionary
        .fillna(nts_households["Ten1_B02ID"])
    )  # fill the NaNs with the original values

    # Create a new column in spc
    spc_edited["tenure_spc_for_matching"] = (
        spc_edited["tenure"]
        .map(matching_dict_spc_tenure)  # map the values to the new dictionary
        .fillna(spc_edited["tenure"])
    )  # fill the NaNs with the original values

    # ## Step 3: Matching at Household Level
    # TODO: remove once refactored into two scripts
    load_households = False
    if not load_households:
        logger.info("Categorical matching: MATCHING HOUSEHOLDS")

        #
        # Now that we've prepared all the columns, we can start matching.

        # ### 3.1 Categorical matching
        #
        # We will match on (a subset of) the following columns:
        #
        # | Matching variable | NTS column | SPC column |
        # | ------------------| ---------- | ---------- |
        # | Household income  | `HHIncome2002_BO2ID` | `salary_yearly_hh_cat` |
        # | Number of adults  | `HHoldNumAdults` | `num_adults` |
        # | Number of children | `HHoldNumChildren` | `num_children` |
        # | Employment status | `HHoldEmploy_B01ID` | `pwkstat_NTS_match` |
        # | Car ownership | `NumCar_SPC_match` | `num_cars` |
        # | Type of tenancy | `tenure_nts_for_matching` | `tenure_spc_for_matching` |
        # | Rural/Urban Classification | `Settlement2011EW_B03ID` | `Settlement2011EW_B03ID_spc_CD` |

        # Prepare SPC df for matching

        # Select multiple columns
        spc_matching = spc_edited[
            [
                "hid",
                "salary_yearly_hh_cat",
                "num_adults",
                "num_children",
                "num_pension_age",
                "pwkstat_NTS_match",
                "num_cars",
                "tenure_spc_for_matching",
                "Settlement2011EW_B03ID_spc_CD",
                "Settlement2011EW_B04ID_spc_CD",
            ]
        ]

        # edit the df so that we have one row per hid
        spc_matching = spc_matching.drop_duplicates(subset="hid")

        spc_matching.head(10)

        # Prepare NTS df for matching

        nts_matching = nts_households[
            [
                "HouseholdID",
                "HHIncome2002_B02ID",
                "HHoldNumAdults",
                "HHoldNumChildren",
                "num_pension_age_nts",
                "HHoldEmploy_B01ID",
                "NumCar_SPC_match",
                "tenure_nts_for_matching",
                "Settlement2011EW_B03ID",
                "Settlement2011EW_B04ID",
            ]
        ]

        # Dictionary of matching columns. We extract column names from this dictioary when matching on a subset of the columns

        # column_names (keys) for the dictionary
        matching_ids = [
            "household_id",
            "yearly_income",
            "number_adults",
            "number_children",
            "num_pension_age",
            "employment_status",
            "number_cars",
            "tenure_status",
            "rural_urban_2_categories",
            "rural_urban_4_categories",
        ]

        # Dict with value qual to a list with spc_matching and nts_matching column names
        matching_dfs_dict = {
            column_name: [spc_value, nts_value]
            for column_name, spc_value, nts_value in zip(
                matching_ids, spc_matching, nts_matching
            )
        }

        # We match iteratively on a subset of columns. We start with all columns, and then remove
        # one of the optionals columns at a time (relaxing the condition). Once a household has over n
        # matches, we stop matching it to more matches. We continue until all optional columns are removed
        matcher_exact = MatcherExact(
            df_pop=spc_matching,
            df_pop_id="hid",
            df_sample=nts_matching,
            df_sample_id="HouseholdID",
            matching_dict=matching_dfs_dict,
            fixed_cols=list(config.matching.required_columns),
            optional_cols=list(config.matching.optional_columns),
            n_matches=config.matching.n_matches,
            chunk_size=config.matching.chunk_size,
            show_progress=True,
        )

        # Match

        matches_hh_level = matcher_exact.iterative_match_categorical()

        # Number of unmatched households

        # no. of keys where value is na
        na_count = sum([1 for v in matches_hh_level.values() if pd.isna(v).all()])

        logger.info(
            f"Categorical matching: {na_count} households in the SPC had no match"
        )
        logger.info(
            f"{round((na_count / len(matches_hh_level)) * 100, 1)}% of households in the SPC had no match"
        )

        # ### Random Sampling from matched households

        logger.info("Categorical matching: Randomly choosing one match per household")
        #
        # In categorical matching, many households in the SPC are matched to more than 1 household in the NTS. Which household to choose? We do random sampling

        # for each key in the dictionary, sample 1 of the values associated with it and store it in a new dictionary

        """
        - iterate over each key-value pair in the matches_hh_result dictionary.
        - For each key-value pair, use np.random.choice(value) to randomly select
        one item from the list of values associated with the current key.
        - create a new dictionary hid_to_HouseholdID_sample where each key from the
        original dictionary is associated with one randomly selected value from the
        original list of values.

        """
        # Randomly sample one match per household if it has one match or more
        matches_hh_level_sample = {
            key: np.random.choice(value)
            for key, value in matches_hh_level.items()
            if value
            and not pd.isna(
                np.random.choice(value)
            )  # Ensure the value list is not empty and the selected match is not NaN
        }

        # Multiple matches in case we want to try stochastic runs

        # Same logic as above, but repeat it multiple times and store each result as a separate dictionary in a list
        matches_hh_level_sample_list = [
            {
                key: np.random.choice(value)
                for key, value in matches_hh_level.items()
                if value and not pd.isna(np.random.choice(value))
            }
            for i in range(25)  # Repeat the process 25 times
        ]

        logger.info("Categorical matching: Random sampling complete")

        # Save results
        logger.info("Categorical matching: Saving results")

        # matching results
        with open(get_interim_path("matches_hh_level_categorical.pkl"), "wb") as f:
            pkl.dump(matches_hh_level, f)

        # random sample
        with open(
            get_interim_path("matches_hh_level_categorical_random_sample.pkl"), "wb"
        ) as f:
            pkl.dump(matches_hh_level_sample, f)

        # multiple random samples
        with open(
            get_interim_path("matches_hh_level_categorical_random_sample_multiple.pkl"),
            "wb",
        ) as f:
            pkl.dump(matches_hh_level_sample_list, f)
    else:
        logger.info("Categorical matching: loading matched households")
        # Load matching result
        with open(
            get_interim_path("matches_hh_level_categorical_random_sample.pkl"), "rb"
        ) as f:
            matches_hh_level_sample = pkl.load(f)

        # multiple random samples
        with open(
            get_interim_path("matches_hh_level_categorical_random_sample_multiple.pkl"),
            "rb",
        ) as f:
            matches_hh_level_sample_list = pkl.load(f)

    ## add matches_hh_level_sample as a column in spc_edited
    spc_edited["nts_hh_id"] = spc_edited["hid"].map(matches_hh_level_sample)

    # Do the same at the df level. Add nts_hh_id_sample column to the spc df

    # # for each hid in spc_edited, sample a value from the nts_hh_id col.
    # spc_edited['nts_hh_id_sample'] = spc_edited['nts_hh_id'].apply(lambda x: np.random.choice(x) if x is not np.nan else np.nan)
    # # All rows with the same 'hid' should have the same value for 'nts_hh_id_sample'. Group by hid and assign the first value to all rows in the group
    # spc_edited['nts_hh_id_sample'] = spc_edited.groupby('hid')['nts_hh_id_sample'].transform('first')

    # spc_edited.head(10)

    # ## Step 4: Matching at Individual Level
    #
    # 1) Prepare columns for matching - they should all be numerical
    #     a) age_years in the SPC -> Convert from actual age to age brackets from the dictionary
    # 2) Filter to specific household
    # 3) Nearest neighbor merge without replacement (edit while function below)
    #
    #

    # Create an 'age' column in the SPC that matches the NTS categories

    # create a dictionary for reference on how the labels for "Age_B04ID" match the actual age brackets

    # dict_nts_ind_age = {-10: 'DEAD',
    #                     -8: 'NA',
    #                     1: '0-4',
    #                     2: '5-10',
    #                     3: '11-16',
    #                     4: '17-20',
    #                     5: '21-29',
    #                     6: '30-39',
    #                     7: '40-49',
    #                     8: '50-59',
    #                     9: '60+'
    #                     }

    # Define the bins and labels based on dict_nts_ind_age
    bins = [0, 4, 10, 16, 20, 29, 39, 49, 59, np.inf]
    labels = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    # Create a new column in spc_edited that maps the age_years to the keys of dict_nts_ind_age
    spc_edited["age_group"] = (
        pd.cut(spc_edited["age_years"], bins=bins, labels=labels)
        .astype("int")
        .fillna(-8)
    )

    # rename nts columns in preparation for matching

    nts_individuals.rename(
        columns={"Age_B04ID": "age_group", "Sex_B01ID": "sex"}, inplace=True
    )

    # TODO: remove once refactored into two scripts
    load_individuals = False
    if not load_individuals:
        logger.info("Statistical matching: MATCHING INDIVIDUALS")

        # PSM matching using internal match_individuals function
        matches_ind = match_individuals(
            df1=spc_edited,
            df2=nts_individuals,
            matching_columns=["age_group", "sex"],
            df1_id="hid",
            df2_id="HouseholdID",
            matches_hh=matches_hh_level_sample,
            show_progress=True,
        )

        # save random sample
        with open(
            get_interim_path("matches_ind_level_categorical_random_sample.pkl"), "wb"
        ) as f:
            pkl.dump(matches_ind, f)
    else:
        logger.info("Statistical matching: loading matched individuals")
        with open(
            get_interim_path("matches_ind_level_categorical_random_sample.pkl"), "rb"
        ) as f:
            matches_ind = pkl.load(f)

    # Add matches_ind values to spc_edited using map
    spc_edited["nts_ind_id"] = spc_edited.index.map(matches_ind)

    # add the nts_individuals.IndividualID to spc_edit. The current nts_ind_id is the row index of nts_individuals
    spc_edited["nts_ind_id"] = spc_edited["nts_ind_id"].map(
        nts_individuals["IndividualID"]
    )

    logger.info("Statistical matching: Matching complete")

    # ### Match on multiple samples

    # logger.info("Statistical matching: Matching on multiple samples")

    # # In household level matching, some households in the SPC are matched to multiple households in the NTS. To have 1:1 match between the SPC and NTS, we randomly sample from the list of matches
    # #
    # # The random sample produces different results each time. In `matches_hh_level_sample_list` we did many iterations of random sampling to produce multiple results of household matching, and saved the output in a list of dictionaries.
    # #
    # # Here, we iterate over the list and do individual matching for each item. The output is a list of n dictionaries, each of which could be used as a synthetic population matched to the NTS

    # # iterate over all items in the matches_hh_level_sample_list and apply the match_individuals function to each
    # parallel = Parallel(n_jobs=-1, return_as="generator")
    # matches_list_of_dict = list(
    #     parallel(
    #         delayed(match_individuals)(
    #             df1=spc_edited,
    #             df2=nts_individuals,
    #             matching_columns=["age_group", "sex"],
    #             df1_id="hid",
    #             df2_id="HouseholdID",
    #             matches_hh=matches_hh_level_sample_list[i],
    #             show_progress=False,
    #         )
    #         for i in trange(len(matches_hh_level_sample_list))
    #     )
    # )

    # # Save the results of individual matching
    # logger.info("Statistical matching: Saving results")

    # # save multiple random samples
    # with open(
    #     get_interim_path("matches_ind_level_categorical_random_sample_multiple.pkl"), "wb"
    # ) as f:
    #     pkl.dump(matches_list_of_dict, f)

    # ### Add trip data
    logger.info("Post-processing: Editing column names")

    # Rename columns and map actual modes and trip purposes to the trip table.
    #
    # Code taken from: https://github.com/arup-group/pam/blob/main/examples/07_travel_survey_to_matsim.ipynb

    nts_trips = nts_trips.rename(
        columns={  # rename data
            "JourSeq": "seq",
            "TripOrigGOR_B02ID": "ozone",
            "TripDestGOR_B02ID": "dzone",
            "TripPurpFrom_B01ID": "oact",
            "TripPurpTo_B01ID": "dact",
            "MainMode_B04ID": "mode",
            "TripStart": "tst",
            "TripEnd": "tet",
        }
    )

    logger.info("Post-processing: Mapping modes and trip purposes")

    mode_mapping = {
        1: "walk",
        2: "bike",
        3: "car",  #'Car/van driver'
        4: "car_passenger",  #'Car/van passenger'
        5: "motorcycle",  #'Motorcycle',
        6: "car",  #'Other private transport',
        7: "pt",  # Bus in London',
        8: "pt",  #'Other local bus',
        9: "pt",  #'Non-local bus',
        10: "pt",  #'London Underground',
        11: "pt",  #'Surface Rail',
        12: "taxi",  #'Taxi/minicab',
        13: "pt",  #'Other public transport',
        -10: "DEAD",
        -8: "NA",
    }

    purp_mapping = {
        1: "work",
        2: "work",  #'In course of work',
        3: "education",
        4: "shop",  #'Food shopping',
        5: "shop",  #'Non food shopping',
        6: "medical",  #'Personal business medical',
        7: "other",  #'Personal business eat/drink',
        8: "other",  #'Personal business other',
        9: "other",  #'Eat/drink with friends',
        10: "visit",  #'Visit friends',
        11: "other",  #'Other social',
        12: "other",  #'Entertain/ public activity',
        13: "other",  #'Sport: participate',
        14: "home",  #'Holiday: base',
        15: "other",  #'Day trip/just walk',
        16: "other",  #'Other non-escort',
        17: "escort",  #'Escort home',
        18: "escort",  #'Escort work',
        19: "escort",  #'Escort in course of work',
        20: "escort",  #'Escort education',
        21: "escort",  #'Escort shopping/personal business',
        22: "escort",  #'Other escort',
        23: "home",  #'Home',
        -10: "DEAD",
        -8: "NA",
    }

    nts_trips["mode"] = nts_trips["mode"].map(mode_mapping)

    nts_trips["oact"] = nts_trips["oact"].map(purp_mapping)

    nts_trips["dact"] = nts_trips["dact"].map(purp_mapping)

    # # For education trips, we use age as an indicator for the type of education facility the individual is most likely to go to. The `age_group_mapping` dictionary maps age groups to education facility types. For each person activity, we use the age_group to determine which education facilities to look at.
    logger.info("Post-processing: Assigning education activities to education types")

    # map the age_group to an education type (age group is from NTS::Age_B04ID)
    # TODO edit osmox config to replace education_college with education_university.
    # We should have mutually exclusive groups only and these two options serve the
    # same age group
    age_group_mapping = {
        1: "education_kg",  # "0-4"
        2: "education_school",  # "5-10"
        3: "education_school",  # "11-16"
        4: "education_university",  # "17-20"
        5: "education_university",  # "21-29"
        6: "education_university",  # "30-39"
        7: "education_university",  # "40-49"
        8: "education_university",  # "50-59"
        9: "education_university",  # "60+"
    }

    # map the age_group_mapping dict to an education type
    # (age group is from NTS::Age_B04ID)
    # TODO: move this further upstream in a preprocessing step so that I don't need
    # to save the df again
    spc_edited["education_type"] = spc_edited["age_group"].map(age_group_mapping)

    # create an independant copy of spc_edited
    spc_edited_copy = spc_edited.copy()

    # replace non-finite values with a default value
    spc_edited_copy["nts_ind_id"] = spc_edited_copy["nts_ind_id"].fillna(-1)
    # convert the nts_ind_id column to int for merging
    spc_edited_copy["nts_ind_id"] = spc_edited_copy["nts_ind_id"].astype(int)

    # Add output columns required for assignment scripts
    spc_output_cols = [
        col for col in spc_edited_copy.columns if col in cols_for_assignment_all()
    ]
    nts_output_cols = [
        col for col in nts_trips.columns if col in cols_for_assignment_all()
    ] + ["IndividualID"]

    # merge the copy with nts_trips using IndividualID
    spc_edited_copy = spc_edited_copy[spc_output_cols].merge(
        nts_trips[nts_output_cols],
        left_on="nts_ind_id",
        right_on="IndividualID",
        how="left",
    )

    # save the file as a parquet file
    spc_edited_copy.to_parquet(config.spc_with_nts_trips_filepath)

    # save the nts data for later use in validation
    nts_individuals.to_parquet(config.output_path / "nts_individuals.parquet")
    nts_households.to_parquet(config.output_path / "nts_households.parquet")
    nts_trips.to_parquet(config.output_path / "nts_trips.parquet")


if __name__ == "__main__":
    main()
