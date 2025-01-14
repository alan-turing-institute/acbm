from datetime import timedelta

import geopandas as gpd
import pandas as pd
from pam import write
from pam.read import load_travel_diary
from pam.samplers.time import apply_jitter_to_plan
from shapely import Point, wkt

from acbm.cli import acbm_cli
from acbm.config import load_and_setup_config
from acbm.postprocessing.matsim import (
    Population,
    add_home_location_to_individuals,
    calculate_percentage_remaining,
    get_hhlIncome,
    get_passengers,
    get_pt_subscription,
    get_students,
    log_row_count,
)


@acbm_cli
def main(config_file):
    config = load_and_setup_config(config_file)
    logger = config.get_logger("converting_to_matsim", __file__)

    # ----- Read the data

    logger.info("1 - Loading data")

    population = Population.read(config)

    # ----- Clean the data

    logger.info("2 - Cleaning data")

    # rename age_years to age in individuals
    population.individuals.rename(columns={"age_years": "age"}, inplace=True)

    # ----- Add some person attributes to the individuals dataframe

    # sex

    # get sex column from spc
    # TODO: add sex column upstream in the beginning of the pipeline
    spc = pd.read_parquet(
        config.spc_combined_filepath,
        columns=["id", "household", "age_years", "sex", "salary_yearly"],
    )

    # change spc["sex"] column: 1 = male, 2 = female
    spc["sex"] = spc["sex"].map({1: "male", 2: "female"})
    # merge it on
    individuals = population.individuals.merge(
        spc[["id", "sex"]], left_on="pid", right_on="id", how="left"
    )
    individuals = individuals.drop(columns="id")

    # isStudent
    individuals = get_students(
        individuals=individuals,
        activities=population.activities,
        age_base_threshold=config.postprocessing.student_age_base,
        # age_upper_threshold = config.postprocessing.student_age_upper,
        activity="education",
    )

    # isPassenger
    individuals = get_passengers(
        legs=population.legs,
        individuals=individuals,
        modes=config.postprocessing.modes_passenger,
    )

    # hasPTsubscription

    individuals = get_pt_subscription(
        individuals=individuals, age_threshold=config.postprocessing.pt_subscription_age
    )

    ## hhlIncome
    individuals = get_hhlIncome(
        individuals=individuals,
        individuals_with_salary=spc,
        pension_age=config.postprocessing.pt_subscription_age,
        pension=config.postprocessing.state_pension,
    )

    # ----- Add vehicle ownership attributes (car, bicycle) to the individuals dataframe
    # TODO: move this upstream

    # a. spc to nts match (used to get nts_id: spc_id match)
    spc_with_nts = pd.read_parquet(config.spc_with_nts_trips_filepath)
    nts_individuals = pd.read_parquet(config.output_path / "nts_individuals.parquet")

    # b. Create a df with the vehicle ownership data (from the nts)

    nts_individuals = nts_individuals[
        ["IndividualID", "OwnCycleN_B01ID", "DrivLic_B02ID", "CarAccess_B01ID"]
    ]

    # Create CarAvailability colum

    car_availability_mapping = {
        1: "yes",  # Main driver of company car
        2: "yes",  # Other main driver
        3: "some",  # Not main driver of household car
    }

    nts_individuals["CarAvailability"] = (
        nts_individuals["CarAccess_B01ID"].map(car_availability_mapping).fillna("no")
    )
    # Create BicycleAvailability column

    bicycle_availability_mapping = {
        1: "yes",  # Own a pedal cycle yourself
        2: "some",  # Have use of household pedal cycle
        3: "no",  # Have use of non-household pedal cycle
    }

    nts_individuals["BicycleAvailability"] = (
        nts_individuals["OwnCycleN_B01ID"]
        .map(bicycle_availability_mapping)
        .fillna("no")
    )

    # Create hasLicence column
    # 1: Full licence, 2: Provisional licence, 3: Other or none
    nts_individuals["hasLicence"] = nts_individuals["DrivLic_B02ID"].apply(
        lambda x: x == 1
    )

    # Keep only the columns we created
    nts_individuals = nts_individuals[
        ["IndividualID", "CarAvailability", "BicycleAvailability", "hasLicence"]
    ]
    nts_individuals.head(10)

    # c. add spc id to nts_individuals

    # create a df with spc_id and nts_id
    spcid_to_ntsid = spc_with_nts[
        ["id", "nts_ind_id"]
    ].drop_duplicates()  # spc_with_nts has one row per travel day

    # add the spc_id column
    nts_individuals = nts_individuals.merge(
        spcid_to_ntsid, left_on="IndividualID", right_on="nts_ind_id", how="inner"
    ).drop(columns=["nts_ind_id"])

    nts_individuals.rename(columns={"id": "spc_id"}, inplace=True)

    # d. merge nts_individuals with individuals to get the vehicle ownership data
    individuals = individuals.merge(
        nts_individuals, left_on="pid", right_on="spc_id", how="left"
    ).drop(columns=["spc_id", "IndividualID"])

    # We will be removing some rows in each planning operation. This function helps keep a
    # record of the number of rows in each table after each operation.

    row_counts = []

    logger.info("2.1 - Record number of rows in each df before cleaning")

    log_row_count(population.individuals, "individuals", "0_initial", row_counts)
    log_row_count(population.households, "households", "0_initial", row_counts)
    log_row_count(population.activities, "activities", "0_initial", row_counts)
    log_row_count(population.legs, "legs", "0_initial", row_counts)
    log_row_count(population.legs_geo, "legs_geo", "0_initial", row_counts)

    logger.info("2.2 - Remove people that don't exist across all dfs")

    # When writing to matsim using pam, we get an error when a pid exists in one dataset
    #  but not in the other. We will remove these people from the datasets.

    population = population.filter_by_pid()

    log_row_count(population.individuals, "individuals", "1_filter_by_pid", row_counts)
    log_row_count(population.households, "households", "1_filter_by_pid", row_counts)
    log_row_count(population.activities, "activities", "1_filter_by_pid", row_counts)
    log_row_count(population.legs, "legs", "1_filter_by_pid", row_counts)
    log_row_count(population.legs_geo, "legs_geo", "1_filter_by_pid", row_counts)

    logger.info("2.3 - Rename geometry columns (for PAM)")

    # TODO: Rename columns upstream in 3.3_assign_facility_all script
    population.legs_geo.rename(
        columns={
            "start_location_geometry_wkt": "start_loc",
            "end_location_geometry_wkt": "end_loc",
        },
        inplace=True,
    )
    logger.info("2.4 - Remove people with missing location data ")

    population = population.filter_no_location()

    log_row_count(
        population.individuals, "individuals", "2_filter_no_location", row_counts
    )
    log_row_count(
        population.households, "households", "2_filter_no_location", row_counts
    )
    log_row_count(
        population.activities, "activities", "2_filter_no_location", row_counts
    )
    log_row_count(population.legs, "legs", "2_filter_no_location", row_counts)
    log_row_count(population.legs_geo, "legs_geo", "2_filter_no_location", row_counts)

    logger.info("2.5 - Log number of rows in each df after cleaning")

    percentages = calculate_percentage_remaining(row_counts=row_counts)

    # Log the percentages
    for stage, df_name, count, percentage in percentages:
        logger.info(
            f"{df_name} - {stage} - {count} rows: {percentage:.1f}% rows remaining"
        )

    logger.info("3a - Convert geometry columns to POINT geometry")

    # Function to convert to Point if not already a Point
    def convert_to_point(value):
        if isinstance(value, Point):
            return value
        return wkt.loads(value)

    # Convert start_loc and end_loc to shapely point objects
    population.legs_geo["start_loc"] = population.legs_geo["start_loc"].apply(
        convert_to_point
    )
    population.legs_geo["end_loc"] = population.legs_geo["end_loc"].apply(
        convert_to_point
    )

    # Convert to GeoDataFrame with start_loc as the active geometry
    legs_geo = gpd.GeoDataFrame(population.legs_geo, geometry="start_loc")

    logger.info("3b - Add home location to individuals")

    # Apply
    individuals_geo = add_home_location_to_individuals(legs_geo, population.individuals)

    logger.info("4 - Write to MATSim XML")

    logger.info("4.1 - Load travel diary to PAM")

    population = load_travel_diary(
        trips=legs_geo,
        persons_attributes=individuals_geo,
        tour_based=False,
        include_loc=True,
        sort_by_seq=True,
    )

    logger.info("4.2 - Jittering plans")
    # TODO: Move this upstream
    for _, _, person in population.people():
        apply_jitter_to_plan(
            person.plan,
            jitter=timedelta(minutes=config.postprocessing.pam_jitter),
            min_duration=timedelta(minutes=config.postprocessing.pam_min_duration),
        )
        # crop to 24-hours
        person.plan.crop()

    logger.info("4.3 - Write to MATSim XML")

    write.write_matsim_population_v6(
        population=population,
        path=config.output_path / "plans.xml",
        coordinate_reference_system=f"EPSG:{config.output_crs}",
    )


if __name__ == "__main__":
    main()
