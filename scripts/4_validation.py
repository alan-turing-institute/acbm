import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from acbm.cli import acbm_cli
from acbm.config import load_and_setup_config
from acbm.validating.plots import (
    plot_activity_sequence_comparison,
    plot_comparison,
    plot_intrazonal_trips,
)
from acbm.validating.utils import calculate_od_distances, process_sequences


@acbm_cli
def main(config_file):
    config = load_and_setup_config(config_file)
    logger = config.get_logger("validation", __file__)

    # ----- Folder for validation plots

    logger.info("1. Creating folder for validation plots")
    validation_plots_path = config.validation_plots_path

    # ----- Reading in the data

    logger.info("2. Reading in the data")

    # NTS data
    legs_nts = pd.read_parquet(config.output_path / "nts_trips.parquet")

    legs_nts = legs_nts[legs_nts["TravDay"].isin(config.parameters.nts_days_of_week)]

    # Model outputs
    legs_acbm = pd.read_csv(config.output_path / "legs.csv")
    legs_acbm_geo = pd.read_parquet(config.output_path / "legs_with_locations.parquet")

    # ----- Preproccessing the data

    logger.info("3a. Preprocessing: renaming columns")

    # rename origin activity and destination activity columns
    legs_acbm = legs_acbm.rename(
        columns={"origin activity": "oact", "destination activity": "dact"}
    )
    legs_acbm_geo = legs_acbm_geo.rename(
        columns={"origin activity": "oact", "destination activity": "dact"}
    )

    # rename distance column in NTS
    legs_nts = legs_nts.rename(columns={"TripDisIncSW": "distance"})

    logger.info("3b. Preprocessing: Edit distance column in NTS")

    # convert legs_nts["distance"] from miles to km
    legs_nts["distance"] = legs_nts["distance"] * 1.60934

    logger.info("3c. Preprocessing: Adding hour of day column")

    # acbm - tst is in datetime format
    # Convert tst to datetime format and extract the hour component in one step
    legs_acbm["tst_hour"] = legs_acbm["tst"].apply(lambda x: pd.to_datetime(x).hour)
    legs_acbm["tet_hour"] = legs_acbm["tet"].apply(lambda x: pd.to_datetime(x).hour)

    # nts - tst is in minutes
    # Convert legs_nts["tst"] from minutes to hours
    legs_nts["tst_hour"] = legs_nts["tst"] // 60
    legs_nts["tet_hour"] = legs_nts["tet"] // 60

    logger.info("3d. Preprocessing: Abbreviating column values for trip purpose")

    # Mapping dictionary
    activity_mapping = {
        "home": "h",
        "other": "o",
        "escort": "e",
        "work": "w",
        "shop": "sh",
        "visit": "v",
        "education": "edu",
        "medical": "m",
    }

    legs_acbm["oact_abr"] = legs_acbm["oact"].replace(activity_mapping)
    legs_acbm["dact_abr"] = legs_acbm["dact"].replace(activity_mapping)

    legs_nts["oact_abr"] = legs_nts["oact"].replace(activity_mapping)
    legs_nts["dact_abr"] = legs_nts["dact"].replace(activity_mapping)

    # ----- Validation Plots

    logger.info("4. Validation plots")

    logger.info("4a.1 Validation (Matching) - Trip Purpose")

    # Get number of trips by mode for legs_nts, and legs_acbm, and plot a comparative bar plot
    # NTS
    purpose_nts = legs_nts.groupby("dact").size().reset_index(name="count")
    purpose_nts["source"] = "nts"

    # ACBM
    purpose_acbm = legs_acbm.groupby("dact").size().reset_index(name="count")
    purpose_acbm["source"] = "acbm"

    # Combine the data
    purpose_compare = pd.concat([purpose_nts, purpose_acbm])

    # Calculate the percentage of trips for each mode within each source
    purpose_compare["percentage"] = purpose_compare.groupby("source")[
        "count"
    ].transform(lambda x: (x / x.sum()) * 100)

    plt.figure()

    sns.barplot(data=purpose_compare, x="dact", y="percentage", hue="source")
    plt.xlabel("Trip purpose")
    plt.ylabel("Percentage of total trips")
    plt.title("Percentage of Trips by Purpose for NTS and ACBM")
    # plt.show()

    # Save the plot
    plt.tight_layout()
    plt.savefig(validation_plots_path / "1_matching_trip_purpose.png")

    logger.info("4a.2 Validation (Matching) - Trip Mode")

    # Get number of trips by mode for legs_nts, and legs_acbm, and plot a comparative bar plot
    # NTS
    modeshare_nts = legs_nts.groupby("mode").size().reset_index(name="count")
    modeshare_nts["source"] = "nts"

    # ACBM
    modeshare_acbm = legs_acbm.groupby("mode").size().reset_index(name="count")
    modeshare_acbm["source"] = "acbm"

    # Combine the data
    modeshare_compare = pd.concat([modeshare_nts, modeshare_acbm])
    # Calculate the percentage of trips for each mode within each source
    modeshare_compare["percentage"] = modeshare_compare.groupby("source")[
        "count"
    ].transform(lambda x: (x / x.sum()) * 100)

    # Plot
    plt.figure()

    sns.barplot(data=modeshare_compare, x="mode", y="percentage", hue="source")
    plt.ylabel("Percentage of total trips")
    plt.title("Percentage of Trips by Mode for NTS and ACBM")
    # plt.show()

    # Save the plot
    plt.tight_layout()
    plt.savefig(validation_plots_path / "2_matching_trip_mode.png")

    logger.info("4a.3 Validation (Matching) - time of day")

    # Plot aggregate
    plot_comparison(
        legs_acbm,
        legs_nts,
        value_column="tst_hour",
        max_y_value=20,
        plot_type="time",
        figsize=(10, 5),
        plot_mode="aggregate",
        save_path=validation_plots_path / "3a_matching_time_of_day_aggregate.png",
    )

    # Plot facet
    plot_comparison(
        legs_acbm,
        legs_nts,
        value_column="tst_hour",
        max_y_value=70,
        plot_type="time",
        plot_mode="facet",
        save_path=validation_plots_path / "3b_matching_time_of_day_facet.png",
    )

    logger.info("4a.4 Validation (Matching) - Activity Sequences")

    # Process the sequences for ACBM and NTS data

    sequence_nts = process_sequences(
        df=legs_nts,
        pid_col="IndividualID",
        seq_col="seq",
        origin_activity_col="oact_abr",
        destination_activity_col="dact_abr",
        suffix="nts",
    )

    sequence_acbm = process_sequences(
        df=legs_acbm,
        pid_col="pid",
        seq_col="seq",
        origin_activity_col="oact_abr",
        destination_activity_col="dact_abr",
        suffix="acbm",
    )

    # Plot the comparison

    plot_activity_sequence_comparison(
        sequence_nts=sequence_nts,
        sequence_acbm=sequence_acbm,
        activity_mapping=activity_mapping,
        perc_cutoff=0.35,
        save_path=validation_plots_path / "4_matching_activity_sequences.png",
    )

    logger.info("4b. Validation (Assigning) - Trip Distance")

    # Apply the function to legs_acbm_geo
    legs_acbm_geo = calculate_od_distances(
        df=legs_acbm_geo,
        start_wkt_col="start_location_geometry_wkt",
        end_wkt_col="end_location_geometry_wkt",
        crs_epsg=config.output_crs,
        detour_factor=1.56,
        decay_rate=0.0001,
    )

    # Plot: Aggregate
    plot_comparison(
        legs_acbm_geo,
        legs_nts,
        value_column="distance",
        bin_size=2,
        value_threshold=50,
        max_y_value=30,
        figsize=(10, 5),
        plot_type="distance",
        plot_mode="aggregate",
        save_path=validation_plots_path / "5a_assigning_distance_aggregate.png",
    )

    # Plot: Facet by activity_type
    plot_comparison(
        legs_acbm_geo,
        legs_nts,
        value_column="distance",
        bin_size=2,
        value_threshold=50,
        max_y_value=30,
        plot_type="distance",
        plot_mode="facet",
        save_path=validation_plots_path / "5b_assigning_distance_facet.png",
    )

    logger.info("4c.1 Validation (Assigning) - Intrazonal Activities")

    # -- Plot: by trip purpose
    # Calculate the percentage of intrazonal trips for each unique purpose

    plot_intrazonal_trips(
        legs_acbm,
        validation_plots_path=validation_plots_path,
        plot_type="purp",
        plot_name="6a_assigning_intrazonal_purp.png",
    )

    # -- Plot: By OD pair
    # Calculate the percentage of intrazonal trips for each unique OD combination
    # (e.g. home - work)

    plot_intrazonal_trips(
        legs_acbm,
        validation_plots_path=validation_plots_path,
        plot_type="od",
        plot_name="6b_assigning_intrazonal_od.png",
    )


if __name__ == "__main__":
    main()
