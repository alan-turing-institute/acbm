import geopandas as gpd
import pandas as pd

from acbm.assigning.plots import (
    plot_workzone_assignment_heatmap,
    plot_workzone_assignment_line,
)
from acbm.assigning.select_zone_work import WorkZoneAssignment
from acbm.assigning.utils import (
    activity_chains_for_assignment,
    cols_for_assignment_work,
    filter_matrix_to_boundary,
)
from acbm.cli import acbm_cli
from acbm.config import load_and_setup_config
from acbm.preprocessing import add_locations_to_activity_chains
from acbm.utils import calculate_rmse


@acbm_cli
def main(config_file):
    config = load_and_setup_config(config_file)
    logger = config.get_logger("assigning_primary_zone", __file__)

    #### LOAD DATA ####

    # --- Possible zones for each activity (calculated in 3.1_assign_possible_zones.py)
    possible_zones_work = pd.read_pickle(config.possible_zones_work)

    # --- boundaries
    logger.info("Loading study area boundaries")
    boundaries = config.get_study_area_boundaries()
    logger.info(f"Study area boundaries loaded and reprojected to {config.output_crs}")

    # osm POI data

    osm_data_gdf = pd.read_pickle(config.osm_poi_with_zones)
    # Convert the DataFrame into a GeoDataFrame, and assign a coordinate reference system (CRS)
    osm_data_gdf = gpd.GeoDataFrame(
        osm_data_gdf, geometry="geometry", crs=f"EPSG:{config.output_crs}"
    )

    # --- Activity chains
    logger.info("Loading activity chains")
    activity_chains = activity_chains_for_assignment(
        config, cols_for_assignment_work(), subset_to_chosen_day=True
    )

    logger.info("Filtering activity chains for trip purpose: work")
    activity_chains_work = activity_chains[activity_chains["dact"] == "work"]

    logger.info("Assigning activity home locations to boundaries zoning system")
    # add home location (based on OA11CD from SPC)
    activity_chains_work = add_locations_to_activity_chains(
        activity_chains=activity_chains_work,
        target_crs=f"EPSG:{config.output_crs}",
        centroid_layer=pd.read_csv(config.centroid_layer_filepath),
    )

    # --- WORK: existing travel demand data

    # Commuting matrices (from 2021 census)

    # "OA" or "MSOA" data: set as config.boundary_geography if not passed
    commute_level = (
        config.boundary_geography
        if config.work_assignment.commute_level is None
        else config.work_assignment.commute_level
    )

    logger.info(f"Loading commuting matrices at {commute_level} level")

    # Clean the data

    if commute_level == "MSOA":
        logger.info("Step 1: Reading in the zipped csv file")
        travel_demand = pd.read_csv(config.travel_demand_filepath)

        logger.info("Step 2: Creating commute_mode_dict")
        commute_mode_dict = {
            "Bus, minibus or coach": "pt",
            "Driving a car or van": "car",
            "Train": "pt",
            "Underground, metro, light rail, tram": "pt",
            "On foot": "walk",
            "Taxi": "taxi",
            "Other method of travel to work": "other",
            "Bicycle": "cycle",
            "Passenger in a car or van": "car_passenger",
            "Motorcycle, scooter or moped": "car",
            "Work mainly at or from home": "home",
        }

        logger.info("Step 3: Mapping commute mode to model mode")
        travel_demand["mode"] = travel_demand[
            "Method used to travel to workplace (12 categories) label"
        ].map(commute_mode_dict)

        logger.info("Step 4: Filtering rows and dropping unnecessary columns")
        travel_demand_clipped = travel_demand[
            travel_demand["Place of work indicator (4 categories) code"].isin([1, 3])
        ]
        travel_demand_clipped = travel_demand_clipped.drop(
            columns=[
                "Middle layer Super Output Areas label",
                "MSOA of workplace label",
                "Method used to travel to workplace (12 categories) label",
                "Method used to travel to workplace (12 categories) code",
                "Place of work indicator (4 categories) code",
                "Place of work indicator (4 categories) label",
            ]
        )

        logger.info("Step 5: Renaming columns and grouping")
        travel_demand_clipped = travel_demand_clipped.rename(
            columns={
                "Middle layer Super Output Areas code": "MSOA21CD_home",
                "MSOA of workplace code": "MSOA21CD_work",
            }
        )
        travel_demand_clipped = (
            travel_demand_clipped.groupby(["MSOA21CD_home", "MSOA21CD_work", "mode"])
            .agg({"Count": "sum"})
            .reset_index()
        )

        logger.info("Step 6: Filtering matrix to boundary")
        travel_demand_clipped = filter_matrix_to_boundary(
            boundary=boundaries,
            matrix=travel_demand_clipped,
            boundary_id_col="MSOA21CD",
            matrix_id_col="MSOA21CD",
            type="both",
        )

    elif commute_level == "OA":
        logger.info("Step 1: Reading in the zipped csv file")
        travel_demand = pd.read_csv(config.travel_demand_filepath)

        logger.info("Step 2: Filtering rows and dropping unnecessary columns")
        travel_demand_clipped = travel_demand[
            travel_demand["Place of work indicator (4 categories) code"].isin([1, 3])
        ]
        travel_demand_clipped = travel_demand_clipped.drop(
            columns=[
                "Place of work indicator (4 categories) code",
                "Place of work indicator (4 categories) label",
            ]
        )

        logger.info("Step 3: Renaming columns and grouping")
        travel_demand_clipped = travel_demand_clipped.rename(
            columns={
                "Output Areas code": "OA21CD_home",
                "OA of workplace code": "OA21CD_work",
            }
        )
        travel_demand_clipped = (
            travel_demand_clipped.groupby(["OA21CD_home", "OA21CD_work"])
            .agg({"Count": "sum"})
            .reset_index()
        )

        logger.info("Step 4: Filtering matrix to boundary")
        travel_demand_clipped = filter_matrix_to_boundary(
            boundary=boundaries,
            matrix=travel_demand_clipped,
            boundary_id_col=config.zone_id,
            matrix_id_col=config.zone_id,
            type="both",
        )

    logger.info(f"Commuting matrices at {commute_level} level loaded")

    # Get dictionary of commuting matrices
    logger.info("Converting commuting matrices to dictionaries")

    if commute_level == "MSOA":
        # TODO: check, currently unsused
        _travel_demand_dict_mode = (
            travel_demand_clipped.groupby(["MSOA21CD_home", "MSOA21CD_work"])
            .apply(lambda x: dict(zip(x["mode"], x["Count"])))
            .to_dict()
        )
        travel_demand_dict_nomode = (
            travel_demand_clipped.groupby(["MSOA21CD_home", "MSOA21CD_work"])["Count"]
            .sum()
            .to_dict()
        )

    elif commute_level == "OA":
        travel_demand_dict_nomode = (
            travel_demand_clipped.groupby(["OA21CD_home", "OA21CD_work"])["Count"]
            .sum()
            .to_dict()
        )

    logger.info("Commuting matrices converted to dictionaries")

    #### ASSIGN TO ZONE FROM FEASIBLE ZONES ####

    zone_assignment = WorkZoneAssignment(
        activities_to_assign=possible_zones_work, actual_flows=travel_demand_dict_nomode
    )

    assignments_df = zone_assignment.select_work_zone_optimization(
        use_percentages=config.work_assignment.use_percentages,
        weight_max_dev=config.work_assignment.weight_max_dev,
        weight_total_dev=config.work_assignment.weight_total_dev,
        max_zones=config.work_assignment.max_zones,
    )

    # Add assigned zones to activity_chains_work. Replace dzone with assigned_zone
    activity_chains_work["dzone"] = activity_chains_work["id"].map(
        assignments_df.set_index("person_id")["assigned_zone"]
    )

    # --- Evaluating assignment quality

    # - RMSE

    # Step 1: Convert both the actual demand and the assigned demand data to the correct format
    # df: origin_zone, assigned_zone, demand_assigned

    # a: Aggregate assignment_opt DataFrame
    assignment_agg = (
        assignments_df.groupby(["origin_zone", "assigned_zone"])
        .size()
        .reset_index(name="demand_assigned")
    )

    # b: Convert travel_demand_dict_no_mode to DataFrame
    demand_df = pd.DataFrame(
        list(travel_demand_dict_nomode.items()), columns=["zone_pair", "demand_actual"]
    )
    demand_df[["origin_zone", "assigned_zone"]] = pd.DataFrame(
        demand_df["zone_pair"].tolist(), index=demand_df.index
    )
    demand_df.drop(columns=["zone_pair"], inplace=True)

    # Step 2: Merge the two DataFrames
    workzone_assignment_opt = pd.merge(
        assignment_agg, demand_df, on=["origin_zone", "assigned_zone"], how="outer"
    ).fillna(0)

    # (1) % of Total Demand
    workzone_assignment_opt["pct_of_total_demand_actual"] = (
        workzone_assignment_opt["demand_actual"]
        / workzone_assignment_opt["demand_actual"].sum()
    ) * 100
    workzone_assignment_opt["pct_of_total_demand_assigned"] = (
        workzone_assignment_opt["demand_assigned"]
        / workzone_assignment_opt["demand_assigned"].sum()
    ) * 100

    # (2) For each OD pair, demand as % of total demand from the same origin
    workzone_assignment_opt["pct_of_o_total_actual"] = workzone_assignment_opt.groupby(
        "origin_zone"
    )["demand_actual"].transform(lambda x: (x / x.sum()) * 100)
    workzone_assignment_opt["pct_of_o_total_assigned"] = (
        workzone_assignment_opt.groupby(
            "origin_zone"
        )["demand_assigned"].transform(lambda x: (x / x.sum()) * 100)
    )

    # (3) For each OD pair, demand as % of total demand to each destination
    workzone_assignment_opt["pct_of_d_total_actual"] = workzone_assignment_opt.groupby(
        "assigned_zone"
    )["demand_actual"].transform(lambda x: (x / x.sum()) * 100)
    workzone_assignment_opt["pct_of_d_total_assigned"] = (
        workzone_assignment_opt.groupby(
            "assigned_zone"
        )["demand_assigned"].transform(lambda x: (x / x.sum()) * 100)
    )

    # Open the file in write mode
    with open(config.workzone_rmse_results_path, "w") as file:
        # (1) RMSE for % of Total Demand
        predictions = workzone_assignment_opt["pct_of_total_demand_assigned"]
        targets = workzone_assignment_opt["pct_of_total_demand_actual"]

        rmse_pct_of_total_demand = calculate_rmse(predictions, targets)
        file.write(f"RMSE for % of Total Demand: {rmse_pct_of_total_demand}\n")

        # (2) RMSE for demand as % of total demand from the same origin
        predictions = workzone_assignment_opt["pct_of_o_total_assigned"]
        targets = workzone_assignment_opt["pct_of_o_total_actual"]

        rmse_pct_of_o_total = calculate_rmse(predictions, targets)
        file.write(
            f"RMSE for % of Total Demand from the Same Origin: {rmse_pct_of_o_total}\n"
        )

        # (3) RMSE for demand as % of total demand to each destination
        predictions = workzone_assignment_opt["pct_of_d_total_assigned"]
        targets = workzone_assignment_opt["pct_of_d_total_actual"]

        rmse_pct_of_d_total = calculate_rmse(predictions, targets)
        file.write(
            f"RMSE for % of Total Demand to Each Destination: {rmse_pct_of_d_total}\n"
        )

    # --- Plots

    # Plot the demand_actual and demand_assigned values as a line graph for n origin_zones.
    plot_workzone_assignment_line(
        assignment_results=workzone_assignment_opt,
        n=10,
        selection_type="top",
        sort_by="actual",
        save_dir=config.assigning_plots_path,
    )

    # Plot the demand_actual and demand_assigned values as a heatmap for n origin_zones.
    plot_workzone_assignment_heatmap(
        workzone_assignment_opt,
        n=20,
        selection_type="top",
        sort_by="assigned",
        save_dir=config.assigning_plots_path,
    )

    # save the activity chains as a pickle
    activity_chains_work.to_pickle(config.activity_chains_work)


if __name__ == "__main__":
    main()
