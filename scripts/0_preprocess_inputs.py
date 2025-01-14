import pandas as pd
from uatk_spc import Reader

from acbm.cli import acbm_cli
from acbm.config import load_and_setup_config
from acbm.preprocessing import edit_boundary_resolution


@acbm_cli
def main(config_file):
    config = load_and_setup_config(config_file)
    logger = config.get_logger("preprocessing", __file__)

    # ----- BOUNDARIES
    logger.info("Preprocessing Boundary Layer")

    ## Read in the boundary layer for the whole of England

    logger.info("1. Reading in the boundary layer for the whole of England")

    boundaries = config.get_boundaries()

    ## --- Dissolve boundaries if resolution is MSOA

    boundary_geography = config.parameters.boundary_geography  # can only be OA or MSOA
    logger.info(f"2. Dissolving boundaries to {boundary_geography} level")

    boundaries = edit_boundary_resolution(
        study_area=boundaries, geography=boundary_geography, zone_id=config.zone_id
    )

    ## --- Filter to study area
    # we filter using msoa21cd values, which exist regardless of the boundary resolution

    logger.info("3. Filtering boundaries to specified study area")

    # Step 1: Get zones from SPC (these will be 2011 MSOAs)
    spc = Reader(config.spc_raw_path, config.region, backend="pandas")
    zones_in_region = list(spc.info_per_msoa.keys())

    # Step 2: Filter boundaries to identified zones

    # a) get MSOA11CD to MSOA21CD lookup
    msoa_lookup = pd.read_csv(config.lookup_filepath)
    # Filter msoa_lookup to include only rows where MSOA11CD is in zones_in_region
    msoa_lookup_filtered = msoa_lookup[msoa_lookup["MSOA11CD"].isin(zones_in_region)]
    # Extract the corresponding MSOA21CD values
    msoa21cd_values = msoa_lookup_filtered["MSOA21CD"].tolist()

    # b) filter boundaries to include only rows where MSOA21CD is in msoa21cd_values
    boundaries_filtered = boundaries[boundaries["MSOA21CD"].isin(msoa21cd_values)]

    ## Save the output as parquet
    logger.info(f"4. Saving the boundaries to {config.study_area_filepath} path")
    boundaries_filtered.to_file(
        config.study_area_filepath,
        driver="GeoJSON",
    )


if __name__ == "__main__":
    main()
