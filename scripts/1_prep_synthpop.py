from uatk_spc.builder import Builder

from acbm.cli import acbm_cli
from acbm.config import load_and_setup_config


@acbm_cli
def main(config_file):
    config = load_and_setup_config(config_file)
    logger = config.get_logger("preprocessing", __file__)

    logger.info("Combine SPC people and houeshold data")
    spc_people_hh = (
        Builder(
            config.spc_raw_path, config.region, backend="pandas", input_type="parquet"
        )
        .add_households()
        .unnest(
            ["health", "employment", "details", "demographics"], rsuffix="_household"
        )
        .build()
    )

    logger.info(f"Write combined SPC data to: {config.spc_combined_filepath}")
    spc_people_hh.to_parquet(config.spc_combined_filepath)


if __name__ == "__main__":
    main()
