import os

import click
import numpy as np
import pandas as pd
import polars as pl

import acbm
from acbm.config import load_config


def print_stats(counts):
    print("Census count:", counts["census_count"].sum())
    print(
        "ACBM count:",
        counts["acbm_count"].sum(),
    )
    r2 = np.corrcoef(counts["census_count"], counts["acbm_count"])[0, 1] ** 2
    print("The R^2 value is: ", r2)
    rmse = np.sqrt((counts["census_count"] - counts["acbm_count"]).pow(2).mean())
    print("The RMSE value is: ", rmse)
    mes = (counts["census_count"] - counts["acbm_count"]).abs().mean()
    print("The MAE value is: ", mes)


@click.command()
@click.option("--id", prompt="Run ID for stats to be generated from", type=str)
def main(id: str):
    pd.options.mode.copy_on_write = True
    os.chdir(acbm.root_path)
    config = load_config(f"data/outputs/{id}/config.toml")
    trav_day = 3
    spc = pl.read_parquet(config.interim_path / "leeds_people_hh.parquet")
    df = pl.read_parquet(
        config.interim_path / "matching" / "spc_with_nts_trips.parquet"
    ).join(spc.select(["id", "pwkstat"]), on="id", how="left", coalesce=True)

    print(
        "% of people with a NTS match: {:.1%}".format(
            df.filter(df["nts_hh_id"].is_not_null()).unique("id").shape[0]
            / df.unique("id").shape[0]
        )
    )
    print(
        "% of people with any travel day: {:.1%}".format(
            df.filter(pl.col("TravDay").is_not_null()).group_by("id").all().shape[0]
            / df.unique("id").shape[0],
        )
    )
    print(
        "% of people with any weekday travel day: {:.1%}".format(
            df.group_by("id")
            .all()
            .filter(
                pl.lit(1).is_in(pl.col("TravDay"))
                | pl.lit(2).is_in(pl.col("TravDay"))
                | pl.lit(3).is_in(pl.col("TravDay"))
                | pl.lit(4).is_in(pl.col("TravDay"))
                | pl.lit(5).is_in(pl.col("TravDay"))
            )
            .shape[0]
            / df.unique("id").shape[0],
        )
    )
    print(
        "% of people with a travel day of {}: {:.1%}".format(
            trav_day,
            df.group_by("id").all().filter(pl.lit(3).is_in(pl.col("TravDay"))).shape[0]
            / df.unique("id").shape[0],
        )
    )

    acbm_matrix = (
        pl.scan_parquet(config.output_path / "legs_with_locations.parquet")
        .filter(pl.col("purp").eq("work"))
        .unique(subset=["pid"])
        .filter(pl.col("ozone").is_not_null() & pl.col("dzone").is_not_null())
        .group_by(["ozone", "dzone"])
        .len()
        .rename({"len": "acbm_count"})
        .collect()
    )

    census_matrix = (
        pl.scan_csv("data/external/cencus/ODWP15EW_MSOA.csv")
        .rename(
            {
                "Middle layer Super Output Areas code": "ozone",
                "MSOA of workplace code": "dzone",
            }
        )
        .filter(pl.col("Place of work indicator (4 categories) code").is_in([1, 3]))
        .rename({"Count": "census_count"})
        .group_by(["ozone", "dzone"])
        .sum()
        .select(["ozone", "dzone", "census_count"])
        .collect()
    )

    both_counts = acbm_matrix.join(census_matrix, on=["ozone", "dzone"])
    print_stats(both_counts)


if __name__ == "__main__":
    main()
