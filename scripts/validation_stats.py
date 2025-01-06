import os

import click
import numpy as np
import pandas as pd
import polars as pl

import acbm
from acbm.config import load_config


def print_stats(counts, lcol, rcol):
    print("Census (total):", counts[lcol].sum())
    print(
        "ACBM (total):",
        counts[rcol].sum(),
    )
    r2 = np.corrcoef(counts[lcol], counts[rcol])[0, 1] ** 2
    print("The R^2 value is: ", r2)
    rmse = np.sqrt((counts[lcol] - counts[rcol]).pow(2).mean())
    print("The RMSE value is: ", rmse)
    mes = (counts[lcol] - counts[rcol]).abs().mean()
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

    print("Summary stats for SPC and NTS matched people and activities...")
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

    # For no normalization
    both_counts = acbm_matrix.join(census_matrix, on=["ozone", "dzone"])

    print(both_counts)
    print("\nStats for counts...")
    print_stats(both_counts, "census_count", "acbm_count")

    # Total flows (by origin): i.e. the sum over dzones for a given ozone
    acbm_matrix_norm = acbm_matrix.join(
        acbm_matrix.select(["ozone", "acbm_count"])
        .group_by("ozone")
        .sum()
        .rename({"acbm_count": "acbm_total"}),
        on=["ozone"],
    ).with_columns((pl.col("acbm_count") / pl.col("acbm_total")).alias("acbm_norm"))

    census_matrix_norm = census_matrix.join(
        census_matrix.select(["ozone", "census_count"])
        .group_by("ozone")
        .sum()
        .rename({"census_count": "census_total"}),
        on=["ozone"],
    ).with_columns(
        (pl.col("census_count") / pl.col("census_total")).alias("census_norm")
    )
    both_norms = acbm_matrix_norm.join(census_matrix_norm, on=["ozone", "dzone"])

    print(both_norms)
    print("\nStats for proportions (by total origin)...")
    print_stats(both_norms, "census_norm", "acbm_norm")


if __name__ == "__main__":
    main()
