from typing import Callable

import click
import pandas as pd


def acbm_cli(c: Callable):
    @click.command()
    @click.option(
        "--config-file", prompt="Filepath relative to repo root of config", type=str
    )
    def main(config_file):
        pd.options.mode.copy_on_write = True
        return c(config_file)

    return main
