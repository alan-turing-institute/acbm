#!/usr/bin/env python

import subprocess

import click


def run_command(script_path, config_file):
    command = ["python", script_path, "--config-file", config_file]
    try:
        subprocess.run(command, capture_output=False, text=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        print(f"Return code: {e.returncode}")
        print(f"Output: {e.output}")
        exit(1)


@click.command()
@click.option(
    "--config-file", help="Filepath relative to repo root of config", type=str
)
@click.option(
    "--skip-preprocess", help="Skip preprocess script", is_flag=True, default=False
)
@click.option("--skip-osmox", help="Skip osmox script", is_flag=True, default=False)
@click.option(
    "--matching-only",
    help="Only run the matching part of AcBM",
    is_flag=True,
    default=False,
)
@click.option(
    "--assigning-only",
    help="Only run the assigning part of AcBM",
    is_flag=True,
    default=False,
)
def main(config_file, skip_preprocess, skip_osmox, matching_only, assigning_only):
    if not skip_preprocess and not assigning_only:
        run_command("scripts/0_preprocess_inputs.py", config_file)
    if not skip_osmox and not matching_only and not assigning_only:
        run_command("scripts/0.1_run_osmox.py", config_file)
    if not assigning_only:
        run_command("scripts/1_prep_synthpop.py", config_file)
        run_command("scripts/2_match_households_and_individuals.py", config_file)
    if not matching_only or assigning_only:
        run_command("scripts/3.1_assign_primary_feasible_zones.py", config_file)
        run_command("scripts/3.2.1_assign_primary_zone_edu.py", config_file)
        run_command("scripts/3.2.2_assign_primary_zone_work.py", config_file)
        run_command("scripts/3.2.3_assign_secondary_zone.py", config_file)
        run_command("scripts/3.3_assign_facility_all.py", config_file)
        run_command("scripts/4_validation.py", config_file)
        run_command("scripts/5_acbm_to_matsim_xml.py", config_file)


if __name__ == "__main__":
    main()
