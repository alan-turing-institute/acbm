import os
import random
from dataclasses import dataclass
from hashlib import sha256
from logging import Logger
from pathlib import Path
from typing import Tuple

import geopandas as gpd
import jcs
import numpy as np
import tomlkit
from pydantic import BaseModel, Field, field_serializer, field_validator

import acbm
from acbm.logger_config import create_logger


@dataclass(frozen=True)
class Parameters(BaseModel):
    seed: int
    region: str
    number_of_households: int | None = None
    zone_id: str
    travel_times: bool
    boundary_geography: str
    nts_years: list[int]
    nts_regions: list[str]
    nts_day_of_week: int
    output_crs: int
    tolerance_work: float | None = None
    tolerance_edu: float | None = None


@dataclass(frozen=True)
class MatchingParams(BaseModel):
    required_columns: Tuple[str, ...]
    optional_columns: Tuple[str, ...]
    n_matches: int | None = None
    chunk_size: int = 50_000


@dataclass(frozen=True)
class FeasibleAssignmentParams(BaseModel):
    detour_factor: float
    decay_rate: float


@dataclass(frozen=True)
class WorkAssignmentParams(BaseModel):
    use_percentages: bool
    weight_max_dev: float
    weight_total_dev: float
    max_zones: int
    commute_level: str

    @field_validator("commute_level")
    def validate_commute_level(commute_level: str) -> str:
        return commute_level.upper()


@dataclass(frozen=True)
class Postprocessing(BaseModel):
    pam_jitter: int
    pam_min_duration: int
    student_age_base: int
    student_age_upper: int
    modes_passenger: list[str]
    pt_subscription_age: int
    state_pension: int


@dataclass(frozen=True)
class PathParams(BaseModel):
    root_path: Path | None = acbm.root_path
    output_path: Path | None = None
    osm_path: Path | None = None
    study_areas_filepath: Path | None = None

    @field_serializer(
        "root_path",
        "output_path",
        "osm_path",
        "study_areas_filepath",
        check_fields=False,
    )
    def serialize_filepath(self, filepath: Path | str | None) -> str | None:
        return None if filepath is None else str(filepath.relative_to(self.root_path))


class Config(BaseModel):
    parameters: Parameters = Field(description="Config: parameters.")
    matching: MatchingParams = Field(description="Config: parameters for matching.")
    feasible_assignment: FeasibleAssignmentParams = Field(
        description="Config: parameters for assignment of feasible zones."
    )
    work_assignment: WorkAssignmentParams = Field(
        description="Config: parameters for work assignment."
    )
    postprocessing: Postprocessing = Field(
        description="Config: parameters for postprocessing."
    )
    paths: PathParams | None = Field(description="Path overrides.", default=None)

    def make_dirs(self):
        """Makes all directories requried from config"""
        os.makedirs(self.output_path, exist_ok=True)
        os.makedirs(self.logs_path, exist_ok=True)
        os.makedirs(self.assigning_plots_path, exist_ok=True)
        os.makedirs(self.validation_plots_path, exist_ok=True)
        os.makedirs(self.activities_per_zone.parent, exist_ok=True)
        os.makedirs(self.study_area_filepath.parent, exist_ok=True)
        os.makedirs(self.interim_path, exist_ok=True)
        os.makedirs(self.travel_times_estimates_filepath.parent, exist_ok=True)
        os.makedirs(self.spc_combined_filepath.parent, exist_ok=True)
        os.makedirs(self.spc_with_nts_trips_filepath.parent, exist_ok=True)
        os.makedirs(self.osmox_path, exist_ok=True)
        os.makedirs(self.osm_path.parent, exist_ok=True)

    @property
    def root_path(self) -> Path:
        return (
            acbm.root_path
            if self.paths is None or self.paths.root_path is None
            else self.paths.root_path
        )

    @property
    def id(self):
        """Since config determines outputs, the SHA256 hash of the config can be used
        as an identifier for outputs.

        See [popgetter](https://github.com/Urban-Analytics-Technology-Platform/popgetter/blob/7da293f4eb2d36480dbd137a27be623aa09449bf/python/popgetter/metadata.py#L83).
        """
        # Take first 10 chars to enable paths to remain not too long
        ID_LENGTH = 10
        return sha256(jcs.canonicalize(self.model_dump(exclude_none=True))).hexdigest()[
            :ID_LENGTH
        ]

    @property
    def logs_path(self) -> Path:
        """Returns logs path."""
        return self.output_path / "logs"

    @property
    def boundaries_filepath(self) -> Path:
        """Returns boundaries path."""
        return self.external_path / "boundaries" / "oa_england.geojson"

    @property
    def lookup_filepath(self) -> Path:
        return (
            self.external_path / "MSOA_2011_MSOA_2021_Lookup_for_England_and_Wales.csv"
        )

    @property
    def spc_raw_path(self) -> Path:
        return self.root_path / "data" / "external" / "spc_output" / "raw"

    @property
    def spc_combined_filepath(self) -> Path:
        return self.interim_path / f"{self.region}_people_hh.parquet"

    @property
    def study_area_filepath(self) -> Path:
        """Returns boundaries path."""
        return (
            self.output_path / "boundaries" / "study_area_zones.geojson"
            if self.paths is None or self.paths.study_areas_filepath is None
            else self.paths.study_areas_filepath
        )

    @property
    def workzone_rmse_results_path(self) -> Path:
        return self.root_path / self.output_path / "workzone_rmse_results.txt"

    @property
    def osmox_path(self) -> Path:
        """Returns osmox path."""
        return self.root_path / self.output_path / "osmox"

    @property
    def osm_path(self) -> Path:
        """Returns osm path."""
        return (
            self.root_path
            / self.osmox_path
            / (self.region + f"_epsg_{self.output_crs}.parquet")
            if self.paths is None or self.paths.osm_path is None
            else self.paths.osm_path
        )

    @property
    def activities_per_zone(self) -> Path:
        """Returns activities per zone filepath."""
        return self.interim_path / "assigning" / "activities_per_zone.parquet"

    @property
    def possible_zones_education(self) -> Path:
        """Returns possible zones for education filepath."""
        return self.interim_path / "assigning" / "possible_zones_education.pkl"

    @property
    def possible_zones_work(self) -> Path:
        """Returns possible zones for work filepath."""
        return self.interim_path / "assigning" / "possible_zones_work.pkl"

    @property
    def osm_poi_with_zones(self) -> Path:
        """Returns OSM POI with zones filepath."""
        return self.interim_path / "assigning" / "osm_poi_with_zones.pkl"

    @property
    def activity_chains_education(self) -> Path:
        """Returns activity chains (education) filepath."""
        return self.interim_path / "assigning" / "activity_chains_education.pkl"

    @property
    def activity_chains_work(self) -> Path:
        """Returns activity chains (work) filepath."""
        return self.interim_path / "assigning" / "activity_chains_work.pkl"

    @property
    def centroids(self) -> Path:
        return self.external_path / "centroids" / "Output_Areas_Dec_2011_PWC_2022.csv"

    @property
    def assigning_plots_path(self) -> str:
        """Returns assigning plots path."""
        return self.output_path / "plots" / "assigning"

    @property
    def validation_plots_path(self) -> str:
        """Returns validation plots path."""
        return self.output_path / "plots" / "validation"

    @property
    def output_path(self) -> str:
        """Returns output path."""
        return (
            self.root_path / "data" / "outputs" / self.id
            if self.paths is None or self.paths.output_path is None
            else self.paths.output_path
        )

    @property
    def external_path(self) -> str:
        """Returns external data path."""
        return self.root_path / "data" / "external"

    @property
    def interim_path(self) -> str:
        """Returns interim data path."""
        return self.output_path / "interim"

    @property
    def psu_filepath(self) -> Path:
        return (
            self.external_path
            / "nts"
            / "UKDA-5340-tab"
            / "tab"
            / "psu_eul_2002-2022.tab"
        )

    @property
    def nts_individuals_filepath(self) -> Path:
        return (
            self.external_path
            / "nts"
            / "UKDA-5340-tab"
            / "tab"
            / "individual_eul_2002-2022.tab"
        )

    @property
    def nts_households_filepath(self) -> Path:
        return (
            self.external_path
            / "nts"
            / "UKDA-5340-tab"
            / "tab"
            / "household_eul_2002-2022.tab"
        )

    @property
    def nts_trips_filepath(self) -> Path:
        return (
            self.external_path
            / "nts"
            / "UKDA-5340-tab"
            / "tab"
            / "trip_eul_2002-2022.tab"
        )

    @property
    def rural_urban_filepath(self) -> Path:
        return self.external_path / "census_2011_rural_urban.csv"

    @property
    def centroid_layer_filepath(self) -> Path:
        return self.external_path / "centroids" / "Output_Areas_Dec_2011_PWC_2022.csv"

    @property
    def travel_demand_filepath(self) -> Path:
        if self.work_assignment.commute_level == "MSOA":
            return self.external_path / "ODWP15EW_MSOA_v1.zip"
        return self.external_path / "ODWP01EW_OA.zip"

    @property
    def travel_times_filepath(self) -> Path:
        if self.work_assignment.commute_level == "MSOA":
            return (
                self.external_path
                / "travel_times"
                / "MSOA"
                / "travel_time_matrix.parquet"
            )
        return self.external_path / "travel_times" / "OA" / "travel_time_matrix.parquet"

    @property
    def travel_times_estimates_filepath(self) -> Path:
        return self.interim_path / "assigning" / "travel_time_estimates.parquet"

    @property
    def spc_with_nts_trips_filepath(self) -> Path:
        return Path(self.interim_path / "matching" / "spc_with_nts_trips.parquet")

    @property
    def seed(self) -> int:
        return self.parameters.seed

    @property
    def region(self) -> str:
        return self.parameters.region

    @property
    def output_crs(self) -> str:
        return self.parameters.output_crs

    @property
    def zone_id(self) -> str:
        return self.parameters.zone_id

    @classmethod
    def origin_zone_id(cls, zone_id: str) -> str:
        return zone_id + "_from"

    @classmethod
    def destination_zone_id(cls, zone_id: str) -> str:
        return zone_id + "_to"

    @property
    def boundary_geography(self) -> str:
        return self.parameters.boundary_geography

    def init_rng(self):
        try:
            np.random.seed(self.seed)
            random.seed(self.seed)
        except Exception as err:
            msg = f"config does not provide a rng seed with err: {err}"
            raise ValueError(msg) from err

    def write(self, filepath: str | Path):
        with open(filepath, "w") as f:
            f.write(tomlkit.dumps(self.model_dump(exclude_none=True)))

    def get_logger(self, name: str, filename: str) -> Logger:
        stem = ".".join(Path(os.path.basename(filename)).name.split(".")[:-1])
        stem_with_log_suffix = stem + ".log"
        return create_logger(
            name,
            stem_with_log_suffix,
            self.logs_path,
        )

    def get_boundaries(self) -> gpd.GeoDataFrame:
        boundaries = gpd.read_file(self.boundaries_filepath)
        return boundaries.to_crs(epsg=self.output_crs)

    def get_study_area_boundaries(self) -> gpd.GeoDataFrame:
        study_area = gpd.read_file(self.study_area_filepath)
        return study_area.to_crs(epsg=self.output_crs)


def load_config(filepath: str | Path) -> Config:
    """Loads config from filepath."""
    with open(filepath, "rb") as f:
        return Config.model_validate(tomlkit.load(f))


def load_and_setup_config(filepath: str | Path) -> Config:
    """Loads config from filepath, makes required dirs, writes config and inits RNG."""
    config = load_config(filepath)
    config.make_dirs()
    config_path = config.output_path / "config.toml"
    if not os.path.isfile(config_path):
        config.write(config_path)
    config.init_rng()
    return config
