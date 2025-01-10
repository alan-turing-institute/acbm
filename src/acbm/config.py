import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tomlkit
from pydantic import BaseModel, Field


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


@dataclass(frozen=True)
class MatchingParams(BaseModel):
    required_columns: list[str]
    optional_columns: list[str]
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
    commute_level: str | None = None


@dataclass(frozen=True)
class Postprocessing(BaseModel):
    pam_jitter: int
    pam_min_duration: int
    student_age_base: int
    student_age_upper: int
    modes_passenger: list[str]
    pt_subscription_age: int
    state_pension: int


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

    # TODO: consider moving to method in config
    def init_rng(self):
        try:
            np.random.seed(self.seed)
            random.seed(self.seed)
        except Exception as err:
            msg = f"config does not provide a rng seed with err: {err}"
            raise ValueError(msg) from err


def load_config(filepath: str | Path) -> Config:
    with open(filepath, "rb") as f:
        return Config.model_validate(tomlkit.load(f))
