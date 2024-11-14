import random
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import jcs
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


@dataclass(frozen=True)
class MatchingParams(BaseModel):
    required_columns: list[str]
    optional_columns: list[str]
    n_matches: int | None = None
    chunk_size: int = 50_000


@dataclass(frozen=True)
class WorkAssignmentParams(BaseModel):
    use_percentages: bool
    weight_max_dev: float
    weight_total_dev: float
    max_zones: int
    commute_level: str | None = None


class Config(BaseModel):
    parameters: Parameters = Field(description="Config: parameters.")
    work_assignment: WorkAssignmentParams = Field(
        description="Config: parameters for work assignment."
    )
    matching: MatchingParams = Field(description="Config: parameters for matching.")

    @property
    def id(self):
        """Since config determines outputs, the SHA256 hash of the config can be used
        as an identifier for outputs.

        See [popgetter](https://github.com/Urban-Analytics-Technology-Platform/popgetter/blob/7da293f4eb2d36480dbd137a27be623aa09449bf/python/popgetter/metadata.py#L83).
        """
        # Since the out paths are not too long, take first 10 chars
        ID_LENGTH = 10

        def serializable_vars(obj: object) -> dict:
            variables = {}
            # Check if variables are serializable
            for key, val in vars(obj).items():
                try:
                    # Try to serialize
                    jcs.canonicalize(val)
                    # Store in dict if serializable
                    variables[key] = val
                except Exception:
                    # If cannot serialize, continue
                    continue

        return sha256(jcs.canonicalize(serializable_vars(self))).hexdigest()[:ID_LENGTH]

    def processed_path(self) -> str:
        """Returns full processed path."""
        return Path("data") / "processed" / self.id

    @property
    def seed(self) -> int:
        return self.parameters.seed

    @property
    def region(self) -> str:
        return self.parameters.region

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
