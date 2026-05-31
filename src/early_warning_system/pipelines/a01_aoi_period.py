from .utils import *

import datetime
from dataclasses import dataclass, fields
from typing import Optional


# ──────────────────────────────────────────────
# BOUNDING BOX
# ──────────────────────────────────────────────

@dataclass
class BoundingBox:
    """Spatial extent container with provider-specific format converters."""
    minx: float
    miny: float
    maxx: float
    maxy: float

    @classmethod
    def from_dict(cls, d: dict) -> "BoundingBox":
        valid_keys = {f.name for f in fields(cls)}
        missing = valid_keys - set(d.keys())
        if missing:
            raise ValueError(f"BoundingBox dict is missing keys: {missing}")
        return cls(**{k: d[k] for k in valid_keys})

    def to_dict(self) -> dict:
        return {"minx": self.minx, "miny": self.miny, "maxx": self.maxx, "maxy": self.maxy}

    def to_era5_area(self) -> list:
        """Returns [N, W, S, E] as required by the CDS API."""
        return [self.maxy, self.minx, self.miny, self.maxx]

    def __repr__(self) -> str:
        return f"BoundingBox(N={self.maxy}, S={self.miny}, W={self.minx}, E={self.maxx})"


# ──────────────────────────────────────────────
# SPATIAL HELPERS
# ──────────────────────────────────────────────

def round_up_values(value, part_per_unit):
    return float(np.ceil(value*part_per_unit)/part_per_unit)

def round_down_values(value, part_per_unit):
    return float(np.floor(value*part_per_unit)/part_per_unit)
    
def get_bounds(minx, maxx, maxy, miny, nodes_by_deg=10, **kwargs):
    '''Get bounds that are compatible with the patch size of model'''
    # Return the bounds
    dict_params = {
        'minx': round_down_values(minx, nodes_by_deg),
        'miny': round_down_values(miny, nodes_by_deg),
        'maxx': round_up_values(maxx, nodes_by_deg), # Substract one node since the middle point is included
        'maxy': round_up_values(maxy, nodes_by_deg), # Substract one node since the middle point is included
    }
    return dict_params


# ──────────────────────────────────────────────
# PERIOD HELPERS
# ──────────────────────────────────────────────

def resolve_period(
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    window: int = 30,
) -> tuple[int, int]:
    """Resolve (start_year, end_year), defaulting to last completed year and a 30-year window."""
    if end_year is None:
        end_year = datetime.date.today().year - 1
    if start_year is None:
        start_year = end_year - window
    return int(start_year), int(end_year)


def get_year_list(start_year: int, end_year: int) -> list[int]:
    return list(range(start_year, end_year + 1))


def describe_period(start_year: int, end_year: int) -> str:
    return f"{start_year} to {end_year} ({end_year - start_year} year period)"