"""Types for silent walking robot metadata."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RobotSpec:
  """Static metadata for a silent walking robot."""

  name: str
  task_id: str
  foot_site_names: tuple[str, ...]
  foot_collision_geom_names: tuple[str, ...]
  asset_status: str
  mass_normalization: float
