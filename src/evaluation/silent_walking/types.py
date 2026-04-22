"""Types for silent walking robot metadata."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

RobotAssetStatus = Literal["missing_assets", "ready"]


@dataclass(frozen=True, slots=True)
class RobotSpec:
  """Static metadata for a silent walking robot."""

  name: str
  task_id: str
  foot_site_names: tuple[str, ...]
  foot_collision_geom_names: tuple[str, ...]
  asset_path: Path
  mass_normalization: float

  @property
  def asset_status(self) -> RobotAssetStatus:
    """Return the current asset availability state."""

    return "ready" if self.asset_path.exists() else "missing_assets"
