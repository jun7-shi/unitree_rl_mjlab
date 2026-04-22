"""Types for silent walking evaluation."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch

RobotAssetStatus = Literal["missing_assets", "ready"]


@dataclass(frozen=True, slots=True)
class RobotSpec:
  """Static metadata for a silent walking robot."""

  name: str
  task_id: str
  action_dim: int
  foot_site_names: tuple[str, ...]
  foot_collision_geom_names: tuple[str, ...]
  asset_path: Path
  mass_normalization: float

  @property
  def asset_status(self) -> RobotAssetStatus:
    """Return the current asset availability state."""

    return "ready" if self.asset_path.exists() else "missing_assets"


@dataclass(frozen=True, slots=True)
class FootContactFrame:
  """Per-foot contact data captured for one simulation frame."""

  foot: str
  positions_w: torch.Tensor
  normals_w: torch.Tensor
  normal_forces_n: torch.Tensor
  tangential_forces_n: torch.Tensor


@dataclass(frozen=True, slots=True)
class EpisodeMetricSummary:
  """Episode-level summary scores for silent walking evaluation."""

  contact_quietness: float
  body_smoothness: float
  task_compliance: float
  total_score: float


@dataclass(frozen=True, slots=True)
class EpisodeMetricTrace:
  """Metric-canonical raw values collected across one evaluation episode."""

  foot_z_force_n: torch.Tensor
  foot_contact_flag: torch.Tensor
  foot_vertical_velocity_m_s: torch.Tensor
  capsule_names: tuple[str, ...]
  capsule_layout_xy_m: torch.Tensor
  capsule_z_force_n: torch.Tensor
  capsule_contact_flag: torch.Tensor
  capsule_vertical_velocity_m_s: torch.Tensor
  peak_force_bw: torch.Tensor
  loading_rate_bw_s: torch.Tensor
  touchdown_peak_force_bw: torch.Tensor
  touchdown_loading_rate_bw_s: torch.Tensor
  touchdown_vertical_speed_m_s: torch.Tensor
  left_touchdown_peak_force_bw: torch.Tensor
  right_touchdown_peak_force_bw: torch.Tensor
  left_touchdown_loading_rate_bw_s: torch.Tensor
  right_touchdown_loading_rate_bw_s: torch.Tensor
  left_touchdown_vertical_speed_m_s: torch.Tensor
  right_touchdown_vertical_speed_m_s: torch.Tensor
  touchdown_peak_asymmetry_bw: torch.Tensor
  capsule_touchdown_peak_force_bw: torch.Tensor
  capsule_touchdown_loading_rate_bw_s: torch.Tensor
  capsule_touchdown_vertical_speed_m_s: torch.Tensor
  capsule_touchdown_count: torch.Tensor
  action_rate_l2: torch.Tensor
  command_velocity: torch.Tensor
  actual_linear_velocity: torch.Tensor
  actual_yaw_rate: torch.Tensor
  linear_velocity_error: torch.Tensor
  yaw_rate_error: torch.Tensor
  contact_quietness_score: torch.Tensor
