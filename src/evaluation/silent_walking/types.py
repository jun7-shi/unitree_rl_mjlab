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
  foot_collision_fromto_xy_m: tuple[tuple[tuple[float, float], tuple[float, float]], ...]
  foot_collision_radius_m: tuple[float, ...]
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
  capsule_outline_fromto_xy_m: torch.Tensor
  capsule_radius_m: torch.Tensor
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
  foot_grid_names: tuple[str, ...] = ()
  foot_grid_shape: tuple[int, int] = (0, 0)
  foot_grid_local_xy_m: torch.Tensor | None = None
  foot_grid_velocity_m_s: torch.Tensor | None = None
  foot_grid_force_n: torch.Tensor | None = None
  foot_loading_rate_n_s: torch.Tensor | None = None
  foot_loading_rate_bw_s: torch.Tensor | None = None
  heel_vertical_velocity_m_s: torch.Tensor | None = None
  toe_vertical_velocity_m_s: torch.Tensor | None = None
  heel_height_m: torch.Tensor | None = None
  toe_height_m: torch.Tensor | None = None
  foot_roll_angle_rad: torch.Tensor | None = None
  touchdown_heel_vz_m_s: torch.Tensor | None = None
  touchdown_toe_vz_m_s: torch.Tensor | None = None
  touchdown_roll_angle_rad: torch.Tensor | None = None
  touchdown_first_contact_region: tuple[str, ...] = ()
  foot_region_names: tuple[str, ...] = ()
  foot_region_contact_flag: torch.Tensor | None = None
  foot_corner_names: tuple[str, ...] = ()
  foot_corner_height_m: torch.Tensor | None = None
  foot_corner_vertical_velocity_m_s: torch.Tensor | None = None
  region_event_type: tuple[str, ...] = ()
  region_event_stance_id: torch.Tensor | None = None
  region_event_step: torch.Tensor | None = None
  region_event_substep_index: torch.Tensor | None = None
  region_event_time_offset_s: torch.Tensor | None = None
  region_event_foot_index: torch.Tensor | None = None
  region_event_foot: tuple[str, ...] = ()
  region_event_regions: tuple[str, ...] = ()
  region_event_peak_force_bw: torch.Tensor | None = None
  region_event_loading_rate_bw_s: torch.Tensor | None = None
  region_event_vertical_speed_m_s: torch.Tensor | None = None
  region_event_corner_downward_speed_m_s: torch.Tensor | None = None
  region_event_corner_downward_speeds_m_s: torch.Tensor | None = None
