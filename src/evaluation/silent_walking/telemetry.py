"""Streaming telemetry helpers for silent walking evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from mjlab.utils.lab_api.math import quat_apply

from .metrics import (
  combine_total_score,
  compute_body_smoothness_score,
  compute_contact_quietness_score,
  compute_task_compliance_score,
  normalize_force_by_body_weight,
)
from .regional_contact import REGION_NAMES, RegionalContactEvent, RegionalContactState
from .robots import get_foot_proxy_spec
from .types import EpisodeMetricSummary, EpisodeMetricTrace

FootRegion = Literal["heel", "toe", "flat", "unknown"]


def detect_touchdown(
  previous_contact: torch.Tensor,
  current_contact: torch.Tensor,
) -> torch.Tensor:
  """Return contact rising edges."""

  return torch.logical_not(previous_contact.bool()) & current_contact.bool()


def positive_loading_rate(force_series: torch.Tensor, dt: float) -> torch.Tensor:
  """Return frame-wise positive force derivative, clamped at zero."""

  if dt <= 0.0:
    raise ValueError("dt must be positive")
  if force_series.ndim < 1:
    raise ValueError("force_series must have at least one dimension")
  if force_series.shape[0] == 0:
    return force_series.clone()

  rate = torch.zeros_like(force_series)
  if force_series.shape[0] == 1:
    return rate
  deltas = force_series[1:] - force_series[:-1]
  rate[1:] = torch.clamp(deltas / dt, min=0.0)
  return rate


def virtual_heel_toe_points(
  *,
  body_pos_w: torch.Tensor,
  body_quat_w: torch.Tensor,
  heel_local_offset_m: tuple[float, float, float],
  toe_local_offset_m: tuple[float, float, float],
) -> tuple[torch.Tensor, torch.Tensor]:
  """Return virtual heel/toe world positions for rigid foot telemetry."""

  heel_offset = torch.tensor(
    heel_local_offset_m,
    dtype=body_pos_w.dtype,
    device=body_pos_w.device,
  ).view(1, 1, 3)
  toe_offset = torch.tensor(
    toe_local_offset_m,
    dtype=body_pos_w.dtype,
    device=body_pos_w.device,
  ).view(1, 1, 3)
  heel_pos_w = body_pos_w + quat_apply(body_quat_w, heel_offset.expand_as(body_pos_w))
  toe_pos_w = body_pos_w + quat_apply(body_quat_w, toe_offset.expand_as(body_pos_w))
  return heel_pos_w, toe_pos_w


def foot_roll_angle_from_points(
  heel_pos_w: torch.Tensor,
  toe_pos_w: torch.Tensor,
) -> torch.Tensor:
  """Return rigid-foot roll proxy; positive means toe-up."""

  horizontal = torch.linalg.norm(toe_pos_w[..., :2] - heel_pos_w[..., :2], dim=-1)
  horizontal = torch.clamp(horizontal, min=1e-6)
  return torch.atan2(toe_pos_w[..., 2] - heel_pos_w[..., 2], horizontal)


@dataclass(frozen=True, slots=True)
class TouchdownEvent:
  """A single foot touchdown event."""

  step: int
  foot_index: int
  foot_name: str
  peak_force_n: torch.Tensor
  loading_rate_n_s: torch.Tensor
  vertical_speed_m_s: torch.Tensor
  heel_vz_m_s: torch.Tensor
  toe_vz_m_s: torch.Tensor
  roll_angle_rad: torch.Tensor
  first_contact_region: FootRegion


@dataclass(frozen=True, slots=True)
class QuietTelemetrySnapshot:
  """Latest values for realtime quiet-walking overlays."""

  step: int
  foot_names: tuple[str, ...]
  foot_contact: torch.Tensor
  foot_force_n: torch.Tensor
  foot_loading_rate_n_s: torch.Tensor
  foot_site_vz_m_s: torch.Tensor
  heel_vz_m_s: torch.Tensor
  toe_vz_m_s: torch.Tensor
  foot_roll_angle_rad: torch.Tensor
  last_touchdown: TouchdownEvent | None
  foot_region_names: tuple[str, ...] = ()
  foot_region_contact: torch.Tensor | None = None
  foot_corner_names: tuple[str, ...] = ()
  foot_corner_vz_m_s: torch.Tensor | None = None
  last_region_event: RegionalContactEvent | None = None


class SilentTelemetryCollector:
  """Collect streaming quiet-walking telemetry and reduce it into episode traces."""

  def __init__(
    self,
    *,
    robot_name: str,
    dt: float,
    body_weight_newton: float,
  ) -> None:
    if dt <= 0.0:
      raise ValueError("dt must be positive")
    if body_weight_newton <= 0.0:
      raise ValueError("body_weight_newton must be positive")

    self.robot_name = robot_name
    self.dt = dt
    self.body_weight_newton = body_weight_newton
    self.proxy = get_foot_proxy_spec(robot_name)

    self._actions: list[torch.Tensor] = []
    self._action_rate_l2: list[torch.Tensor] = []
    self._command_velocity: list[torch.Tensor] = []
    self._actual_linear_velocity: list[torch.Tensor] = []
    self._actual_yaw_rate: list[torch.Tensor] = []
    self._linear_velocity_error: list[torch.Tensor] = []
    self._yaw_rate_error: list[torch.Tensor] = []
    self._foot_force_n: list[torch.Tensor] = []
    self._foot_contact: list[torch.Tensor] = []
    self._foot_site_vz_m_s: list[torch.Tensor] = []
    self._foot_loading_rate_n_s: list[torch.Tensor] = []
    self._heel_height_m: list[torch.Tensor] = []
    self._toe_height_m: list[torch.Tensor] = []
    self._heel_vz_m_s: list[torch.Tensor] = []
    self._toe_vz_m_s: list[torch.Tensor] = []
    self._foot_corner_height_m: list[torch.Tensor] = []
    self._foot_corner_vz_m_s: list[torch.Tensor] = []
    self._foot_region_contact: list[torch.Tensor] = []
    self._foot_roll_angle_rad: list[torch.Tensor] = []
    self._foot_grid_velocity_m_s: list[torch.Tensor] = []
    self._foot_grid_force_n: list[torch.Tensor] = []
    self._foot_grid_names: tuple[str, ...] = ()
    self._foot_grid_local_xy_m: torch.Tensor | None = None
    self._capsule_force_n: list[torch.Tensor] = []
    self._capsule_contact: list[torch.Tensor] = []
    self._capsule_vz_m_s: list[torch.Tensor] = []
    self._capsule_names: tuple[str, ...] = ()
    self._touchdowns: list[TouchdownEvent] = []
    self._region_events: list[RegionalContactEvent] = []
    self._region_state = RegionalContactState(num_feet=len(self.proxy.foot_names))

    self._previous_action: torch.Tensor | None = None
    self._previous_foot_contact: torch.Tensor | None = None
    self._previous_foot_force_n: torch.Tensor | None = None
    self._previous_heel_pos_w: torch.Tensor | None = None
    self._previous_toe_pos_w: torch.Tensor | None = None
    self._previous_foot_corner_pos_w: torch.Tensor | None = None
    self._latest_snapshot: QuietTelemetrySnapshot | None = None

  @property
  def latest_snapshot(self) -> QuietTelemetrySnapshot | None:
    """Return the latest realtime telemetry snapshot."""

    return self._latest_snapshot

  def record_sample(
    self,
    *,
    action: torch.Tensor,
    command_velocity: torch.Tensor,
    actual_linear_velocity: torch.Tensor,
    actual_yaw_rate: torch.Tensor,
    foot_force_n: torch.Tensor,
    foot_contact: torch.Tensor,
    foot_site_vz_m_s: torch.Tensor,
    capsule_names: tuple[str, ...],
    capsule_force_n: torch.Tensor,
    capsule_contact: torch.Tensor,
    capsule_vz_m_s: torch.Tensor,
    heel_pos_w: torch.Tensor,
    toe_pos_w: torch.Tensor,
    foot_roll_angle_rad: torch.Tensor,
    foot_corner_pos_w: torch.Tensor | None = None,
    foot_region_contact: torch.Tensor | None = None,
    foot_grid_names: tuple[str, ...] = (),
    foot_grid_local_xy_m: torch.Tensor | None = None,
    foot_grid_velocity_m_s: torch.Tensor | None = None,
    foot_grid_force_n: torch.Tensor | None = None,
    substep_dt: float | None = None,
    substep_foot_force_n: torch.Tensor | None = None,
    substep_foot_site_vz_m_s: torch.Tensor | None = None,
    substep_foot_region_contact: torch.Tensor | None = None,
    substep_corner_downward_speeds_m_s: torch.Tensor | None = None,
  ) -> None:
    """Record one control-step telemetry sample."""

    action_2d = _as_2d(action)
    command_velocity_2d = _as_2d(command_velocity)
    actual_linear_velocity_2d = _as_2d(actual_linear_velocity)
    actual_yaw_rate_1d = _as_1d(actual_yaw_rate)
    foot_force_2d = _as_2d(foot_force_n)
    foot_contact_2d = _as_2d(foot_contact).bool()
    foot_site_vz_2d = _as_2d(foot_site_vz_m_s)
    capsule_force_2d = _as_2d(capsule_force_n)
    capsule_contact_2d = _as_2d(capsule_contact).bool()
    capsule_vz_2d = _as_2d(capsule_vz_m_s)
    heel_pos_3d = _as_3d(heel_pos_w)
    toe_pos_3d = _as_3d(toe_pos_w)
    roll_angle_2d = _as_2d(foot_roll_angle_rad)
    if foot_corner_pos_w is None:
      foot_corner_pos_4d = torch.zeros(
        (*heel_pos_3d.shape[:2], len(self.proxy.foot_corner_names), 3),
        dtype=heel_pos_3d.dtype,
        device=heel_pos_3d.device,
      )
    else:
      foot_corner_pos_4d = _as_4d(foot_corner_pos_w)
    if foot_region_contact is None:
      foot_region_contact_3d = torch.zeros(
        (*heel_pos_3d.shape[:2], len(REGION_NAMES)),
        dtype=torch.bool,
        device=heel_pos_3d.device,
      )
    else:
      foot_region_contact_3d = _as_3d(foot_region_contact).bool()
    foot_grid_velocity_4d = (
      _as_4d(foot_grid_velocity_m_s)
      if foot_grid_velocity_m_s is not None
      else None
    )
    foot_grid_force_3d = (
      _as_3d(foot_grid_force_n)
      if foot_grid_force_n is not None
      else None
    )

    selected = 0
    current_action = action_2d[selected].detach().clone()
    current_foot_force = foot_force_2d[selected].detach().clone()
    current_foot_contact = foot_contact_2d[selected].detach().clone()
    current_heel_pos = heel_pos_3d[selected].detach().clone()
    current_toe_pos = toe_pos_3d[selected].detach().clone()
    current_corner_pos = foot_corner_pos_4d[selected].detach().clone()
    current_region_contact = foot_region_contact_3d[selected].detach().clone()
    current_grid_velocity = (
      foot_grid_velocity_4d[selected].detach().clone()
      if foot_grid_velocity_4d is not None
      else None
    )
    current_grid_force = (
      foot_grid_force_3d[selected].detach().clone()
      if foot_grid_force_3d is not None
      else None
    )
    if current_grid_velocity is not None:
      if current_grid_force is None:
        current_grid_force = torch.zeros(
          current_grid_velocity.shape[:2],
          dtype=current_grid_velocity.dtype,
          device=current_grid_velocity.device,
        )
      if current_grid_force.shape != current_grid_velocity.shape[:2]:
        raise ValueError("foot_grid_force_n must have shape [B, F, P]")
      if not foot_grid_names:
        raise ValueError("foot_grid_names are required when foot_grid_velocity_m_s is set")
      if foot_grid_local_xy_m is None:
        raise ValueError("foot_grid_local_xy_m is required when foot_grid_velocity_m_s is set")
      local_xy = foot_grid_local_xy_m.detach().clone().to(
        dtype=current_grid_velocity.dtype,
        device=current_grid_velocity.device,
      )
      if local_xy.ndim != 2 or local_xy.shape != current_grid_velocity.shape[1:2] + (2,):
        raise ValueError("foot_grid_local_xy_m must have shape [P, 2]")
      if self._foot_grid_names and self._foot_grid_names != foot_grid_names:
        raise ValueError("foot-grid point ordering changed during telemetry collection")
      if (
        self._foot_grid_local_xy_m is not None
        and self._foot_grid_local_xy_m.shape != local_xy.shape
      ):
        raise ValueError("foot-grid point layout changed during telemetry collection")
      self._foot_grid_names = foot_grid_names
      self._foot_grid_local_xy_m = local_xy

    if self._previous_action is None:
      action_rate = torch.zeros(1, dtype=current_action.dtype, device=current_action.device)
    else:
      action_rate = torch.linalg.norm(current_action - self._previous_action).reshape(1)

    if self._previous_foot_force_n is None:
      foot_loading_rate = torch.zeros_like(current_foot_force)
    else:
      foot_loading_rate = torch.clamp(
        (current_foot_force - self._previous_foot_force_n) / self.dt,
        min=0.0,
      )

    if self._previous_heel_pos_w is None:
      heel_vz = torch.zeros_like(current_heel_pos[:, 2])
      toe_vz = torch.zeros_like(current_toe_pos[:, 2])
    else:
      heel_vz = (current_heel_pos[:, 2] - self._previous_heel_pos_w[:, 2]) / self.dt
      toe_vz = (current_toe_pos[:, 2] - self._previous_toe_pos_w[:, 2]) / self.dt

    if self._previous_foot_corner_pos_w is None:
      corner_vz = torch.zeros_like(current_corner_pos[..., 2])
    else:
      corner_vz = (
        current_corner_pos[..., 2] - self._previous_foot_corner_pos_w[..., 2]
      ) / self.dt
    corner_downward_speeds = torch.clamp(-corner_vz, min=0.0)

    if self._previous_foot_contact is not None:
      touchdown = detect_touchdown(self._previous_foot_contact, current_foot_contact)
      for foot_index in torch.where(touchdown)[0].tolist():
        foot_name = self.proxy.foot_names[foot_index]
        event = TouchdownEvent(
          step=len(self._foot_force_n),
          foot_index=int(foot_index),
          foot_name=foot_name,
          peak_force_n=current_foot_force[foot_index].detach().clone(),
          loading_rate_n_s=foot_loading_rate[foot_index].detach().clone(),
          vertical_speed_m_s=torch.clamp(
            -foot_site_vz_2d[selected, foot_index],
            min=0.0,
          ).detach().clone(),
          heel_vz_m_s=heel_vz[foot_index].detach().clone(),
          toe_vz_m_s=toe_vz[foot_index].detach().clone(),
          roll_angle_rad=roll_angle_2d[selected, foot_index].detach().clone(),
          first_contact_region=_first_contact_region(
            float(current_heel_pos[foot_index, 2].item()),
            float(current_toe_pos[foot_index, 2].item()),
          ),
        )
        self._touchdowns.append(event)

    if substep_foot_region_contact is None:
      foot_vertical_speed = torch.clamp(-foot_site_vz_2d[selected], min=0.0)
      self._region_events.extend(
        self._region_state.update(
          step=len(self._foot_force_n),
          region_contact=current_region_contact,
          foot_names=self.proxy.foot_names,
          foot_force_n=current_foot_force,
          foot_loading_rate_n_s=foot_loading_rate,
          foot_vertical_speed_m_s=foot_vertical_speed,
          corner_downward_speeds_m_s=corner_downward_speeds,
        )
      )
    else:
      substep_region_contact_3d = _as_substep_3d(substep_foot_region_contact).bool()
      num_substeps = int(substep_region_contact_3d.shape[0])
      if num_substeps > 0:
        resolved_substep_dt = substep_dt if substep_dt is not None else self.dt / num_substeps
        substep_force_2d = (
          _as_substep_2d(substep_foot_force_n)
          if substep_foot_force_n is not None
          else current_foot_force.expand(num_substeps, -1)
        )
        substep_site_vz_2d = (
          _as_substep_2d(substep_foot_site_vz_m_s)
          if substep_foot_site_vz_m_s is not None
          else foot_site_vz_2d[selected].expand(num_substeps, -1)
        )
        substep_corner_downward_3d = (
          _as_substep_3d(substep_corner_downward_speeds_m_s)
          if substep_corner_downward_speeds_m_s is not None
          else corner_downward_speeds.expand(num_substeps, -1, -1)
        )
        previous_substep_force = self._previous_foot_force_n
        for substep_index in range(num_substeps):
          substep_force = substep_force_2d[substep_index].detach().clone()
          if previous_substep_force is None:
            substep_loading_rate = torch.zeros_like(substep_force)
          else:
            substep_loading_rate = torch.clamp(
              (substep_force - previous_substep_force) / resolved_substep_dt,
              min=0.0,
            )
          self._region_events.extend(
            self._region_state.update(
              step=len(self._foot_force_n),
              substep_index=substep_index,
              time_offset_s=(substep_index + 1) * resolved_substep_dt,
              region_contact=substep_region_contact_3d[substep_index],
              foot_names=self.proxy.foot_names,
              foot_force_n=substep_force,
              foot_loading_rate_n_s=substep_loading_rate,
              foot_vertical_speed_m_s=torch.clamp(
                -substep_site_vz_2d[substep_index],
                min=0.0,
              ),
              corner_downward_speeds_m_s=substep_corner_downward_3d[substep_index],
            )
          )
          previous_substep_force = substep_force

    linear_error = torch.linalg.norm(
      command_velocity_2d[:, :2] - actual_linear_velocity_2d[:, :2],
      dim=1,
    )
    yaw_error = torch.abs(command_velocity_2d[:, 2] - actual_yaw_rate_1d)

    self._actions.append(current_action)
    self._action_rate_l2.append(action_rate.detach().clone())
    self._command_velocity.append(command_velocity_2d[selected].detach().clone())
    self._actual_linear_velocity.append(actual_linear_velocity_2d[selected].detach().clone())
    self._actual_yaw_rate.append(actual_yaw_rate_1d[selected].detach().clone().reshape(1))
    self._linear_velocity_error.append(linear_error[selected].detach().clone().reshape(1))
    self._yaw_rate_error.append(yaw_error[selected].detach().clone().reshape(1))
    self._foot_force_n.append(current_foot_force)
    self._foot_contact.append(current_foot_contact.float())
    self._foot_site_vz_m_s.append(foot_site_vz_2d[selected].detach().clone())
    self._foot_loading_rate_n_s.append(foot_loading_rate.detach().clone())
    self._heel_height_m.append(current_heel_pos[:, 2].detach().clone())
    self._toe_height_m.append(current_toe_pos[:, 2].detach().clone())
    self._heel_vz_m_s.append(heel_vz.detach().clone())
    self._toe_vz_m_s.append(toe_vz.detach().clone())
    self._foot_corner_height_m.append(current_corner_pos[..., 2].detach().clone())
    self._foot_corner_vz_m_s.append(corner_vz.detach().clone())
    self._foot_region_contact.append(current_region_contact.float().detach().clone())
    self._foot_roll_angle_rad.append(roll_angle_2d[selected].detach().clone())
    if current_grid_velocity is not None:
      self._foot_grid_velocity_m_s.append(current_grid_velocity)
      assert current_grid_force is not None
      self._foot_grid_force_n.append(current_grid_force)
    self._capsule_force_n.append(capsule_force_2d[selected].detach().clone())
    self._capsule_contact.append(capsule_contact_2d[selected].float().detach().clone())
    self._capsule_vz_m_s.append(capsule_vz_2d[selected].detach().clone())
    self._capsule_names = capsule_names

    self._previous_action = current_action
    self._previous_foot_contact = current_foot_contact
    self._previous_foot_force_n = current_foot_force
    self._previous_heel_pos_w = current_heel_pos
    self._previous_toe_pos_w = current_toe_pos
    self._previous_foot_corner_pos_w = current_corner_pos
    self._latest_snapshot = QuietTelemetrySnapshot(
      step=len(self._foot_force_n) - 1,
      foot_names=self.proxy.foot_names,
      foot_contact=current_foot_contact.float(),
      foot_force_n=current_foot_force,
      foot_loading_rate_n_s=foot_loading_rate,
      foot_site_vz_m_s=foot_site_vz_2d[selected].detach().clone(),
      heel_vz_m_s=heel_vz.detach().clone(),
      toe_vz_m_s=toe_vz.detach().clone(),
      foot_roll_angle_rad=roll_angle_2d[selected].detach().clone(),
      last_touchdown=self._touchdowns[-1] if self._touchdowns else None,
      foot_region_names=REGION_NAMES,
      foot_region_contact=current_region_contact.float(),
      foot_corner_names=self.proxy.foot_corner_names,
      foot_corner_vz_m_s=corner_vz.detach().clone(),
      last_region_event=self._region_events[-1] if self._region_events else None,
    )

  def to_trace_and_summary(
    self,
    *,
    capsule_layout_xy_m: torch.Tensor | None = None,
    capsule_outline_fromto_xy_m: torch.Tensor | None = None,
    capsule_radius_m: torch.Tensor | None = None,
  ) -> tuple[EpisodeMetricTrace, EpisodeMetricSummary]:
    """Reduce collected telemetry into the canonical trace and summary objects."""

    if not self._foot_force_n:
      raise RuntimeError("Cannot summarize an empty telemetry collector")

    foot_force = torch.stack(self._foot_force_n)
    foot_contact = torch.stack(self._foot_contact)
    foot_vz = torch.stack(self._foot_site_vz_m_s)
    foot_loading = torch.stack(self._foot_loading_rate_n_s)
    capsule_force = torch.stack(self._capsule_force_n)
    capsule_contact = torch.stack(self._capsule_contact)
    capsule_vz = torch.stack(self._capsule_vz_m_s)
    action_rate = torch.cat(self._action_rate_l2)
    command_velocity = torch.stack(self._command_velocity)
    actual_linear_velocity = torch.stack(self._actual_linear_velocity)
    actual_yaw_rate = torch.cat(self._actual_yaw_rate)
    linear_velocity_error = torch.cat(self._linear_velocity_error)
    yaw_rate_error = torch.cat(self._yaw_rate_error)
    heel_height = torch.stack(self._heel_height_m)
    toe_height = torch.stack(self._toe_height_m)
    heel_vz = torch.stack(self._heel_vz_m_s)
    toe_vz = torch.stack(self._toe_vz_m_s)
    corner_height = torch.stack(self._foot_corner_height_m)
    corner_vz = torch.stack(self._foot_corner_vz_m_s)
    region_contact = torch.stack(self._foot_region_contact)
    roll_angle = torch.stack(self._foot_roll_angle_rad)
    foot_grid_velocity = (
      torch.stack(self._foot_grid_velocity_m_s)
      if self._foot_grid_velocity_m_s
      else None
    )
    foot_grid_force = (
      torch.stack(self._foot_grid_force_n)
      if self._foot_grid_force_n
      else None
    )

    touchdown_peak_force = _event_tensor(
      self._touchdowns,
      "peak_force_n",
      device=foot_force.device,
    )
    touchdown_loading_rate = _event_tensor(
      self._touchdowns,
      "loading_rate_n_s",
      device=foot_force.device,
    )
    touchdown_vertical_speed = _event_tensor(
      self._touchdowns,
      "vertical_speed_m_s",
      device=foot_force.device,
    )
    touchdown_heel_vz = _event_tensor(
      self._touchdowns,
      "heel_vz_m_s",
      device=foot_force.device,
    )
    touchdown_toe_vz = _event_tensor(
      self._touchdowns,
      "toe_vz_m_s",
      device=foot_force.device,
    )
    touchdown_roll = _event_tensor(
      self._touchdowns,
      "roll_angle_rad",
      device=foot_force.device,
    )
    touchdown_regions = tuple(event.first_contact_region for event in self._touchdowns)
    region_event_peak_force = _event_tensor(
      self._region_events,
      "peak_force_n",
      device=foot_force.device,
    )
    region_event_loading_rate = _event_tensor(
      self._region_events,
      "loading_rate_n_s",
      device=foot_force.device,
    )
    region_event_vertical_speed = _event_tensor(
      self._region_events,
      "vertical_speed_m_s",
      device=foot_force.device,
    )
    region_event_corner_downward_speed = _event_tensor(
      self._region_events,
      "corner_downward_speed_m_s",
      device=foot_force.device,
    )
    region_event_corner_downward_speeds = _event_matrix_tensor(
      self._region_events,
      "corner_downward_speeds_m_s",
      width=len(self.proxy.foot_corner_names),
      device=foot_force.device,
    )

    peak_force_bw = normalize_force_by_body_weight(
      _safe_mean(touchdown_peak_force),
      self.body_weight_newton,
    )
    loading_rate_bw_s = normalize_force_by_body_weight(
      _safe_mean(touchdown_loading_rate),
      self.body_weight_newton,
    )
    touchdown_peak_force_bw = normalize_force_by_body_weight(
      touchdown_peak_force,
      self.body_weight_newton,
    )
    touchdown_loading_rate_bw_s = normalize_force_by_body_weight(
      touchdown_loading_rate,
      self.body_weight_newton,
    )
    foot_loading_rate_bw_s = normalize_force_by_body_weight(
      foot_loading,
      self.body_weight_newton,
    )

    per_foot_peak_bw = [
      normalize_force_by_body_weight(
        _event_tensor(
          [event for event in self._touchdowns if event.foot_index == foot_index],
          "peak_force_n",
          device=foot_force.device,
        ),
        self.body_weight_newton,
      )
      for foot_index in range(len(self.proxy.foot_names))
    ]
    per_foot_loading_bw = [
      normalize_force_by_body_weight(
        _event_tensor(
          [event for event in self._touchdowns if event.foot_index == foot_index],
          "loading_rate_n_s",
          device=foot_force.device,
        ),
        self.body_weight_newton,
      )
      for foot_index in range(len(self.proxy.foot_names))
    ]
    per_foot_speed = [
      _event_tensor(
        [event for event in self._touchdowns if event.foot_index == foot_index],
        "vertical_speed_m_s",
        device=foot_force.device,
      )
      for foot_index in range(len(self.proxy.foot_names))
    ]

    capsule_peak_bw = _capsule_touchdown_metric(
      capsule_force,
      capsule_contact,
      self.body_weight_newton,
      metric="peak",
      dt=self.dt,
    )
    capsule_loading_bw_s = _capsule_touchdown_metric(
      capsule_force,
      capsule_contact,
      self.body_weight_newton,
      metric="loading",
      dt=self.dt,
    )
    capsule_speed = _capsule_touchdown_speed(capsule_vz, capsule_contact)
    capsule_count = _capsule_touchdown_count(capsule_contact)

    contact_quietness = compute_contact_quietness_score(
      peak_force_bw,
      loading_rate_bw_s,
    )
    body_smoothness = compute_body_smoothness_score(action_rate.mean())
    task_compliance = compute_task_compliance_score(
      linear_velocity_error.mean(),
      yaw_rate_error.mean(),
    )
    total_score = combine_total_score(
      contact_quietness,
      body_smoothness,
      task_compliance,
    )
    peak_asymmetry = torch.abs(
      _safe_mean(per_foot_peak_bw[0]) - _safe_mean(per_foot_peak_bw[1])
    ).reshape(1)

    trace = EpisodeMetricTrace(
      foot_z_force_n=foot_force,
      foot_contact_flag=foot_contact,
      foot_vertical_velocity_m_s=foot_vz,
      capsule_names=self._capsule_names,
      capsule_layout_xy_m=(
        capsule_layout_xy_m.detach().clone().to(device=foot_force.device, dtype=foot_force.dtype)
        if capsule_layout_xy_m is not None
        else torch.zeros((len(self._capsule_names), 2), dtype=foot_force.dtype, device=foot_force.device)
      ),
      capsule_outline_fromto_xy_m=(
        capsule_outline_fromto_xy_m.detach().clone().to(device=foot_force.device, dtype=foot_force.dtype)
        if capsule_outline_fromto_xy_m is not None
        else torch.zeros((0, 2, 2), dtype=foot_force.dtype, device=foot_force.device)
      ),
      capsule_radius_m=(
        capsule_radius_m.detach().clone().to(device=foot_force.device, dtype=foot_force.dtype)
        if capsule_radius_m is not None
        else torch.zeros((0,), dtype=foot_force.dtype, device=foot_force.device)
      ),
      capsule_z_force_n=capsule_force,
      capsule_contact_flag=capsule_contact,
      capsule_vertical_velocity_m_s=capsule_vz,
      peak_force_bw=peak_force_bw.reshape(1),
      loading_rate_bw_s=loading_rate_bw_s.reshape(1),
      touchdown_peak_force_bw=touchdown_peak_force_bw.flatten(),
      touchdown_loading_rate_bw_s=touchdown_loading_rate_bw_s.flatten(),
      touchdown_vertical_speed_m_s=touchdown_vertical_speed.flatten(),
      left_touchdown_peak_force_bw=per_foot_peak_bw[0].flatten(),
      right_touchdown_peak_force_bw=per_foot_peak_bw[1].flatten(),
      left_touchdown_loading_rate_bw_s=per_foot_loading_bw[0].flatten(),
      right_touchdown_loading_rate_bw_s=per_foot_loading_bw[1].flatten(),
      left_touchdown_vertical_speed_m_s=per_foot_speed[0].flatten(),
      right_touchdown_vertical_speed_m_s=per_foot_speed[1].flatten(),
      touchdown_peak_asymmetry_bw=peak_asymmetry,
      capsule_touchdown_peak_force_bw=capsule_peak_bw.flatten(),
      capsule_touchdown_loading_rate_bw_s=capsule_loading_bw_s.flatten(),
      capsule_touchdown_vertical_speed_m_s=capsule_speed.flatten(),
      capsule_touchdown_count=capsule_count.flatten(),
      action_rate_l2=action_rate.flatten(),
      command_velocity=command_velocity,
      actual_linear_velocity=actual_linear_velocity,
      actual_yaw_rate=actual_yaw_rate.flatten(),
      linear_velocity_error=linear_velocity_error.flatten(),
      yaw_rate_error=yaw_rate_error.flatten(),
      contact_quietness_score=contact_quietness.reshape(1),
      foot_grid_names=self._foot_grid_names,
      foot_grid_shape=self.proxy.foot_grid_shape,
      foot_grid_local_xy_m=self._foot_grid_local_xy_m,
      foot_grid_velocity_m_s=foot_grid_velocity,
      foot_grid_force_n=foot_grid_force,
      foot_loading_rate_n_s=foot_loading,
      foot_loading_rate_bw_s=foot_loading_rate_bw_s,
      heel_vertical_velocity_m_s=heel_vz,
      toe_vertical_velocity_m_s=toe_vz,
      heel_height_m=heel_height,
      toe_height_m=toe_height,
      foot_roll_angle_rad=roll_angle,
      touchdown_heel_vz_m_s=touchdown_heel_vz.flatten(),
      touchdown_toe_vz_m_s=touchdown_toe_vz.flatten(),
      touchdown_roll_angle_rad=touchdown_roll.flatten(),
      touchdown_first_contact_region=touchdown_regions,
      foot_region_names=REGION_NAMES,
      foot_region_contact_flag=region_contact,
      foot_corner_names=self.proxy.foot_corner_names,
      foot_corner_height_m=corner_height,
      foot_corner_vertical_velocity_m_s=corner_vz,
      region_event_type=tuple(event.event_type for event in self._region_events),
      region_event_stance_id=torch.tensor(
        [event.stance_id for event in self._region_events],
        dtype=torch.int64,
        device=foot_force.device,
      ),
      region_event_step=torch.tensor(
        [event.step for event in self._region_events],
        dtype=torch.int64,
        device=foot_force.device,
      ),
      region_event_substep_index=torch.tensor(
        [event.substep_index for event in self._region_events],
        dtype=torch.int64,
        device=foot_force.device,
      ),
      region_event_time_offset_s=torch.tensor(
        [event.time_offset_s for event in self._region_events],
        dtype=foot_force.dtype,
        device=foot_force.device,
      ),
      region_event_foot_index=torch.tensor(
        [event.foot_index for event in self._region_events],
        dtype=torch.int64,
        device=foot_force.device,
      ),
      region_event_foot=tuple(event.foot_name for event in self._region_events),
      region_event_regions=tuple("+".join(event.regions) for event in self._region_events),
      region_event_peak_force_bw=normalize_force_by_body_weight(
        region_event_peak_force.flatten(),
        self.body_weight_newton,
      ),
      region_event_loading_rate_bw_s=normalize_force_by_body_weight(
        region_event_loading_rate.flatten(),
        self.body_weight_newton,
      ),
      region_event_vertical_speed_m_s=region_event_vertical_speed.flatten(),
      region_event_corner_downward_speed_m_s=region_event_corner_downward_speed.flatten(),
      region_event_corner_downward_speeds_m_s=region_event_corner_downward_speeds,
    )
    summary = EpisodeMetricSummary(
      contact_quietness=float(contact_quietness.item()),
      body_smoothness=float(body_smoothness.item()),
      task_compliance=float(task_compliance.item()),
      total_score=float(total_score.item()),
    )
    return trace, summary


def _as_1d(value: torch.Tensor) -> torch.Tensor:
  if value.ndim == 0:
    return value.reshape(1)
  if value.ndim == 1:
    return value
  if value.ndim == 2 and value.shape[1] == 1:
    return value[:, 0]
  raise ValueError(f"Expected 1D tensor-compatible value, got shape {tuple(value.shape)}")


def _as_2d(value: torch.Tensor) -> torch.Tensor:
  if value.ndim == 1:
    return value.unsqueeze(0)
  if value.ndim == 2:
    return value
  raise ValueError(f"Expected 2D tensor-compatible value, got shape {tuple(value.shape)}")


def _as_3d(value: torch.Tensor) -> torch.Tensor:
  if value.ndim == 2:
    return value.unsqueeze(0)
  if value.ndim == 3:
    return value
  raise ValueError(f"Expected 3D tensor-compatible value, got shape {tuple(value.shape)}")


def _as_4d(value: torch.Tensor) -> torch.Tensor:
  if value.ndim == 3:
    return value.unsqueeze(0)
  if value.ndim == 4:
    return value
  raise ValueError(f"Expected 4D tensor-compatible value, got shape {tuple(value.shape)}")


def _as_substep_2d(value: torch.Tensor) -> torch.Tensor:
  """Normalize substep tensors to [S, F] for the first evaluated environment."""

  if value.ndim == 2:
    return value
  if value.ndim == 3:
    return value[:, 0]
  raise ValueError(
    f"Expected [S, F] or [S, B, F] substep tensor, got shape {tuple(value.shape)}"
  )


def _as_substep_3d(value: torch.Tensor) -> torch.Tensor:
  """Normalize substep tensors to [S, F, D] for the first evaluated environment."""

  if value.ndim == 3:
    return value
  if value.ndim == 4:
    return value[:, 0]
  raise ValueError(
    f"Expected [S, F, D] or [S, B, F, D] substep tensor, got shape {tuple(value.shape)}"
  )


def _event_tensor(
  events: list,
  attr: str,
  *,
  device: torch.device,
) -> torch.Tensor:
  if not events:
    return torch.zeros(0, dtype=torch.float32, device=device)
  return torch.stack([getattr(event, attr).to(device=device).float() for event in events])


def _event_matrix_tensor(
  events: list,
  attr: str,
  *,
  width: int,
  device: torch.device,
) -> torch.Tensor:
  if not events:
    return torch.zeros((0, width), dtype=torch.float32, device=device)
  rows = []
  for event in events:
    value = getattr(event, attr)
    if value is None:
      rows.append(torch.zeros(width, dtype=torch.float32, device=device))
    else:
      rows.append(value.to(device=device).float().reshape(width))
  return torch.stack(rows)


def _safe_mean(values: torch.Tensor) -> torch.Tensor:
  if values.numel() == 0:
    return torch.zeros((), dtype=torch.float32, device=values.device)
  return values.float().mean()


def _first_contact_region(heel_z: float, toe_z: float, tolerance_m: float = 0.005) -> FootRegion:
  if abs(heel_z - toe_z) <= tolerance_m:
    return "flat"
  if heel_z < toe_z:
    return "heel"
  if toe_z < heel_z:
    return "toe"
  return "unknown"


def _capsule_touchdown_metric(
  capsule_force: torch.Tensor,
  capsule_contact: torch.Tensor,
  body_weight_newton: float,
  *,
  metric: Literal["peak", "loading"],
  dt: float,
) -> torch.Tensor:
  per_capsule: list[torch.Tensor] = []
  loading = positive_loading_rate(capsule_force, dt=dt)
  for capsule_idx in range(capsule_force.shape[1]):
    touchdowns = _rising_edge_indices(capsule_contact[:, capsule_idx].bool())
    if not touchdowns:
      per_capsule.append(torch.zeros((), dtype=capsule_force.dtype, device=capsule_force.device))
      continue
    source = capsule_force if metric == "peak" else loading
    per_capsule.append(source[touchdowns, capsule_idx].float().mean())
  return normalize_force_by_body_weight(torch.stack(per_capsule), body_weight_newton)


def _capsule_touchdown_speed(
  capsule_vz: torch.Tensor,
  capsule_contact: torch.Tensor,
) -> torch.Tensor:
  per_capsule: list[torch.Tensor] = []
  for capsule_idx in range(capsule_vz.shape[1]):
    touchdowns = _rising_edge_indices(capsule_contact[:, capsule_idx].bool())
    if not touchdowns:
      per_capsule.append(torch.zeros((), dtype=capsule_vz.dtype, device=capsule_vz.device))
      continue
    per_capsule.append(torch.clamp(-capsule_vz[touchdowns, capsule_idx], min=0.0).float().mean())
  return torch.stack(per_capsule)


def _capsule_touchdown_count(capsule_contact: torch.Tensor) -> torch.Tensor:
  counts = [
    len(_rising_edge_indices(capsule_contact[:, capsule_idx].bool()))
    for capsule_idx in range(capsule_contact.shape[1])
  ]
  return torch.tensor(counts, dtype=torch.float32, device=capsule_contact.device)


def _rising_edge_indices(contact: torch.Tensor) -> list[int]:
  indices: list[int] = []
  previous = False
  for idx, value in enumerate(contact.tolist()):
    current = bool(value)
    if current and not previous:
      indices.append(idx)
    previous = current
  return indices
