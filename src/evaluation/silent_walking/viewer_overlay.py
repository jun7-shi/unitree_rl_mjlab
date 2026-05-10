"""Realtime viewer overlays for silent-walking telemetry."""

from __future__ import annotations

import html
import traceback
from typing import Any

import mujoco
import torch

from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer
from mjlab.viewer.base import VerbosityLevel

from .contact_backends import (
  extract_capsule_contact_forces,
  extract_capsule_vertical_velocities,
  extract_feet_contact_flags,
  extract_feet_net_forces,
  extract_foot_region_contact_flags,
  extract_foot_vertical_velocities,
)
from .robots import get_foot_proxy_spec, get_robot_spec
from .runner import _extract_virtual_foot_corner_points, _extract_virtual_heel_toe_points
from .telemetry import QuietTelemetrySnapshot, SilentTelemetryCollector


class _ActionRecordingPolicy:
  """Policy wrapper that exposes the last action for telemetry."""

  def __init__(self, policy) -> None:
    self._policy = policy
    self.last_action: torch.Tensor | None = None

  def __call__(self, obs) -> torch.Tensor:
    action = self._policy(obs)
    self.last_action = action.detach().clone()
    return action

  def reset(self) -> None:
    reset = getattr(self._policy, "reset", None)
    if reset is not None:
      reset()
    self.last_action = None


class _QuietTelemetryMixin:
  """Shared telemetry recording behavior for native and Viser viewers."""

  _quiet_policy: _ActionRecordingPolicy

  def _init_quiet_telemetry(self, robot_name: str) -> None:
    self._quiet_robot_name = robot_name
    self._quiet_spec = get_robot_spec(robot_name)
    self._quiet_proxy = get_foot_proxy_spec(robot_name)
    self._quiet_foot_body_ids: list[int] | None = None
    self._quiet_collector = SilentTelemetryCollector(
      robot_name=robot_name,
      dt=self.env.unwrapped.step_dt,
      body_weight_newton=self._quiet_spec.mass_normalization,
    )

  def _reset_quiet_telemetry(self) -> None:
    self._init_quiet_telemetry(self._quiet_robot_name)

  def _execute_step(self) -> bool:
    ok = super()._execute_step()
    if not ok:
      return False
    try:
      self._record_quiet_sample()
      return True
    except Exception:
      self._last_error = traceback.format_exc()
      self.log(
        f"[ERROR] Exception during quiet telemetry:\n{self._last_error}",
        VerbosityLevel.SILENT,
      )
      self.pause()
      return False

  def reset_environment(self) -> None:
    super().reset_environment()
    self._reset_quiet_telemetry()

  def _record_quiet_sample(self) -> None:
    env = self.env.unwrapped
    robot = env.scene["robot"]
    if self._quiet_foot_body_ids is None:
      body_ids, body_names = robot.find_bodies(
        self._quiet_proxy.foot_body_names,
        preserve_order=True,
      )
      if tuple(body_names) != self._quiet_proxy.foot_body_names:
        raise RuntimeError("Foot body ordering does not match telemetry proxy spec")
      self._quiet_foot_body_ids = list(body_ids)

    action = self._quiet_policy.last_action
    if action is None:
      return

    capsule_names, capsule_force, capsule_contact = extract_capsule_contact_forces(
      env,
      robot_name=self._quiet_robot_name,
    )
    capsule_velocity_names, capsule_vz = extract_capsule_vertical_velocities(
      env,
      robot_name=self._quiet_robot_name,
    )
    if capsule_names != capsule_velocity_names:
      raise RuntimeError("Capsule force and velocity ordering mismatch")

    heel_pos_w, toe_pos_w, foot_roll_angle = _extract_virtual_heel_toe_points(
      env,
      self._quiet_robot_name,
      self._quiet_foot_body_ids,
    )
    foot_corner_pos_w = _extract_virtual_foot_corner_points(
      env,
      self._quiet_robot_name,
      self._quiet_foot_body_ids,
    )
    foot_region_contact = extract_foot_region_contact_flags(
      env,
      self._quiet_robot_name,
      self._quiet_foot_body_ids,
    )
    self._quiet_collector.record_sample(
      action=action,
      command_velocity=_get_velocity_command(env, robot),
      actual_linear_velocity=robot.data.root_link_lin_vel_b[:, :2],
      actual_yaw_rate=robot.data.root_link_ang_vel_b[:, 2],
      foot_force_n=extract_feet_net_forces(env, self._quiet_robot_name)[..., 2].abs(),
      foot_contact=extract_feet_contact_flags(env, self._quiet_robot_name),
      foot_site_vz_m_s=extract_foot_vertical_velocities(env, self._quiet_robot_name),
      capsule_names=capsule_names,
      capsule_force_n=capsule_force,
      capsule_contact=capsule_contact,
      capsule_vz_m_s=capsule_vz,
      heel_pos_w=heel_pos_w,
      toe_pos_w=toe_pos_w,
      foot_roll_angle_rad=foot_roll_angle,
      foot_corner_pos_w=foot_corner_pos_w,
      foot_region_contact=foot_region_contact,
    )

  @property
  def quiet_snapshot(self) -> QuietTelemetrySnapshot | None:
    return self._quiet_collector.latest_snapshot


class QuietNativeMujocoViewer(_QuietTelemetryMixin, NativeMujocoViewer):
  """Native MuJoCo viewer with live quiet-walking HUD rows."""

  def __init__(self, env, policy, *, robot_name: str = "g1", **kwargs) -> None:
    self._quiet_policy = _ActionRecordingPolicy(policy)
    super().__init__(env, self._quiet_policy, **kwargs)
    self._init_quiet_telemetry(robot_name)

  def _set_status_overlay(self, viewer) -> None:
    status = self.get_status()
    capped = " [CAPPED]" if status.capped else ""
    base_labels = "Env\nStep\nStatus\nSpeed\nTarget RT\nActual RT"
    base_values = (
      f"{self.env_idx + 1}/{self.env.num_envs}\n"
      f"{status.step_count}\n"
      f"{'PAUSED' if status.paused else 'RUNNING'}{capped}\n"
      f"{status.speed_label}\n"
      f"{status.target_realtime:.2f}x\n"
      f"{status.actual_realtime:.2f}x ({status.smoothed_fps:.0f} FPS)"
    )
    quiet_labels, quiet_values = format_quiet_native_rows(
      self.quiet_snapshot,
      body_weight_newton=self._quiet_spec.mass_normalization,
    )
    overlay = (
      mujoco.mjtFontScale.mjFONTSCALE_150.value,
      mujoco.mjtGridPos.mjGRID_TOPLEFT.value,
      f"{base_labels}\n{quiet_labels}",
      f"{base_values}\n{quiet_values}",
    )
    viewer.set_texts(overlay)


class QuietViserPlayViewer(_QuietTelemetryMixin, ViserPlayViewer):
  """Viser viewer with a live quiet-walking metrics panel."""

  def __init__(self, env, policy, *, robot_name: str = "g1", **kwargs) -> None:
    self._quiet_policy = _ActionRecordingPolicy(policy)
    self._quiet_html_handle = None
    super().__init__(env, self._quiet_policy, **kwargs)
    self._init_quiet_telemetry(robot_name)

  def setup(self) -> None:
    super().setup()
    with self._server.gui.add_folder("Quiet Metrics", expand_by_default=True):
      self._quiet_html_handle = self._server.gui.add_html(
        format_quiet_html(None, self._quiet_spec.mass_normalization)
      )

  def sync_env_to_viewer(self) -> None:
    super().sync_env_to_viewer()
    if self._quiet_html_handle is not None and self._counter % 5 == 0:
      self._quiet_html_handle.content = format_quiet_html(
        self.quiet_snapshot,
        self._quiet_spec.mass_normalization,
      )


def format_quiet_native_rows(
  snapshot: QuietTelemetrySnapshot | None,
  body_weight_newton: float,
) -> tuple[str, str]:
  """Format quiet telemetry for the native MuJoCo text overlay."""

  if snapshot is None:
    return "Quiet", "waiting for first step"

  labels: list[str] = []
  values: list[str] = []
  for foot_idx, foot_name in enumerate(snapshot.foot_names):
    labels.append(f"Quiet {foot_name}")
    values.append(
      _format_foot_line(
        force_n=snapshot.foot_force_n[foot_idx],
        loading_rate_n_s=snapshot.foot_loading_rate_n_s[foot_idx],
        foot_vz_m_s=snapshot.foot_site_vz_m_s[foot_idx],
        heel_vz_m_s=snapshot.heel_vz_m_s[foot_idx],
        toe_vz_m_s=snapshot.toe_vz_m_s[foot_idx],
        roll_angle_rad=snapshot.foot_roll_angle_rad[foot_idx],
        body_weight_newton=body_weight_newton,
        foot_region_names=snapshot.foot_region_names,
        foot_region_contact=(
          snapshot.foot_region_contact[foot_idx]
          if snapshot.foot_region_contact is not None
          else None
        ),
        foot_corner_names=snapshot.foot_corner_names,
        foot_corner_vz_m_s=(
          snapshot.foot_corner_vz_m_s[foot_idx]
          if snapshot.foot_corner_vz_m_s is not None
          else None
        ),
      )
    )

  labels.append("Last TD")
  values.append(_format_touchdown_line(snapshot, body_weight_newton))
  labels.append("Last Region")
  values.append(_format_region_event_line(snapshot, body_weight_newton))
  return "\n".join(labels), "\n".join(values)


def format_quiet_html(
  snapshot: QuietTelemetrySnapshot | None,
  body_weight_newton: float,
) -> str:
  """Format quiet telemetry for Viser's HTML GUI."""

  if snapshot is None:
    return (
      "<div style='font-size:0.85em;line-height:1.35;padding:0 1em 0.5em 1em;'>"
      "<strong>Quiet Metrics:</strong> waiting for first step"
      "</div>"
    )

  rows = []
  for foot_idx, foot_name in enumerate(snapshot.foot_names):
    rows.append(
      "<tr>"
      f"<td>{html.escape(foot_name)}</td>"
      f"<td>{_float(snapshot.foot_force_n[foot_idx]) / body_weight_newton:.2f}</td>"
      f"<td>{_float(snapshot.foot_loading_rate_n_s[foot_idx]) / body_weight_newton:.2f}</td>"
      f"<td>{_float(snapshot.foot_site_vz_m_s[foot_idx]):.2f}</td>"
      f"<td>{_float(snapshot.heel_vz_m_s[foot_idx]):.2f}</td>"
      f"<td>{_float(snapshot.toe_vz_m_s[foot_idx]):.2f}</td>"
      f"<td>{_float(snapshot.foot_roll_angle_rad[foot_idx]):.2f}</td>"
      f"<td>{html.escape(_format_active_regions(snapshot, foot_idx))}</td>"
      f"<td>{html.escape(_format_corner_vz(snapshot, foot_idx))}</td>"
      "</tr>"
    )

  touchdown = html.escape(_format_touchdown_line(snapshot, body_weight_newton))
  region_event = html.escape(_format_region_event_line(snapshot, body_weight_newton))
  return f"""
    <div style="font-size:0.85em;line-height:1.35;padding:0 1em 0.5em 1em;">
      <strong>Quiet Metrics</strong>
      <table style="width:100%;border-collapse:collapse;margin-top:0.35em;">
        <thead>
          <tr>
            <th align="left">Foot</th>
            <th align="right">F/BW</th>
            <th align="right">LR/BW/s</th>
            <th align="right">Vz</th>
            <th align="right">Heel Vz</th>
            <th align="right">Toe Vz</th>
            <th align="right">Roll</th>
            <th align="right">Regions</th>
            <th align="right">Corner Vz</th>
          </tr>
        </thead>
        <tbody>{''.join(rows)}</tbody>
      </table>
      <div style="margin-top:0.35em;"><strong>Last TD:</strong> {touchdown}</div>
      <div style="margin-top:0.20em;"><strong>Last Region:</strong> {region_event}</div>
    </div>
  """


def _get_velocity_command(env, robot) -> torch.Tensor:
  try:
    command = env.command_manager.get_command("twist")
  except Exception:
    command = None
  if command is not None:
    return command
  return torch.zeros(
    (env.num_envs, 3),
    dtype=robot.data.root_link_lin_vel_b.dtype,
    device=robot.data.root_link_lin_vel_b.device,
  )


def _format_foot_line(
  *,
  force_n: torch.Tensor,
  loading_rate_n_s: torch.Tensor,
  foot_vz_m_s: torch.Tensor,
  heel_vz_m_s: torch.Tensor,
  toe_vz_m_s: torch.Tensor,
  roll_angle_rad: torch.Tensor,
  body_weight_newton: float,
  foot_region_names: tuple[str, ...] = (),
  foot_region_contact: torch.Tensor | None = None,
  foot_corner_names: tuple[str, ...] = (),
  foot_corner_vz_m_s: torch.Tensor | None = None,
) -> str:
  del foot_corner_names
  return (
    f"F {_float(force_n) / body_weight_newton:.2f}BW | "
    f"LR {_float(loading_rate_n_s) / body_weight_newton:.2f}BW/s | "
    f"Vz {_float(foot_vz_m_s):.2f} | "
    f"H/T {_float(heel_vz_m_s):.2f}/{_float(toe_vz_m_s):.2f} | "
    f"roll {_float(roll_angle_rad):.2f} | "
    f"R {_format_region_tensor(foot_region_names, foot_region_contact)} | "
    f"C {_format_corner_tensor(foot_corner_vz_m_s)}"
  )


def _format_touchdown_line(
  snapshot: QuietTelemetrySnapshot,
  body_weight_newton: float,
) -> str:
  event = snapshot.last_touchdown
  if event is None:
    return "none"
  return (
    f"{event.foot_name} step {event.step} "
    f"{event.first_contact_region} "
    f"F {_float(event.peak_force_n) / body_weight_newton:.2f}BW "
    f"LR {_float(event.loading_rate_n_s) / body_weight_newton:.2f}BW/s "
    f"Vz {_float(event.vertical_speed_m_s):.2f} "
    f"H/T {_float(event.heel_vz_m_s):.2f}/{_float(event.toe_vz_m_s):.2f} "
    f"roll {_float(event.roll_angle_rad):.2f}"
  )


def _format_region_event_line(
  snapshot: QuietTelemetrySnapshot,
  body_weight_newton: float,
) -> str:
  event = snapshot.last_region_event
  if event is None:
    return "none"
  regions = "+".join(event.regions)
  return (
    f"{event.foot_name} step {event.step} "
    f"{event.event_type} {regions} "
    f"F {_float(event.peak_force_n) / body_weight_newton:.2f}BW "
    f"LR {_float(event.loading_rate_n_s) / body_weight_newton:.2f}BW/s "
    f"Vz {_float(event.vertical_speed_m_s):.2f} "
    f"C {_float(event.corner_downward_speed_m_s):.2f}"
  )


def _format_active_regions(snapshot: QuietTelemetrySnapshot, foot_idx: int) -> str:
  if snapshot.foot_region_contact is None:
    return "none"
  return _format_region_tensor(
    snapshot.foot_region_names,
    snapshot.foot_region_contact[foot_idx],
  )


def _format_corner_vz(snapshot: QuietTelemetrySnapshot, foot_idx: int) -> str:
  if snapshot.foot_corner_vz_m_s is None:
    return ""
  return _format_corner_tensor(snapshot.foot_corner_vz_m_s[foot_idx])


def _format_region_tensor(
  region_names: tuple[str, ...],
  region_contact: torch.Tensor | None,
) -> str:
  if region_contact is None or not region_names:
    return "none"
  active = [
    region_names[idx]
    for idx in range(min(len(region_names), int(region_contact.numel())))
    if _float(region_contact[idx]) > 0.5
  ]
  return "+".join(active) if active else "none"


def _format_corner_tensor(corner_vz_m_s: torch.Tensor | None) -> str:
  if corner_vz_m_s is None or corner_vz_m_s.numel() == 0:
    return ""
  return "/".join(f"{_float(value):.2f}" for value in corner_vz_m_s)


def _float(value: torch.Tensor | float | int) -> float:
  if isinstance(value, torch.Tensor):
    return float(value.detach().cpu().reshape(()).item())
  return float(value)
