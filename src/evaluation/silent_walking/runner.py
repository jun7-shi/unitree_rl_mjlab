"""Runtime runner for silent walking evaluation."""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

import torch

_RUNTIME_CACHE_ROOT = Path(
  os.environ.get("SILENT_WALKING_CACHE_ROOT", "/tmp/unitree_rl_mjlab/silent_walking")
)
_MPL_CACHE_DIR = _RUNTIME_CACHE_ROOT / "matplotlib"
_WARP_CACHE_DIR = _RUNTIME_CACHE_ROOT / "warp"

for cache_dir in (_MPL_CACHE_DIR, _WARP_CACHE_DIR):
  cache_dir.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE_DIR))
os.environ.setdefault("WARP_CACHE_PATH", str(_WARP_CACHE_DIR))

import mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.registry import load_env_cfg
from mjlab.tasks.registry import load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
import src.tasks  # noqa: F401

from .contact_backends import (
  extract_capsule_contact_point_forces,
  extract_capsule_contact_forces,
  extract_capsule_layout_positions,
  extract_capsule_vertical_velocities,
  extract_feet_contact_flags,
  extract_feet_net_forces,
  extract_foot_grid_velocities,
  extract_foot_region_contact_flags,
  extract_foot_vertical_velocities,
  supports_contact_backend,
)
from .foot_grid import distribute_contact_forces_to_foot_grid
from .metrics import (
  combine_total_score,
  compute_body_smoothness_score,
  compute_contact_quietness_score,
  compute_loading_rate,
  compute_task_compliance_score,
  normalize_force_by_body_weight,
)
from .policy_adapters import PolicyAdapter, adapt_policy_path, is_checkpoint_path
from .regional_contact import virtual_foot_corner_points
from .regional_contact import world_to_local_foot_points
from .robots import get_foot_proxy_spec, get_robot_spec
from .telemetry import (
  SilentTelemetryCollector,
  foot_roll_angle_from_points,
  virtual_heel_toe_points,
)
from .types import EpisodeMetricSummary, EpisodeMetricTrace


@dataclass(frozen=True, slots=True)
class SilentEvalResult:
  """Evaluation result bundle for one policy rollout."""

  robot_name: str
  task_id: str
  steps: int
  step_dt: float
  trace: EpisodeMetricTrace
  summary: EpisodeMetricSummary


@dataclass(slots=True)
class _SubstepTelemetryWindow:
  """Substep telemetry collected inside one policy/control step."""

  foot_force_n: list[torch.Tensor]
  foot_site_vz_m_s: list[torch.Tensor]
  foot_region_contact: list[torch.Tensor]
  corner_downward_speeds_m_s: list[torch.Tensor]

  @classmethod
  def empty(cls) -> "_SubstepTelemetryWindow":
    return cls([], [], [], [])

  def append(
    self,
    *,
    foot_force_n: torch.Tensor,
    foot_site_vz_m_s: torch.Tensor,
    foot_region_contact: torch.Tensor,
    corner_downward_speeds_m_s: torch.Tensor,
  ) -> None:
    self.foot_force_n.append(foot_force_n.detach().clone())
    self.foot_site_vz_m_s.append(foot_site_vz_m_s.detach().clone())
    self.foot_region_contact.append(foot_region_contact.detach().clone())
    self.corner_downward_speeds_m_s.append(corner_downward_speeds_m_s.detach().clone())

  def as_record_kwargs(self) -> dict[str, torch.Tensor] | dict[str, None]:
    if not self.foot_region_contact:
      return {}
    return {
      "substep_foot_force_n": torch.stack(self.foot_force_n),
      "substep_foot_site_vz_m_s": torch.stack(self.foot_site_vz_m_s),
      "substep_foot_region_contact": torch.stack(self.foot_region_contact),
      "substep_corner_downward_speeds_m_s": torch.stack(self.corner_downward_speeds_m_s),
    }


class _CallablePolicyAdapter:
  """Wrap a plain callable policy with the evaluator adapter interface."""

  def __init__(self, policy_callable, use_raw_obs: bool = False):
    self._policy_callable = policy_callable
    self.use_raw_obs = use_raw_obs

  def reset(self) -> None:
    return None

  def act(self, obs) -> torch.Tensor:
    with torch.inference_mode():
      return self._policy_callable(obs)


def _validate_runtime_support(robot_name: str) -> None:
  """Fail fast when the selected robot is not ready for runtime evaluation."""

  spec = get_robot_spec(robot_name)
  if spec.asset_status != "ready":
    raise FileNotFoundError(
      f"Robot '{spec.name}' assets are not available at {spec.asset_path}"
    )
  if not supports_contact_backend(robot_name):
    raise NotImplementedError(
      f"Silent walking runtime evaluation is not wired yet for robot '{spec.name}'"
    )


def _ensure_capsule_contact_sensor(env_cfg, robot_name: str) -> None:
  """Attach a per-capsule ground contact sensor to the evaluation env config."""

  spec = get_robot_spec(robot_name)
  existing_sensors = cfg_sensors = env_cfg.scene.sensors or ()
  existing_names = {sensor.name for sensor in cfg_sensors}

  new_sensors: list[ContactSensorCfg] = []
  if "foot_capsule_ground_contact" not in existing_names:
    new_sensors.append(
      ContactSensorCfg(
        name="foot_capsule_ground_contact",
        primary=ContactMatch(
          mode="geom",
          pattern=spec.foot_collision_geom_names,
          entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
      )
    )
  if "foot_capsule_ground_contact_points" not in existing_names:
    new_sensors.append(
      ContactSensorCfg(
        name="foot_capsule_ground_contact_points",
        primary=ContactMatch(
          mode="geom",
          pattern=spec.foot_collision_geom_names,
          entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force", "pos", "normal", "tangent"),
        reduce="maxforce",
        num_slots=1,
        global_frame=True,
      )
    )
  if not new_sensors:
    return

  env_cfg.scene.sensors = existing_sensors + tuple(new_sensors)


def _disable_foot_phase_observation(env_cfg) -> None:
  """Remove motion foot phase terms for legacy tracking checkpoints."""

  for group_name in ("actor", "critic"):
    group = env_cfg.observations.get(group_name)
    if group is not None:
      group.terms.pop("motion_foot_phase", None)


def _configure_motion_file(env_cfg, motion_file: str | None) -> None:
  """Inject a local tracking motion file when evaluating tracking policies."""

  if motion_file is None:
    return
  motion_cmd = env_cfg.commands.get("motion")
  if motion_cmd is None or not hasattr(motion_cmd, "motion_file"):
    raise ValueError("--motion-file requires a task with a 'motion' command")
  motion_cmd.motion_file = motion_file


def _configure_fixed_velocity_command(
  env_cfg,
  fixed_command: tuple[float, float, float] | None,
) -> None:
  """Force a velocity task to use one constant command throughout evaluation."""

  if fixed_command is None:
    return
  twist_cmd = env_cfg.commands.get("twist")
  if twist_cmd is None or not hasattr(twist_cmd, "ranges"):
    raise ValueError("--fixed-command requires a task with a 'twist' velocity command")
  lin_x, lin_y, yaw = fixed_command
  twist_cmd.ranges.lin_vel_x = (lin_x, lin_x)
  twist_cmd.ranges.lin_vel_y = (lin_y, lin_y)
  twist_cmd.ranges.ang_vel_z = (yaw, yaw)
  if hasattr(twist_cmd, "rel_standing_envs"):
    twist_cmd.rel_standing_envs = 0.0
  if hasattr(twist_cmd, "heading_command"):
    twist_cmd.heading_command = False
  if getattr(twist_cmd.ranges, "heading", None) is not None:
    twist_cmd.ranges.heading = None


def _get_eval_command_velocity(env: ManagerBasedRlEnv, robot) -> torch.Tensor:
  """Return velocity command when present; zeros for non-velocity evaluation tasks."""

  try:
    return env.command_manager.get_command("twist")
  except KeyError:
    return torch.zeros(
      (env.num_envs, 3),
      dtype=robot.data.root_link_lin_vel_b.dtype,
      device=robot.data.root_link_lin_vel_b.device,
    )


def _unwrap_obs(obs: Any) -> Any:
  """Strip wrapper tuples from env observations while preserving inner structure."""

  if isinstance(obs, tuple):
    if not obs:
      raise TypeError("Received empty observation tuple.")
    return _unwrap_obs(obs[0])
  return obs


def _extract_actor_obs(obs: Any) -> torch.Tensor:
  """Normalize env observation outputs to a single actor observation tensor."""

  obs = _unwrap_obs(obs)
  if isinstance(obs, dict):
    actor_obs = obs.get("policy") or obs.get("actor")
    if isinstance(actor_obs, torch.Tensor):
      return actor_obs
  if isinstance(obs, torch.Tensor):
    return obs
  raise TypeError(f"Unsupported observation type: {type(obs)!r}")


def _step_env(env: ManagerBasedRlEnv, action: torch.Tensor) -> tuple[Any, torch.Tensor]:
  """Step the environment and return the raw and actor observations."""

  step_result = env.step(action)
  if not isinstance(step_result, tuple):
    raise TypeError(f"Unsupported env.step return type: {type(step_result)!r}")
  if len(step_result) == 4:
    obs, _, _, _ = step_result
  elif len(step_result) == 5:
    obs, _, _, _, _ = step_result
  else:
    raise TypeError(f"Unsupported env.step tuple length: {len(step_result)}")
  return obs, _extract_actor_obs(obs)


def _step_env_with_substep_telemetry(
  env: ManagerBasedRlEnv,
  action: torch.Tensor,
  *,
  robot_name: str,
  foot_body_ids: list[int],
) -> tuple[Any, torch.Tensor, _SubstepTelemetryWindow]:
  """Step the env while sampling contact telemetry at physics-substep resolution.

  This mirrors ``ManagerBasedRlEnv.step`` but inserts an evaluation-only sample
  after every physics step. The policy still runs at the normal control rate.
  """

  substeps = _SubstepTelemetryWindow.empty()
  env.action_manager.process_action(action.to(env.device))

  for _ in range(env.cfg.decimation):
    env._sim_step_counter += 1
    env.action_manager.apply_action()
    env.scene.write_data_to_sim()
    env.sim.step()
    env.sim.forward()
    env.scene.update(dt=env.physics_dt)
    _append_substep_telemetry(
      substeps,
      env=env,
      robot_name=robot_name,
      foot_body_ids=foot_body_ids,
    )

  env.episode_length_buf += 1
  env.common_step_counter += 1

  env.reset_buf = env.termination_manager.compute()
  env.reset_terminated = env.termination_manager.terminated
  env.reset_time_outs = env.termination_manager.time_outs

  env.reward_buf = env.reward_manager.compute(dt=env.step_dt)
  env.metrics_manager.compute()

  reset_env_ids = env.reset_buf.nonzero(as_tuple=False).squeeze(-1)
  if len(reset_env_ids) > 0:
    env._reset_idx(reset_env_ids)
    env.scene.write_data_to_sim()

  env.sim.forward()
  env.command_manager.compute(dt=env.step_dt)

  if "step" in env.event_manager.available_modes:
    env.event_manager.apply(mode="step", dt=env.step_dt)
  if "interval" in env.event_manager.available_modes:
    env.event_manager.apply(mode="interval", dt=env.step_dt)

  env.sim.sense()
  env.obs_buf = env.observation_manager.compute(update_history=True)
  return env.obs_buf, _extract_actor_obs(env.obs_buf), substeps


def _compute_touchdown_metrics(
  normal_force_series: torch.Tensor,
  contact_flag_series: torch.Tensor,
  dt: float,
  vertical_velocity_series: torch.Tensor | None = None,
  loading_window_steps: int = 5,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
  """Compute per-touchdown peak force, loading rate, and vertical speed."""

  touchdown_peaks: list[torch.Tensor] = []
  touchdown_rates: list[torch.Tensor] = []
  touchdown_vertical_speeds: list[torch.Tensor] = []
  num_steps, num_feet = normal_force_series.shape

  for foot_idx in range(num_feet):
    foot_touchdown_peaks: list[torch.Tensor] = []
    foot_touchdown_rates: list[torch.Tensor] = []
    foot_touchdown_vertical_speeds: list[torch.Tensor] = []
    foot_forces = normal_force_series[:, foot_idx]
    foot_contacts = contact_flag_series[:, foot_idx]
    foot_vertical_velocity = (
      vertical_velocity_series[:, foot_idx]
      if vertical_velocity_series is not None
      else None
    )
    previous_contact = False

    for step_idx in range(num_steps):
      in_contact = bool(foot_contacts[step_idx].item())
      if in_contact and not previous_contact:
        end_idx = min(step_idx + loading_window_steps, num_steps)
        window = foot_forces[step_idx:end_idx]
        foot_touchdown_peaks.append(window.max())
        foot_touchdown_rates.append(compute_loading_rate(window, dt=dt))
        if foot_vertical_velocity is not None:
          foot_touchdown_vertical_speeds.append(
            torch.clamp(-foot_vertical_velocity[step_idx], min=0.0)
          )
      previous_contact = in_contact

    if foot_touchdown_peaks:
      touchdown_peaks.append(torch.stack(foot_touchdown_peaks))
      touchdown_rates.append(torch.stack(foot_touchdown_rates))
      if foot_touchdown_vertical_speeds:
        touchdown_vertical_speeds.append(torch.stack(foot_touchdown_vertical_speeds))
      else:
        touchdown_vertical_speeds.append(torch.zeros(1, device=normal_force_series.device))
    else:
      zero = torch.zeros(1, device=normal_force_series.device)
      touchdown_peaks.append(zero)
      touchdown_rates.append(zero)
      touchdown_vertical_speeds.append(zero)

  return touchdown_peaks, touchdown_rates, touchdown_vertical_speeds


def _summarize_channel_events(channel_events: list[torch.Tensor]) -> torch.Tensor:
  """Reduce variable-length event lists into one mean value per channel."""

  return torch.stack([events.mean() for events in channel_events])


def _extract_virtual_heel_toe_points(
  env: ManagerBasedRlEnv,
  robot_name: str,
  foot_body_ids: list[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Extract virtual heel/toe points and a rigid-foot roll proxy."""

  proxy = get_foot_proxy_spec(robot_name)
  robot = env.scene["robot"]
  body_pos_w = robot.data.body_link_pos_w[:, foot_body_ids]
  body_quat_w = robot.data.body_link_quat_w[:, foot_body_ids]
  heel_pos_w, toe_pos_w = virtual_heel_toe_points(
    body_pos_w=body_pos_w,
    body_quat_w=body_quat_w,
    heel_local_offset_m=proxy.heel_local_offset_m,
    toe_local_offset_m=proxy.toe_local_offset_m,
  )
  return heel_pos_w, toe_pos_w, foot_roll_angle_from_points(heel_pos_w, toe_pos_w)


def _extract_virtual_foot_corner_points(
  env: ManagerBasedRlEnv,
  robot_name: str,
  foot_body_ids: list[int],
) -> torch.Tensor:
  """Extract evaluator-local virtual 4-corner foot points."""

  proxy = get_foot_proxy_spec(robot_name)
  robot = env.scene["robot"]
  body_pos_w = robot.data.body_link_pos_w[:, foot_body_ids]
  body_quat_w = robot.data.body_link_quat_w[:, foot_body_ids]
  return virtual_foot_corner_points(
    body_pos_w=body_pos_w,
    body_quat_w=body_quat_w,
    corner_local_offsets_m=proxy.foot_corner_local_offsets_m,
  )


def _extract_virtual_foot_corner_downward_speeds(
  env: ManagerBasedRlEnv,
  robot_name: str,
  foot_body_ids: list[int],
) -> torch.Tensor:
  """Extract instantaneous vertical speeds for evaluator-local foot corners."""

  robot = env.scene["robot"]
  body_pos_w = robot.data.body_link_pos_w[:, foot_body_ids]
  body_lin_vel_w = robot.data.body_link_lin_vel_w[:, foot_body_ids]
  body_ang_vel_w = robot.data.body_link_ang_vel_w[:, foot_body_ids]
  corner_pos_w = _extract_virtual_foot_corner_points(env, robot_name, foot_body_ids)
  rel_pos_w = corner_pos_w - body_pos_w.unsqueeze(2)
  angular_velocity = torch.cross(
    body_ang_vel_w.unsqueeze(2).expand_as(rel_pos_w),
    rel_pos_w,
    dim=-1,
  )
  corner_velocity_w = body_lin_vel_w.unsqueeze(2) + angular_velocity
  return torch.clamp(-corner_velocity_w[..., 2], min=0.0)


def _append_substep_telemetry(
  window: _SubstepTelemetryWindow,
  *,
  env: ManagerBasedRlEnv,
  robot_name: str,
  foot_body_ids: list[int],
) -> None:
  """Append one physics-substep contact sample to the current control window."""

  foot_force = extract_feet_net_forces(env, robot_name=robot_name)[..., 2].abs()
  foot_site_vz = extract_foot_vertical_velocities(env, robot_name=robot_name)
  foot_region_contact = extract_foot_region_contact_flags(
    env,
    robot_name,
    foot_body_ids,
  )
  corner_downward_speeds = _extract_virtual_foot_corner_downward_speeds(
    env,
    robot_name,
    foot_body_ids,
  )
  window.append(
    foot_force_n=foot_force,
    foot_site_vz_m_s=foot_site_vz,
    foot_region_contact=foot_region_contact,
    corner_downward_speeds_m_s=corner_downward_speeds,
  )


def _load_checkpoint_policy(
  task_id: str,
  env: ManagerBasedRlEnv,
  checkpoint_path: str,
  device: str,
):
  """Load an inference policy from an RSL-RL checkpoint."""

  agent_cfg = load_rl_cfg(task_id)
  wrapped_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
  runner = runner_cls(wrapped_env, asdict(agent_cfg), device=device)
  runner.load(
    checkpoint_path,
    load_cfg={"actor": True},
    strict=True,
    map_location=device,
  )
  return runner.get_inference_policy(device=device)


def _resolve_policy(
  task_id: str,
  env: ManagerBasedRlEnv,
  policy_source: str | PolicyAdapter,
  device: str,
):
  """Resolve a runtime policy source to a callable rollout policy."""

  if isinstance(policy_source, str):
    if is_checkpoint_path(policy_source):
      return _CallablePolicyAdapter(
        _load_checkpoint_policy(task_id, env, policy_source, device),
        use_raw_obs=True,
      )
    return adapt_policy_path(policy_source, device=device)
  return policy_source


def run_silent_eval(
  robot_name: str,
  policy_adapter: str | PolicyAdapter,
  steps: int,
  device: str,
  *,
  task_id: str | None = None,
  motion_file: str | None = None,
  disable_foot_phase_observation: bool = False,
  no_terminations: bool = False,
  fixed_command: tuple[float, float, float] | None = None,
) -> SilentEvalResult:
  """Run a minimal silent-walking evaluation rollout."""

  configure_torch_backends()

  spec = get_robot_spec(robot_name)
  resolved_task_id = task_id or spec.task_id
  _validate_runtime_support(robot_name)
  env_cfg = load_env_cfg(resolved_task_id, play=True)
  env_cfg.scene.num_envs = 1
  if no_terminations:
    env_cfg.terminations = {}
  _configure_motion_file(env_cfg, motion_file)
  _configure_fixed_velocity_command(env_cfg, fixed_command)
  if disable_foot_phase_observation:
    _disable_foot_phase_observation(env_cfg)
  _ensure_capsule_contact_sensor(env_cfg, robot_name)
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)

  try:
    policy = _resolve_policy(resolved_task_id, env, policy_adapter, device)
    raw_obs = env.reset()
    obs = _extract_actor_obs(raw_obs)
    policy.reset()

    proxy = get_foot_proxy_spec(robot_name)
    robot = env.scene["robot"]
    foot_body_ids, foot_body_names = robot.find_bodies(
      proxy.foot_body_names,
      preserve_order=True,
    )
    if tuple(foot_body_names) != proxy.foot_body_names:
      raise RuntimeError("Foot body ordering does not match the telemetry proxy spec")
    num_feet = len(proxy.foot_names)

    collector = SilentTelemetryCollector(
      robot_name=robot_name,
      dt=env.step_dt,
      body_weight_newton=spec.mass_normalization,
    )
    capsule_layout_xy: torch.Tensor | None = None

    for _ in range(steps):
      policy_obs = _unwrap_obs(raw_obs) if getattr(policy, "use_raw_obs", False) else obs
      action = policy.act(policy_obs)
      raw_obs, obs, substeps = _step_env_with_substep_telemetry(
        env,
        action,
        robot_name=robot_name,
        foot_body_ids=foot_body_ids,
      )
      foot_force = extract_feet_net_forces(env, robot_name=robot_name)
      foot_contact = extract_feet_contact_flags(env, robot_name=robot_name)
      foot_vertical_velocity = extract_foot_vertical_velocities(env, robot_name=robot_name)
      capsule_names, capsule_force, capsule_contact = extract_capsule_contact_forces(
        env,
        robot_name=robot_name,
      )
      contact_point_names, contact_point_force, contact_point_mask, contact_point_pos = (
        extract_capsule_contact_point_forces(env, robot_name=robot_name)
      )
      capsule_velocity_names, capsule_vertical_velocity = extract_capsule_vertical_velocities(
        env,
        robot_name=robot_name,
      )
      foot_grid_names, foot_grid_local_xy, foot_grid_velocity = (
        extract_foot_grid_velocities(env, robot_name=robot_name)
      )
      capsule_layout_names, capsule_layout = extract_capsule_layout_positions(
        env,
        robot_name=robot_name,
      )
      if capsule_names != capsule_velocity_names:
        raise RuntimeError("Capsule force and velocity ordering mismatch")
      if capsule_names != capsule_layout_names:
        raise RuntimeError("Capsule force and layout ordering mismatch")
      if capsule_names != contact_point_names:
        raise RuntimeError("Capsule force and contact-point ordering mismatch")
      if capsule_layout_xy is None:
        capsule_layout_xy = capsule_layout.squeeze(0).detach().clone()

      command_velocity = _get_eval_command_velocity(env, robot)
      heel_pos_w, toe_pos_w, foot_roll_angle = _extract_virtual_heel_toe_points(
        env,
        robot_name,
        foot_body_ids,
      )
      foot_corner_pos_w = _extract_virtual_foot_corner_points(
        env,
        robot_name,
        foot_body_ids,
      )
      foot_region_contact = extract_foot_region_contact_flags(
        env,
        robot_name,
        foot_body_ids,
      )
      foot_grid_force = (
        distribute_contact_forces_to_foot_grid(
          contact_force_n=contact_point_force.reshape(
            contact_point_force.shape[0],
            num_feet,
            -1,
          ),
          contact_pos_local_xy_m=world_to_local_foot_points(
            body_pos_w=robot.data.body_link_pos_w[:, foot_body_ids],
            body_quat_w=robot.data.body_link_quat_w[:, foot_body_ids],
            points_w=contact_point_pos.reshape(
              contact_point_pos.shape[0],
              num_feet,
              -1,
              3,
            ),
          )[..., :2],
          contact_mask=contact_point_mask.reshape(
            contact_point_mask.shape[0],
            num_feet,
            -1,
          ),
          local_xy_m=foot_grid_local_xy,
        )
        if foot_grid_names and spec.foot_collision_fromto_xy_m
        else None
      )
      collector.record_sample(
        action=action,
        command_velocity=command_velocity,
        actual_linear_velocity=robot.data.root_link_lin_vel_b[:, :2],
        actual_yaw_rate=robot.data.root_link_ang_vel_b[:, 2],
        foot_force_n=foot_force[..., 2].abs(),
        foot_contact=foot_contact,
        foot_site_vz_m_s=foot_vertical_velocity,
        capsule_names=capsule_names,
        capsule_force_n=capsule_force,
        capsule_contact=capsule_contact,
        capsule_vz_m_s=capsule_vertical_velocity,
        heel_pos_w=heel_pos_w,
        toe_pos_w=toe_pos_w,
        foot_roll_angle_rad=foot_roll_angle,
        foot_corner_pos_w=foot_corner_pos_w,
        foot_region_contact=foot_region_contact,
        foot_grid_names=foot_grid_names,
        foot_grid_local_xy_m=foot_grid_local_xy,
        foot_grid_velocity_m_s=foot_grid_velocity if foot_grid_names else None,
        foot_grid_force_n=foot_grid_force,
        substep_dt=env.physics_dt,
        **substeps.as_record_kwargs(),
      )

    capsule_outline_fromto_xy_m = (
      torch.tensor(
        spec.foot_collision_fromto_xy_m,
        dtype=torch.float32,
      )
      if spec.foot_collision_fromto_xy_m
      else torch.zeros((0, 2, 2), dtype=torch.float32)
    )
    capsule_radius_m = (
      torch.tensor(
        spec.foot_collision_radius_m,
        dtype=torch.float32,
      )
      if spec.foot_collision_radius_m
      else torch.zeros((0,), dtype=torch.float32)
    )
    trace, summary = collector.to_trace_and_summary(
      capsule_layout_xy_m=capsule_layout_xy,
      capsule_outline_fromto_xy_m=capsule_outline_fromto_xy_m,
      capsule_radius_m=capsule_radius_m,
    )
    return SilentEvalResult(
      robot_name=robot_name,
      task_id=resolved_task_id,
      steps=steps,
      step_dt=env.step_dt,
      trace=trace,
      summary=summary,
    )
  finally:
    env.close()
