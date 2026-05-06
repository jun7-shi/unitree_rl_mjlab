from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_error_magnitude

from .commands import MotionCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _get_body_indexes(
  command: MotionCommand, body_names: tuple[str, ...] | None
) -> list[int]:
  return [
    i
    for i, name in enumerate(command.cfg.body_names)
    if (body_names is None) or (name in body_names)
  ]


def infer_reference_contacts_from_heights(
  reference_heights: torch.Tensor,
  clearance_threshold: float,
) -> torch.Tensor:
  """Infer stance feet from per-foot reference heights.

  Each foot uses its own minimum height as the local floor proxy so clips with
  slight left/right height offsets do not bias contact inference.
  """
  min_height = torch.amin(reference_heights, dim=0, keepdim=True)
  return reference_heights <= (min_height + clearance_threshold)


def swing_contact_cost_from_masks(
  reference_contact: torch.Tensor,
  actual_contact: torch.Tensor,
) -> torch.Tensor:
  """Return per-env count of feet touching while reference says swing."""
  reference_swing = torch.logical_not(reference_contact)
  return torch.sum((reference_swing & actual_contact).float(), dim=1)


def landing_force_cost_from_contact_events(
  first_contact: torch.Tensor,
  force_magnitude: torch.Tensor,
) -> torch.Tensor:
  """Return per-env landing force summed over first-contact feet."""
  return torch.sum(force_magnitude * first_contact.float(), dim=1)


def single_support_reward_from_masks(
  reference_contact: torch.Tensor,
  actual_contact: torch.Tensor,
) -> torch.Tensor:
  """Reward matching the reference support foot during single-support samples."""
  reference_single_support = torch.sum(reference_contact.float(), dim=1) == 1.0
  actual_matches_reference = torch.all(reference_contact == actual_contact, dim=1)
  return (reference_single_support & actual_matches_reference).float()


def swing_clearance_margin_cost_from_heights(
  reference_contact: torch.Tensor,
  actual_heights: torch.Tensor,
  reference_floor_heights: torch.Tensor,
  margin_m: float,
) -> torch.Tensor:
  """Penalize swing feet whose body-height clearance is below a margin."""
  reference_swing = torch.logical_not(reference_contact)
  actual_clearance = actual_heights - reference_floor_heights
  clearance_deficit = torch.clamp(margin_m - actual_clearance, min=0.0)
  return torch.sum(clearance_deficit * reference_swing.float(), dim=1)


def swing_phase_from_reference_contact(reference_contact: torch.Tensor) -> torch.Tensor:
  """Return normalized 0..1 phase for contiguous reference swing runs."""
  phase = torch.zeros(
    reference_contact.shape,
    dtype=torch.float32,
    device=reference_contact.device,
  )
  for foot_index in range(reference_contact.shape[1]):
    swing = torch.logical_not(reference_contact[:, foot_index])
    swing_indices = torch.where(swing)[0]
    if swing_indices.numel() == 0:
      continue

    run_start = int(swing_indices[0].item())
    previous = run_start
    for index in swing_indices[1:].tolist():
      if index != previous + 1:
        _fill_swing_phase_run(phase[:, foot_index], run_start, previous)
        run_start = index
      previous = index
    _fill_swing_phase_run(phase[:, foot_index], run_start, previous)
  return phase


def _fill_swing_phase_run(
  phase: torch.Tensor,
  start_index: int,
  end_index: int,
) -> None:
  run_length = end_index - start_index + 1
  if run_length == 1:
    phase[start_index] = 0.0
    return
  phase[start_index : end_index + 1] = torch.linspace(
    0.0,
    1.0,
    run_length,
    dtype=phase.dtype,
    device=phase.device,
  )


def phase_swing_clearance_target(
  swing_phase: torch.Tensor,
  reference_clearance: torch.Tensor,
  base_clearance_m: float,
  lift_m: float,
  landing_phase_start: float,
) -> torch.Tensor:
  """Build a phase-dependent swing clearance target in meters."""
  active_lift = torch.where(
    swing_phase < landing_phase_start,
    torch.sin(swing_phase * torch.pi),
    torch.zeros_like(swing_phase),
  )
  phase_target = base_clearance_m + lift_m * active_lift
  return torch.maximum(reference_clearance, phase_target)


def phase_swing_clearance_cost_from_heights(
  reference_contact: torch.Tensor,
  actual_heights: torch.Tensor,
  reference_floor_heights: torch.Tensor,
  reference_clearance: torch.Tensor,
  swing_phase: torch.Tensor,
  base_clearance_m: float,
  lift_m: float,
  landing_phase_start: float,
) -> torch.Tensor:
  """Penalize swing feet below a phase-dependent clearance envelope."""
  reference_swing = torch.logical_not(reference_contact)
  actual_clearance = actual_heights - reference_floor_heights
  target_clearance = phase_swing_clearance_target(
    swing_phase=swing_phase,
    reference_clearance=reference_clearance,
    base_clearance_m=base_clearance_m,
    lift_m=lift_m,
    landing_phase_start=landing_phase_start,
  )
  clearance_deficit = torch.clamp(target_clearance - actual_clearance, min=0.0)
  return torch.sum(clearance_deficit * reference_swing.float(), dim=1)


def early_swing_contact_cost_from_masks(
  reference_contact: torch.Tensor,
  actual_contact: torch.Tensor,
  swing_phase: torch.Tensor,
  landing_phase_start: float,
) -> torch.Tensor:
  """Return per-env count of early swing feet already touching terrain."""
  early_swing = torch.logical_not(reference_contact) & (
    swing_phase < landing_phase_start
  )
  return torch.sum((early_swing & actual_contact).float(), dim=1)


def foot_lift_trajectory_cost_from_heights(
  reference_contact: torch.Tensor,
  actual_heights: torch.Tensor,
  reference_floor_heights: torch.Tensor,
  swing_phase: torch.Tensor,
  base_clearance_m: float,
  lift_m: float,
  phase_power: float,
  deadband_m: float,
) -> torch.Tensor:
  """Penalize deficits against a smooth phase-conditioned foot-lift target."""
  reference_swing = torch.logical_not(reference_contact)
  actual_clearance = actual_heights - reference_floor_heights
  lift_shape = torch.clamp(torch.sin(swing_phase * torch.pi), min=0.0) ** phase_power
  target_clearance = base_clearance_m + lift_m * lift_shape
  clearance_deficit = torch.clamp(
    target_clearance - actual_clearance - deadband_m,
    min=0.0,
  )
  return torch.sum(clearance_deficit * reference_swing.float(), dim=1)


def late_swing_downward_velocity_cost_from_velocity(
  reference_contact: torch.Tensor,
  vertical_velocity_m_s: torch.Tensor,
  swing_phase: torch.Tensor,
  velocity_phase_start: float,
  max_downward_velocity_m_s: float,
) -> torch.Tensor:
  """Penalize excessive downward foot speed near reference touchdown."""
  late_swing = torch.logical_not(reference_contact) & (
    swing_phase >= velocity_phase_start
  )
  downward_speed = torch.clamp(-vertical_velocity_m_s, min=0.0)
  speed_excess = torch.clamp(
    downward_speed - max_downward_velocity_m_s,
    min=0.0,
  )
  return torch.sum(speed_excess * late_swing.float(), dim=1)


def _get_reference_phase_context(
  command: MotionCommand,
  body_indexes: list[int],
  clearance_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  cache = getattr(command, "_reference_phase_prior_cache", {})
  cache_key = (tuple(body_indexes), float(clearance_threshold))
  if cache_key not in cache:
    all_reference_heights = command.motion.body_pos_w[:, body_indexes, 2]
    reference_floor_heights = torch.amin(
      all_reference_heights,
      dim=0,
      keepdim=True,
    )
    all_reference_contact = infer_reference_contacts_from_heights(
      all_reference_heights,
      clearance_threshold=clearance_threshold,
    )
    all_swing_phase = swing_phase_from_reference_contact(all_reference_contact)
    all_reference_clearance = all_reference_heights - reference_floor_heights
    cache[cache_key] = (
      reference_floor_heights,
      all_reference_contact,
      all_reference_clearance,
      all_swing_phase,
    )
    setattr(command, "_reference_phase_prior_cache", cache)

  (
    reference_floor_heights,
    all_reference_contact,
    all_reference_clearance,
    all_swing_phase,
  ) = cache[cache_key]
  time_steps = torch.clamp(command.time_steps, 0, command.motion.time_step_total - 1)
  return (
    reference_floor_heights,
    all_reference_contact[time_steps],
    all_reference_clearance[time_steps],
    all_swing_phase[time_steps],
  )


def motion_global_anchor_position_error_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  error = torch.sum(
    torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1
  )
  return torch.exp(-error / std**2)


def motion_global_anchor_orientation_error_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2
  return torch.exp(-error / std**2)


def motion_relative_body_position_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  error = torch.sum(
    torch.square(
      command.body_pos_relative_w[:, body_indexes]
      - command.robot_body_pos_w[:, body_indexes]
    ),
    dim=-1,
  )
  return torch.exp(-error.mean(-1) / std**2)


def motion_relative_body_orientation_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  error = (
    quat_error_magnitude(
      command.body_quat_relative_w[:, body_indexes],
      command.robot_body_quat_w[:, body_indexes],
    )
    ** 2
  )
  return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_linear_velocity_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  error = torch.sum(
    torch.square(
      command.body_lin_vel_w[:, body_indexes]
      - command.robot_body_lin_vel_w[:, body_indexes]
    ),
    dim=-1,
  )
  return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_angular_velocity_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  error = torch.sum(
    torch.square(
      command.body_ang_vel_w[:, body_indexes]
      - command.robot_body_ang_vel_w[:, body_indexes]
    ),
    dim=-1,
  )
  return torch.exp(-error.mean(-1) / std**2)


def motion_swing_contact_penalty(
  env: ManagerBasedRlEnv,
  command_name: str,
  sensor_name: str,
  foot_body_names: tuple[str, ...],
  clearance_threshold: float = 0.03,
) -> torch.Tensor:
  """Penalize touching terrain when the reference foot is in swing phase."""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, foot_body_names)
  if len(body_indexes) != len(foot_body_names):
    raise ValueError(
      "Could not resolve all foot body names for contact-aware tracking reward: "
      f"{foot_body_names}"
    )

  reference_heights = command.body_pos_w[:, body_indexes, 2]
  reference_floor_heights = torch.amin(
    command.motion.body_pos_w[:, body_indexes, 2], dim=0, keepdim=True
  )
  reference_contact = reference_heights <= (
    reference_floor_heights + clearance_threshold
  )

  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  actual_contact = sensor.data.found > 0
  cost = swing_contact_cost_from_masks(reference_contact, actual_contact)
  env.extras["log"]["Metrics/tracking_swing_contact_mean"] = torch.mean(cost)
  return cost


def motion_single_support_reward(
  env: ManagerBasedRlEnv,
  command_name: str,
  sensor_name: str,
  foot_body_names: tuple[str, ...],
  clearance_threshold: float = 0.03,
) -> torch.Tensor:
  """Reward matching the reference contact foot during single support."""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, foot_body_names)
  if len(body_indexes) != len(foot_body_names):
    raise ValueError(
      "Could not resolve all foot body names for single-support reward: "
      f"{foot_body_names}"
    )

  reference_heights = command.body_pos_w[:, body_indexes, 2]
  reference_floor_heights = torch.amin(
    command.motion.body_pos_w[:, body_indexes, 2], dim=0, keepdim=True
  )
  reference_contact = reference_heights <= (
    reference_floor_heights + clearance_threshold
  )

  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  actual_contact = sensor.data.found > 0
  reward = single_support_reward_from_masks(reference_contact, actual_contact)
  env.extras["log"]["Metrics/tracking_single_support_match_mean"] = torch.mean(reward)
  return reward


def motion_swing_clearance_penalty(
  env: ManagerBasedRlEnv,
  command_name: str,
  foot_body_names: tuple[str, ...],
  clearance_threshold: float = 0.03,
  margin_m: float = 0.10,
) -> torch.Tensor:
  """Penalize low swing-foot body height before contact occurs."""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, foot_body_names)
  if len(body_indexes) != len(foot_body_names):
    raise ValueError(
      "Could not resolve all foot body names for swing-clearance reward: "
      f"{foot_body_names}"
    )

  reference_floor_heights = torch.amin(
    command.motion.body_pos_w[:, body_indexes, 2], dim=0, keepdim=True
  )
  reference_heights = command.body_pos_w[:, body_indexes, 2]
  reference_contact = reference_heights <= (
    reference_floor_heights + clearance_threshold
  )
  actual_heights = command.robot_body_pos_w[:, body_indexes, 2]
  cost = swing_clearance_margin_cost_from_heights(
    reference_contact=reference_contact,
    actual_heights=actual_heights,
    reference_floor_heights=reference_floor_heights,
    margin_m=margin_m,
  )
  env.extras["log"]["Metrics/tracking_swing_clearance_cost_mean"] = torch.mean(cost)
  return cost


def motion_phase_swing_clearance_penalty(
  env: ManagerBasedRlEnv,
  command_name: str,
  foot_body_names: tuple[str, ...],
  clearance_threshold: float = 0.03,
  base_clearance_m: float = 0.04,
  lift_m: float = 0.08,
  landing_phase_start: float = 0.85,
) -> torch.Tensor:
  """Penalize low swing-foot height using a reference-phase clearance envelope."""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, foot_body_names)
  if len(body_indexes) != len(foot_body_names):
    raise ValueError(
      "Could not resolve all foot body names for phase swing-clearance reward: "
      f"{foot_body_names}"
    )

  (
    reference_floor_heights,
    reference_contact,
    reference_clearance,
    swing_phase,
  ) = _get_reference_phase_context(
    command=command,
    body_indexes=body_indexes,
    clearance_threshold=clearance_threshold,
  )
  actual_heights = command.robot_body_pos_w[:, body_indexes, 2]
  cost = phase_swing_clearance_cost_from_heights(
    reference_contact=reference_contact,
    actual_heights=actual_heights,
    reference_floor_heights=reference_floor_heights,
    reference_clearance=reference_clearance,
    swing_phase=swing_phase,
    base_clearance_m=base_clearance_m,
    lift_m=lift_m,
    landing_phase_start=landing_phase_start,
  )
  env.extras["log"][
    "Metrics/tracking_phase_swing_clearance_cost_mean"
  ] = torch.mean(cost)
  return cost


def motion_early_swing_contact_penalty(
  env: ManagerBasedRlEnv,
  command_name: str,
  sensor_name: str,
  foot_body_names: tuple[str, ...],
  clearance_threshold: float = 0.03,
  landing_phase_start: float = 0.85,
) -> torch.Tensor:
  """Penalize contact before the reference swing foot reaches landing phase."""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, foot_body_names)
  if len(body_indexes) != len(foot_body_names):
    raise ValueError(
      "Could not resolve all foot body names for early swing-contact reward: "
      f"{foot_body_names}"
    )

  _, reference_contact, _, swing_phase = _get_reference_phase_context(
    command=command,
    body_indexes=body_indexes,
    clearance_threshold=clearance_threshold,
  )
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  actual_contact = sensor.data.found > 0
  cost = early_swing_contact_cost_from_masks(
    reference_contact=reference_contact,
    actual_contact=actual_contact,
    swing_phase=swing_phase,
    landing_phase_start=landing_phase_start,
  )
  env.extras["log"]["Metrics/tracking_early_swing_contact_mean"] = torch.mean(cost)
  return cost


def motion_foot_lift_trajectory_penalty(
  env: ManagerBasedRlEnv,
  command_name: str,
  foot_body_names: tuple[str, ...],
  clearance_threshold: float = 0.03,
  base_clearance_m: float = 0.04,
  lift_m: float = 0.08,
  phase_power: float = 1.0,
  deadband_m: float = 0.01,
) -> torch.Tensor:
  """Penalize missing foot lift against a smooth swing-phase trajectory."""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, foot_body_names)
  if len(body_indexes) != len(foot_body_names):
    raise ValueError(
      "Could not resolve all foot body names for foot-lift trajectory reward: "
      f"{foot_body_names}"
    )

  reference_floor_heights, reference_contact, _, swing_phase = (
    _get_reference_phase_context(
      command=command,
      body_indexes=body_indexes,
      clearance_threshold=clearance_threshold,
    )
  )
  actual_heights = command.robot_body_pos_w[:, body_indexes, 2]
  cost = foot_lift_trajectory_cost_from_heights(
    reference_contact=reference_contact,
    actual_heights=actual_heights,
    reference_floor_heights=reference_floor_heights,
    swing_phase=swing_phase,
    base_clearance_m=base_clearance_m,
    lift_m=lift_m,
    phase_power=phase_power,
    deadband_m=deadband_m,
  )
  env.extras["log"]["Metrics/tracking_foot_lift_deficit_mean"] = torch.mean(cost)
  return cost


def motion_late_swing_velocity_penalty(
  env: ManagerBasedRlEnv,
  command_name: str,
  foot_body_names: tuple[str, ...],
  clearance_threshold: float = 0.03,
  velocity_phase_start: float = 0.80,
  max_downward_velocity_m_s: float = 0.40,
) -> torch.Tensor:
  """Penalize high downward foot speed near the reference landing phase."""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, foot_body_names)
  if len(body_indexes) != len(foot_body_names):
    raise ValueError(
      "Could not resolve all foot body names for late-swing velocity reward: "
      f"{foot_body_names}"
    )

  _, reference_contact, _, swing_phase = _get_reference_phase_context(
    command=command,
    body_indexes=body_indexes,
    clearance_threshold=clearance_threshold,
  )
  vertical_velocity = command.robot_body_lin_vel_w[:, body_indexes, 2]
  cost = late_swing_downward_velocity_cost_from_velocity(
    reference_contact=reference_contact,
    vertical_velocity_m_s=vertical_velocity,
    swing_phase=swing_phase,
    velocity_phase_start=velocity_phase_start,
    max_downward_velocity_m_s=max_downward_velocity_m_s,
  )
  env.extras["log"][
    "Metrics/tracking_late_swing_downward_velocity_mean"
  ] = torch.mean(cost)
  return cost


def motion_soft_landing_penalty(
  env: ManagerBasedRlEnv,
  sensor_name: str,
) -> torch.Tensor:
  """Penalize high terrain contact force at touchdown."""
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.force is not None
  force_magnitude = torch.norm(sensor.data.force, dim=-1)
  first_contact = sensor.compute_first_contact(dt=env.step_dt)
  cost = landing_force_cost_from_contact_events(first_contact, force_magnitude)
  num_landings = torch.sum(first_contact.float())
  mean_landing_force = torch.sum(cost) / torch.clamp(num_landings, min=1)
  env.extras["log"]["Metrics/tracking_landing_force_mean"] = mean_landing_force
  return cost


def self_collision_cost(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 10.0,
) -> torch.Tensor:
  """Penalize self-collisions.

  When the sensor provides force history (from ``history_length > 0``),
  counts substeps where any contact force exceeds *force_threshold*.
  Falls back to the instantaneous ``found`` count otherwise.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  if data.force_history is not None:
    # force_history: [B, N, H, 3]
    force_mag = torch.norm(data.force_history, dim=-1)  # [B, N, H]
    hit = (force_mag > force_threshold).any(dim=1)  # [B, H]
    return hit.sum(dim=-1).float()  # [B]
  assert data.found is not None
  return data.found.squeeze(-1)
