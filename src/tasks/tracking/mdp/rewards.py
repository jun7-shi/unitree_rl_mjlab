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
