from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  subtract_frame_transforms,
)

from .commands import MotionCommand
from .rewards import _get_body_indexes, _get_reference_phase_context

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def motion_anchor_pos_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))

  pos, _ = subtract_frame_transforms(
    command.robot_anchor_pos_w,
    command.robot_anchor_quat_w,
    command.anchor_pos_w,
    command.anchor_quat_w,
  )

  return pos.view(env.num_envs, -1)


def motion_anchor_ori_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))

  _, ori = subtract_frame_transforms(
    command.robot_anchor_pos_w,
    command.robot_anchor_quat_w,
    command.anchor_pos_w,
    command.anchor_quat_w,
  )
  mat = matrix_from_quat(ori)
  return mat[..., :2].reshape(mat.shape[0], -1)


def robot_body_pos_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))

  num_bodies = len(command.cfg.body_names)
  pos_b, _ = subtract_frame_transforms(
    command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
    command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
    command.robot_body_pos_w,
    command.robot_body_quat_w,
  )

  return pos_b.view(env.num_envs, -1)


def robot_body_ori_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))

  num_bodies = len(command.cfg.body_names)
  _, ori_b = subtract_frame_transforms(
    command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
    command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
    command.robot_body_pos_w,
    command.robot_body_quat_w,
  )
  mat = matrix_from_quat(ori_b)
  return mat[..., :2].reshape(mat.shape[0], -1)


def foot_phase_observation_from_context(
  reference_contact: torch.Tensor,
  swing_phase: torch.Tensor,
) -> torch.Tensor:
  """Encode each foot as sin phase, cos phase, and reference swing flag."""
  reference_swing = torch.logical_not(reference_contact).float()
  phase_sin = torch.sin(swing_phase * torch.pi)
  phase_cos = torch.cos(swing_phase * torch.pi)
  return torch.stack((phase_sin, phase_cos, reference_swing), dim=-1).reshape(
    reference_contact.shape[0],
    -1,
  )


def motion_foot_phase(
  env: ManagerBasedRlEnv,
  command_name: str,
  foot_body_names: tuple[str, ...],
  clearance_threshold: float = 0.03,
) -> torch.Tensor:
  """Return command-derived foot phase observation for left/right swing timing."""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, foot_body_names)
  if len(body_indexes) != len(foot_body_names):
    raise ValueError(
      "Could not resolve all foot body names for foot-phase observation: "
      f"{foot_body_names}"
    )

  _, reference_contact, _, swing_phase = _get_reference_phase_context(
    command=command,
    body_indexes=body_indexes,
    clearance_threshold=clearance_threshold,
  )
  return foot_phase_observation_from_context(reference_contact, swing_phase)
