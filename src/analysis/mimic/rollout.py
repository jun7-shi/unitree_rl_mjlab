"""Checkpoint rollout collection for mimic gait analysis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass(frozen=True, slots=True)
class PolicyGaitTrace:
  """Policy rollout gait signals aligned to reference metrics."""

  contact_flags: np.ndarray
  foot_clearance_m: np.ndarray
  foot_vertical_velocity_m_s: np.ndarray
  action_rate_l2: np.ndarray
  step_dt: float


def run_checkpoint_gait_trace(
  *,
  task_id: str,
  motion_file: str | Path,
  checkpoint: str | Path,
  steps: int,
  device: str,
  robot_name: str = "g1",
  disable_randomization: bool = True,
) -> PolicyGaitTrace:
  """Run a checkpoint policy and collect foot contact and clearance signals."""

  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.registry import load_env_cfg
  from mjlab.tasks.tracking.mdp import MotionCommandCfg

  from src.evaluation.silent_walking.contact_backends import (
    extract_capsule_contact_forces,
    extract_foot_vertical_velocities,
  )
  from src.evaluation.silent_walking.robots import get_robot_spec
  from src.evaluation.silent_walking.runner import (
    _ensure_capsule_contact_sensor,
    _extract_actor_obs,
    _load_checkpoint_policy,
    _step_env,
    _unwrap_obs,
  )

  spec = get_robot_spec(robot_name)
  env_cfg = load_env_cfg(task_id, play=True)
  env_cfg.scene.num_envs = 1
  if "motion" not in env_cfg.commands:
    raise ValueError(f"Task {task_id} does not define a motion command")
  motion_cmd = env_cfg.commands["motion"]
  if not isinstance(motion_cmd, MotionCommandCfg):
    raise TypeError("Task motion command is not MotionCommandCfg")
  motion_path = Path(motion_file).expanduser().resolve()
  if not motion_path.exists():
    raise FileNotFoundError(f"Motion file not found: {motion_path}")
  motion_cmd.motion_file = str(motion_path)
  motion_cmd.sampling_mode = "start"
  motion_cmd.pose_range = {}
  motion_cmd.velocity_range = {}
  if disable_randomization:
    env_cfg.events = {}
  _ensure_capsule_contact_sensor(env_cfg, robot_name)

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
  try:
    policy = _load_checkpoint_policy(
      task_id=task_id,
      env=env,
      checkpoint_path=str(Path(checkpoint).expanduser().resolve()),
      device=device,
    )
    raw_obs = env.reset()
    obs = _extract_actor_obs(raw_obs)
    robot = env.scene["robot"]
    site_ids, _ = robot.find_sites(spec.foot_site_names, preserve_order=True)

    contact_samples: list[np.ndarray] = []
    clearance_samples: list[np.ndarray] = []
    vertical_velocity_samples: list[np.ndarray] = []
    action_rate_samples: list[np.ndarray] = []
    previous_action: torch.Tensor | None = None

    for _ in range(steps):
      with torch.inference_mode():
        action = policy(_unwrap_obs(raw_obs))
      if previous_action is None:
        action_rate_samples.append(np.zeros((1,), dtype=float))
      else:
        action_rate_samples.append(
          torch.linalg.norm(action - previous_action, dim=1).detach().cpu().numpy()
        )
      previous_action = action.detach().clone()
      raw_obs, obs = _step_env(env, action)
      del obs
      _, _, capsule_contact = extract_capsule_contact_forces(env, robot_name=robot_name)
      capsule_contact_np = (
        capsule_contact.squeeze(0).detach().cpu().numpy().astype(bool)
      )
      if capsule_contact_np.shape[0] % 2 != 0:
        raise RuntimeError("Expected an even number of left/right foot capsules")
      half = capsule_contact_np.shape[0] // 2
      contact_samples.append(
        np.asarray(
          [capsule_contact_np[:half].any(), capsule_contact_np[half:].any()],
          dtype=bool,
        )
      )
      site_z = robot.data.site_pos_w[:, site_ids, 2].squeeze(0).detach().cpu().numpy()
      clearance_samples.append(np.maximum(site_z, 0.0))
      vertical_velocity_samples.append(
        extract_foot_vertical_velocities(env, robot_name=robot_name)
        .squeeze(0)
        .detach()
        .cpu()
        .numpy()
      )

    return PolicyGaitTrace(
      contact_flags=np.stack(contact_samples),
      foot_clearance_m=np.stack(clearance_samples),
      foot_vertical_velocity_m_s=np.stack(vertical_velocity_samples),
      action_rate_l2=np.concatenate(action_rate_samples),
      step_dt=float(env.step_dt),
    )
  finally:
    env.close()
