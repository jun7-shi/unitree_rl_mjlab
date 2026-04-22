"""Contact extraction helpers for silent walking evaluation."""

from __future__ import annotations

import torch

from mjlab.sensor import ContactSensor

from .robots import get_robot_spec


def supports_contact_backend(robot_name: str) -> bool:
  """Return whether the robot has a wired contact extraction backend."""

  return robot_name == "g1"


def extract_feet_net_forces(env, robot_name: str) -> torch.Tensor:
  """Extract per-foot net force vectors from the configured contact sensor."""

  if not supports_contact_backend(robot_name):
    spec = get_robot_spec(robot_name)
    raise NotImplementedError(
      f"Contact extraction is not implemented yet for robot '{spec.name}'"
    )

  sensor: ContactSensor = env.scene["feet_ground_contact"]
  force = sensor.data.force
  if force is None:
    raise RuntimeError("feet_ground_contact force data is unavailable")
  return force


def extract_feet_contact_flags(env, robot_name: str) -> torch.Tensor:
  """Extract per-foot contact presence flags from the configured contact sensor."""

  if not supports_contact_backend(robot_name):
    spec = get_robot_spec(robot_name)
    raise NotImplementedError(
      f"Contact extraction is not implemented yet for robot '{spec.name}'"
    )

  sensor: ContactSensor = env.scene["feet_ground_contact"]
  found = sensor.data.found
  if found is None:
    raise RuntimeError("feet_ground_contact found data is unavailable")
  return found > 0


def extract_g1_feet_net_forces(env) -> torch.Tensor:
  """Return G1 foot contact net forces with shape [num_envs, 2, 3]."""

  return extract_feet_net_forces(env, robot_name="g1")


def extract_foot_vertical_velocities(env, robot_name: str) -> torch.Tensor:
  """Extract per-foot site vertical velocities with shape [num_envs, 2]."""

  if not supports_contact_backend(robot_name):
    spec = get_robot_spec(robot_name)
    raise NotImplementedError(
      f"Foot vertical velocity extraction is not implemented yet for robot '{spec.name}'"
    )

  spec = get_robot_spec(robot_name)
  robot = env.scene["robot"]
  site_ids, _ = robot.find_sites(spec.foot_site_names, preserve_order=True)
  return robot.data.site_lin_vel_w[:, site_ids, 2]


def extract_capsule_vertical_velocities(env, robot_name: str) -> tuple[tuple[str, ...], torch.Tensor]:
  """Extract per-capsule vertical velocities with shape [num_envs, num_capsules]."""

  if not supports_contact_backend(robot_name):
    spec = get_robot_spec(robot_name)
    raise NotImplementedError(
      f"Capsule vertical velocity extraction is not implemented yet for robot '{spec.name}'"
    )

  spec = get_robot_spec(robot_name)
  robot = env.scene["robot"]
  geom_ids, geom_names = robot.find_geoms(spec.foot_collision_geom_names, preserve_order=True)
  return tuple(geom_names), robot.data.geom_lin_vel_w[:, geom_ids, 2]


def extract_capsule_layout_positions(env, robot_name: str) -> tuple[tuple[str, ...], torch.Tensor]:
  """Extract per-capsule horizontal positions for footprint plotting."""

  if not supports_contact_backend(robot_name):
    spec = get_robot_spec(robot_name)
    raise NotImplementedError(
      f"Capsule layout extraction is not implemented yet for robot '{spec.name}'"
    )

  spec = get_robot_spec(robot_name)
  robot = env.scene["robot"]
  geom_ids, geom_names = robot.find_geoms(spec.foot_collision_geom_names, preserve_order=True)
  return tuple(geom_names), robot.data.geom_pos_w[:, geom_ids, :2]


def extract_capsule_contact_forces(
  env,
  robot_name: str,
) -> tuple[tuple[str, ...], torch.Tensor, torch.Tensor]:
  """Extract per-capsule world-z contact forces and contact flags."""

  if not supports_contact_backend(robot_name):
    spec = get_robot_spec(robot_name)
    raise NotImplementedError(
      f"Capsule contact extraction is not implemented yet for robot '{spec.name}'"
    )

  spec = get_robot_spec(robot_name)
  sensor: ContactSensor = env.scene["foot_capsule_ground_contact"]
  force = sensor.data.force
  found = sensor.data.found
  if force is None or found is None:
    raise RuntimeError("foot_capsule_ground_contact data is unavailable")
  if force.shape[1] != len(spec.foot_collision_geom_names):
    raise RuntimeError(
      "foot_capsule_ground_contact shape does not match robot foot collision geoms"
    )
  return spec.foot_collision_geom_names, force[..., 2].abs(), found > 0
