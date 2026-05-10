"""Contact extraction helpers for silent walking evaluation."""

from __future__ import annotations

import torch

from mjlab.sensor import ContactSensor

from .foot_grid import foot_grid_point_velocities
from .regional_contact import classify_contact_regions, world_to_local_foot_points
from .robots import get_foot_proxy_spec, get_robot_spec


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


def extract_foot_grid_velocities(
  env,
  robot_name: str,
) -> tuple[tuple[str, ...], torch.Tensor, torch.Tensor]:
  """Extract virtual foot-grid point velocities.

  Returns `(point_names, local_xy_m, velocity_m_s)` where velocity has shape
  `[num_envs, num_feet, num_points, 3]`.
  """

  if not supports_contact_backend(robot_name):
    spec = get_robot_spec(robot_name)
    raise NotImplementedError(
      f"Foot-grid velocity extraction is not implemented yet for robot '{spec.name}'"
    )

  proxy = get_foot_proxy_spec(robot_name)
  if not proxy.foot_body_names or not proxy.foot_grid_local_offsets_m:
    return (
      (),
      torch.zeros((0, 2), dtype=torch.float32),
      torch.zeros((0, 0, 0, 3), dtype=torch.float32),
    )

  robot = env.scene["robot"]
  body_ids, _ = robot.find_bodies(proxy.foot_body_names, preserve_order=True)
  local_offsets = torch.tensor(
    proxy.foot_grid_local_offsets_m,
    dtype=robot.data.body_link_pos_w.dtype,
    device=robot.data.body_link_pos_w.device,
  )
  grid_velocity = foot_grid_point_velocities(
    body_pos_w=robot.data.body_link_pos_w[:, body_ids],
    body_quat_w=robot.data.body_link_quat_w[:, body_ids],
    body_lin_vel_w=robot.data.body_link_lin_vel_w[:, body_ids],
    body_ang_vel_w=robot.data.body_link_ang_vel_w[:, body_ids],
    local_offsets_m=local_offsets,
  )
  return (
    proxy.foot_grid_names,
    local_offsets[:, :2].detach().clone(),
    grid_velocity,
  )


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


def extract_capsule_contact_point_forces(
  env,
  robot_name: str,
) -> tuple[tuple[str, ...], torch.Tensor, torch.Tensor, torch.Tensor]:
  """Extract strongest per-capsule contact-point force magnitudes and positions."""

  if not supports_contact_backend(robot_name):
    spec = get_robot_spec(robot_name)
    raise NotImplementedError(
      f"Capsule contact point extraction is not implemented yet for robot '{spec.name}'"
    )

  spec = get_robot_spec(robot_name)
  sensor: ContactSensor = env.scene["foot_capsule_ground_contact_points"]
  found = sensor.data.found
  force = sensor.data.force
  pos = sensor.data.pos
  if found is None or force is None or pos is None:
    raise RuntimeError("foot_capsule_ground_contact_points data is unavailable")
  if found.shape[1] != len(spec.foot_collision_geom_names):
    raise RuntimeError(
      "foot_capsule_ground_contact_points shape does not match robot foot collision geoms"
    )
  return spec.foot_collision_geom_names, force[..., 2].abs(), found > 0, pos


def extract_capsule_contact_points(
  env,
  robot_name: str,
) -> tuple[tuple[str, ...], torch.Tensor, torch.Tensor]:
  """Extract per-capsule strongest contact point positions in world frame."""

  if not supports_contact_backend(robot_name):
    spec = get_robot_spec(robot_name)
    raise NotImplementedError(
      f"Capsule contact point extraction is not implemented yet for robot '{spec.name}'"
    )

  spec = get_robot_spec(robot_name)
  sensor: ContactSensor = env.scene["foot_capsule_ground_contact_points"]
  found = sensor.data.found
  pos = sensor.data.pos
  if found is None or pos is None:
    raise RuntimeError("foot_capsule_ground_contact_points data is unavailable")
  if found.shape[1] != len(spec.foot_collision_geom_names):
    raise RuntimeError(
      "foot_capsule_ground_contact_points shape does not match robot foot collision geoms"
    )
  return spec.foot_collision_geom_names, found > 0, pos


def contact_points_to_foot_region_flags(
  *,
  robot_name: str,
  contact_pos_w: torch.Tensor,
  contact_mask: torch.Tensor,
  foot_body_pos_w: torch.Tensor,
  foot_body_quat_w: torch.Tensor,
) -> torch.Tensor:
  """Convert ordered capsule contact points into per-foot region contact flags."""

  proxy = get_foot_proxy_spec(robot_name)
  num_feet = len(proxy.foot_names)
  if contact_pos_w.ndim != 3 or contact_pos_w.shape[-1] != 3:
    raise ValueError("contact_pos_w must have shape [B, C, 3]")
  if contact_mask.shape != contact_pos_w.shape[:2]:
    raise ValueError("contact_mask must have shape [B, C]")
  if contact_pos_w.shape[1] % num_feet != 0:
    raise ValueError("number of capsule contacts must divide evenly by foot count")

  capsules_per_foot = contact_pos_w.shape[1] // num_feet
  contact_pos_by_foot = contact_pos_w.reshape(
    contact_pos_w.shape[0],
    num_feet,
    capsules_per_foot,
    3,
  )
  contact_mask_by_foot = contact_mask.reshape(
    contact_mask.shape[0],
    num_feet,
    capsules_per_foot,
  )
  local_pos = world_to_local_foot_points(
    body_pos_w=foot_body_pos_w,
    body_quat_w=foot_body_quat_w,
    points_w=contact_pos_by_foot,
  )
  return classify_contact_regions(
    contact_pos_local_m=local_pos,
    contact_mask=contact_mask_by_foot,
    heel_region_max_x_m=proxy.heel_region_max_x_m,
    toe_region_min_x_m=proxy.toe_region_min_x_m,
  )


def extract_foot_region_contact_flags(
  env,
  robot_name: str,
  foot_body_ids: list[int],
) -> torch.Tensor:
  """Extract per-foot heel/midfoot/toe region contact flags."""

  capsule_names, contact_mask, contact_pos = extract_capsule_contact_points(
    env,
    robot_name=robot_name,
  )
  spec = get_robot_spec(robot_name)
  if capsule_names != spec.foot_collision_geom_names:
    raise RuntimeError("Capsule contact point ordering mismatch")
  robot = env.scene["robot"]
  return contact_points_to_foot_region_flags(
    robot_name=robot_name,
    contact_pos_w=contact_pos,
    contact_mask=contact_mask,
    foot_body_pos_w=robot.data.body_link_pos_w[:, foot_body_ids],
    foot_body_quat_w=robot.data.body_link_quat_w[:, foot_body_ids],
  )
