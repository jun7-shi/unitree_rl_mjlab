"""Contact extraction helpers for silent walking evaluation."""

from __future__ import annotations

import numpy as np
import torch
import mujoco

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


def _resolve_sim_geom_ids(env, geom_names: tuple[str, ...]) -> list[int]:
  """Resolve local evaluator geom names to MuJoCo geom ids."""

  sim_geom_names = {
    env.sim.mj_model.geom(geom_id).name: geom_id
    for geom_id in range(env.sim.mj_model.ngeom)
  }
  resolved_ids: list[int] = []
  for geom_name in geom_names:
    candidates = (
      sim_geom_names.get(geom_name),
      sim_geom_names.get(f"robot/{geom_name}"),
    )
    resolved_id = next((candidate for candidate in candidates if candidate is not None), None)
    if resolved_id is None:
      raise KeyError(f"Unable to resolve MuJoCo geom id for '{geom_name}'")
    resolved_ids.append(resolved_id)

  return resolved_ids


def extract_capsule_contact_forces(
  env,
  robot_name: str,
) -> tuple[tuple[str, ...], torch.Tensor, torch.Tensor]:
  """Extract per-capsule summed normal contact forces and contact flags."""

  if not supports_contact_backend(robot_name):
    spec = get_robot_spec(robot_name)
    raise NotImplementedError(
      f"Capsule contact extraction is not implemented yet for robot '{spec.name}'"
    )

  spec = get_robot_spec(robot_name)
  geom_names = spec.foot_collision_geom_names
  sim_geom_ids = _resolve_sim_geom_ids(env, geom_names)
  sim_geom_to_index = {geom_id: idx for idx, geom_id in enumerate(sim_geom_ids)}

  normal_forces = torch.zeros((1, len(geom_names)), dtype=torch.float32)
  contact_flags = torch.zeros((1, len(geom_names)), dtype=torch.bool)
  contact_wrench = np.zeros(6, dtype=np.float64)

  for contact_idx in range(env.sim.mj_data.ncon):
    contact = env.sim.mj_data.contact[contact_idx]
    geom_id = None
    if contact.geom1 in sim_geom_to_index:
      geom_id = contact.geom1
    elif contact.geom2 in sim_geom_to_index:
      geom_id = contact.geom2
    if geom_id is None:
      continue

    mujoco.mj_contactForce(env.sim.mj_model, env.sim.mj_data, contact_idx, contact_wrench)
    geom_index = sim_geom_to_index[geom_id]
    normal_forces[0, geom_index] += float(max(contact_wrench[0], 0.0))
    contact_flags[0, geom_index] = True

  return geom_names, normal_forces, contact_flags
