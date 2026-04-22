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


def extract_g1_feet_net_forces(env) -> torch.Tensor:
  """Return G1 foot contact net forces with shape [num_envs, 2, 3]."""

  return extract_feet_net_forces(env, robot_name="g1")
