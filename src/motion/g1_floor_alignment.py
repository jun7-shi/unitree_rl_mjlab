from __future__ import annotations

from dataclasses import dataclass

import mujoco
from mjlab.scene import Scene

from src.motion.seed_bones import (
  G1_29DOF_JOINT_COLUMNS,
  apply_root_z_offset,
  compute_floor_alignment_offset,
)
from src.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg


G1_29DOF_JOINT_NAMES = tuple(
  column.removesuffix("_dof") for column in G1_29DOF_JOINT_COLUMNS
)


@dataclass(frozen=True)
class FootFloorAlignmentReport:
  offset_m: float
  floor_height_m: float
  clearance_m: float
  quantile: float
  min_lower_bound_m: float
  selected_lower_bound_m: float
  frames_below_floor: int
  frame_count: int


def _find_named_id(model: mujoco.MjModel, obj_type: mujoco.mjtObj, name: str) -> int:
  for candidate in (f"robot/{name}", name):
    obj_id = mujoco.mj_name2id(model, obj_type, candidate)
    if obj_id >= 0:
      return int(obj_id)
  raise ValueError(f"Could not find {obj_type.name} named '{name}' in G1 model")


def _foot_collision_geom_ids(model: mujoco.MjModel) -> list[int]:
  geom_ids: list[int] = []
  for geom_id in range(model.ngeom):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
    if "_foot" in name and name.endswith("_collision"):
      geom_ids.append(geom_id)
  if not geom_ids:
    raise ValueError("Could not find G1 foot collision geoms in compiled model")
  return geom_ids


def _capsule_lower_bound_m(
  model: mujoco.MjModel, data: mujoco.MjData, geom_id: int
) -> float:
  center = data.geom_xpos[geom_id]
  rotation = data.geom_xmat[geom_id].reshape(3, 3)
  radius = float(model.geom_size[geom_id, 0])
  half_length = float(model.geom_size[geom_id, 1])
  capsule_axis = rotation[:, 2]
  return float(center[2] - abs(capsule_axis[2]) * half_length - radius)


def _set_qpos_from_motion_row(
  model: mujoco.MjModel,
  data: mujoco.MjData,
  joint_qpos_addresses: list[int],
  row: list[float],
) -> None:
  data.qpos[:] = 0.0
  data.qpos[0:3] = row[0:3]
  # Motion rows store root quaternions as xyzw; MuJoCo freejoint qpos uses wxyz.
  data.qpos[3:7] = [row[6], row[3], row[4], row[5]]
  for qpos_address, joint_position in zip(joint_qpos_addresses, row[7:]):
    data.qpos[qpos_address] = joint_position


def compute_g1_foot_lower_bounds_m(
  rows: list[list[float]], device: str = "cpu"
) -> list[float]:
  """Return the lowest G1 foot collision point for each motion row."""
  if not rows:
    raise ValueError("Cannot align an empty motion")

  scene = Scene(unitree_g1_flat_tracking_env_cfg().scene, device=device)
  model = scene.compile()
  data = mujoco.MjData(model)

  joint_qpos_addresses = [
    int(
      model.jnt_qposadr[
        _find_named_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
      ]
    )
    for joint_name in G1_29DOF_JOINT_NAMES
  ]
  foot_geom_ids = _foot_collision_geom_ids(model)

  lower_bounds_m: list[float] = []
  for row in rows:
    _set_qpos_from_motion_row(model, data, joint_qpos_addresses, row)
    mujoco.mj_forward(model, data)
    lower_bounds_m.append(
      min(_capsule_lower_bound_m(model, data, geom_id) for geom_id in foot_geom_ids)
    )
  return lower_bounds_m


def align_g1_motion_rows_to_floor(
  rows: list[list[float]],
  floor_height_m: float = 0.0,
  clearance_m: float = 0.0,
  quantile: float = 0.0,
  device: str = "cpu",
) -> tuple[list[list[float]], FootFloorAlignmentReport]:
  """Lift G1 motion rows so selected foot lower bound reaches the target floor."""
  lower_bounds_m = compute_g1_foot_lower_bounds_m(rows, device=device)
  offset_m = compute_floor_alignment_offset(
    lower_bounds_m,
    floor_height_m=floor_height_m,
    clearance_m=clearance_m,
    quantile=quantile,
  )
  selected_lower_bound_m = floor_height_m + clearance_m - offset_m
  report = FootFloorAlignmentReport(
    offset_m=offset_m,
    floor_height_m=floor_height_m,
    clearance_m=clearance_m,
    quantile=quantile,
    min_lower_bound_m=min(lower_bounds_m),
    selected_lower_bound_m=selected_lower_bound_m,
    frames_below_floor=sum(1 for value in lower_bounds_m if value < floor_height_m),
    frame_count=len(lower_bounds_m),
  )
  return apply_root_z_offset(rows, offset_m), report
