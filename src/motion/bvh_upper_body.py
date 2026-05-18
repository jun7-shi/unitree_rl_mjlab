from __future__ import annotations

from pathlib import Path

import numpy as np

from src.motion.bvh_sew import load_bvh, soma_mujoco_to_mjlab_rotation
from src.motion.sew_mimic import ArmKeypointTarget
from src.motion.sew_upper_body import UpperBodyTarget


def soma_to_g1_upper_body_orientation_offsets() -> dict[str, np.ndarray]:
  """Return SOMA-to-G1 orientation offsets copied from soma-retargeter config."""
  return {
    "chest": _rotation_matrix_from_quat_xyzw([0.478, 0.521, 0.521, 0.478]),
    "left_hand": _rotation_matrix_from_quat_xyzw([-0.7071, 0.0, 0.0, 0.7071]),
    "right_hand": _rotation_matrix_from_quat_xyzw([0.0, 0.707, -0.707, 0.0]),
  }


def load_soma_bvh_upper_body_targets(
  path: str | Path,
  *,
  scale: float = 0.01,
  frame_slice: slice | None = None,
  joint_names: dict[str, str] | None = None,
  apply_orientation_offsets: bool = False,
  align_upper_arm_axes_to_g1: bool = False,
  remove_initial_heading: bool = False,
) -> list[UpperBodyTarget]:
  """Extract chest and bilateral arm targets from a SOMA BVH in mjlab coordinates."""
  names = {
    "chest": "Chest",
    "left_shoulder": "LeftArm",
    "left_elbow": "LeftForeArm",
    "left_wrist": "LeftHand",
    "left_hand": "LeftHand",
    "right_shoulder": "RightArm",
    "right_elbow": "RightForeArm",
    "right_wrist": "RightHand",
    "right_hand": "RightHand",
  }
  if joint_names is not None:
    names.update(joint_names)

  motion = load_bvh(path, scale=scale)
  missing = [name for name in names.values() if name not in motion.joints]
  if missing:
    raise ValueError(f"BVH is missing upper-body joints: {', '.join(sorted(set(missing)))}")

  indices = range(len(motion.frames))[frame_slice or slice(None)]
  orientation_offsets = soma_to_g1_upper_body_orientation_offsets() if apply_orientation_offsets else None
  targets: list[UpperBodyTarget] = []
  for frame_index in indices:
    pose = motion.frame_pose(frame_index, coordinate_rotation=soma_mujoco_to_mjlab_rotation())
    chest_orientation = pose.rotations[names["chest"]]
    left_hand_orientation = pose.rotations[names["left_hand"]]
    right_hand_orientation = pose.rotations[names["right_hand"]]
    if orientation_offsets is not None:
      chest_orientation = chest_orientation @ orientation_offsets["chest"]
      left_hand_orientation = left_hand_orientation @ orientation_offsets["left_hand"]
      right_hand_orientation = right_hand_orientation @ orientation_offsets["right_hand"]
    left_arm = ArmKeypointTarget(
      shoulder=pose.positions[names["left_shoulder"]],
      elbow=pose.positions[names["left_elbow"]],
      wrist=pose.positions[names["left_wrist"]],
      hand_orientation=left_hand_orientation,
    )
    right_arm = ArmKeypointTarget(
      shoulder=pose.positions[names["right_shoulder"]],
      elbow=pose.positions[names["right_elbow"]],
      wrist=pose.positions[names["right_wrist"]],
      hand_orientation=right_hand_orientation,
    )
    if align_upper_arm_axes_to_g1:
      left_arm = synthesize_g1_axis_proxy_arm_target(left_arm)
      right_arm = synthesize_g1_axis_proxy_arm_target(right_arm)

    targets.append(
      UpperBodyTarget(
        chest_position=pose.positions[names["chest"]],
        chest_orientation=chest_orientation,
        left_arm=left_arm,
        right_arm=right_arm,
      )
    )
  if remove_initial_heading:
    targets = _remove_initial_heading(targets)
  return targets


def load_soma_bvh_upper_body_human_keypoint_targets(
  path: str | Path,
  *,
  scale: float = 0.01,
  frame_slice: slice | None = None,
  joint_names: dict[str, str] | None = None,
  apply_orientation_offsets: bool = False,
  remove_initial_heading: bool = False,
) -> list[UpperBodyTarget]:
  """Load BVH upper-body targets that preserve human shoulder/elbow/wrist keypoints.

  This is the paper-pure entry point: ``elbow - shoulder`` matches the
  human's physical upper-arm direction. Use this when you want the SEW
  solver to see the paper's defined input semantics.

  Use ``load_soma_bvh_upper_body_targets`` with
  ``align_upper_arm_axes_to_g1=True`` instead if you want the G1 axis proxy
  target that drives the closed-form solver to zero axis residual against
  a reachable robot pose at the cost of losing human keypoint semantics.
  """
  return load_soma_bvh_upper_body_targets(
    path,
    scale=scale,
    frame_slice=frame_slice,
    joint_names=joint_names,
    apply_orientation_offsets=apply_orientation_offsets,
    align_upper_arm_axes_to_g1=False,
    remove_initial_heading=remove_initial_heading,
  )


def synthesize_g1_axis_proxy_arm_target(target: ArmKeypointTarget) -> ArmKeypointTarget:
  shoulder = target.elbow.copy()
  elbow = shoulder + (target.shoulder - target.elbow)
  wrist = elbow + (target.wrist - target.elbow)
  return ArmKeypointTarget(
    shoulder=shoulder,
    elbow=elbow,
    wrist=wrist,
    hand_orientation=target.hand_orientation,
  )


def _remove_initial_heading(targets: list[UpperBodyTarget]) -> list[UpperBodyTarget]:
  if not targets:
    return []
  heading = _heading_yaw(targets[0].chest_orientation)
  correction = _z_axis_rotation(-heading)
  return [_rotate_target(target, correction) for target in targets]


def _heading_yaw(rotation: np.ndarray) -> float:
  matrix = np.asarray(rotation, dtype=float).reshape(3, 3)
  return float(np.arctan2(matrix[1, 0], matrix[0, 0]))


def _z_axis_rotation(angle: float) -> np.ndarray:
  cosine = float(np.cos(angle))
  sine = float(np.sin(angle))
  return np.array(
    [
      [cosine, -sine, 0.0],
      [sine, cosine, 0.0],
      [0.0, 0.0, 1.0],
    ],
    dtype=float,
  )


def _rotate_target(target: UpperBodyTarget, rotation: np.ndarray) -> UpperBodyTarget:
  return UpperBodyTarget(
    chest_position=rotation @ target.chest_position,
    chest_orientation=rotation @ target.chest_orientation,
    left_arm=_rotate_arm_target(target.left_arm, rotation),
    right_arm=_rotate_arm_target(target.right_arm, rotation),
  )


def _rotate_arm_target(target: ArmKeypointTarget, rotation: np.ndarray) -> ArmKeypointTarget:
  return ArmKeypointTarget(
    shoulder=rotation @ target.shoulder,
    elbow=rotation @ target.elbow,
    wrist=rotation @ target.wrist,
    hand_orientation=rotation @ target.hand_orientation,
  )


def _rotation_matrix_from_quat_xyzw(quat_xyzw: list[float]) -> np.ndarray:
  quat = np.asarray(quat_xyzw, dtype=float)
  norm = float(np.linalg.norm(quat))
  if norm <= 1e-12:
    raise ValueError("Cannot build a rotation matrix from a zero-length quaternion")
  x, y, z, w = quat / norm
  return np.array(
    [
      [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
      [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
      [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ],
    dtype=float,
  )
