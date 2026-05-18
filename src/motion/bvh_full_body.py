from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.motion.bvh_sew import load_bvh, soma_mujoco_to_mjlab_rotation
from src.motion.bvh_upper_body import (
  _rotation_matrix_from_quat_xyzw,
  soma_to_g1_upper_body_orientation_offsets,
  synthesize_g1_axis_proxy_arm_target,
)
from src.motion.sew_full_body import FullBodyTarget
from src.motion.sew_lower_body import LegKeypointTarget, LowerBodyTarget
from src.motion.sew_mimic import ArmKeypointTarget, normalize
from src.motion.sew_upper_body import UpperBodyTarget


@dataclass(frozen=True)
class LowerBodyEffectorConfig:
  joint_scales: dict[str, float]
  position_offsets: dict[str, np.ndarray]
  orientation_offsets: dict[str, np.ndarray]


def soma_to_g1_lower_body_effector_config() -> LowerBodyEffectorConfig:
  """Return the SOMA-to-G1 lower-body scaler values from soma-retargeter."""
  human_height_assumption = 1.8
  model_height = 1.70
  ratio = model_height / human_height_assumption
  joint_scales = {
    "Hips": 0.82 * ratio,
    "LeftLeg": 0.86 * ratio,
    "RightLeg": 0.86 * ratio,
    "LeftShin": 0.86 * ratio,
    "RightShin": 0.86 * ratio,
    "LeftFoot": 0.82 * ratio,
    "RightFoot": 0.82 * ratio,
  }
  raw_offsets = {
    "LeftLeg": ([-0.03, 0.036, -0.05], [0.459, -0.538, -0.538, 0.459]),
    "LeftShin": ([0.0, 0.04, -0.005], [0.5, -0.5, -0.5, 0.5]),
    "LeftFoot": ([-0.017, 0.04, -0.017], [0.695, -0.128, -0.128, 0.695]),
    "RightLeg": ([-0.03, -0.036, -0.05], [0.538, 0.459, 0.459, 0.538]),
    "RightShin": ([0.0, -0.04, -0.005], [0.5, 0.5, 0.5, 0.5]),
    "RightFoot": ([-0.017, -0.04, -0.017], [0.128, 0.695, 0.695, 0.128]),
  }
  return LowerBodyEffectorConfig(
    joint_scales=joint_scales,
    position_offsets={
      joint_name: np.asarray(position, dtype=float)
      for joint_name, (position, _orientation) in raw_offsets.items()
    },
    orientation_offsets={
      joint_name: _rotation_matrix_from_quat_xyzw(orientation)
      for joint_name, (_position, orientation) in raw_offsets.items()
    },
  )


def load_soma_bvh_full_body_targets(
  path: str | Path,
  *,
  scale: float = 0.01,
  frame_slice: slice | None = None,
  joint_names: dict[str, str] | None = None,
  apply_orientation_offsets: bool = False,
  align_upper_arm_axes_to_g1: bool = False,
  apply_lower_body_offsets: bool = False,
  remove_initial_heading: bool = False,
  localize_to_body_frame: bool = False,
) -> list[FullBodyTarget]:
  """Extract upper-body and bilateral leg SEW targets from a SOMA BVH."""
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
    "left_hip": "LeftLeg",
    "left_knee": "LeftShin",
    "left_ankle": "LeftFoot",
    "left_foot": "LeftFoot",
    "right_hip": "RightLeg",
    "right_knee": "RightShin",
    "right_ankle": "RightFoot",
    "right_foot": "RightFoot",
  }
  if apply_lower_body_offsets:
    names["hips"] = "Hips"
  if joint_names is not None:
    names.update(joint_names)

  motion = load_bvh(path, scale=scale)
  missing = [name for name in names.values() if name not in motion.joints]
  if missing:
    raise ValueError(f"BVH is missing full-body joints: {', '.join(sorted(set(missing)))}")

  indices = range(len(motion.frames))[frame_slice or slice(None)]
  orientation_offsets = soma_to_g1_upper_body_orientation_offsets() if apply_orientation_offsets else None
  lower_body_offsets = soma_to_g1_lower_body_effector_config() if apply_lower_body_offsets else None
  targets: list[FullBodyTarget] = []
  body_frames: list[np.ndarray] = []
  for frame_index in indices:
    pose = motion.frame_pose(frame_index, coordinate_rotation=soma_mujoco_to_mjlab_rotation())
    if localize_to_body_frame:
      body_frames.append(_body_frame_from_pose(pose, names))
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
    lower = (
      _lower_body_target_from_scaled_effectors(pose, names, lower_body_offsets)
      if lower_body_offsets is not None
      else _lower_body_target_from_raw_keypoints(pose, names)
    )

    targets.append(
      FullBodyTarget(
        lower=lower,
        upper=UpperBodyTarget(
          chest_position=pose.positions[names["chest"]],
          chest_orientation=chest_orientation,
          left_arm=left_arm,
          right_arm=right_arm,
        ),
      )
    )

  if localize_to_body_frame:
    targets = _localize_targets_to_body_frame(
      targets,
      body_frames,
      remove_initial_heading=remove_initial_heading,
    )
  elif remove_initial_heading:
    targets = _remove_initial_heading(targets)
  return targets


def load_soma_bvh_full_body_human_keypoint_targets(
  path: str | Path,
  *,
  scale: float = 0.01,
  frame_slice: slice | None = None,
  joint_names: dict[str, str] | None = None,
  apply_orientation_offsets: bool = False,
  apply_lower_body_offsets: bool = False,
  remove_initial_heading: bool = False,
  localize_to_body_frame: bool = False,
) -> list[FullBodyTarget]:
  """Load BVH full-body targets that preserve human shoulder/elbow/wrist keypoints.

  This is the paper-pure entry point for the upper arms: ``elbow - shoulder``
  matches the human's physical upper-arm direction. Use
  ``load_soma_bvh_full_body_targets`` with ``align_upper_arm_axes_to_g1=True``
  if you want the G1 axis proxy convention instead.
  """
  return load_soma_bvh_full_body_targets(
    path,
    scale=scale,
    frame_slice=frame_slice,
    joint_names=joint_names,
    apply_orientation_offsets=apply_orientation_offsets,
    align_upper_arm_axes_to_g1=False,
    apply_lower_body_offsets=apply_lower_body_offsets,
    remove_initial_heading=remove_initial_heading,
    localize_to_body_frame=localize_to_body_frame,
  )


def _body_frame_from_pose(pose, names: dict[str, str]) -> np.ndarray:
  left_hip = pose.positions[names["left_hip"]]
  right_hip = pose.positions[names["right_hip"]]
  hips = pose.positions[names["hips"]] if "hips" in names else 0.5 * (left_hip + right_hip)
  left_axis = normalize(left_hip - right_hip)
  up_axis = _project_body_up_axis(pose.positions[names["chest"]] - hips, left_axis)
  forward_axis = normalize(np.cross(left_axis, up_axis))
  up_axis = normalize(np.cross(forward_axis, left_axis))
  return np.column_stack([forward_axis, left_axis, up_axis])


def _project_body_up_axis(candidate: np.ndarray, left_axis: np.ndarray) -> np.ndarray:
  for axis in (candidate, np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0])):
    projected = axis - left_axis * float(np.dot(axis, left_axis))
    norm = float(np.linalg.norm(projected))
    if norm > 1e-12:
      return projected / norm
  raise ValueError("Cannot construct a body frame from collinear hip and chest keypoints")


def _localize_targets_to_body_frame(
  targets: list[FullBodyTarget],
  body_frames: list[np.ndarray],
  *,
  remove_initial_heading: bool,
) -> list[FullBodyTarget]:
  if not targets:
    return []
  reference_frame = body_frames[0]
  reference_correction = reference_frame
  if remove_initial_heading:
    reference_correction = _z_axis_rotation(-_heading_yaw(reference_frame)) @ reference_frame
  return [
    _rotate_full_body_target(target, reference_correction @ body_frame.T)
    for target, body_frame in zip(targets, body_frames)
  ]


def _lower_body_target_from_raw_keypoints(pose, names: dict[str, str]) -> LowerBodyTarget:
  return LowerBodyTarget(
    left_leg=LegKeypointTarget(
      hip=pose.positions[names["left_hip"]],
      knee=pose.positions[names["left_knee"]],
      ankle=pose.positions[names["left_ankle"]],
      foot_orientation=pose.rotations[names["left_foot"]],
    ),
    right_leg=LegKeypointTarget(
      hip=pose.positions[names["right_hip"]],
      knee=pose.positions[names["right_knee"]],
      ankle=pose.positions[names["right_ankle"]],
      foot_orientation=pose.rotations[names["right_foot"]],
    ),
  )


def _lower_body_target_from_scaled_effectors(
  pose,
  names: dict[str, str],
  config: LowerBodyEffectorConfig,
) -> LowerBodyTarget:
  left_hip, _left_hip_orientation = _scaled_effector(
    pose, names, "hips", "left_hip", "LeftLeg", config
  )
  left_knee, _left_knee_orientation = _scaled_effector(
    pose, names, "hips", "left_knee", "LeftShin", config
  )
  left_ankle, left_foot_orientation = _scaled_effector(
    pose, names, "hips", "left_ankle", "LeftFoot", config
  )
  right_hip, _right_hip_orientation = _scaled_effector(
    pose, names, "hips", "right_hip", "RightLeg", config
  )
  right_knee, _right_knee_orientation = _scaled_effector(
    pose, names, "hips", "right_knee", "RightShin", config
  )
  right_ankle, right_foot_orientation = _scaled_effector(
    pose, names, "hips", "right_ankle", "RightFoot", config
  )
  return LowerBodyTarget(
    left_leg=LegKeypointTarget(
      hip=left_hip,
      knee=left_knee,
      ankle=left_ankle,
      foot_orientation=left_foot_orientation,
    ),
    right_leg=LegKeypointTarget(
      hip=right_hip,
      knee=right_knee,
      ankle=right_ankle,
      foot_orientation=right_foot_orientation,
    ),
  )


def _scaled_effector(
  pose,
  names: dict[str, str],
  root_role: str,
  joint_role: str,
  config_joint_name: str,
  config: LowerBodyEffectorConfig,
) -> tuple[np.ndarray, np.ndarray]:
  root_name = names[root_role]
  joint_name = names[joint_role]
  root_position = pose.positions[root_name]
  orientation = pose.rotations[joint_name] @ config.orientation_offsets[config_joint_name]
  position = (
    (pose.positions[joint_name] - root_position) * config.joint_scales[config_joint_name]
    + root_position * config.joint_scales["Hips"]
    + orientation @ config.position_offsets[config_joint_name]
  )
  return position, orientation


def _remove_initial_heading(targets: list[FullBodyTarget]) -> list[FullBodyTarget]:
  if not targets:
    return []
  heading = _heading_yaw(targets[0].upper.chest_orientation)
  correction = _z_axis_rotation(-heading)
  return [_rotate_full_body_target(target, correction) for target in targets]


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


def _rotate_full_body_target(target: FullBodyTarget, rotation: np.ndarray) -> FullBodyTarget:
  return FullBodyTarget(
    lower=LowerBodyTarget(
      left_leg=_rotate_leg_target(target.lower.left_leg, rotation),
      right_leg=_rotate_leg_target(target.lower.right_leg, rotation),
    ),
    upper=UpperBodyTarget(
      chest_position=rotation @ target.upper.chest_position,
      chest_orientation=rotation @ target.upper.chest_orientation,
      left_arm=_rotate_arm_target(target.upper.left_arm, rotation),
      right_arm=_rotate_arm_target(target.upper.right_arm, rotation),
    ),
  )


def _rotate_leg_target(target: LegKeypointTarget, rotation: np.ndarray) -> LegKeypointTarget:
  return LegKeypointTarget(
    hip=rotation @ target.hip,
    knee=rotation @ target.knee,
    ankle=rotation @ target.ankle,
    foot_orientation=rotation @ target.foot_orientation,
  )


def _rotate_arm_target(target: ArmKeypointTarget, rotation: np.ndarray) -> ArmKeypointTarget:
  return ArmKeypointTarget(
    shoulder=rotation @ target.shoulder,
    elbow=rotation @ target.elbow,
    wrist=rotation @ target.wrist,
    hand_orientation=rotation @ target.hand_orientation,
  )
