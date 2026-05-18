from __future__ import annotations

from copy import copy
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from src.motion.bvh_upper_body import synthesize_g1_axis_proxy_arm_target
from src.motion.sew_full_body import FullBodyTarget
from src.motion.sew_lower_body import LegKeypointTarget, LowerBodyTarget
from src.motion.sew_mimic import ArmKeypointTarget, normalize
from src.motion.sew_upper_body import UpperBodyTarget


MEDIAPIPE_POSE_LANDMARKS = {
  "left_shoulder": 11,
  "right_shoulder": 12,
  "left_elbow": 13,
  "right_elbow": 14,
  "left_wrist": 15,
  "right_wrist": 16,
  "left_hip": 23,
  "right_hip": 24,
  "left_knee": 25,
  "right_knee": 26,
  "left_ankle": 27,
  "right_ankle": 28,
  "left_foot": 31,
  "right_foot": 32,
}


@dataclass
class ExponentialJointFilter:
  """Stateful exponential smoother for realtime retargeted joint angles."""

  alpha: float = 0.35
  max_delta: float | None = None
  _state: np.ndarray | None = None

  def __post_init__(self) -> None:
    if not 0.0 < self.alpha <= 1.0:
      raise ValueError(f"alpha must be in (0, 1], got {self.alpha}")
    if self.max_delta is not None and self.max_delta <= 0.0:
      raise ValueError(f"max_delta must be positive when set, got {self.max_delta}")

  def reset(self) -> None:
    self._state = None

  def update(self, values: Sequence[float] | np.ndarray) -> np.ndarray:
    current = np.asarray(values, dtype=float)
    if self._state is None or self._state.shape != current.shape:
      self._state = current.copy()
      return self._state.copy()
    filtered = (1.0 - self.alpha) * self._state + self.alpha * current
    if self.max_delta is not None:
      delta = np.clip(filtered - self._state, -self.max_delta, self.max_delta)
      filtered = self._state + delta
    self._state = filtered
    return self._state.copy()


@dataclass
class ExponentialLandmarkFilter:
  """Stateful exponential smoother for MediaPipe landmark xyz coordinates."""

  alpha: float = 0.35
  _state: np.ndarray | None = None

  def __post_init__(self) -> None:
    if not 0.0 < self.alpha <= 1.0:
      raise ValueError(f"alpha must be in (0, 1], got {self.alpha}")

  def reset(self) -> None:
    self._state = None

  def update(self, landmarks) -> list:
    current = np.asarray(
      [[float(landmark.x), float(landmark.y), float(landmark.z)] for landmark in landmarks],
      dtype=float,
    )
    if self._state is None or self._state.shape != current.shape:
      self._state = current.copy()
    else:
      self._state = (1.0 - self.alpha) * self._state + self.alpha * current

    filtered_landmarks = []
    for landmark, coordinates in zip(landmarks, self._state):
      filtered = copy(landmark)
      filtered.x = float(coordinates[0])
      filtered.y = float(coordinates[1])
      filtered.z = float(coordinates[2])
      filtered_landmarks.append(filtered)
    return filtered_landmarks


def full_body_target_from_mediapipe_landmarks(
  landmarks,
  *,
  scale: float = 1.0,
  min_visibility: float = 0.5,
  align_upper_arm_axes_to_g1: bool = True,
  flip_depth: bool = False,
) -> FullBodyTarget:
  """Convert MediaPipe Pose landmarks into the existing full-body SEW target."""
  points = _extract_pose_points(
    landmarks,
    scale=scale,
    min_visibility=min_visibility,
    flip_depth=flip_depth,
  )
  upper = _upper_body_target_from_points(
    points,
    align_upper_arm_axes_to_g1=align_upper_arm_axes_to_g1,
  )
  left_foot_orientation = _limb_orientation(
    points["left_foot"] - points["left_ankle"],
    upper.chest_orientation[:, 2],
  )
  right_foot_orientation = _limb_orientation(
    points["right_foot"] - points["right_ankle"],
    upper.chest_orientation[:, 2],
  )

  return FullBodyTarget(
    lower=LowerBodyTarget(
      left_leg=LegKeypointTarget(
        hip=points["left_hip"],
        knee=points["left_knee"],
        ankle=points["left_ankle"],
        foot_orientation=left_foot_orientation,
      ),
      right_leg=LegKeypointTarget(
        hip=points["right_hip"],
        knee=points["right_knee"],
        ankle=points["right_ankle"],
        foot_orientation=right_foot_orientation,
      ),
    ),
    upper=upper,
  )


def upper_body_target_from_mediapipe_landmarks(
  landmarks,
  *,
  scale: float = 1.0,
  min_visibility: float = 0.5,
  align_upper_arm_axes_to_g1: bool = True,
  flip_depth: bool = False,
) -> UpperBodyTarget:
  """Convert MediaPipe Pose landmarks into an upper-body SEW target."""
  roles = (
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
  )
  points = _extract_pose_points(
    landmarks,
    scale=scale,
    min_visibility=min_visibility,
    roles=roles,
    flip_depth=flip_depth,
  )
  return _upper_body_target_from_points(
    points,
    align_upper_arm_axes_to_g1=align_upper_arm_axes_to_g1,
  )


def _upper_body_target_from_points(
  points: dict[str, np.ndarray],
  *,
  align_upper_arm_axes_to_g1: bool,
) -> UpperBodyTarget:
  chest_orientation = _body_orientation(points)
  left_hand_orientation = _limb_orientation(
    points["left_wrist"] - points["left_elbow"],
    chest_orientation[:, 2],
  )
  right_hand_orientation = _limb_orientation(
    points["right_wrist"] - points["right_elbow"],
    chest_orientation[:, 2],
  )
  left_arm = ArmKeypointTarget(
    shoulder=points["left_shoulder"],
    elbow=points["left_elbow"],
    wrist=points["left_wrist"],
    hand_orientation=left_hand_orientation,
  )
  right_arm = ArmKeypointTarget(
    shoulder=points["right_shoulder"],
    elbow=points["right_elbow"],
    wrist=points["right_wrist"],
    hand_orientation=right_hand_orientation,
  )
  if align_upper_arm_axes_to_g1:
    left_arm = synthesize_g1_axis_proxy_arm_target(left_arm)
    right_arm = synthesize_g1_axis_proxy_arm_target(right_arm)
  return UpperBodyTarget(
    chest_position=0.5 * (points["left_shoulder"] + points["right_shoulder"]),
    chest_orientation=chest_orientation,
    left_arm=left_arm,
    right_arm=right_arm,
  )


def _extract_pose_points(
  landmarks,
  *,
  scale: float,
  min_visibility: float,
  roles: Sequence[str] | None = None,
  flip_depth: bool = False,
) -> dict[str, np.ndarray]:
  points: dict[str, np.ndarray] = {}
  selected_roles = MEDIAPIPE_POSE_LANDMARKS if roles is None else {
    role: MEDIAPIPE_POSE_LANDMARKS[role]
    for role in roles
  }
  for name, index in selected_roles.items():
    landmark = landmarks[index]
    visibility = float(getattr(landmark, "visibility", 1.0))
    if visibility < min_visibility:
      raise ValueError(f"MediaPipe landmark '{name}' visibility {visibility:.3f} is below {min_visibility:.3f}")
    points[name] = _mediapipe_point_to_robot_frame(landmark, scale=scale, flip_depth=flip_depth)
  return points


def _mediapipe_point_to_robot_frame(landmark, *, scale: float, flip_depth: bool = False) -> np.ndarray:
  robot_x = float(landmark.z) if flip_depth else -float(landmark.z)
  point = np.array(
    [
      robot_x,
      -float(landmark.x),
      -float(landmark.y),
    ],
    dtype=float,
  )
  return point * float(scale)


def _body_orientation(points: dict[str, np.ndarray]) -> np.ndarray:
  shoulder_mid = 0.5 * (points["left_shoulder"] + points["right_shoulder"])
  hip_mid = 0.5 * (points["left_hip"] + points["right_hip"]) if (
    "left_hip" in points and "right_hip" in points
  ) else shoulder_mid - np.array([0.0, 0.0, 1.0], dtype=float)
  y_axis = normalize(points["left_shoulder"] - points["right_shoulder"])
  z_axis = normalize(shoulder_mid - hip_mid)
  return _orthonormal_frame(y_axis=y_axis, z_axis=z_axis)


def _limb_orientation(direction: np.ndarray, up_hint: np.ndarray) -> np.ndarray:
  x_axis = normalize(direction)
  z_hint = np.asarray(up_hint, dtype=float)
  z_axis = z_hint - float(np.dot(z_hint, x_axis)) * x_axis
  if np.linalg.norm(z_axis) <= 1e-8:
    z_axis = np.array([0.0, 0.0, 1.0], dtype=float)
    z_axis = z_axis - float(np.dot(z_axis, x_axis)) * x_axis
  if np.linalg.norm(z_axis) <= 1e-8:
    z_axis = np.array([0.0, 1.0, 0.0], dtype=float)
    z_axis = z_axis - float(np.dot(z_axis, x_axis)) * x_axis
  z_axis = normalize(z_axis)
  y_axis = normalize(np.cross(z_axis, x_axis))
  z_axis = normalize(np.cross(x_axis, y_axis))
  return np.column_stack((x_axis, y_axis, z_axis))


def _orthonormal_frame(*, y_axis: np.ndarray, z_axis: np.ndarray) -> np.ndarray:
  y = normalize(y_axis)
  z_hint = normalize(z_axis)
  x = normalize(np.cross(y, z_hint))
  z = normalize(np.cross(x, y))
  return np.column_stack((x, y, z))
