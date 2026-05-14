from types import SimpleNamespace

import numpy as np

from src.motion.sew_full_body import FullBodyTarget
from src.motion.webcam_pose import (
  ExponentialJointFilter,
  ExponentialLandmarkFilter,
  _mediapipe_point_to_robot_frame,
  full_body_target_from_mediapipe_landmarks,
  upper_body_target_from_mediapipe_landmarks,
)


def _landmark(x, y, z, visibility=1.0):
  return SimpleNamespace(x=x, y=y, z=z, visibility=visibility)


def _synthetic_mediapipe_pose():
  landmarks = [_landmark(0.0, 0.0, 0.0) for _ in range(33)]
  landmarks[11] = _landmark(-0.25, -0.30, -0.10)  # left shoulder
  landmarks[12] = _landmark(0.25, -0.30, -0.10)  # right shoulder
  landmarks[13] = _landmark(-0.45, 0.05, -0.05)  # left elbow
  landmarks[14] = _landmark(0.45, 0.05, -0.05)  # right elbow
  landmarks[15] = _landmark(-0.55, 0.35, -0.05)  # left wrist
  landmarks[16] = _landmark(0.55, 0.35, -0.05)  # right wrist
  landmarks[23] = _landmark(-0.16, 0.35, 0.00)  # left hip
  landmarks[24] = _landmark(0.16, 0.35, 0.00)  # right hip
  landmarks[25] = _landmark(-0.15, 0.90, 0.02)  # left knee
  landmarks[26] = _landmark(0.15, 0.90, 0.02)  # right knee
  landmarks[27] = _landmark(-0.14, 1.35, 0.04)  # left ankle
  landmarks[28] = _landmark(0.14, 1.35, 0.04)  # right ankle
  landmarks[31] = _landmark(-0.14, 1.45, -0.15)  # left foot index
  landmarks[32] = _landmark(0.14, 1.45, -0.15)  # right foot index
  return landmarks


def test_full_body_target_from_mediapipe_landmarks_returns_retargeting_target():
  target = full_body_target_from_mediapipe_landmarks(_synthetic_mediapipe_pose())

  assert isinstance(target, FullBodyTarget)
  assert target.upper.left_arm.shoulder.shape == (3,)
  assert target.upper.chest_orientation.shape == (3, 3)
  assert target.lower.left_leg.foot_orientation.shape == (3, 3)
  np.testing.assert_allclose(
    target.upper.chest_orientation.T @ target.upper.chest_orientation,
    np.eye(3),
    atol=1e-7,
  )


def test_full_body_target_from_mediapipe_landmarks_rejects_low_visibility():
  landmarks = _synthetic_mediapipe_pose()
  landmarks[15].visibility = 0.1

  try:
    full_body_target_from_mediapipe_landmarks(landmarks, min_visibility=0.5)
  except ValueError as exc:
    assert "left_wrist" in str(exc)
  else:
    raise AssertionError("expected low visibility landmark to be rejected")


def test_upper_body_target_from_mediapipe_landmarks_allows_missing_lower_body():
  landmarks = _synthetic_mediapipe_pose()
  for index in (23, 24, 25, 26, 27, 28, 31, 32):
    landmarks[index].visibility = 0.0

  target = upper_body_target_from_mediapipe_landmarks(landmarks, min_visibility=0.5)

  assert target.left_arm.shoulder.shape == (3,)
  assert target.right_arm.wrist.shape == (3,)
  np.testing.assert_allclose(
    target.chest_orientation.T @ target.chest_orientation,
    np.eye(3),
    atol=1e-7,
  )


def test_mediapipe_depth_mapping_can_be_flipped():
  landmark = _landmark(0.25, -0.5, -0.75)

  default_point = _mediapipe_point_to_robot_frame(landmark, scale=2.0)
  flipped_point = _mediapipe_point_to_robot_frame(landmark, scale=2.0, flip_depth=True)

  np.testing.assert_allclose(default_point, [1.5, -0.5, 1.0])
  np.testing.assert_allclose(flipped_point, [-1.5, -0.5, 1.0])


def test_exponential_joint_filter_smooths_joint_angles():
  joint_filter = ExponentialJointFilter(alpha=0.25)

  first = joint_filter.update(np.array([0.0, 1.0]))
  second = joint_filter.update(np.array([1.0, 3.0]))

  np.testing.assert_allclose(first, [0.0, 1.0])
  np.testing.assert_allclose(second, [0.25, 1.5])


def test_exponential_joint_filter_limits_single_frame_joint_delta():
  joint_filter = ExponentialJointFilter(alpha=1.0, max_delta=0.5)

  first = joint_filter.update(np.array([0.0, 0.0]))
  second = joint_filter.update(np.array([2.0, -2.0]))

  np.testing.assert_allclose(first, [0.0, 0.0])
  np.testing.assert_allclose(second, [0.5, -0.5])


def test_exponential_landmark_filter_smooths_xyz_and_keeps_visibility():
  landmark_filter = ExponentialLandmarkFilter(alpha=0.25)
  first_landmarks = [_landmark(0.0, 1.0, 2.0, visibility=0.8)]
  second_landmarks = [_landmark(4.0, 5.0, 6.0, visibility=0.2)]

  first = landmark_filter.update(first_landmarks)
  second = landmark_filter.update(second_landmarks)

  assert first[0].visibility == 0.8
  np.testing.assert_allclose([second[0].x, second[0].y, second[0].z], [1.0, 2.0, 3.0])
  assert second[0].visibility == 0.2
