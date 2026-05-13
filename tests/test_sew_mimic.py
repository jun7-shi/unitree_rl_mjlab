import math

import numpy as np
import pytest

from src.motion.sew_mimic import (
  G1SEWMimicRetargeter,
  normalize,
  retarget_motion_rows_from_targets,
  rotation_about_axis,
  solve_two_axis_rotation,
  subproblem1,
  subproblem4,
)


def test_normalize_rejects_zero_length_vectors():
  with pytest.raises(ValueError, match="zero-length"):
    normalize([0.0, 0.0, 0.0])


def test_subproblem1_aligns_vector_by_rotation_about_axis():
  theta = subproblem1([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0])

  rotated = rotation_about_axis([0.0, 0.0, 1.0], theta) @ np.array([1.0, 0.0, 0.0])

  assert theta == pytest.approx(math.pi / 2.0, abs=1e-7)
  np.testing.assert_allclose(rotated, [0.0, 1.0, 0.0], atol=1e-7)


def test_subproblem4_returns_angles_that_place_rotated_vector_on_plane():
  angles = subproblem4(
    p=[1.0, 0.0, 0.0],
    h=[0.0, 1.0, 0.0],
    k=[0.0, 0.0, 1.0],
    d=1.0,
  )

  assert len(angles) == 1
  rotated = rotation_about_axis([0.0, 0.0, 1.0], angles[0]) @ np.array([1.0, 0.0, 0.0])

  assert angles[0] == pytest.approx(math.pi / 2.0, abs=1e-7)
  assert float(np.dot([0.0, 1.0, 0.0], rotated)) == pytest.approx(1.0, abs=1e-7)


def test_solve_two_axis_rotation_matches_paper_subproblem2_composition():
  first_axis = np.array([0.0, 0.0, 1.0])
  second_axis = np.array([0.0, 1.0, 0.0])
  initial_axis = normalize([1.0, 0.3, 0.2])
  expected_first = 0.7
  expected_second = -0.4
  target_axis = (
    rotation_about_axis(first_axis, expected_first)
    @ rotation_about_axis(second_axis, expected_second)
    @ initial_axis
  )

  candidates = solve_two_axis_rotation(
    initial_axis,
    target_axis,
    first_axis,
    second_axis,
  )

  assert any(
    np.linalg.norm(
      rotation_about_axis(first_axis, first)
      @ rotation_about_axis(second_axis, second)
      @ initial_axis
      - target_axis
    ) <= 1e-8
    for first, second in candidates
  )


def test_g1_retargeter_recovers_targets_generated_from_reachable_pose():
  retargeter = G1SEWMimicRetargeter(side="left")
  q_init = np.zeros(7)
  q_target = np.array([0.2, 0.35, -0.45, 0.9, 0.25, -0.35, 0.4])

  target = retargeter.target_from_configuration(q_target)
  result = retargeter.retarget(
    q_init,
    shoulder=target.shoulder,
    elbow=target.elbow,
    wrist=target.wrist,
    hand_orientation=target.hand_orientation,
  )

  assert result.success
  assert result.joint_angles.shape == (7,)
  assert np.all(result.joint_angles >= retargeter.joint_limits[:, 0] - 1e-8)
  assert np.all(result.joint_angles <= retargeter.joint_limits[:, 1] + 1e-8)
  assert result.upper_arm_error <= 1e-3
  assert result.lower_arm_error <= 1e-3
  assert result.wrist_error <= 1e-3


def test_retarget_motion_rows_from_targets_updates_g1_arm_columns_for_mocap_handoff():
  retargeter = G1SEWMimicRetargeter(side="left")
  q_target = np.array([0.1, 0.2, -0.2, 0.5, 0.1, -0.1, 0.2])
  target = retargeter.target_from_configuration(q_target)
  row = [0.0] * 36
  row[6] = 1.0

  output = retarget_motion_rows_from_targets(
    [row], side="left", targets=[target], retargeter=retargeter
  )

  assert len(output) == 1
  assert len(output[0]) == 36
  np.testing.assert_allclose(output[0][22:29], q_target, atol=1e-6)
  assert output[0][:22] == row[:22]
