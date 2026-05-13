import numpy as np

from src.motion.sew_lower_body import G1LowerBodySEWRetargeter, retarget_lower_body_targets
from src.motion.sew_mimic import normalize


def test_g1_lower_body_target_uses_g1_thigh_axis_with_hip_knee_sign():
  retargeter = G1LowerBodySEWRetargeter()
  q_target = np.array(
    [
      0.10,
      0.12,
      0.20,
      0.55,
      -0.20,
      0.10,
      -0.15,
      -0.12,
      -0.25,
      0.65,
      -0.10,
      -0.08,
    ],
    dtype=float,
  )
  target = retargeter.target_from_configuration(q_target)
  retargeter._set_lower_body_joint_angles(q_target)

  expected_left_thigh = -normalize(retargeter.data.xaxis[retargeter._axis_joint_ids["left"]["thigh"]])
  expected_right_thigh = -normalize(retargeter.data.xaxis[retargeter._axis_joint_ids["right"]["thigh"]])

  np.testing.assert_allclose(
    normalize(target.left_leg.knee - target.left_leg.hip),
    expected_left_thigh,
    atol=1e-8,
  )
  np.testing.assert_allclose(
    normalize(target.right_leg.knee - target.right_leg.hip),
    expected_right_thigh,
    atol=1e-8,
  )


def test_g1_lower_body_retargeter_recovers_reachable_leg_pose_without_nonlinear_ik():
  retargeter = G1LowerBodySEWRetargeter()
  q_target = np.array(
    [
      0.10,
      0.12,
      0.20,
      0.55,
      -0.20,
      0.10,
      -0.15,
      -0.12,
      -0.25,
      0.65,
      -0.10,
      -0.08,
    ],
    dtype=float,
  )
  target = retargeter.target_from_configuration(q_target)

  result = retargeter.retarget(np.zeros(12), target)

  assert result.success
  assert result.joint_angles.shape == (12,)
  assert result.full_qpos.shape == (retargeter.model.nq,)
  assert result.errors["left_thigh"] <= 1e-3
  assert result.errors["left_shank"] <= 1e-3
  assert result.errors["left_foot"] <= 1e-3
  assert result.errors["right_thigh"] <= 1e-3
  assert result.errors["right_shank"] <= 1e-3
  assert result.errors["right_foot"] <= 1e-3


def test_retarget_lower_body_targets_uses_previous_frame_as_warm_start():
  retargeter = G1LowerBodySEWRetargeter()
  q_target = np.array(
    [
      0.05,
      0.08,
      0.10,
      0.40,
      -0.12,
      0.05,
      -0.04,
      -0.08,
      -0.10,
      0.45,
      -0.10,
      -0.04,
    ],
    dtype=float,
  )
  target = retargeter.target_from_configuration(q_target)

  results = retarget_lower_body_targets([target, target], retargeter=retargeter)

  assert len(results) == 2
  np.testing.assert_allclose(results[1].joint_angles, results[0].joint_angles, atol=1e-8)
  np.testing.assert_allclose(
    results[0].full_qpos[retargeter.controlled_qpos_addresses],
    results[0].joint_angles,
  )
