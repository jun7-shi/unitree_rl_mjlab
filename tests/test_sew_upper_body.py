import numpy as np

from src.motion.sew_upper_body import G1UpperBodySEWRetargeter, retarget_upper_body_targets


def test_g1_upper_body_retargeter_recovers_reachable_upper_body_pose():
  retargeter = G1UpperBodySEWRetargeter()
  q_target = np.array(
    [
      0.12,
      0.08,
      -0.10,
      0.20,
      0.35,
      -0.45,
      0.90,
      0.25,
      -0.35,
      0.40,
      -0.15,
      -0.30,
      0.35,
      0.70,
      -0.20,
      0.25,
      -0.30,
    ],
    dtype=float,
  )
  target = retargeter.target_from_configuration(q_target)

  result = retargeter.retarget(np.zeros(17), target)

  assert result.success
  assert result.joint_angles.shape == (17,)
  assert result.full_qpos.shape == (retargeter.model.nq,)
  assert result.errors["waist"] <= 1e-3
  assert result.errors["left_upper_arm"] <= 1e-3
  assert result.errors["left_lower_arm"] <= 1e-3
  assert result.errors["left_wrist"] <= 1e-3
  assert result.errors["right_upper_arm"] <= 1e-3
  assert result.errors["right_lower_arm"] <= 1e-3
  assert result.errors["right_wrist"] <= 1e-3


def test_retarget_upper_body_targets_uses_previous_frame_as_warm_start():
  retargeter = G1UpperBodySEWRetargeter()
  q_target = np.array(
    [
      0.05,
      -0.04,
      0.03,
      0.10,
      0.20,
      -0.20,
      0.50,
      0.10,
      -0.10,
      0.20,
      -0.10,
      -0.20,
      0.20,
      0.50,
      -0.10,
      0.10,
      -0.20,
    ],
    dtype=float,
  )
  target = retargeter.target_from_configuration(q_target)

  results = retarget_upper_body_targets([target, target], retargeter=retargeter)

  assert len(results) == 2
  np.testing.assert_allclose(results[1].joint_angles, results[0].joint_angles, atol=1e-8)
  np.testing.assert_allclose(
    results[0].full_qpos[retargeter.controlled_qpos_addresses],
    results[0].joint_angles,
  )
