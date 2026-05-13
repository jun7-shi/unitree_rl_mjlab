import numpy as np

from src.motion.sew_full_body import G1FullBodySEWRetargeter, retarget_full_body_targets


def _reachable_full_body_pose():
  lower = np.array(
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
  upper = np.array(
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
  return np.concatenate([lower, upper])


def test_g1_full_body_retargeter_recovers_reachable_pose():
  retargeter = G1FullBodySEWRetargeter()
  q_target = _reachable_full_body_pose()
  target = retargeter.target_from_configuration(q_target)

  result = retargeter.retarget(np.zeros(29), target)

  assert result.success
  assert result.joint_angles.shape == (29,)
  assert result.full_qpos.shape == (retargeter.model.nq,)
  assert result.errors["left_thigh"] <= 1e-3
  assert result.errors["right_shank"] <= 1e-3
  assert result.errors["waist"] <= 1e-3
  assert result.errors["left_wrist"] <= 1e-3
  assert result.errors["right_wrist"] <= 1e-3


def test_retarget_full_body_targets_uses_previous_frame_as_warm_start():
  retargeter = G1FullBodySEWRetargeter()
  target = retargeter.target_from_configuration(_reachable_full_body_pose())

  results = retarget_full_body_targets([target, target], retargeter=retargeter)

  assert len(results) == 2
  np.testing.assert_allclose(results[1].joint_angles, results[0].joint_angles, atol=1e-8)
