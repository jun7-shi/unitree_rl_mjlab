from pathlib import Path

import numpy as np
import pytest

from src.motion.bvh_full_body import load_soma_bvh_full_body_targets
from src.motion.sew_full_body import G1FullBodySEWRetargeter, retarget_full_body_targets
from src.motion.sew_upper_body import G1UpperBodySEWRetargeter, retarget_upper_body_targets


def test_full_body_retargeting_avoids_right_elbow_branch_jitter_on_bvh_clip():
  bvh_path = (
    Path("/data/jun7.shi/datasets/bones-seed/soma_uniform/bvh/230413")
    / "dance_vouge_dancehall_open_close_boogle_180_R_fast_002__A320_M.bvh"
  )
  if not bvh_path.exists():
    pytest.skip("local bones-seed A320 fixture is not available")
  targets = load_soma_bvh_full_body_targets(
    bvh_path,
    frame_slice=slice(0, 10),
    apply_orientation_offsets=True,
    align_upper_arm_axes_to_g1=True,
    apply_lower_body_offsets=True,
    remove_initial_heading=True,
  )

  results = retarget_full_body_targets(targets, retargeter=G1FullBodySEWRetargeter())
  q_deg = np.rad2deg(np.asarray([result.joint_angles for result in results], dtype=float))
  right_shoulder_yaw = q_deg[:, 24]
  right_elbow = q_deg[:, 25]

  assert np.max(np.abs(np.diff(right_shoulder_yaw))) <= 1.0
  assert np.max(np.abs(np.diff(right_elbow))) <= 1.0


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
