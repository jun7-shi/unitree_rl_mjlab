from pathlib import Path

import numpy as np
import pytest

from src.motion.bvh_full_body import load_soma_bvh_full_body_targets
from src.motion.sew_mimic import (
  bounded_equivalent_angles,
  resolve_sew_algorithm_config,
  select_sew_candidate,
)
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


def test_full_body_retargeting_keeps_left_shoulder_branch_near_joint_limit_on_bvh_clip():
  bvh_path = (
    Path("/data/jun7.shi/datasets/bones-seed/soma_uniform/bvh/231006")
    / "dance_blinding_lights_004__A464.bvh"
  )
  if not bvh_path.exists():
    pytest.skip("local bones-seed A464 fixture is not available")
  targets = load_soma_bvh_full_body_targets(
    bvh_path,
    frame_slice=slice(650, 682),
    apply_orientation_offsets=True,
    align_upper_arm_axes_to_g1=True,
    apply_lower_body_offsets=True,
    remove_initial_heading=True,
    localize_to_body_frame=True,
  )

  results = retarget_full_body_targets(targets, retargeter=G1FullBodySEWRetargeter())
  q_deg = np.rad2deg(np.asarray([result.joint_angles for result in results], dtype=float))
  left_shoulder = q_deg[-4:, 15:18]

  assert np.max(np.abs(np.diff(left_shoulder, axis=0))) <= 15.0


def test_paper_v1_tracks_continuous_raw_left_shoulder_sew_branch_without_limit_projection():
  bvh_path = (
    Path("/data/jun7.shi/datasets/bones-seed/soma_uniform/bvh/231006")
    / "dance_blinding_lights_004__A464.bvh"
  )
  if not bvh_path.exists():
    pytest.skip("local bones-seed A464 fixture is not available")
  targets = load_soma_bvh_full_body_targets(
    bvh_path,
    frame_slice=slice(425, 526),
    apply_orientation_offsets=True,
    align_upper_arm_axes_to_g1=True,
    apply_lower_body_offsets=True,
    remove_initial_heading=True,
    localize_to_body_frame=True,
  )

  results = retarget_upper_body_targets(
    [target.upper for target in targets],
    retargeter=G1UpperBodySEWRetargeter(algorithm_version="paper_v1"),
  )
  q_deg = np.rad2deg(np.asarray([result.joint_angles for result in results], dtype=float))
  left_shoulder_pitch_roll = q_deg[:, 3:5]

  assert np.max(np.abs(np.diff(left_shoulder_pitch_roll, axis=0))) <= 15.0


def test_paper_v1_wrist_solver_does_not_apply_g1_limits_before_final_processing():
  retargeter = G1UpperBodySEWRetargeter(algorithm_version="paper_v1")
  q_target = np.zeros(17, dtype=float)
  q_target[10:17] = np.deg2rad(
    [
      362.31359148,
      -18.94043648,
      -340.91519644,
      -49.48623630,
      -36.53660000,
      -16.49030000,
      -38.55870000,
    ]
  )
  target = retargeter.target_from_configuration(q_target)
  q_init = q_target.copy()
  q_init[14:17] = np.array(
    [
      retargeter.joint_limits[14, 0] + 1e-6,
      retargeter.joint_limits[15, 0] + 1e-6,
      retargeter.joint_limits[16, 1] - 1e-6,
    ],
    dtype=float,
  )

  result = retargeter.retarget(q_init, target)

  assert result.errors["right_wrist"] <= 1e-8
  assert result.joint_angles[14] < retargeter.joint_limits[14, 0]
  assert result.joint_angles[15] < retargeter.joint_limits[15, 0]
  assert result.joint_angles[16] > retargeter.joint_limits[16, 1]


def test_branch_v2_reflects_out_of_range_shoulder_branch_without_degrading_wrist_branch():
  bvh_path = (
    Path("/data/jun7.shi/datasets/bones-seed/soma_uniform/bvh/231006")
    / "dance_blinding_lights_004__A464.bvh"
  )
  if not bvh_path.exists():
    pytest.skip("local bones-seed A464 fixture is not available")
  targets = load_soma_bvh_full_body_targets(
    bvh_path,
    frame_slice=slice(425, 526),
    apply_orientation_offsets=True,
    align_upper_arm_axes_to_g1=True,
    apply_lower_body_offsets=True,
    remove_initial_heading=True,
    localize_to_body_frame=True,
  )
  retargeter = G1UpperBodySEWRetargeter(algorithm_version="branch_v2")

  results = retarget_upper_body_targets(
    [target.upper for target in targets],
    retargeter=retargeter,
  )
  q_deg = np.rad2deg(np.asarray([result.joint_angles for result in results], dtype=float))
  left_shoulder_pitch = q_deg[:, 3]
  left_shoulder_yaw = q_deg[:, 5]
  left_wrist_roll_yaw = q_deg[:, [7, 9]]

  assert np.max(np.abs(np.diff(left_shoulder_pitch))) <= 15.0
  assert np.max(np.abs(np.diff(left_shoulder_yaw))) <= 15.0
  np.testing.assert_allclose(left_shoulder_pitch[75], -139.5502730848912, atol=1e-3)
  np.testing.assert_allclose(left_shoulder_yaw[75], -124.83731114938365, atol=1e-1)
  assert np.max(np.abs(left_wrist_roll_yaw[:, 0])) <= 30.0
  assert np.max(np.abs(left_wrist_roll_yaw[:, 1])) <= 30.0


def test_algorithm_versions_have_distinct_limit_and_branch_selection_semantics():
  joint_limits = np.array([[0.0, 1.0]], dtype=float)
  paper_v1 = resolve_sew_algorithm_config("paper_v1")
  branch_v2 = resolve_sew_algorithm_config("branch_v2")
  near_saturated = np.array([1.0, 2.0])
  far_exact = np.array([5.0, 6.0])

  np.testing.assert_allclose(
    bounded_equivalent_angles(1.1, joint_limits, 0, config=paper_v1),
    [1.1 - 2.0 * np.pi, 1.1, 1.1 + 2.0 * np.pi],
  )
  np.testing.assert_allclose(
    bounded_equivalent_angles(1.1, joint_limits, 0, config=branch_v2),
    [1.0],
  )

  selected_v1 = select_sew_candidate(
    [
      (near_saturated, (1.5e-3, 0.1), 0.1),
      (far_exact, (0.0, 4.0), 4.0),
    ],
    config=paper_v1,
  )
  selected_v2 = select_sew_candidate(
    [
      (near_saturated, (1.5e-3, 0.1), 0.1),
      (far_exact, (0.0, 4.0), 4.0),
    ],
    config=branch_v2,
  )

  np.testing.assert_allclose(selected_v1, far_exact)
  np.testing.assert_allclose(selected_v2, near_saturated)


def test_full_body_retargeting_keeps_left_wrist_reachable_after_a464_branch_change():
  bvh_path = (
    Path("/data/jun7.shi/datasets/bones-seed/soma_uniform/bvh/231006")
    / "dance_blinding_lights_004__A464.bvh"
  )
  if not bvh_path.exists():
    pytest.skip("local bones-seed A464 fixture is not available")
  targets = load_soma_bvh_full_body_targets(
    bvh_path,
    frame_slice=slice(650, 710),
    apply_orientation_offsets=True,
    align_upper_arm_axes_to_g1=True,
    apply_lower_body_offsets=True,
    remove_initial_heading=True,
    localize_to_body_frame=True,
  )

  results = retarget_full_body_targets(targets, retargeter=G1FullBodySEWRetargeter())
  q_deg = np.rad2deg(np.asarray([result.joint_angles for result in results], dtype=float))
  left_wrist_roll = q_deg[:, 19]
  left_wrist_yaw = q_deg[:, 21]

  assert np.max(np.abs(left_wrist_roll[-15:])) <= 25.0
  assert np.max(np.abs(np.diff(left_wrist_roll[-15:]))) <= 3.0
  assert np.max(np.abs(left_wrist_yaw[-15:])) <= 15.0
  assert np.max(np.abs(np.diff(left_wrist_yaw[-15:]))) <= 3.0


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
