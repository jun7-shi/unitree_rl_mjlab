import textwrap

import numpy as np

from src.motion.bvh_upper_body import (
  load_soma_bvh_upper_body_human_keypoint_targets,
  load_soma_bvh_upper_body_targets,
  soma_to_g1_upper_body_orientation_offsets,
)


def _write_tiny_upper_body_bvh(path):
  path.write_text(
    textwrap.dedent(
      """\
      HIERARCHY
      ROOT Root
      {
        OFFSET 0.0 0.0 0.0
        CHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation
        JOINT Chest
        {
          OFFSET 0.0 100.0 0.0
          CHANNELS 3 Zrotation Yrotation Xrotation
          JOINT LeftArm
          {
            OFFSET 10.0 0.0 0.0
            CHANNELS 3 Zrotation Yrotation Xrotation
            JOINT LeftForeArm
            {
              OFFSET 20.0 0.0 0.0
              CHANNELS 3 Zrotation Yrotation Xrotation
              JOINT LeftHand
              {
                OFFSET 30.0 0.0 0.0
                CHANNELS 3 Zrotation Yrotation Xrotation
              }
            }
          }
          JOINT RightArm
          {
            OFFSET -10.0 0.0 0.0
            CHANNELS 3 Zrotation Yrotation Xrotation
            JOINT RightForeArm
            {
              OFFSET -20.0 0.0 0.0
              CHANNELS 3 Zrotation Yrotation Xrotation
              JOINT RightHand
              {
                OFFSET -30.0 0.0 0.0
                CHANNELS 3 Zrotation Yrotation Xrotation
              }
            }
          }
        }
      }
      MOTION
      Frames: 2
      Frame Time: 0.008333
      1 2 3 0 0 0 0 0 0 0 0 0 90 0 0 0 0 0 0 0 0 0 0 0 0 0 0
      1 2 3 0 0 0 0 0 0 0 0 0 90 0 0 0 0 0 0 0 0 0 0 0 0 0 0
      """
    ),
    encoding="utf-8",
  )


def _heading_yaw(matrix):
  return np.arctan2(matrix[1, 0], matrix[0, 0])


def test_load_soma_bvh_upper_body_targets_extracts_chest_and_both_arms(tmp_path):
  bvh_path = tmp_path / "upper.bvh"
  _write_tiny_upper_body_bvh(bvh_path)

  targets = load_soma_bvh_upper_body_targets(bvh_path, frame_slice=slice(0, 1))

  assert len(targets) == 1
  target = targets[0]
  np.testing.assert_allclose(target.chest_position, [0.01, -0.03, 1.02])
  np.testing.assert_allclose(target.chest_orientation.T @ target.chest_orientation, np.eye(3), atol=1e-12)

  np.testing.assert_allclose(target.left_arm.shoulder, [0.11, -0.03, 1.02])
  np.testing.assert_allclose(target.left_arm.elbow, [0.31, -0.03, 1.02])
  np.testing.assert_allclose(target.left_arm.wrist, [0.31, -0.03, 1.32], atol=1e-8)
  np.testing.assert_allclose(target.left_arm.hand_orientation[:, 0], [0.0, 0.0, 1.0], atol=1e-8)

  np.testing.assert_allclose(target.right_arm.shoulder, [-0.09, -0.03, 1.02])
  np.testing.assert_allclose(target.right_arm.elbow, [-0.29, -0.03, 1.02])
  np.testing.assert_allclose(target.right_arm.wrist, [-0.59, -0.03, 1.02])


def test_load_soma_bvh_upper_body_targets_can_apply_soma_to_g1_orientation_offsets(tmp_path):
  bvh_path = tmp_path / "upper.bvh"
  _write_tiny_upper_body_bvh(bvh_path)

  base = load_soma_bvh_upper_body_targets(bvh_path, frame_slice=slice(0, 1))[0]
  aligned = load_soma_bvh_upper_body_targets(
    bvh_path,
    frame_slice=slice(0, 1),
    apply_orientation_offsets=True,
  )[0]
  offsets = soma_to_g1_upper_body_orientation_offsets()

  np.testing.assert_allclose(aligned.chest_orientation, base.chest_orientation @ offsets["chest"])
  np.testing.assert_allclose(
    aligned.left_arm.hand_orientation,
    base.left_arm.hand_orientation @ offsets["left_hand"],
  )
  np.testing.assert_allclose(
    aligned.right_arm.hand_orientation,
    base.right_arm.hand_orientation @ offsets["right_hand"],
  )


def test_load_soma_bvh_upper_body_targets_can_synthesize_g1_axis_proxy_targets(tmp_path):
  bvh_path = tmp_path / "upper.bvh"
  _write_tiny_upper_body_bvh(bvh_path)

  base = load_soma_bvh_upper_body_targets(bvh_path, frame_slice=slice(0, 1))[0]
  proxy = load_soma_bvh_upper_body_targets(
    bvh_path,
    frame_slice=slice(0, 1),
    align_upper_arm_axes_to_g1=True,
  )[0]

  base_left_upper = base.left_arm.elbow - base.left_arm.shoulder
  proxy_left_upper = proxy.left_arm.elbow - proxy.left_arm.shoulder
  base_left_lower = base.left_arm.wrist - base.left_arm.elbow
  proxy_left_lower = proxy.left_arm.wrist - proxy.left_arm.elbow
  assert not np.allclose(proxy.left_arm.shoulder, base.left_arm.shoulder)
  np.testing.assert_allclose(proxy_left_upper, -base_left_upper)
  np.testing.assert_allclose(proxy_left_lower, base_left_lower)

  base_right_upper = base.right_arm.elbow - base.right_arm.shoulder
  proxy_right_upper = proxy.right_arm.elbow - proxy.right_arm.shoulder
  base_right_lower = base.right_arm.wrist - base.right_arm.elbow
  proxy_right_lower = proxy.right_arm.wrist - proxy.right_arm.elbow
  assert not np.allclose(proxy.right_arm.shoulder, base.right_arm.shoulder)
  np.testing.assert_allclose(proxy_right_upper, -base_right_upper)
  np.testing.assert_allclose(proxy_right_lower, base_right_lower)


def test_human_keypoint_loader_preserves_shoulder_while_proxy_loader_replaces_it(tmp_path):
  bvh_path = tmp_path / "upper.bvh"
  _write_tiny_upper_body_bvh(bvh_path)

  human = load_soma_bvh_upper_body_human_keypoint_targets(bvh_path, frame_slice=slice(0, 1))[0]
  default = load_soma_bvh_upper_body_targets(bvh_path, frame_slice=slice(0, 1))[0]
  proxy = load_soma_bvh_upper_body_targets(
    bvh_path,
    frame_slice=slice(0, 1),
    align_upper_arm_axes_to_g1=True,
  )[0]

  np.testing.assert_allclose(human.left_arm.shoulder, default.left_arm.shoulder)
  np.testing.assert_allclose(human.right_arm.shoulder, default.right_arm.shoulder)

  assert not np.allclose(proxy.left_arm.shoulder, human.left_arm.shoulder)
  assert not np.allclose(proxy.right_arm.shoulder, human.right_arm.shoulder)
  np.testing.assert_allclose(proxy.left_arm.shoulder, human.left_arm.elbow)
  np.testing.assert_allclose(proxy.right_arm.shoulder, human.right_arm.elbow)


def test_load_soma_bvh_upper_body_targets_can_remove_initial_heading(tmp_path):
  bvh_path = tmp_path / "upper.bvh"
  _write_tiny_upper_body_bvh(bvh_path)

  base = load_soma_bvh_upper_body_targets(
    bvh_path,
    frame_slice=slice(0, 1),
    apply_orientation_offsets=True,
  )[0]
  aligned = load_soma_bvh_upper_body_targets(
    bvh_path,
    frame_slice=slice(0, 1),
    apply_orientation_offsets=True,
    remove_initial_heading=True,
  )[0]

  assert abs(_heading_yaw(aligned.chest_orientation)) <= 1e-12
  assert abs(_heading_yaw(base.chest_orientation)) > 1e-3
