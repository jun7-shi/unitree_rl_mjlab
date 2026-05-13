import textwrap

import numpy as np

from src.motion.bvh_sew import (
  load_bvh,
  load_bvh_arm_targets,
  load_soma_bvh_arm_targets,
  soma_mujoco_to_mjlab_rotation,
)


def _write_tiny_bvh(path):
  path.write_text(
    textwrap.dedent(
      """\
      HIERARCHY
      ROOT Root
      {
        OFFSET 0.0 0.0 0.0
        CHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation
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
              End Site
              {
                OFFSET 5.0 0.0 0.0
              }
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
      MOTION
      Frames: 2
      Frame Time: 0.008333
      1 2 3 0 0 0 0 0 0 90 0 0 0 0 0 0 0 0 0 0 0 0 0 0
      1 2 3 0 0 0 0 0 0 90 0 0 0 0 0 0 0 0 0 0 0 0 0 0
      """
    ),
    encoding="utf-8",
  )


def test_load_bvh_computes_world_positions_and_rotations(tmp_path):
  bvh_path = tmp_path / "tiny.bvh"
  _write_tiny_bvh(bvh_path)

  motion = load_bvh(bvh_path, scale=0.01)
  frame = motion.frame_pose(0)

  np.testing.assert_allclose(frame.positions["Root"], [0.01, 0.02, 0.03])
  np.testing.assert_allclose(frame.positions["LeftArm"], [0.11, 0.02, 0.03])
  np.testing.assert_allclose(frame.positions["LeftForeArm"], [0.31, 0.02, 0.03])
  np.testing.assert_allclose(frame.positions["LeftHand"], [0.31, 0.32, 0.03], atol=1e-8)
  np.testing.assert_allclose(frame.rotations["LeftForeArm"][:, 0], [0.0, 1.0, 0.0], atol=1e-8)


def test_load_bvh_arm_targets_maps_left_and_right_keypoints(tmp_path):
  bvh_path = tmp_path / "tiny.bvh"
  _write_tiny_bvh(bvh_path)

  left_targets = load_bvh_arm_targets(bvh_path, side="left", scale=0.01)
  right_targets = load_bvh_arm_targets(bvh_path, side="right", scale=0.01)

  assert len(left_targets) == 2
  np.testing.assert_allclose(left_targets[0].shoulder, [0.11, 0.02, 0.03])
  np.testing.assert_allclose(left_targets[0].elbow, [0.31, 0.02, 0.03])
  np.testing.assert_allclose(left_targets[0].wrist, [0.31, 0.32, 0.03], atol=1e-8)
  np.testing.assert_allclose(left_targets[0].hand_orientation[:, 0], [0.0, 1.0, 0.0], atol=1e-8)

  np.testing.assert_allclose(right_targets[0].shoulder, [-0.09, 0.02, 0.03])
  np.testing.assert_allclose(right_targets[0].elbow, [-0.29, 0.02, 0.03])
  np.testing.assert_allclose(right_targets[0].wrist, [-0.59, 0.02, 0.03])


def test_soma_mujoco_rotation_matches_soma_retargeter_space_converter():
  rotation = soma_mujoco_to_mjlab_rotation()

  np.testing.assert_allclose(rotation @ np.array([1.0, 2.0, 3.0]), [1.0, -3.0, 2.0])
  np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
  assert np.linalg.det(rotation) == np.float64(1.0)


def test_load_soma_bvh_arm_targets_uses_mujoco_coordinate_conversion(tmp_path):
  bvh_path = tmp_path / "tiny.bvh"
  _write_tiny_bvh(bvh_path)

  targets = load_soma_bvh_arm_targets(bvh_path, side="left")

  np.testing.assert_allclose(targets[0].shoulder, [0.11, -0.03, 0.02])
  np.testing.assert_allclose(targets[0].elbow, [0.31, -0.03, 0.02])
  np.testing.assert_allclose(targets[0].wrist, [0.31, -0.03, 0.32], atol=1e-8)
  np.testing.assert_allclose(targets[0].hand_orientation[:, 0], [0.0, 0.0, 1.0], atol=1e-8)
