import textwrap

import numpy as np

from src.motion.bvh_full_body import (
  load_soma_bvh_full_body_targets,
  soma_to_g1_lower_body_effector_config,
)
from src.motion.bvh_sew import load_bvh, soma_mujoco_to_mjlab_rotation
from src.motion.sew_mimic import normalize


def _write_tiny_full_body_bvh(path):
  frame_values = " ".join(str(value) for value in [1, 2, 3, *([0] * 42)])
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
        JOINT LeftLeg
        {
          OFFSET 0.0 10.0 0.0
          CHANNELS 3 Zrotation Yrotation Xrotation
          JOINT LeftShin
          {
            OFFSET 0.0 -40.0 0.0
            CHANNELS 3 Zrotation Yrotation Xrotation
            JOINT LeftFoot
            {
              OFFSET 0.0 -40.0 0.0
              CHANNELS 3 Zrotation Yrotation Xrotation
            }
          }
        }
        JOINT RightLeg
        {
          OFFSET 0.0 -10.0 0.0
          CHANNELS 3 Zrotation Yrotation Xrotation
          JOINT RightShin
          {
            OFFSET 0.0 -40.0 0.0
            CHANNELS 3 Zrotation Yrotation Xrotation
            JOINT RightFoot
            {
              OFFSET 0.0 -40.0 0.0
              CHANNELS 3 Zrotation Yrotation Xrotation
            }
          }
        }
      }
      MOTION
      Frames: 1
      Frame Time: 0.008333
      __FRAME_VALUES__
      """
    ).replace("__FRAME_VALUES__", frame_values),
    encoding="utf-8",
  )


def _write_tiny_humanoid_full_body_bvh(path):
  frame_values = " ".join(str(value) for value in [1, 2, 3, *([0] * 45)])
  path.write_text(
    textwrap.dedent(
      """\
      HIERARCHY
      ROOT Root
      {
        OFFSET 0.0 0.0 0.0
        CHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation
        JOINT Hips
        {
          OFFSET 0.0 90.0 0.0
          CHANNELS 3 Zrotation Yrotation Xrotation
          JOINT Chest
          {
            OFFSET 0.0 60.0 0.0
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
          JOINT LeftLeg
          {
            OFFSET 8.0 -5.0 0.0
            CHANNELS 3 Zrotation Yrotation Xrotation
            JOINT LeftShin
            {
              OFFSET 0.0 -40.0 0.0
              CHANNELS 3 Zrotation Yrotation Xrotation
              JOINT LeftFoot
              {
                OFFSET 0.0 -40.0 0.0
                CHANNELS 3 Zrotation Yrotation Xrotation
              }
            }
          }
          JOINT RightLeg
          {
            OFFSET -8.0 -5.0 0.0
            CHANNELS 3 Zrotation Yrotation Xrotation
            JOINT RightShin
            {
              OFFSET 0.0 -40.0 0.0
              CHANNELS 3 Zrotation Yrotation Xrotation
              JOINT RightFoot
              {
                OFFSET 0.0 -40.0 0.0
                CHANNELS 3 Zrotation Yrotation Xrotation
              }
            }
          }
        }
      }
      MOTION
      Frames: 1
      Frame Time: 0.008333
      __FRAME_VALUES__
      """
    ).replace("__FRAME_VALUES__", frame_values),
    encoding="utf-8",
  )


def _write_turning_humanoid_full_body_bvh(path):
  frame0 = [0, 0, 0, 0, 0, 0, *([0] * 42)]
  frame1 = [0, 0, 0, 90, 0, 0, *([0] * 42)]
  path.write_text(
    textwrap.dedent(
      """\
      HIERARCHY
      ROOT Root
      {
        OFFSET 0.0 0.0 0.0
        CHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation
        JOINT Hips
        {
          OFFSET 0.0 90.0 0.0
          CHANNELS 3 Zrotation Yrotation Xrotation
          JOINT Chest
          {
            OFFSET 0.0 60.0 0.0
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
          JOINT LeftLeg
          {
            OFFSET 8.0 -5.0 0.0
            CHANNELS 3 Zrotation Yrotation Xrotation
            JOINT LeftShin
            {
              OFFSET 0.0 -40.0 0.0
              CHANNELS 3 Zrotation Yrotation Xrotation
              JOINT LeftFoot
              {
                OFFSET 0.0 -40.0 0.0
                CHANNELS 3 Zrotation Yrotation Xrotation
              }
            }
          }
          JOINT RightLeg
          {
            OFFSET -8.0 -5.0 0.0
            CHANNELS 3 Zrotation Yrotation Xrotation
            JOINT RightShin
            {
              OFFSET 0.0 -40.0 0.0
              CHANNELS 3 Zrotation Yrotation Xrotation
              JOINT RightFoot
              {
                OFFSET 0.0 -40.0 0.0
                CHANNELS 3 Zrotation Yrotation Xrotation
              }
            }
          }
        }
      }
      MOTION
      Frames: 2
      Frame Time: 0.008333
      __FRAME0__
      __FRAME1__
      """
    )
    .replace("__FRAME0__", " ".join(str(value) for value in frame0))
    .replace("__FRAME1__", " ".join(str(value) for value in frame1)),
    encoding="utf-8",
  )


def test_load_soma_bvh_full_body_targets_extracts_upper_and_lower_body(tmp_path):
  bvh_path = tmp_path / "full.bvh"
  _write_tiny_full_body_bvh(bvh_path)

  targets = load_soma_bvh_full_body_targets(bvh_path, frame_slice=slice(0, 1))

  assert len(targets) == 1
  target = targets[0]
  np.testing.assert_allclose(target.upper.chest_position, [0.01, -0.03, 1.02])
  np.testing.assert_allclose(target.lower.left_leg.hip, [0.01, -0.03, 0.12])
  np.testing.assert_allclose(target.lower.left_leg.knee, [0.01, -0.03, -0.28])
  np.testing.assert_allclose(target.lower.left_leg.ankle, [0.01, -0.03, -0.68])
  np.testing.assert_allclose(target.lower.right_leg.hip, [0.01, -0.03, -0.08])
  np.testing.assert_allclose(target.lower.right_leg.knee, [0.01, -0.03, -0.48])
  np.testing.assert_allclose(target.lower.right_leg.ankle, [0.01, -0.03, -0.88])


def test_load_soma_bvh_full_body_targets_can_remove_initial_heading(tmp_path):
  bvh_path = tmp_path / "full.bvh"
  _write_tiny_full_body_bvh(bvh_path)

  target = load_soma_bvh_full_body_targets(
    bvh_path,
    frame_slice=slice(0, 1),
    apply_orientation_offsets=True,
    remove_initial_heading=True,
  )[0]

  heading = np.arctan2(target.upper.chest_orientation[1, 0], target.upper.chest_orientation[0, 0])
  assert abs(heading) <= 1e-12


def test_load_soma_bvh_full_body_targets_can_apply_soma_to_g1_lower_body_offsets(tmp_path):
  bvh_path = tmp_path / "humanoid_full.bvh"
  _write_tiny_humanoid_full_body_bvh(bvh_path)
  motion = load_bvh(bvh_path, scale=0.01)
  pose = motion.frame_pose(0, coordinate_rotation=soma_mujoco_to_mjlab_rotation())
  config = soma_to_g1_lower_body_effector_config()
  root = pose.positions["Hips"]
  expected_left_hip_orientation = (
    pose.rotations["LeftLeg"] @ config.orientation_offsets["LeftLeg"]
  )
  expected_left_hip = (
    (pose.positions["LeftLeg"] - root) * config.joint_scales["LeftLeg"]
    + root * config.joint_scales["Hips"]
    + expected_left_hip_orientation @ config.position_offsets["LeftLeg"]
  )

  target = load_soma_bvh_full_body_targets(
    bvh_path,
    frame_slice=slice(0, 1),
    apply_lower_body_offsets=True,
  )[0]

  np.testing.assert_allclose(target.lower.left_leg.hip, expected_left_hip)
  np.testing.assert_allclose(
    target.lower.left_leg.foot_orientation,
    pose.rotations["LeftFoot"] @ config.orientation_offsets["LeftFoot"],
  )


def test_load_soma_bvh_full_body_targets_can_localize_to_body_frame(tmp_path):
  bvh_path = tmp_path / "turning_humanoid.bvh"
  _write_turning_humanoid_full_body_bvh(bvh_path)

  targets = load_soma_bvh_full_body_targets(
    bvh_path,
    apply_lower_body_offsets=True,
    localize_to_body_frame=True,
    remove_initial_heading=True,
  )

  assert len(targets) == 2
  first_left_thigh = normalize(targets[0].lower.left_leg.knee - targets[0].lower.left_leg.hip)
  second_left_thigh = normalize(targets[1].lower.left_leg.knee - targets[1].lower.left_leg.hip)
  first_right_thigh = normalize(targets[0].lower.right_leg.knee - targets[0].lower.right_leg.hip)
  second_right_thigh = normalize(targets[1].lower.right_leg.knee - targets[1].lower.right_leg.hip)

  np.testing.assert_allclose(second_left_thigh, first_left_thigh, atol=1e-10)
  np.testing.assert_allclose(second_right_thigh, first_right_thigh, atol=1e-10)
