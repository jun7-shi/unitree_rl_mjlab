import textwrap

import numpy as np

from src.motion.bvh_full_body import load_soma_bvh_full_body_targets


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
