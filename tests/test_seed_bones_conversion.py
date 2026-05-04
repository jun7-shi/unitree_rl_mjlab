import math
import tempfile
import textwrap
import unittest
from pathlib import Path

from src.motion.seed_bones import (
  G1_29DOF_JOINT_COLUMNS,
  apply_root_z_offset,
  compute_floor_alignment_offset,
  convert_seed_bones_csv_to_motion_rows,
)


class SeedBonesConversionTests(unittest.TestCase):
  def test_convert_seed_bones_csv_to_motion_rows_normalizes_units_and_layout(self):
    joint_values_deg = [str(i) for i in range(1, len(G1_29DOF_JOINT_COLUMNS) + 1)]
    csv_text = textwrap.dedent(
      """\
      Frame,root_translateX,root_translateY,root_translateZ,root_rotateX,root_rotateY,root_rotateZ,{joint_headers}
      0,100,200,300,0,0,90,{joint_values}
      """
    ).format(
      joint_headers=",".join(G1_29DOF_JOINT_COLUMNS),
      joint_values=",".join(joint_values_deg),
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
      input_path = Path(tmp_dir) / "seed.csv"
      input_path.write_text(csv_text, encoding="utf-8")

      rows = convert_seed_bones_csv_to_motion_rows(str(input_path))

    self.assertEqual(len(rows), 1)
    self.assertEqual(len(rows[0]), 36)

    self.assertAlmostEqual(rows[0][0], 1.0)
    self.assertAlmostEqual(rows[0][1], 2.0)
    self.assertAlmostEqual(rows[0][2], 3.0)

    quat_xyzw = rows[0][3:7]
    quat_norm = math.sqrt(sum(value * value for value in quat_xyzw))
    self.assertAlmostEqual(quat_norm, 1.0, places=6)
    self.assertAlmostEqual(quat_xyzw[0], 0.0, places=6)
    self.assertAlmostEqual(quat_xyzw[1], 0.0, places=6)
    self.assertAlmostEqual(quat_xyzw[2], math.sqrt(0.5), places=6)
    self.assertAlmostEqual(quat_xyzw[3], math.sqrt(0.5), places=6)

    self.assertAlmostEqual(rows[0][7], math.radians(1.0), places=6)
    self.assertAlmostEqual(
      rows[0][-1], math.radians(float(len(G1_29DOF_JOINT_COLUMNS))), places=6
    )

  def test_convert_seed_bones_csv_to_motion_rows_remaps_pitch_and_roll_axes(self):
    joint_values_deg = ",".join(["0"] * len(G1_29DOF_JOINT_COLUMNS))
    csv_text = textwrap.dedent(
      """\
      Frame,root_translateX,root_translateY,root_translateZ,root_rotateX,root_rotateY,root_rotateZ,{joint_headers}
      0,0,0,100,90,0,0,{joint_values}
      1,0,0,100,0,90,0,{joint_values}
      """
    ).format(
      joint_headers=",".join(G1_29DOF_JOINT_COLUMNS),
      joint_values=joint_values_deg,
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
      input_path = Path(tmp_dir) / "seed_axes.csv"
      input_path.write_text(csv_text, encoding="utf-8")

      rows = convert_seed_bones_csv_to_motion_rows(str(input_path))

    pitch_quat = rows[0][3:7]
    self.assertAlmostEqual(pitch_quat[0], 0.0, places=6)
    self.assertAlmostEqual(pitch_quat[1], -math.sqrt(0.5), places=6)
    self.assertAlmostEqual(pitch_quat[2], 0.0, places=6)
    self.assertAlmostEqual(pitch_quat[3], math.sqrt(0.5), places=6)

    roll_quat = rows[1][3:7]
    self.assertAlmostEqual(roll_quat[0], math.sqrt(0.5), places=6)
    self.assertAlmostEqual(roll_quat[1], 0.0, places=6)
    self.assertAlmostEqual(roll_quat[2], 0.0, places=6)
    self.assertAlmostEqual(roll_quat[3], math.sqrt(0.5), places=6)

  def test_compute_floor_alignment_offset_raises_lowest_foot_to_clearance(self):
    offset = compute_floor_alignment_offset(
      foot_lower_bounds_m=[-0.04, -0.06, 0.02],
      floor_height_m=0.0,
      clearance_m=0.005,
    )

    self.assertAlmostEqual(offset, 0.065)

  def test_apply_root_z_offset_only_changes_root_height(self):
    rows = [
      [1.0, 2.0, 0.7, 0.0, 0.0, 0.0, 1.0, 0.1],
      [1.5, 2.5, 0.8, 0.0, 0.0, 0.0, 1.0, 0.2],
    ]

    shifted = apply_root_z_offset(rows, 0.06)

    self.assertEqual(rows[0][2], 0.7)
    self.assertEqual(shifted[0][:2], rows[0][:2])
    self.assertAlmostEqual(shifted[0][2], 0.76)
    self.assertEqual(shifted[0][3:], rows[0][3:])
    self.assertAlmostEqual(shifted[1][2], 0.86)


if __name__ == "__main__":
  unittest.main()
