import csv
import subprocess
import sys
from pathlib import Path

import numpy as np

from src.motion.seed_bones import G1_29DOF_JOINT_COLUMNS
from tests.test_bvh_full_body import _write_tiny_full_body_bvh


def test_seed_vs_paper_raw_script_help_lists_visualization_options():
  result = subprocess.run(
    [
      sys.executable,
      "scripts/visualize_seed_vs_paper_raw_g1.py",
      "--help",
    ],
    cwd=Path(__file__).resolve().parents[1],
    check=False,
    capture_output=True,
    text=True,
  )

  assert result.returncode == 0, result.stderr
  assert "seed G1 CSV against paper_v1 raw upper-body retargeting" in result.stdout
  assert "--paper-raw-offset-to-seed" in result.stdout
  assert "--global-root" in result.stdout


def test_paper_raw_motion_row_preserves_seed_root_and_lower_body():
  from scripts.visualize_seed_vs_paper_raw_g1 import (
    UPPER_BODY_DOF,
    UPPER_BODY_START,
    paper_raw_motion_row_from_seed,
  )

  seed_row = [float(index) for index in range(7 + 29)]
  paper_upper_q = np.arange(UPPER_BODY_DOF, dtype=float) + 100.0

  paper_row = paper_raw_motion_row_from_seed(seed_row, paper_upper_q)

  assert paper_row[: 7 + UPPER_BODY_START] == seed_row[: 7 + UPPER_BODY_START]
  assert paper_row[7 + UPPER_BODY_START :] == list(paper_upper_q)


def test_unwrap_to_moves_angle_to_reference_neighborhood():
  from scripts.visualize_seed_vs_paper_raw_g1 import unwrap_to

  assert unwrap_to(-343.0, -50.0) == 17.0
  assert unwrap_to(197.0, 0.0) == -163.0


def test_comparison_offsets_arrange_columns_on_side_axis():
  from scripts.visualize_seed_vs_paper_raw_g1 import comparison_offsets

  offsets = comparison_offsets(spacing=1.25)

  np.testing.assert_allclose(offsets["bvh"], [0.0, -1.25, 0.0])
  np.testing.assert_allclose(offsets["seed"], [0.0, 0.0, 0.0])
  np.testing.assert_allclose(offsets["paper"], [0.0, 1.25, 0.0])


def test_comparison_frames_include_raw_bvh_full_body_skeleton(tmp_path):
  from scripts.visualize_seed_vs_paper_raw_g1 import VisualizerConfig, build_comparison_frames

  bvh_path = tmp_path / "full.bvh"
  _write_tiny_full_body_bvh(bvh_path)
  csv_path = tmp_path / "seed.csv"
  with csv_path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.writer(handle)
    writer.writerow(
      [
        "Frame",
        "root_translateX",
        "root_translateY",
        "root_translateZ",
        "root_rotateX",
        "root_rotateY",
        "root_rotateZ",
        *G1_29DOF_JOINT_COLUMNS,
      ]
    )
    writer.writerow([0, 0, 0, 100, 0, 0, 0, *([0] * len(G1_29DOF_JOINT_COLUMNS))])

  frame = build_comparison_frames(
    VisualizerConfig(
      bvh_path=bvh_path,
      seed_csv_path=csv_path,
      start_frame=0,
      end_frame=0,
      apply_orientation_offsets=False,
      align_upper_arm_axes_to_g1=False,
      remove_initial_heading=False,
    )
  )[0]

  assert frame.bvh_full_target.lower.left_leg.hip.shape == (3,)
  assert frame.bvh_full_target.upper.left_arm.shoulder.shape == (3,)
