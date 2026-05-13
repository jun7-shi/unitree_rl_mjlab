import csv
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import scripts.compare_sew_with_seed_csv as compare
from scripts.compare_sew_with_seed_csv import CompareConfig, compare_retarget_with_seed_csv
from src.motion.seed_bones import G1_29DOF_JOINT_COLUMNS


def _write_seed_csv(path: Path, upper_values_by_frame: list[list[float]]):
  fieldnames = [
    "Frame",
    "root_translateX",
    "root_translateY",
    "root_translateZ",
    "root_rotateX",
    "root_rotateY",
    "root_rotateZ",
    *G1_29DOF_JOINT_COLUMNS,
  ]
  with path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    for frame, upper_values in enumerate(upper_values_by_frame):
      row = {name: "0" for name in fieldnames}
      row["Frame"] = str(frame)
      for name, value in zip(compare.G1_UPPER_BODY_JOINT_COLUMNS, upper_values):
        row[name] = str(value)
      writer.writerow(row)


def test_load_seed_upper_body_joint_angles_reads_degrees_as_radians(tmp_path):
  csv_path = tmp_path / "seed.csv"
  _write_seed_csv(csv_path, [[0.0, 10.0, *([0.0] * 15)]])

  q = compare.load_seed_upper_body_joint_angles(csv_path)

  assert q.shape == (1, 17)
  assert q[0, 0] == 0.0
  assert math.isclose(q[0, 1], math.radians(10.0))


def test_compare_retarget_with_seed_csv_reports_joint_differences(monkeypatch, tmp_path):
  csv_path = tmp_path / "seed.csv"
  _write_seed_csv(
    csv_path,
    [
      [0.0] * 17,
      [10.0] + [0.0] * 16,
      [20.0] + [0.0] * 16,
    ],
  )
  captured = {}

  def fake_load_targets(
    path,
    *,
    frame_slice,
    apply_orientation_offsets,
    align_upper_arm_axes_to_g1,
    remove_initial_heading,
  ):
    captured["frame_slice"] = frame_slice
    captured["apply_orientation_offsets"] = apply_orientation_offsets
    captured["align_upper_arm_axes_to_g1"] = align_upper_arm_axes_to_g1
    captured["remove_initial_heading"] = remove_initial_heading
    return [object(), object()]

  def fake_retarget_targets(targets, *, retargeter):
    return [
      SimpleNamespace(success=True, errors={"waist": 0.0}, joint_angles=np.zeros(17)),
      SimpleNamespace(success=True, errors={"waist": 0.0}, joint_angles=np.zeros(17)),
    ]

  monkeypatch.setattr(compare, "load_soma_bvh_upper_body_targets", fake_load_targets)
  monkeypatch.setattr(compare, "retarget_upper_body_targets", fake_retarget_targets)
  monkeypatch.setattr(compare, "G1UpperBodySEWRetargeter", lambda: object())

  summary = compare_retarget_with_seed_csv(
    CompareConfig(
      bvh_path=tmp_path / "unused.bvh",
      seed_csv_path=csv_path,
      start_frame=1,
      max_frames=2,
    )
  )

  assert captured["frame_slice"] == slice(1, 3)
  assert captured["apply_orientation_offsets"] is True
  assert captured["align_upper_arm_axes_to_g1"] is True
  assert captured["remove_initial_heading"] is True
  assert summary.frame_count == 2
  assert summary.success_count == 2
  assert math.isclose(summary.max_abs_joint_diff_deg, 20.0)
  assert math.isclose(summary.mean_abs_joint_diff_deg, 30.0 / 34.0)
