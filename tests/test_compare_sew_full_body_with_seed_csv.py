import csv
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import scripts.compare_sew_full_body_with_seed_csv as compare
from scripts.compare_sew_full_body_with_seed_csv import (
  CompareConfig,
  compare_retarget_with_seed_csv,
)
from src.motion.seed_bones import G1_29DOF_JOINT_COLUMNS


def _write_seed_csv(path: Path, values_by_frame: list[list[float]]):
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
    for frame, values in enumerate(values_by_frame):
      row = {name: "0" for name in fieldnames}
      row["Frame"] = str(frame)
      for name, value in zip(G1_29DOF_JOINT_COLUMNS, values):
        row[name] = str(value)
      writer.writerow(row)


def test_load_seed_full_body_joint_angles_reads_degrees_as_radians(tmp_path):
  csv_path = tmp_path / "seed.csv"
  _write_seed_csv(csv_path, [[0.0, 10.0, *([0.0] * 27)]])

  q = compare.load_seed_full_body_joint_angles(csv_path)

  assert q.shape == (1, 29)
  assert q[0, 0] == 0.0
  assert math.isclose(q[0, 1], math.radians(10.0))


def test_compare_retarget_with_seed_csv_uses_full_body_loader_defaults(monkeypatch, tmp_path):
  csv_path = tmp_path / "seed.csv"
  _write_seed_csv(
    csv_path,
    [
      [0.0] * 29,
      [10.0] + [0.0] * 28,
      [20.0] + [0.0] * 28,
    ],
  )
  captured = {}

  def fake_load_targets(
    path,
    *,
    frame_slice,
    apply_orientation_offsets,
    align_upper_arm_axes_to_g1,
    apply_lower_body_offsets,
    remove_initial_heading,
    localize_to_body_frame,
  ):
    captured["frame_slice"] = frame_slice
    captured["apply_orientation_offsets"] = apply_orientation_offsets
    captured["align_upper_arm_axes_to_g1"] = align_upper_arm_axes_to_g1
    captured["apply_lower_body_offsets"] = apply_lower_body_offsets
    captured["remove_initial_heading"] = remove_initial_heading
    captured["localize_to_body_frame"] = localize_to_body_frame
    return [object(), object()]

  def fake_retarget_targets(targets, *, retargeter):
    return [
      SimpleNamespace(success=True, errors={"left_thigh": 0.0}, joint_angles=np.zeros(29)),
      SimpleNamespace(success=True, errors={"left_thigh": 0.0}, joint_angles=np.zeros(29)),
    ]

  monkeypatch.setattr(compare, "load_soma_bvh_full_body_targets", fake_load_targets)
  monkeypatch.setattr(compare, "retarget_full_body_targets", fake_retarget_targets)
  def fake_retargeter(*, algorithm_version):
    captured["algorithm_version"] = algorithm_version
    return object()

  monkeypatch.setattr(compare, "G1FullBodySEWRetargeter", fake_retargeter)

  summary = compare_retarget_with_seed_csv(
    CompareConfig(
      bvh_path=tmp_path / "unused.bvh",
      seed_csv_path=csv_path,
      start_frame=1,
      max_frames=2,
      algorithm_version="paper_v1",
    )
  )

  assert captured["frame_slice"] == slice(1, 3)
  assert captured["apply_orientation_offsets"] is True
  assert captured["align_upper_arm_axes_to_g1"] is True
  assert captured["apply_lower_body_offsets"] is True
  assert captured["remove_initial_heading"] is True
  assert captured["localize_to_body_frame"] is True
  assert captured["algorithm_version"] == "paper_v1"
  assert summary.frame_count == 2
  assert summary.success_count == 2
  assert math.isclose(summary.max_abs_joint_diff_deg, 20.0)
  assert math.isclose(summary.mean_abs_joint_diff_deg, 30.0 / 58.0)
