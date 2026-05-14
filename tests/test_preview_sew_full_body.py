import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import scripts.preview_sew_full_body as preview
from scripts.preview_sew_full_body import PreviewConfig, run_preview
from tests.test_bvh_full_body import _write_tiny_full_body_bvh


def test_run_full_body_preview_headless_retargets_bvh_and_reports_errors(tmp_path):
  bvh_path = tmp_path / "full.bvh"
  _write_tiny_full_body_bvh(bvh_path)

  summary = run_preview(
    PreviewConfig(
      bvh_path=bvh_path,
      max_frames=1,
      no_viewer=True,
      apply_lower_body_offsets=False,
    )
  )

  assert summary.frame_count == 1
  assert "left_thigh" in summary.max_errors
  assert "right_foot" in summary.max_errors
  assert "waist" in summary.max_errors
  assert "left_wrist" in summary.max_errors


def test_full_body_preview_script_runs_directly_from_repo_root(tmp_path):
  bvh_path = tmp_path / "full.bvh"
  _write_tiny_full_body_bvh(bvh_path)

  result = subprocess.run(
    [
      sys.executable,
      "scripts/preview_sew_full_body.py",
      "--bvh",
      str(bvh_path),
      "--max-frames",
      "1",
      "--no-viewer",
      "--raw-lower-body-offsets",
    ],
    cwd=Path(__file__).resolve().parents[1],
    check=False,
    capture_output=True,
    text=True,
  )

  assert result.returncode == 0, result.stderr
  assert "frames: 1" in result.stdout


def test_webcam_preview_script_help_does_not_require_webcam_dependencies():
  result = subprocess.run(
    [
      sys.executable,
      "scripts/preview_webcam_sew_full_body.py",
      "--help",
    ],
    cwd=Path(__file__).resolve().parents[1],
    check=False,
    capture_output=True,
    text=True,
  )

  assert result.returncode == 0, result.stderr
  assert "Retarget a live webcam human pose" in result.stdout


def test_webcam_dependency_loader_supports_mediapipe_without_top_level_solutions(monkeypatch):
  import scripts.preview_webcam_sew_full_body as webcam_preview

  fake_cv2 = types.ModuleType("cv2")
  fake_mediapipe = types.ModuleType("mediapipe")
  fake_pose = types.ModuleType("mediapipe.python.solutions.pose")
  fake_drawing = types.ModuleType("mediapipe.python.solutions.drawing_utils")
  fake_viewer = types.ModuleType("mujoco.viewer")
  monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
  monkeypatch.setitem(sys.modules, "mediapipe", fake_mediapipe)
  monkeypatch.setitem(sys.modules, "mediapipe.python", types.ModuleType("mediapipe.python"))
  monkeypatch.setitem(sys.modules, "mediapipe.python.solutions", types.ModuleType("mediapipe.python.solutions"))
  monkeypatch.setitem(sys.modules, "mediapipe.python.solutions.pose", fake_pose)
  monkeypatch.setitem(sys.modules, "mediapipe.python.solutions.drawing_utils", fake_drawing)
  monkeypatch.setitem(sys.modules, "mujoco.viewer", fake_viewer)

  cv2, pose, drawing, viewer = webcam_preview._load_realtime_dependencies()

  assert cv2 is fake_cv2
  assert pose is fake_pose
  assert drawing is fake_drawing
  assert viewer is fake_viewer


def test_run_full_body_preview_uses_full_body_loader_defaults(monkeypatch, tmp_path):
  captured = {}

  def fake_load_targets(
    path,
    *,
    frame_slice,
    apply_orientation_offsets,
    align_upper_arm_axes_to_g1,
    apply_lower_body_offsets,
    remove_initial_heading,
  ):
    captured["frame_slice"] = frame_slice
    captured["apply_orientation_offsets"] = apply_orientation_offsets
    captured["align_upper_arm_axes_to_g1"] = align_upper_arm_axes_to_g1
    captured["apply_lower_body_offsets"] = apply_lower_body_offsets
    captured["remove_initial_heading"] = remove_initial_heading
    return [object()]

  def fake_retarget_targets(targets, *, retargeter):
    return [
      SimpleNamespace(
        success=True,
        errors={"waist": 0.0, "left_thigh": 0.0},
      )
    ]

  monkeypatch.setattr(preview, "load_soma_bvh_full_body_targets", fake_load_targets)
  monkeypatch.setattr(preview, "retarget_full_body_targets", fake_retarget_targets)
  monkeypatch.setattr(preview, "G1FullBodySEWRetargeter", lambda: object())

  run_preview(
    PreviewConfig(
      bvh_path=tmp_path / "unused.bvh",
      start_frame=10,
      max_frames=5,
      no_viewer=True,
    )
  )

  assert captured["frame_slice"] == slice(10, 15)
  assert captured["apply_orientation_offsets"] is True
  assert captured["align_upper_arm_axes_to_g1"] is True
  assert captured["apply_lower_body_offsets"] is True
  assert captured["remove_initial_heading"] is True
