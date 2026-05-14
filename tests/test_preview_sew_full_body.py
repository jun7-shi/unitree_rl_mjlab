import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import scripts.preview_sew_full_body as preview
import scripts.preview_webcam_sew_full_body as webcam_preview
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
  fake_pose.POSE_CONNECTIONS = object()
  fake_drawing = types.ModuleType("mediapipe.python.solutions.drawing_utils")
  fake_viewer = types.ModuleType("mujoco.viewer")
  monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
  monkeypatch.setitem(sys.modules, "mediapipe", fake_mediapipe)
  monkeypatch.setitem(sys.modules, "mediapipe.python", types.ModuleType("mediapipe.python"))
  monkeypatch.setitem(sys.modules, "mediapipe.python.solutions", types.ModuleType("mediapipe.python.solutions"))
  monkeypatch.setitem(sys.modules, "mediapipe.python.solutions.pose", fake_pose)
  monkeypatch.setitem(sys.modules, "mediapipe.python.solutions.drawing_utils", fake_drawing)
  monkeypatch.setitem(sys.modules, "mujoco.viewer", fake_viewer)

  cv2, pose_factory, drawing, viewer = webcam_preview._load_realtime_dependencies()

  assert cv2 is fake_cv2
  assert pose_factory._mp_pose is fake_pose
  assert drawing is fake_drawing
  assert viewer is fake_viewer


def test_webcam_dependency_loader_supports_direct_mediapipe_solutions_import(monkeypatch):
  import scripts.preview_webcam_sew_full_body as webcam_preview

  fake_cv2 = types.ModuleType("cv2")
  fake_mediapipe = types.ModuleType("mediapipe")
  fake_pose = types.ModuleType("mediapipe.solutions.pose")
  fake_pose.POSE_CONNECTIONS = object()
  fake_drawing = types.ModuleType("mediapipe.solutions.drawing_utils")
  fake_viewer = types.ModuleType("mujoco.viewer")
  monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
  monkeypatch.setitem(sys.modules, "mediapipe", fake_mediapipe)
  monkeypatch.delitem(sys.modules, "mediapipe.python", raising=False)
  monkeypatch.delitem(sys.modules, "mediapipe.python.solutions", raising=False)
  monkeypatch.setitem(sys.modules, "mediapipe.solutions", types.ModuleType("mediapipe.solutions"))
  monkeypatch.setitem(sys.modules, "mediapipe.solutions.pose", fake_pose)
  monkeypatch.setitem(sys.modules, "mediapipe.solutions.drawing_utils", fake_drawing)
  monkeypatch.setitem(sys.modules, "mujoco.viewer", fake_viewer)

  cv2, pose_factory, drawing, viewer = webcam_preview._load_realtime_dependencies()

  assert cv2 is fake_cv2
  assert pose_factory._mp_pose is fake_pose
  assert drawing is fake_drawing
  assert viewer is fake_viewer


def test_webcam_dependency_loader_supports_mediapipe_tasks_backend(monkeypatch, tmp_path):
  import scripts.preview_webcam_sew_full_body as webcam_preview

  fake_cv2 = types.ModuleType("cv2")
  fake_mediapipe = types.ModuleType("mediapipe")
  fake_mediapipe.__version__ = "0.10.35"
  fake_mediapipe.__file__ = "fake-mediapipe"
  fake_viewer = types.ModuleType("mujoco.viewer")
  fake_mediapipe.tasks = SimpleNamespace(
    BaseOptions=object,
    vision=SimpleNamespace(
      PoseLandmarker=object,
    ),
  )
  model_path = tmp_path / "pose_landmarker_lite.task"
  model_path.write_bytes(b"fake")
  monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
  monkeypatch.setitem(sys.modules, "mediapipe", fake_mediapipe)
  for module_name in (
    "mediapipe.solutions",
    "mediapipe.solutions.pose",
    "mediapipe.solutions.drawing_utils",
    "mediapipe.python",
    "mediapipe.python.solutions",
    "mediapipe.python.solutions.pose",
    "mediapipe.python.solutions.drawing_utils",
  ):
    monkeypatch.delitem(sys.modules, module_name, raising=False)
  monkeypatch.setitem(sys.modules, "mujoco.viewer", fake_viewer)

  cv2, pose_factory, drawing, viewer = webcam_preview._load_realtime_dependencies(model_path)

  assert cv2 is fake_cv2
  assert pose_factory._pose_model == model_path
  assert drawing is None
  assert viewer is fake_viewer


def test_webcam_dependency_loader_requires_pose_model_for_tasks_only_mediapipe(monkeypatch):
  import scripts.preview_webcam_sew_full_body as webcam_preview

  fake_cv2 = types.ModuleType("cv2")
  fake_mediapipe = types.ModuleType("mediapipe")
  fake_mediapipe.__version__ = "0.10.35"
  fake_mediapipe.__file__ = "fake-mediapipe"
  fake_mediapipe.tasks = SimpleNamespace()
  fake_viewer = types.ModuleType("mujoco.viewer")
  monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
  monkeypatch.setitem(sys.modules, "mediapipe", fake_mediapipe)
  for module_name in (
    "mediapipe.solutions",
    "mediapipe.solutions.pose",
    "mediapipe.solutions.drawing_utils",
    "mediapipe.python",
    "mediapipe.python.solutions",
    "mediapipe.python.solutions.pose",
    "mediapipe.python.solutions.drawing_utils",
  ):
    monkeypatch.delitem(sys.modules, module_name, raising=False)
  monkeypatch.setitem(sys.modules, "mujoco.viewer", fake_viewer)

  try:
    webcam_preview._load_realtime_dependencies()
  except RuntimeError as exc:
    assert "--pose-model" in str(exc)
    assert "pose_landmarker_lite.task" in str(exc)
  else:
    raise AssertionError("expected tasks-only mediapipe to require --pose-model")


def test_webcam_overlay_draws_tasks_landmark_lists_without_mediapipe_drawing():
  calls = []
  fake_cv2 = SimpleNamespace(
    circle=lambda *args: calls.append(("circle", args)),
    line=lambda *args: calls.append(("line", args)),
    putText=lambda *args: calls.append(("putText", args)),
    imshow=lambda *args: calls.append(("imshow", args)),
    FONT_HERSHEY_SIMPLEX=0,
    LINE_AA=16,
  )
  landmarks = [SimpleNamespace(x=0.0, y=0.0, visibility=0.0) for _ in range(33)]
  for index, xy in {
    11: (0.25, 0.25),
    12: (0.75, 0.25),
    13: (0.20, 0.50),
    14: (0.80, 0.50),
    15: (0.15, 0.75),
    16: (0.85, 0.75),
  }.items():
    landmarks[index] = SimpleNamespace(x=xy[0], y=xy[1], visibility=1.0)
  pose_result = SimpleNamespace(pose_landmarks=[landmarks])
  frame = __import__("numpy").zeros((100, 200, 3), dtype="uint8")

  webcam_preview._draw_camera_overlay(
    fake_cv2,
    SimpleNamespace(connections=None),
    None,
    frame,
    pose_result,
    upper_body_only=True,
  )

  assert any(name == "circle" for name, _args in calls)
  assert any(name == "line" for name, _args in calls)
  assert calls[-1][0] == "imshow"


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
