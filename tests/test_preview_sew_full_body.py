import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np

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
  assert "--video" in result.stdout
  assert "--flip-depth" in result.stdout
  assert "--algorithm-version" in result.stdout
  assert "--raw-human-arm-axes" in result.stdout
  assert "--mediapipe-delegate" in result.stdout
  assert "paper_v1" in result.stdout
  assert "branch_v2" in result.stdout


def test_webcam_preview_defaults_to_paper_v1_full_body_retargeter(monkeypatch):
  captured = {}

  def fake_full_body_retargeter(*, algorithm_version):
    captured["full_body_algorithm_version"] = algorithm_version
    return SimpleNamespace(controlled_joint_names=tuple(range(29)))

  def fake_upper_body_retargeter(*, algorithm_version):
    captured["upper_body_algorithm_version"] = algorithm_version
    return SimpleNamespace(controlled_joint_names=tuple(range(17)))

  monkeypatch.setattr(webcam_preview, "G1FullBodySEWRetargeter", fake_full_body_retargeter)
  monkeypatch.setattr(webcam_preview, "G1UpperBodySEWRetargeter", fake_upper_body_retargeter)

  retargeter = webcam_preview._create_retargeter(webcam_preview.WebcamPreviewConfig())

  assert len(retargeter.controlled_joint_names) == 29
  assert captured == {"full_body_algorithm_version": "paper_v1"}


def test_webcam_perf_stats_report_includes_tracking_and_retargeting(capsys):
  stats = webcam_preview.PerfStats(report_interval_sec=0.0)
  stats.add(
    read=0.002,
    tracking=0.030,
    target=0.003,
    retarget=0.004,
    render=0.005,
    overlay=0.006,
    frame=0.050,
    had_landmarks=True,
    retargeted=True,
  )

  stats.maybe_report(now=1.0)

  output = capsys.readouterr().out
  assert "tracking" in output
  assert "retarget" in output
  assert "fps" in output
  assert "30.00ms" in output
  assert "4.00ms" in output


def test_webcam_perf_stats_report_includes_retarget_breakdown(capsys):
  stats = webcam_preview.PerfStats(report_interval_sec=0.0)
  stats.add(
    read=0.001,
    tracking=0.002,
    target=0.003,
    retarget=0.040,
    retarget_lower=0.012,
    retarget_upper=0.027,
    retarget_combine=0.001,
    render=0.004,
    overlay=0.005,
    frame=0.050,
    had_landmarks=True,
    retargeted=True,
  )

  stats.maybe_report(now=1.0)

  output = capsys.readouterr().out
  assert "lower=12.00ms" in output
  assert "upper=27.00ms" in output
  assert "combine=1.00ms" in output


def test_webcam_perf_stats_report_includes_target_failures(capsys):
  stats = webcam_preview.PerfStats(report_interval_sec=0.0)
  stats.add(
    read=0.010,
    tracking=0.020,
    target=0.001,
    retarget=0.0,
    render=0.0,
    overlay=0.002,
    frame=0.040,
    had_landmarks=True,
    target_failed=True,
    retargeted=False,
  )

  stats.maybe_report(now=1.0)

  output = capsys.readouterr().out
  assert "target_failed=1/1" in output


def test_webcam_retarget_timing_splits_full_body_retargeter():
  calls = []

  class FakeLower:
    def retarget(self, q, target):
      calls.append(("lower", q.copy(), target))
      return SimpleNamespace(name="lower-result")

  class FakeUpper:
    def retarget(self, q, target):
      calls.append(("upper", q.copy(), target))
      return SimpleNamespace(name="upper-result")

  class FakeFullBodyRetargeter:
    lower = FakeLower()
    upper = FakeUpper()

    def _combine_results(self, lower_result, upper_result):
      calls.append(("combine", lower_result.name, upper_result.name))
      return SimpleNamespace(name="combined")

  result, timing = webcam_preview._retarget_with_timing(
    FakeFullBodyRetargeter(),
    np.arange(29, dtype=float),
    SimpleNamespace(lower="lower-target", upper="upper-target"),
  )

  assert result.name == "combined"
  assert calls[0][0] == "lower"
  assert calls[0][1].tolist() == list(range(12))
  assert calls[0][2] == "lower-target"
  assert calls[1][0] == "upper"
  assert calls[1][1].tolist() == list(range(12, 29))
  assert calls[1][2] == "upper-target"
  assert calls[2] == ("combine", "lower-result", "upper-result")
  assert timing.lower >= 0.0
  assert timing.upper >= 0.0
  assert timing.combine >= 0.0


def test_webcam_preview_upper_body_mode_keeps_selected_algorithm(monkeypatch):
  captured = {}

  def fake_upper_body_retargeter(*, algorithm_version):
    captured["upper_body_algorithm_version"] = algorithm_version
    return SimpleNamespace(controlled_joint_names=tuple(range(17)))

  monkeypatch.setattr(webcam_preview, "G1UpperBodySEWRetargeter", fake_upper_body_retargeter)

  retargeter = webcam_preview._create_retargeter(
    webcam_preview.WebcamPreviewConfig(
      upper_body_only=True,
      algorithm_version="branch_v2",
    )
  )

  assert len(retargeter.controlled_joint_names) == 17
  assert captured == {"upper_body_algorithm_version": "branch_v2"}


def test_webcam_target_builder_passes_raw_arm_axis_switch(monkeypatch):
  captured = {}

  def fake_full_body_target(
    landmarks,
    *,
    scale,
    min_visibility,
    flip_depth,
    align_upper_arm_axes_to_g1,
  ):
    captured["landmarks"] = landmarks
    captured["scale"] = scale
    captured["min_visibility"] = min_visibility
    captured["flip_depth"] = flip_depth
    captured["align_upper_arm_axes_to_g1"] = align_upper_arm_axes_to_g1
    return "target"

  monkeypatch.setattr(webcam_preview, "full_body_target_from_mediapipe_landmarks", fake_full_body_target)

  target = webcam_preview._target_from_landmarks(
    ["landmark"],
    scale=1.25,
    min_visibility=0.4,
    upper_body_only=False,
    flip_depth=True,
    align_upper_arm_axes_to_g1=False,
  )

  assert target == "target"
  assert captured == {
    "landmarks": ["landmark"],
    "scale": 1.25,
    "min_visibility": 0.4,
    "flip_depth": True,
    "align_upper_arm_axes_to_g1": False,
  }


def test_upper_body_version_plot_script_help_lists_algorithm_versions():
  result = subprocess.run(
    [
      sys.executable,
      "scripts/plot_upper_body_retarget_versions.py",
      "--help",
    ],
    cwd=Path(__file__).resolve().parents[1],
    check=False,
    capture_output=True,
    text=True,
  )

  assert result.returncode == 0, result.stderr
  assert "--versions" in result.stdout
  assert "paper_v1" in result.stdout
  assert "branch_v2" in result.stdout


def test_webcam_preview_opens_video_file_source(tmp_path):
  opened_sources = []
  video_path = tmp_path / "input.mp4"
  video_path.write_bytes(b"fake mp4")

  class FakeCapture:
    def __init__(self, source):
      opened_sources.append(source)
      self.set_calls = []

    def isOpened(self):
      return True

    def set(self, key, value):
      self.set_calls.append((key, value))

  fake_cv2 = SimpleNamespace(VideoCapture=FakeCapture)

  capture, source_is_file = webcam_preview._open_capture(
    fake_cv2,
    webcam_preview.WebcamPreviewConfig(video_path=video_path),
  )

  assert opened_sources == [str(video_path)]
  assert source_is_file is True
  assert capture.set_calls == []


def test_webcam_preview_opens_camera_source_with_requested_size():
  opened_sources = []

  class FakeCapture:
    def __init__(self, source):
      opened_sources.append(source)
      self.set_calls = []

    def isOpened(self):
      return True

    def set(self, key, value):
      self.set_calls.append((key, value))

  fake_cv2 = SimpleNamespace(
    VideoCapture=FakeCapture,
    CAP_PROP_FRAME_WIDTH=3,
    CAP_PROP_FRAME_HEIGHT=4,
    CAP_PROP_FPS=5,
    CAP_PROP_BUFFERSIZE=6,
  )

  capture, source_is_file = webcam_preview._open_capture(
    fake_cv2,
    webcam_preview.WebcamPreviewConfig(camera=2, width=640, height=480),
  )

  assert opened_sources == [2]
  assert source_is_file is False
  assert capture.set_calls == [(3, 640), (4, 480), (5, 30.0), (6, 1)]


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


def test_tasks_pose_factory_passes_gpu_delegate_to_base_options(tmp_path):
  captured = {}

  class FakeDelegate:
    CPU = "CPU"
    GPU = "GPU"

  class FakeBaseOptions:
    Delegate = FakeDelegate

    def __init__(self, *, model_asset_path, delegate):
      captured["model_asset_path"] = model_asset_path
      captured["delegate"] = delegate

  class FakePoseLandmarkerOptions:
    def __init__(self, **kwargs):
      captured["options"] = kwargs

  class FakePoseLandmarker:
    @staticmethod
    def create_from_options(options):
      captured["created_options"] = options
      return "pose"

  fake_mp = SimpleNamespace(
    tasks=SimpleNamespace(
      BaseOptions=FakeBaseOptions,
      vision=SimpleNamespace(
        PoseLandmarkerOptions=FakePoseLandmarkerOptions,
        RunningMode=SimpleNamespace(VIDEO="VIDEO"),
        PoseLandmarker=FakePoseLandmarker,
      ),
    )
  )
  model_path = tmp_path / "pose.task"
  model_path.write_bytes(b"fake")

  pose = webcam_preview._TasksPoseFactory(fake_mp, model_path, "gpu").create(
    min_detection_confidence=0.25,
    min_tracking_confidence=0.75,
  )

  assert pose == "pose"
  assert captured["model_asset_path"] == str(model_path)
  assert captured["delegate"] == "GPU"
  assert captured["options"]["running_mode"] == "VIDEO"


def test_webcam_dependency_loader_downloads_default_tasks_model(monkeypatch, tmp_path):
  import scripts.preview_webcam_sew_full_body as webcam_preview

  fake_cv2 = types.ModuleType("cv2")
  fake_mediapipe = types.ModuleType("mediapipe")
  fake_mediapipe.__version__ = "0.10.35"
  fake_mediapipe.__file__ = "fake-mediapipe"
  fake_mediapipe.tasks = SimpleNamespace(
    BaseOptions=object,
    vision=SimpleNamespace(
      PoseLandmarker=object,
    ),
  )
  fake_viewer = types.ModuleType("mujoco.viewer")
  default_model_path = tmp_path / "pose_landmarker_lite.task"
  downloads = []

  def fake_download_pose_model(path, url):
    downloads.append((path, url))
    path.write_bytes(b"fake task model")
    return path

  monkeypatch.setattr(webcam_preview, "DEFAULT_POSE_MODEL_PATH", default_model_path)
  monkeypatch.setattr(webcam_preview, "_download_pose_model", fake_download_pose_model)
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

  cv2, pose_factory, drawing, viewer = webcam_preview._load_realtime_dependencies()

  assert cv2 is fake_cv2
  assert pose_factory._pose_model == default_model_path
  assert drawing is None
  assert viewer is fake_viewer
  assert default_model_path.read_bytes() == b"fake task model"
  assert downloads == [(default_model_path, webcam_preview.DEFAULT_POSE_MODEL_URL)]


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
    localize_to_body_frame,
  ):
    captured["frame_slice"] = frame_slice
    captured["apply_orientation_offsets"] = apply_orientation_offsets
    captured["align_upper_arm_axes_to_g1"] = align_upper_arm_axes_to_g1
    captured["apply_lower_body_offsets"] = apply_lower_body_offsets
    captured["remove_initial_heading"] = remove_initial_heading
    captured["localize_to_body_frame"] = localize_to_body_frame
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
  monkeypatch.setattr(
    preview,
    "G1FullBodySEWRetargeter",
    lambda *, algorithm_version: {"algorithm_version": algorithm_version},
  )

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
  assert captured["localize_to_body_frame"] is True


def test_full_body_preview_script_help_lists_algorithm_versions():
  result = subprocess.run(
    [
      sys.executable,
      "scripts/preview_sew_full_body.py",
      "--help",
    ],
    cwd=Path(__file__).resolve().parents[1],
    check=False,
    capture_output=True,
    text=True,
  )

  assert result.returncode == 0, result.stderr
  assert "--algorithm-version" in result.stdout
  assert "paper_v1" in result.stdout
  assert "branch_v2" in result.stdout
