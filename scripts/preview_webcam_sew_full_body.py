from __future__ import annotations

import argparse
import importlib
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from src.motion.sew_full_body import G1FullBodySEWRetargeter
from src.motion.sew_mimic import BRANCH_V2_ALGORITHM, SEW_ALGORITHM_CONFIGS
from src.motion.sew_upper_body import G1UpperBodySEWRetargeter
from src.motion.webcam_pose import (
  ExponentialJointFilter,
  ExponentialLandmarkFilter,
  full_body_target_from_mediapipe_landmarks,
  upper_body_target_from_mediapipe_landmarks,
)


@dataclass(frozen=True)
class WebcamPreviewConfig:
  camera: int = 0
  video_path: Path | None = None
  width: int = 1280
  height: int = 720
  fps: float = 30.0
  mirror: bool = True
  landmark_scale: float = 1.0
  min_visibility: float = 0.5
  min_detection_confidence: float = 0.5
  min_tracking_confidence: float = 0.5
  landmark_smoothing_alpha: float = 0.35
  smoothing_alpha: float = 0.35
  max_joint_delta_deg: float | None = 12.0
  no_camera_window: bool = False
  pose_model: Path | None = None
  upper_body_only: bool = False
  flip_depth: bool = False
  algorithm_version: str = BRANCH_V2_ALGORITHM.name


def run_webcam_preview(config: WebcamPreviewConfig) -> None:
  if config.width <= 0 or config.height <= 0:
    raise ValueError(f"width and height must be positive, got {config.width}x{config.height}")
  if config.fps <= 0.0:
    raise ValueError(f"fps must be positive, got {config.fps}")
  if not 0.0 < config.landmark_smoothing_alpha <= 1.0:
    raise ValueError(f"landmark_smoothing_alpha must be in (0, 1], got {config.landmark_smoothing_alpha}")
  cv2, pose_factory, mp_drawing, viewer_module = _load_realtime_dependencies(config.pose_model)

  capture, source_is_file = _open_capture(cv2, config)

  retargeter = (
    G1UpperBodySEWRetargeter(algorithm_version=config.algorithm_version)
    if config.upper_body_only
    else G1FullBodySEWRetargeter(algorithm_version=config.algorithm_version)
  )
  data = mujoco.MjData(retargeter.model)
  max_delta = None if config.max_joint_delta_deg is None else np.deg2rad(config.max_joint_delta_deg)
  joint_filter = ExponentialJointFilter(alpha=config.smoothing_alpha, max_delta=max_delta)
  landmark_filter = ExponentialLandmarkFilter(alpha=config.landmark_smoothing_alpha)
  q_previous = np.zeros(len(retargeter.controlled_joint_names), dtype=float)
  frame_dt = 1.0 / config.fps
  frame_index = 0

  with pose_factory.create(
    min_detection_confidence=config.min_detection_confidence,
    min_tracking_confidence=config.min_tracking_confidence,
  ) as pose, viewer_module.launch_passive(retargeter.model, data) as viewer:
    try:
      while viewer.is_running():
        frame_start = time.time()
        ok, frame = capture.read()
        if not ok:
          if source_is_file:
            break
          time.sleep(frame_dt)
          continue
        if config.mirror and not source_is_file:
          frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        timestamp_ms = int(frame_index * frame_dt * 1000) if source_is_file else int(time.time() * 1000)
        frame_index += 1
        pose_result = pose_factory.process(pose, rgb, timestamp_ms)
        landmarks = _select_pose_landmarks(pose_result)
        if landmarks is not None:
          landmarks = landmark_filter.update(landmarks)
          try:
            target = _target_from_landmarks(
              landmarks,
              scale=config.landmark_scale,
              min_visibility=config.min_visibility,
              upper_body_only=config.upper_body_only,
              flip_depth=config.flip_depth,
            )
          except ValueError:
            target = None
          if target is not None:
            retarget_result = retargeter.retarget(q_previous, target)
            q_filtered = joint_filter.update(retarget_result.joint_angles)
            full_qpos = retarget_result.full_qpos.copy()
            full_qpos[retargeter.controlled_qpos_addresses] = q_filtered
            data.qpos[:] = full_qpos
            mujoco.mj_forward(retargeter.model, data)
            viewer.sync()
            q_previous = q_filtered

        if not config.no_camera_window:
          _draw_camera_overlay(cv2, pose_factory, mp_drawing, frame, pose_result, config.upper_body_only)
          key = cv2.waitKey(1) & 0xFF
          if key in (27, ord("q")):
            break

        elapsed = time.time() - frame_start
        if elapsed < frame_dt:
          time.sleep(frame_dt - elapsed)
    finally:
      capture.release()
      if not config.no_camera_window:
        cv2.destroyAllWindows()


def _open_capture(cv2, config: WebcamPreviewConfig):
  if config.video_path is not None:
    video_path = Path(config.video_path)
    if not video_path.exists():
      raise RuntimeError(f"Video file does not exist: {video_path}")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
      raise RuntimeError(f"Could not open video file: {video_path}")
    return capture, True

  capture = cv2.VideoCapture(config.camera)
  if not capture.isOpened():
    raise RuntimeError(f"Could not open webcam index {config.camera}")
  capture.set(cv2.CAP_PROP_FRAME_WIDTH, config.width)
  capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config.height)
  return capture, False


def _target_from_landmarks(
  landmarks,
  *,
  scale: float,
  min_visibility: float,
  upper_body_only: bool,
  flip_depth: bool,
):
  if upper_body_only:
    return upper_body_target_from_mediapipe_landmarks(
      landmarks,
      scale=scale,
      min_visibility=min_visibility,
      flip_depth=flip_depth,
    )
  return full_body_target_from_mediapipe_landmarks(
    landmarks,
    scale=scale,
    min_visibility=min_visibility,
    flip_depth=flip_depth,
  )


def _select_pose_landmarks(pose_result):
  world_landmarks = getattr(pose_result, "pose_world_landmarks", None)
  if world_landmarks is not None:
    return _first_landmark_list(world_landmarks)
  image_landmarks = getattr(pose_result, "pose_landmarks", None)
  if image_landmarks is not None:
    return _first_landmark_list(image_landmarks)
  return None


def _first_landmark_list(landmarks):
  if hasattr(landmarks, "landmark"):
    return landmarks.landmark
  if isinstance(landmarks, list) and landmarks:
    first = landmarks[0]
    return first.landmark if hasattr(first, "landmark") else first
  return None


def _draw_camera_overlay(cv2, pose_factory, mp_drawing, frame, pose_result, upper_body_only: bool = False) -> None:
  landmarks = getattr(pose_result, "pose_landmarks", None)
  if (
    mp_drawing is not None
    and pose_factory.connections is not None
    and hasattr(landmarks, "landmark")
  ):
    mp_drawing.draw_landmarks(
      frame,
      landmarks,
      pose_factory.connections,
    )
  else:
    landmark_list = _first_landmark_list(landmarks)
    if landmark_list is not None:
      _draw_landmarks_with_cv2(cv2, frame, landmark_list, upper_body_only=upper_body_only)
  cv2.imshow("SEW-Mimic webcam pose", frame)


def _draw_landmarks_with_cv2(cv2, frame, landmarks, *, upper_body_only: bool) -> None:
  height, width = frame.shape[:2]
  landmark_indexes = _overlay_landmark_indexes(upper_body_only)
  for start, end in _overlay_connections(upper_body_only):
    start_point = _landmark_pixel(landmarks, start, width, height)
    end_point = _landmark_pixel(landmarks, end, width, height)
    if start_point is not None and end_point is not None:
      cv2.line(frame, start_point, end_point, (70, 220, 80), 2, cv2.LINE_AA)
  for index in landmark_indexes:
    point = _landmark_pixel(landmarks, index, width, height)
    if point is not None:
      cv2.circle(frame, point, 4, (40, 140, 255), -1, cv2.LINE_AA)
  cv2.putText(
    frame,
    "upper-body" if upper_body_only else "full-body",
    (12, 28),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.75,
    (255, 255, 255),
    2,
    cv2.LINE_AA,
  )


def _overlay_landmark_indexes(upper_body_only: bool) -> tuple[int, ...]:
  upper = (11, 12, 13, 14, 15, 16)
  lower = (23, 24, 25, 26, 27, 28, 31, 32)
  return upper if upper_body_only else (*upper, *lower)


def _overlay_connections(upper_body_only: bool) -> tuple[tuple[int, int], ...]:
  upper = (
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
  )
  lower = (
    (11, 23),
    (12, 24),
    (23, 24),
    (23, 25),
    (25, 27),
    (27, 31),
    (24, 26),
    (26, 28),
    (28, 32),
  )
  return upper if upper_body_only else (*upper, *lower)


def _landmark_pixel(landmarks, index: int, width: int, height: int):
  if index >= len(landmarks):
    return None
  landmark = landmarks[index]
  visibility = float(getattr(landmark, "visibility", 1.0))
  if visibility < 0.2:
    return None
  x = int(np.clip(float(landmark.x), 0.0, 1.0) * (width - 1))
  y = int(np.clip(float(landmark.y), 0.0, 1.0) * (height - 1))
  return x, y


def _load_realtime_dependencies(pose_model: Path | None = None):
  missing = []
  try:
    import cv2
  except ModuleNotFoundError:
    cv2 = None
    missing.append("opencv-python")
  try:
    import mediapipe as mp
  except ModuleNotFoundError:
    mp = None
    missing.append("mediapipe")
  viewer_module = sys.modules.get("mujoco.viewer")
  if viewer_module is None:
    try:
      import mujoco.viewer as viewer_module
    except Exception as exc:
      raise RuntimeError(f"Could not import mujoco.viewer: {exc}") from exc

  if missing:
    raise RuntimeError(
      "Missing realtime webcam dependencies: "
      + ", ".join(missing)
      + ". Install them with `pip install -e '.[webcam]'` inside the unitree_rl_mjlab conda env."
    )
  pose_factory, mp_drawing = _load_mediapipe_pose_factory(mp, pose_model)
  return cv2, pose_factory, mp_drawing, viewer_module


class _LegacyPoseFactory:
  def __init__(self, mp_pose):
    self._mp_pose = mp_pose
    self.connections = mp_pose.POSE_CONNECTIONS

  def create(self, *, min_detection_confidence: float, min_tracking_confidence: float):
    return self._mp_pose.Pose(
      model_complexity=1,
      smooth_landmarks=True,
      min_detection_confidence=min_detection_confidence,
      min_tracking_confidence=min_tracking_confidence,
    )

  def process(self, pose, rgb: np.ndarray, timestamp_ms: int):
    del timestamp_ms
    return pose.process(rgb)


class _TasksPoseFactory:
  def __init__(self, mp, pose_model: Path):
    self._mp = mp
    self._pose_model = pose_model
    self.connections = None

  def create(self, *, min_detection_confidence: float, min_tracking_confidence: float):
    base_options = self._mp.tasks.BaseOptions(model_asset_path=str(self._pose_model))
    options = self._mp.tasks.vision.PoseLandmarkerOptions(
      base_options=base_options,
      running_mode=self._mp.tasks.vision.RunningMode.VIDEO,
      min_pose_detection_confidence=min_detection_confidence,
      min_tracking_confidence=min_tracking_confidence,
      num_poses=1,
    )
    return self._mp.tasks.vision.PoseLandmarker.create_from_options(options)

  def process(self, pose, rgb: np.ndarray, timestamp_ms: int):
    image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
    return pose.detect_for_video(image, timestamp_ms)


def _load_mediapipe_pose_factory(mp, pose_model: Path | None):
  legacy = _load_legacy_mediapipe_solutions(mp)
  if legacy is not None:
    mp_pose, mp_drawing = legacy
    return _LegacyPoseFactory(mp_pose), mp_drawing

  return _load_tasks_pose_factory(mp, pose_model), None


def _load_legacy_mediapipe_solutions(mp):
  solutions = getattr(mp, "solutions", None)
  if solutions is not None:
    return solutions.pose, solutions.drawing_utils

  for pose_module, drawing_module in (
    ("mediapipe.solutions.pose", "mediapipe.solutions.drawing_utils"),
    ("mediapipe.python.solutions.pose", "mediapipe.python.solutions.drawing_utils"),
  ):
    try:
      pose = importlib.import_module(pose_module)
      drawing_utils = importlib.import_module(drawing_module)
      return pose, drawing_utils
    except ModuleNotFoundError:
      pass

  return None


def _load_tasks_pose_factory(mp, pose_model: Path | None):
  if pose_model is None:
    version = getattr(mp, "__version__", "unknown")
    location = getattr(mp, "__file__", "unknown")
    raise RuntimeError(
      "This mediapipe build does not include the legacy Pose Solutions API; "
      "use the MediaPipe Tasks backend by passing `--pose-model path/to/pose_landmarker_lite.task`. "
      f"mediapipe version={version}, file={location}. "
      "Official lite model URL: "
      "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
    )

  model_path = Path(pose_model)
  if not model_path.exists():
    raise RuntimeError(f"Pose landmarker model does not exist: {model_path}")

  try:
    tasks = getattr(mp, "tasks")
    _base_options = tasks.BaseOptions
    _pose_landmarker = tasks.vision.PoseLandmarker
  except AttributeError as exc:
    version = getattr(mp, "__version__", "unknown")
    location = getattr(mp, "__file__", "unknown")
    raise RuntimeError(
      "Installed mediapipe package exposes neither legacy Pose Solutions nor Tasks PoseLandmarker. "
      f"mediapipe version={version}, file={location}."
    ) from exc

  return _TasksPoseFactory(mp, model_path)

def build_arg_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
    description="Retarget a live webcam human pose to Unitree G1 and preview it in MuJoCo."
  )
  parser.add_argument("--camera", type=int, default=0, help="OpenCV webcam index.")
  parser.add_argument("--video", type=Path, default=None, help="Read frames from an MP4/video file instead of webcam.")
  parser.add_argument("--width", type=int, default=1280, help="Requested camera width.")
  parser.add_argument("--height", type=int, default=720, help="Requested camera height.")
  parser.add_argument("--fps", type=float, default=30.0, help="Realtime loop rate.")
  parser.add_argument("--no-mirror", action="store_true", help="Do not mirror the webcam image.")
  parser.add_argument("--landmark-scale", type=float, default=1.0, help="Scale MediaPipe world landmarks.")
  parser.add_argument("--min-visibility", type=float, default=0.5, help="Minimum landmark visibility.")
  parser.add_argument("--min-detection-confidence", type=float, default=0.5, help="MediaPipe detection confidence.")
  parser.add_argument("--min-tracking-confidence", type=float, default=0.5, help="MediaPipe tracking confidence.")
  parser.add_argument("--landmark-smoothing-alpha", type=float, default=0.35, help="MediaPipe landmark low-pass filter alpha.")
  parser.add_argument("--smoothing-alpha", type=float, default=0.35, help="Joint low-pass filter alpha.")
  parser.add_argument("--max-joint-delta-deg", type=float, default=12.0, help="Maximum filtered joint change per frame in degrees.")
  parser.add_argument(
    "--algorithm-version",
    choices=tuple(SEW_ALGORITHM_CONFIGS),
    default=BRANCH_V2_ALGORITHM.name,
    help="SEW candidate-selection version to use.",
  )
  parser.add_argument("--flip-depth", action="store_true", help="Flip MediaPipe z depth if hands in front retarget behind the robot.")
  parser.add_argument("--no-camera-window", action="store_true", help="Only show MuJoCo, not the webcam overlay.")
  parser.add_argument(
    "--upper-body-only",
    action="store_true",
    help="Retarget only the waist and arms; useful when legs are outside the camera frame.",
  )
  parser.add_argument(
    "--pose-model",
    type=Path,
    default=None,
    help="MediaPipe Tasks pose_landmarker .task model path; required for mediapipe builds without mp.solutions.",
  )
  return parser


def main(argv: Sequence[str] | None = None) -> int:
  args = build_arg_parser().parse_args(argv)
  run_webcam_preview(
    WebcamPreviewConfig(
      camera=args.camera,
      video_path=args.video,
      width=args.width,
      height=args.height,
      fps=args.fps,
      mirror=not args.no_mirror,
      landmark_scale=args.landmark_scale,
      min_visibility=args.min_visibility,
      min_detection_confidence=args.min_detection_confidence,
      min_tracking_confidence=args.min_tracking_confidence,
      landmark_smoothing_alpha=args.landmark_smoothing_alpha,
      smoothing_alpha=args.smoothing_alpha,
      max_joint_delta_deg=args.max_joint_delta_deg if args.max_joint_delta_deg > 0.0 else None,
      no_camera_window=args.no_camera_window,
      pose_model=args.pose_model,
      upper_body_only=args.upper_body_only,
      flip_depth=args.flip_depth,
      algorithm_version=args.algorithm_version,
    )
  )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
