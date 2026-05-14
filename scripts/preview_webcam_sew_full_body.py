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
from src.motion.webcam_pose import ExponentialJointFilter, full_body_target_from_mediapipe_landmarks


@dataclass(frozen=True)
class WebcamPreviewConfig:
  camera: int = 0
  width: int = 1280
  height: int = 720
  fps: float = 30.0
  mirror: bool = True
  landmark_scale: float = 1.0
  min_visibility: float = 0.5
  min_detection_confidence: float = 0.5
  min_tracking_confidence: float = 0.5
  smoothing_alpha: float = 0.35
  no_camera_window: bool = False


def run_webcam_preview(config: WebcamPreviewConfig) -> None:
  if config.width <= 0 or config.height <= 0:
    raise ValueError(f"width and height must be positive, got {config.width}x{config.height}")
  if config.fps <= 0.0:
    raise ValueError(f"fps must be positive, got {config.fps}")
  cv2, mp_pose, mp_drawing, viewer_module = _load_realtime_dependencies()

  capture = cv2.VideoCapture(config.camera)
  if not capture.isOpened():
    raise RuntimeError(f"Could not open webcam index {config.camera}")
  capture.set(cv2.CAP_PROP_FRAME_WIDTH, config.width)
  capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config.height)

  retargeter = G1FullBodySEWRetargeter()
  data = mujoco.MjData(retargeter.model)
  joint_filter = ExponentialJointFilter(alpha=config.smoothing_alpha)
  q_previous = np.zeros(len(retargeter.controlled_joint_names), dtype=float)
  frame_dt = 1.0 / config.fps

  with mp_pose.Pose(
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=config.min_detection_confidence,
    min_tracking_confidence=config.min_tracking_confidence,
  ) as pose, viewer_module.launch_passive(retargeter.model, data) as viewer:
    try:
      while viewer.is_running():
        frame_start = time.time()
        ok, frame = capture.read()
        if not ok:
          time.sleep(frame_dt)
          continue
        if config.mirror:
          frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        pose_result = pose.process(rgb)
        landmarks = _select_pose_landmarks(pose_result)
        if landmarks is not None:
          try:
            target = full_body_target_from_mediapipe_landmarks(
              landmarks,
              scale=config.landmark_scale,
              min_visibility=config.min_visibility,
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
          _draw_camera_overlay(cv2, mp_pose, mp_drawing, frame, pose_result)
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


def _select_pose_landmarks(pose_result):
  if getattr(pose_result, "pose_world_landmarks", None) is not None:
    return pose_result.pose_world_landmarks.landmark
  if getattr(pose_result, "pose_landmarks", None) is not None:
    return pose_result.pose_landmarks.landmark
  return None


def _draw_camera_overlay(cv2, mp_pose, mp_drawing, frame, pose_result) -> None:
  if getattr(pose_result, "pose_landmarks", None) is not None:
    mp_drawing.draw_landmarks(
      frame,
      pose_result.pose_landmarks,
      mp_pose.POSE_CONNECTIONS,
    )
  cv2.imshow("SEW-Mimic webcam pose", frame)


def _load_realtime_dependencies():
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
  mp_pose, mp_drawing = _load_mediapipe_solutions(mp)
  return cv2, mp_pose, mp_drawing, viewer_module


def _load_mediapipe_solutions(mp):
  solutions = getattr(mp, "solutions", None)
  if solutions is not None:
    return solutions.pose, solutions.drawing_utils

  try:
    pose = importlib.import_module("mediapipe.python.solutions.pose")
    drawing_utils = importlib.import_module("mediapipe.python.solutions.drawing_utils")
  except ModuleNotFoundError as exc:
    raise RuntimeError(
      "Installed mediapipe package does not expose the legacy Pose Solutions API. "
      "Install the webcam extra with `pip install -e '.[webcam]'`."
    ) from exc
  return pose, drawing_utils


def build_arg_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
    description="Retarget a live webcam human pose to Unitree G1 and preview it in MuJoCo."
  )
  parser.add_argument("--camera", type=int, default=0, help="OpenCV webcam index.")
  parser.add_argument("--width", type=int, default=1280, help="Requested camera width.")
  parser.add_argument("--height", type=int, default=720, help="Requested camera height.")
  parser.add_argument("--fps", type=float, default=30.0, help="Realtime loop rate.")
  parser.add_argument("--no-mirror", action="store_true", help="Do not mirror the webcam image.")
  parser.add_argument("--landmark-scale", type=float, default=1.0, help="Scale MediaPipe world landmarks.")
  parser.add_argument("--min-visibility", type=float, default=0.5, help="Minimum landmark visibility.")
  parser.add_argument("--min-detection-confidence", type=float, default=0.5, help="MediaPipe detection confidence.")
  parser.add_argument("--min-tracking-confidence", type=float, default=0.5, help="MediaPipe tracking confidence.")
  parser.add_argument("--smoothing-alpha", type=float, default=0.35, help="Joint low-pass filter alpha.")
  parser.add_argument("--no-camera-window", action="store_true", help="Only show MuJoCo, not the webcam overlay.")
  return parser


def main(argv: Sequence[str] | None = None) -> int:
  args = build_arg_parser().parse_args(argv)
  run_webcam_preview(
    WebcamPreviewConfig(
      camera=args.camera,
      width=args.width,
      height=args.height,
      fps=args.fps,
      mirror=not args.no_mirror,
      landmark_scale=args.landmark_scale,
      min_visibility=args.min_visibility,
      min_detection_confidence=args.min_detection_confidence,
      min_tracking_confidence=args.min_tracking_confidence,
      smoothing_alpha=args.smoothing_alpha,
      no_camera_window=args.no_camera_window,
    )
  )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
