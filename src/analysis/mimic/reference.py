"""Reference motion analysis for mimic training."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .gait_metrics import (
  ClearanceSummary,
  GaitPhaseSummary,
  clearance_summary,
  gait_phase_summary,
)


@dataclass(frozen=True, slots=True)
class ReferenceMotion:
  """Loaded motion NPZ arrays needed for gait analysis."""

  path: Path
  fps: float
  joint_pos: np.ndarray
  joint_vel: np.ndarray
  body_pos_w: np.ndarray
  body_lin_vel_w: np.ndarray


@dataclass(frozen=True, slots=True)
class ReferenceAnalysis:
  """Reference gait analysis bundle."""

  motion_file: str
  fps: float
  frame_count: int
  foot_body_indices: tuple[int, int]
  foot_height_m: np.ndarray
  foot_clearance_m: np.ndarray
  foot_vertical_velocity_m_s: np.ndarray
  contact_flags: np.ndarray
  phase_summary: GaitPhaseSummary
  clearance: ClearanceSummary

  def summary_dict(self) -> dict[str, float | int | str | tuple[int, int]]:
    data = {
      "motion_file": self.motion_file,
      "fps": self.fps,
      "frame_count": self.frame_count,
      "foot_body_indices": self.foot_body_indices,
    }
    data.update({f"reference_{k}": v for k, v in self.phase_summary.to_dict().items()})
    data.update({f"reference_{k}": v for k, v in self.clearance.to_dict().items()})
    return data


def load_reference_motion(path: str | Path) -> ReferenceMotion:
  """Load a tracking motion NPZ."""

  motion_path = Path(path)
  with np.load(motion_path) as data:
    required = ("fps", "joint_pos", "joint_vel", "body_pos_w", "body_lin_vel_w")
    missing = [key for key in required if key not in data]
    if missing:
      raise ValueError(f"Motion file is missing required arrays: {missing}")
    fps_array = np.asarray(data["fps"]).reshape(-1)
    if fps_array.size == 0:
      raise ValueError("Motion file fps array is empty")
    return ReferenceMotion(
      path=motion_path,
      fps=float(fps_array[0]),
      joint_pos=np.asarray(data["joint_pos"], dtype=float),
      joint_vel=np.asarray(data["joint_vel"], dtype=float),
      body_pos_w=np.asarray(data["body_pos_w"], dtype=float),
      body_lin_vel_w=np.asarray(data["body_lin_vel_w"], dtype=float),
    )


def derive_foot_contact_from_height(
  foot_height_m: np.ndarray,
  foot_vertical_velocity_m_s: np.ndarray,
  height_threshold_m: float = 0.025,
  vertical_speed_threshold_m_s: float = 0.35,
) -> np.ndarray:
  """Infer contact from low foot height and limited vertical speed."""

  height = np.asarray(foot_height_m, dtype=float)
  velocity = np.asarray(foot_vertical_velocity_m_s, dtype=float)
  if height.shape != velocity.shape:
    raise ValueError("foot_height_m and foot_vertical_velocity_m_s must match")
  if height.ndim != 2 or height.shape[1] != 2:
    raise ValueError("foot arrays must have shape (frames, 2)")
  return (height <= height_threshold_m) & (
    np.abs(velocity) <= vertical_speed_threshold_m_s
  )


def foot_signals_from_motion(
  motion: ReferenceMotion,
  foot_body_indices: Sequence[int] = (6, 12),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Return foot height, clearance, and vertical velocity arrays."""

  if len(foot_body_indices) != 2:
    raise ValueError("foot_body_indices must contain left and right indices")
  indices = tuple(int(index) for index in foot_body_indices)
  max_index = motion.body_pos_w.shape[1] - 1
  if min(indices) < 0 or max(indices) > max_index:
    raise ValueError(
      f"foot_body_indices {indices} exceed body_pos_w body range 0..{max_index}"
    )
  foot_height = motion.body_pos_w[:, indices, 2]
  floor_estimate = np.min(foot_height, axis=0, keepdims=True)
  foot_clearance = foot_height - floor_estimate
  foot_vertical_velocity = motion.body_lin_vel_w[:, indices, 2]
  return foot_height, foot_clearance, foot_vertical_velocity


def analyze_reference_motion(
  path: str | Path,
  foot_body_indices: Sequence[int] = (6, 12),
  height_threshold_m: float = 0.025,
  vertical_speed_threshold_m_s: float = 0.35,
) -> ReferenceAnalysis:
  """Load and summarize a reference motion NPZ."""

  motion = load_reference_motion(path)
  foot_height, foot_clearance, foot_vertical_velocity = foot_signals_from_motion(
    motion, foot_body_indices=foot_body_indices
  )
  contact = derive_foot_contact_from_height(
    foot_clearance,
    foot_vertical_velocity,
    height_threshold_m=height_threshold_m,
    vertical_speed_threshold_m_s=vertical_speed_threshold_m_s,
  )
  return ReferenceAnalysis(
    motion_file=str(motion.path),
    fps=motion.fps,
    frame_count=int(motion.joint_pos.shape[0]),
    foot_body_indices=tuple(int(index) for index in foot_body_indices),
    foot_height_m=foot_height,
    foot_clearance_m=foot_clearance,
    foot_vertical_velocity_m_s=foot_vertical_velocity,
    contact_flags=contact,
    phase_summary=gait_phase_summary(contact, fps=motion.fps),
    clearance=clearance_summary(foot_clearance, contact),
  )


def reference_rows(analysis: ReferenceAnalysis) -> list[dict[str, float | int]]:
  """Convert frame-level reference signals to CSV rows."""

  rows: list[dict[str, float | int]] = []
  for idx in range(analysis.frame_count):
    rows.append(
      {
        "frame": idx,
        "time_s": idx / analysis.fps,
        "left_contact": int(analysis.contact_flags[idx, 0]),
        "right_contact": int(analysis.contact_flags[idx, 1]),
        "left_clearance_m": float(analysis.foot_clearance_m[idx, 0]),
        "right_clearance_m": float(analysis.foot_clearance_m[idx, 1]),
        "left_vertical_velocity_m_s": float(
          analysis.foot_vertical_velocity_m_s[idx, 0]
        ),
        "right_vertical_velocity_m_s": float(
          analysis.foot_vertical_velocity_m_s[idx, 1]
        ),
      }
    )
  return rows
