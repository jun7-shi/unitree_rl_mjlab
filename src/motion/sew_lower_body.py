from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import mujoco
import numpy as np
from scipy.optimize import least_squares

from src import SRC_PATH
from src.motion.sew_mimic import (
  matrix_orientation_error,
  normalize,
  orientation_error,
  rotation_vector_from_matrix,
)


@dataclass(frozen=True)
class LegKeypointTarget:
  """SEW-Mimic target for one leg, treating hip-knee-ankle as shoulder-elbow-wrist."""

  hip: np.ndarray
  knee: np.ndarray
  ankle: np.ndarray
  foot_orientation: np.ndarray


@dataclass(frozen=True)
class LowerBodyTarget:
  left_leg: LegKeypointTarget
  right_leg: LegKeypointTarget


@dataclass(frozen=True)
class LowerBodyRetargetResult:
  joint_angles: np.ndarray
  full_qpos: np.ndarray
  success: bool
  errors: dict[str, float]
  message: str


class G1LowerBodySEWRetargeter:
  """Retarget hip-knee-ankle leg targets to Unitree G1 lower-body joints."""

  def __init__(self, xml_path: str | Path | None = None, foot_orientation_weight: float = 0.25):
    self.xml_path = Path(xml_path) if xml_path is not None else (
      SRC_PATH / "assets" / "robots" / "unitree_g1" / "xmls" / "g1.xml"
    )
    if foot_orientation_weight < 0.0:
      raise ValueError(f"foot_orientation_weight must be non-negative, got {foot_orientation_weight}")
    self.foot_orientation_weight = float(foot_orientation_weight)
    self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))
    self.data = mujoco.MjData(self.model)
    self.left_leg_joint_names = self._leg_joint_names("left")
    self.right_leg_joint_names = self._leg_joint_names("right")
    self.controlled_joint_names = (*self.left_leg_joint_names, *self.right_leg_joint_names)
    self.controlled_joint_ids = tuple(
      self._named_id(mujoco.mjtObj.mjOBJ_JOINT, name)
      for name in self.controlled_joint_names
    )
    self.controlled_qpos_addresses = np.array(
      [self.model.jnt_qposadr[joint_id] for joint_id in self.controlled_joint_ids],
      dtype=int,
    )
    self.joint_limits = np.array(
      [self.model.jnt_range[joint_id] for joint_id in self.controlled_joint_ids],
      dtype=float,
    )
    self._keypoint_joint_ids = {
      "left": {
        "hip": self._named_id(mujoco.mjtObj.mjOBJ_JOINT, "left_hip_pitch_joint"),
        "knee": self._named_id(mujoco.mjtObj.mjOBJ_JOINT, "left_knee_joint"),
        "ankle": self._named_id(mujoco.mjtObj.mjOBJ_JOINT, "left_ankle_pitch_joint"),
      },
      "right": {
        "hip": self._named_id(mujoco.mjtObj.mjOBJ_JOINT, "right_hip_pitch_joint"),
        "knee": self._named_id(mujoco.mjtObj.mjOBJ_JOINT, "right_knee_joint"),
        "ankle": self._named_id(mujoco.mjtObj.mjOBJ_JOINT, "right_ankle_pitch_joint"),
      },
    }
    self._foot_site_ids = {
      "left": self._named_id(mujoco.mjtObj.mjOBJ_SITE, "left_foot"),
      "right": self._named_id(mujoco.mjtObj.mjOBJ_SITE, "right_foot"),
    }
    self._qpos0 = self.model.qpos0.copy()

  def target_from_configuration(self, joint_angles: Sequence[float]) -> LowerBodyTarget:
    q = np.asarray(joint_angles, dtype=float)
    if q.shape != (12,):
      raise ValueError(f"joint_angles must have shape (12,), got {q.shape}")
    self._set_lower_body_joint_angles(q)
    return LowerBodyTarget(
      left_leg=self._target_leg_from_current_configuration("left"),
      right_leg=self._target_leg_from_current_configuration("right"),
    )

  def retarget(
    self,
    q_init: Sequence[float],
    target: LowerBodyTarget,
  ) -> LowerBodyRetargetResult:
    q = np.clip(np.asarray(q_init, dtype=float), self.joint_limits[:, 0], self.joint_limits[:, 1])
    if q.shape != (12,):
      raise ValueError(f"q_init must have shape (12,), got {q.shape}")

    q = self._solve_leg(q, "left", 0, target.left_leg)
    q = self._solve_leg(q, "right", 6, target.right_leg)
    errors = self._diagnostics(q, target)
    success = all(value <= 1e-3 for value in errors.values())
    return LowerBodyRetargetResult(
      joint_angles=q,
      full_qpos=self.full_qpos_from_joint_angles(q),
      success=success,
      errors=errors,
      message="converged" if success else "retargeting residual above tolerance",
    )

  def full_qpos_from_joint_angles(self, joint_angles: Sequence[float]) -> np.ndarray:
    q = np.asarray(joint_angles, dtype=float)
    if q.shape != (12,):
      raise ValueError(f"joint_angles must have shape (12,), got {q.shape}")
    full_qpos = self._qpos0.copy()
    full_qpos[self.controlled_qpos_addresses] = q
    return full_qpos

  def _solve_leg(
    self,
    q: np.ndarray,
    side: str,
    start_index: int,
    target: LegKeypointTarget,
  ) -> np.ndarray:
    indexes = np.arange(start_index, start_index + 6)
    bounds = (self.joint_limits[indexes, 0], self.joint_limits[indexes, 1])
    thigh = normalize(target.knee - target.hip)
    shank = normalize(target.ankle - target.knee)
    foot = np.asarray(target.foot_orientation, dtype=float).reshape(3, 3)

    def residual(values: np.ndarray) -> np.ndarray:
      candidate = q.copy()
      candidate[indexes] = values
      self._set_lower_body_joint_angles(candidate)
      current_thigh = self._current_leg_segment_axis(side, "thigh")
      current_shank = self._current_leg_segment_axis(side, "shank")
      current_foot = self.data.site_xmat[self._foot_site_ids[side]].reshape(3, 3)
      return np.concatenate(
        [
          current_thigh - thigh,
          current_shank - shank,
          self.foot_orientation_weight * rotation_vector_from_matrix(foot @ current_foot.T),
        ]
      )

    result = least_squares(
      residual,
      q[indexes],
      bounds=bounds,
      xtol=1e-11,
      ftol=1e-11,
      gtol=1e-11,
      max_nfev=800,
    )
    solved = q.copy()
    solved[indexes] = result.x
    return np.clip(solved, self.joint_limits[:, 0], self.joint_limits[:, 1])

  def _diagnostics(self, q: np.ndarray, target: LowerBodyTarget) -> dict[str, float]:
    self._set_lower_body_joint_angles(q)
    errors = {}
    errors.update(self._leg_diagnostics("left", target.left_leg))
    errors.update(self._leg_diagnostics("right", target.right_leg))
    return errors

  def _leg_diagnostics(self, side: str, target: LegKeypointTarget) -> dict[str, float]:
    target_thigh = normalize(target.knee - target.hip)
    target_shank = normalize(target.ankle - target.knee)
    current_foot = self.data.site_xmat[self._foot_site_ids[side]].reshape(3, 3)
    return {
      f"{side}_thigh": orientation_error(self._current_leg_segment_axis(side, "thigh"), target_thigh),
      f"{side}_shank": orientation_error(self._current_leg_segment_axis(side, "shank"), target_shank),
      f"{side}_foot": matrix_orientation_error(current_foot, target.foot_orientation),
    }

  def _target_leg_from_current_configuration(self, side: str) -> LegKeypointTarget:
    ids = self._keypoint_joint_ids[side]
    return LegKeypointTarget(
      hip=self.data.xanchor[ids["hip"]].copy(),
      knee=self.data.xanchor[ids["knee"]].copy(),
      ankle=self.data.xanchor[ids["ankle"]].copy(),
      foot_orientation=self.data.site_xmat[self._foot_site_ids[side]].reshape(3, 3).copy(),
    )

  def _current_leg_segment_axis(self, side: str, segment: str) -> np.ndarray:
    ids = self._keypoint_joint_ids[side]
    if segment == "thigh":
      return normalize(self.data.xanchor[ids["knee"]] - self.data.xanchor[ids["hip"]])
    if segment == "shank":
      return normalize(self.data.xanchor[ids["ankle"]] - self.data.xanchor[ids["knee"]])
    raise ValueError(f"Unknown leg segment: {segment}")

  def _set_lower_body_joint_angles(self, joint_angles: np.ndarray) -> None:
    self.data.qpos[:] = self._qpos0
    self.data.qpos[self.controlled_qpos_addresses] = joint_angles
    mujoco.mj_forward(self.model, self.data)

  def _named_id(self, obj_type: mujoco.mjtObj, name: str) -> int:
    for candidate in (f"robot/{name}", name):
      obj_id = mujoco.mj_name2id(self.model, obj_type, candidate)
      if obj_id >= 0:
        return int(obj_id)
    raise ValueError(f"Could not find {obj_type.name} named '{name}'")

  @staticmethod
  def _leg_joint_names(side: str) -> tuple[str, ...]:
    return tuple(
      f"{side}_{suffix}"
      for suffix in (
        "hip_pitch_joint",
        "hip_roll_joint",
        "hip_yaw_joint",
        "knee_joint",
        "ankle_pitch_joint",
        "ankle_roll_joint",
      )
    )


def retarget_lower_body_targets(
  targets: Sequence[LowerBodyTarget],
  *,
  q_init: Sequence[float] | None = None,
  retargeter: G1LowerBodySEWRetargeter | None = None,
) -> list[LowerBodyRetargetResult]:
  adapter = retargeter or G1LowerBodySEWRetargeter()
  q_previous = np.zeros(12) if q_init is None else np.asarray(q_init, dtype=float)
  results: list[LowerBodyRetargetResult] = []
  for target in targets:
    result = adapter.retarget(q_previous, target)
    results.append(result)
    q_previous = result.joint_angles
  return results
