from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import mujoco
import numpy as np

from src import SRC_PATH
from src.motion.sew_mimic import (
  candidate_sort_key,
  matrix_orientation_error,
  normalize,
  orientation_error,
  solve_two_axis_rotation,
  subproblem1,
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
    self._axis_joint_ids = {
      "left": {
        "thigh": self._named_id(mujoco.mjtObj.mjOBJ_JOINT, "left_hip_yaw_joint"),
      },
      "right": {
        "thigh": self._named_id(mujoco.mjtObj.mjOBJ_JOINT, "right_hip_yaw_joint"),
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
    thigh = normalize(target.knee - target.hip)
    shank = normalize(target.ankle - target.knee)
    q = self._solve_segment_group(
      q,
      slice(start_index, start_index + 2),
      side,
      "thigh",
      thigh,
    )
    q = self._solve_segment_group(
      q,
      slice(start_index + 2, start_index + 4),
      side,
      "shank",
      shank,
    )
    return self._solve_foot_pointing_group(
      q,
      slice(start_index + 4, start_index + 6),
      self._foot_site_ids[side],
      target.foot_orientation,
    )

  def _solve_segment_group(
    self,
    q: np.ndarray,
    group: slice,
    side: str,
    segment: str,
    target_segment_axis: np.ndarray,
  ) -> np.ndarray:
    indexes = np.arange(group.start or 0, group.stop or len(q))
    base = q.copy()
    base[indexes] = 0.0
    self._set_lower_body_joint_angles(base)
    first_axis = normalize(self.data.xaxis[self.controlled_joint_ids[indexes[0]]])
    second_axis = normalize(self.data.xaxis[self.controlled_joint_ids[indexes[1]]])
    initial_axis = self._current_leg_segment_axis(side, segment)
    candidates = solve_two_axis_rotation(
      initial_axis,
      target_segment_axis,
      first_axis,
      second_axis,
    )
    return self._select_segment_group_candidate(q, indexes, candidates, side, segment, target_segment_axis)

  def _solve_foot_pointing_group(
    self,
    q: np.ndarray,
    group: slice,
    foot_site_id: int,
    desired_foot_orientation: np.ndarray,
  ) -> np.ndarray:
    indexes = np.arange(group.start or 0, group.stop or len(q))
    desired = np.asarray(desired_foot_orientation, dtype=float).reshape(3, 3)
    base = q.copy()
    base[indexes] = 0.0
    self._set_lower_body_joint_angles(base)
    pitch_axis = normalize(self.data.xaxis[self.controlled_joint_ids[indexes[0]]])
    initial_x_axis = normalize(self.data.site_xmat[foot_site_id].reshape(3, 3)[:, 0])
    pitch = self._select_single_axis_angle(
      q,
      indexes[0],
      self._single_axis_candidates(initial_x_axis, desired[:, 0], pitch_axis),
      lambda candidate_q: orientation_error(
        self._site_axis_after_setting(candidate_q, foot_site_id, 0),
        desired[:, 0],
      ),
    )

    pitched = q.copy()
    pitched[indexes[0]] = pitch
    pitched[indexes[1]] = 0.0
    self._set_lower_body_joint_angles(pitched)
    roll_axis = normalize(self.data.xaxis[self.controlled_joint_ids[indexes[1]]])
    initial_y_axis = normalize(self.data.site_xmat[foot_site_id].reshape(3, 3)[:, 1])
    roll = self._select_single_axis_angle(
      pitched,
      indexes[1],
      self._single_axis_candidates(initial_y_axis, desired[:, 1], roll_axis),
      lambda candidate_q: orientation_error(
        self._site_axis_after_setting(candidate_q, foot_site_id, 1),
        desired[:, 1],
      ),
    )
    solved = q.copy()
    solved[indexes[0]] = pitch
    solved[indexes[1]] = roll
    return np.clip(solved, self.joint_limits[:, 0], self.joint_limits[:, 1])

  def _select_segment_group_candidate(
    self,
    q: np.ndarray,
    indexes: np.ndarray,
    candidates: list[tuple[float, float]],
    side: str,
    segment: str,
    target_segment_axis: np.ndarray,
  ) -> np.ndarray:
    def score(candidate_q: np.ndarray) -> tuple[float, float]:
      self._set_lower_body_joint_angles(candidate_q)
      return candidate_sort_key(
        axis_error=orientation_error(self._current_leg_segment_axis(side, segment), target_segment_axis),
        joint_distance=float(np.linalg.norm(candidate_q[indexes] - q[indexes])),
      )

    return self._select_candidate(q, indexes, candidates, score)

  def _single_axis_candidates(
    self,
    initial_axis: np.ndarray,
    target_axis: np.ndarray,
    rotation_axis: np.ndarray,
  ) -> list[float]:
    angle = subproblem1(initial_axis, target_axis, rotation_axis)
    return [angle]

  def _select_single_axis_angle(
    self,
    q: np.ndarray,
    joint_index: int,
    candidates: list[float],
    score,
  ) -> float:
    best_angle: float | None = None
    best_score: tuple[float, float] | None = None
    for angle in candidates:
      for bounded_angle in self._bounded_equivalent_angles(angle, joint_index):
        candidate_q = q.copy()
        candidate_q[joint_index] = bounded_angle
        candidate_score = candidate_sort_key(
          axis_error=float(score(candidate_q)),
          joint_distance=abs(float(bounded_angle - q[joint_index])),
        )
        if best_score is None or candidate_score < best_score:
          best_score = candidate_score
          best_angle = bounded_angle
    if best_angle is not None:
      return best_angle
    return float(np.clip(candidates[0] if candidates else q[joint_index], *self.joint_limits[joint_index]))

  def _site_axis_after_setting(
    self,
    q: np.ndarray,
    site_id: int,
    axis_index: int,
  ) -> np.ndarray:
    self._set_lower_body_joint_angles(q)
    return normalize(self.data.site_xmat[site_id].reshape(3, 3)[:, axis_index])

  def _select_candidate(
    self,
    q: np.ndarray,
    indexes: np.ndarray,
    relative_candidates: list[tuple[float, float]],
    score,
  ) -> np.ndarray:
    best_q: np.ndarray | None = None
    best_score: tuple[float, float] | None = None
    for first, second in relative_candidates:
      for candidate_values in self._bounded_angle_pairs(indexes, first, second):
        candidate_q = q.copy()
        candidate_q[indexes] = candidate_values
        candidate_score = score(candidate_q)
        if best_score is None or candidate_score < best_score:
          best_score = candidate_score
          best_q = candidate_q
    if best_q is not None:
      return np.clip(best_q, self.joint_limits[:, 0], self.joint_limits[:, 1])

    fallback = q.copy()
    if relative_candidates:
      fallback[indexes] = np.array(relative_candidates[0], dtype=float)
    return np.clip(fallback, self.joint_limits[:, 0], self.joint_limits[:, 1])

  def _bounded_angle_pairs(
    self,
    indexes: np.ndarray,
    first: float,
    second: float,
  ) -> list[np.ndarray]:
    first_values = self._bounded_equivalent_angles(first, indexes[0])
    second_values = self._bounded_equivalent_angles(second, indexes[1])
    return [
      np.array([first_value, second_value], dtype=float)
      for first_value in first_values
      for second_value in second_values
    ]

  def _bounded_equivalent_angles(self, angle: float, joint_index: int) -> list[float]:
    lower, upper = self.joint_limits[joint_index]
    values = [
      float(angle + 2.0 * np.pi * offset)
      for offset in range(-2, 3)
      if lower - 1e-9 <= angle + 2.0 * np.pi * offset <= upper + 1e-9
    ]
    if values:
      return values
    return []

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
    hip = self.data.xanchor[ids["hip"]].copy()
    raw_knee = self.data.xanchor[ids["knee"]].copy()
    raw_ankle = self.data.xanchor[ids["ankle"]].copy()
    thigh_length = float(np.linalg.norm(raw_knee - hip))
    shank_length = float(np.linalg.norm(raw_ankle - raw_knee))
    thigh_axis = self._current_leg_segment_axis(side, "thigh")
    shank_axis = normalize(raw_ankle - raw_knee)
    knee = hip + thigh_length * thigh_axis
    ankle = knee + shank_length * shank_axis
    return LegKeypointTarget(
      hip=hip,
      knee=knee,
      ankle=ankle,
      foot_orientation=self.data.site_xmat[self._foot_site_ids[side]].reshape(3, 3).copy(),
    )

  def _current_leg_segment_axis(self, side: str, segment: str) -> np.ndarray:
    ids = self._keypoint_joint_ids[side]
    if segment == "thigh":
      return -normalize(self.data.xaxis[self._axis_joint_ids[side]["thigh"]])
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
