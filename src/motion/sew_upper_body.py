from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import mujoco
import numpy as np
from scipy.optimize import least_squares

from src import SRC_PATH
from src.motion.sew_mimic import (
  ArmKeypointTarget,
  BRANCH_V2_ALGORITHM,
  PAPER_V1_ALGORITHM,
  SEWAlgorithmConfig,
  bounded_equivalent_angle_candidates,
  bounded_equivalent_angles,
  candidate_sort_key,
  matrix_orientation_error,
  normalize,
  orientation_error,
  rotation_vector_from_matrix,
  resolve_sew_algorithm_config,
  select_limit_projected_sew_candidate,
  solve_g1_perpendicular_wrist_angles,
  solve_two_axis_rotation,
)


@dataclass(frozen=True)
class UpperBodyTarget:
  """One frame of upper-body targets for SEW-Mimic retargeting."""

  chest_position: np.ndarray
  chest_orientation: np.ndarray
  left_arm: ArmKeypointTarget
  right_arm: ArmKeypointTarget


@dataclass(frozen=True)
class UpperBodyRetargetResult:
  joint_angles: np.ndarray
  full_qpos: np.ndarray
  success: bool
  errors: dict[str, float]
  message: str
  solver_joint_angles: np.ndarray | None = None


class G1UpperBodySEWRetargeter:
  """Retarget SOMA upper-body targets to G1 waist and bilateral arm joints.

  Scope vs the paper: the per-arm axis groups use the closed-form
  Subproblem 2 selection from ``sew_mimic`` and the G1 perpendicular wrists
  use the paper appendix's Euler decomposition. The waist (chest orientation)
  group is still solved numerically with ``scipy.optimize.least_squares``.
  The ``branch_v2`` algorithm additionally reflects four joint angles through
  the nearest joint-limit boundary as a post-processing step; that step has no
  analogue in the paper.

  Each ``ArmKeypointTarget`` is consumed as a direction-only target. Loaders
  that synthesize G1 axis proxy targets (see
  ``synthesize_g1_axis_proxy_arm_target``) can drive this solver to a zero
  axis residual against a reachable robot pose, but the resulting joint
  angles do not preserve the human's physical shoulder-to-elbow direction.
  """

  def __init__(
    self,
    xml_path: str | Path | None = None,
    algorithm_version: str | SEWAlgorithmConfig = BRANCH_V2_ALGORITHM,
  ):
    self.xml_path = Path(xml_path) if xml_path is not None else (
      SRC_PATH / "assets" / "robots" / "unitree_g1" / "xmls" / "g1.xml"
    )
    self.algorithm_config = resolve_sew_algorithm_config(algorithm_version)
    self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))
    self.data = mujoco.MjData(self.model)
    self.waist_joint_names = (
      "waist_yaw_joint",
      "waist_roll_joint",
      "waist_pitch_joint",
    )
    self.left_arm_joint_names = self._arm_joint_names("left")
    self.right_arm_joint_names = self._arm_joint_names("right")
    self.controlled_joint_names = (
      *self.waist_joint_names,
      *self.left_arm_joint_names,
      *self.right_arm_joint_names,
    )
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
    self._torso_body_id = self._named_id(mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    self._axis_joint_ids = {
      "left": {
        "upper": self._named_id(mujoco.mjtObj.mjOBJ_JOINT, "left_shoulder_yaw_joint"),
        "lower": self._named_id(mujoco.mjtObj.mjOBJ_JOINT, "left_wrist_roll_joint"),
      },
      "right": {
        "upper": self._named_id(mujoco.mjtObj.mjOBJ_JOINT, "right_shoulder_yaw_joint"),
        "lower": self._named_id(mujoco.mjtObj.mjOBJ_JOINT, "right_wrist_roll_joint"),
      },
    }
    self._keypoint_joint_ids = {
      "left": {
        "shoulder": self._named_id(mujoco.mjtObj.mjOBJ_JOINT, "left_shoulder_pitch_joint"),
        "elbow": self._named_id(mujoco.mjtObj.mjOBJ_JOINT, "left_elbow_joint"),
        "wrist": self._named_id(mujoco.mjtObj.mjOBJ_JOINT, "left_wrist_pitch_joint"),
      },
      "right": {
        "shoulder": self._named_id(mujoco.mjtObj.mjOBJ_JOINT, "right_shoulder_pitch_joint"),
        "elbow": self._named_id(mujoco.mjtObj.mjOBJ_JOINT, "right_elbow_joint"),
        "wrist": self._named_id(mujoco.mjtObj.mjOBJ_JOINT, "right_wrist_pitch_joint"),
      },
    }
    self._palm_site_ids = {
      "left": self._named_id(mujoco.mjtObj.mjOBJ_SITE, "left_palm"),
      "right": self._named_id(mujoco.mjtObj.mjOBJ_SITE, "right_palm"),
    }
    self._qpos0 = self.model.qpos0.copy()

  def target_from_configuration(self, joint_angles: Sequence[float]) -> UpperBodyTarget:
    """Build a reachable target from a G1 upper-body joint configuration."""
    q = np.asarray(joint_angles, dtype=float)
    if q.shape != (17,):
      raise ValueError(f"joint_angles must have shape (17,), got {q.shape}")
    self._set_upper_body_joint_angles(q)
    return UpperBodyTarget(
      chest_position=self.data.xpos[self._torso_body_id].copy(),
      chest_orientation=self.data.xmat[self._torso_body_id].reshape(3, 3).copy(),
      left_arm=self._target_arm_from_current_configuration("left"),
      right_arm=self._target_arm_from_current_configuration("right"),
    )

  def retarget(
    self,
    q_init: Sequence[float],
    target: UpperBodyTarget,
  ) -> UpperBodyRetargetResult:
    """Retarget one upper-body frame into G1 waist and arm joint angles."""
    if self.algorithm_config.name == BRANCH_V2_ALGORITHM.name:
      original_config = self.algorithm_config
      self.algorithm_config = PAPER_V1_ALGORITHM
      try:
        raw_result = self._retarget_raw(q_init, target)
      finally:
        self.algorithm_config = original_config
      raw_q = raw_result.joint_angles
      q = self._postprocess_branch_v2_joint_angles(raw_q)
      errors = self._diagnostics(q, target)
      success = all(value <= 1e-3 for value in errors.values())
      return UpperBodyRetargetResult(
        joint_angles=q,
        full_qpos=self.full_qpos_from_joint_angles(q),
        success=success,
        errors=errors,
        message="converged" if success else "retargeting residual above tolerance",
        solver_joint_angles=raw_q,
      )
    return self._retarget_raw(q_init, target)

  def _retarget_raw(
    self,
    q_init: Sequence[float],
    target: UpperBodyTarget,
  ) -> UpperBodyRetargetResult:
    q = self._clip_joint_angles(np.asarray(q_init, dtype=float))
    if q.shape != (17,):
      raise ValueError(f"q_init must have shape (17,), got {q.shape}")

    q = self._solve_orientation_group(q, slice(0, 3), self._torso_body_id, target.chest_orientation)
    q = self._solve_arm(q, "left", 3, target.left_arm)
    q = self._solve_arm(q, "right", 10, target.right_arm)
    errors = self._diagnostics(q, target)
    success = all(value <= 1e-3 for value in errors.values())
    return UpperBodyRetargetResult(
      joint_angles=q,
      full_qpos=self.full_qpos_from_joint_angles(q),
      success=success,
      errors=errors,
      message="converged" if success else "retargeting residual above tolerance",
      solver_joint_angles=q,
    )

  def full_qpos_from_joint_angles(self, joint_angles: Sequence[float]) -> np.ndarray:
    q = np.asarray(joint_angles, dtype=float)
    if q.shape != (17,):
      raise ValueError(f"joint_angles must have shape (17,), got {q.shape}")
    full_qpos = self._qpos0.copy()
    full_qpos[self.controlled_qpos_addresses] = q
    return full_qpos

  def _solve_arm(
    self,
    q: np.ndarray,
    side: str,
    start_index: int,
    target: ArmKeypointTarget,
  ) -> np.ndarray:
    upper_arm = normalize(target.elbow - target.shoulder)
    lower_arm = normalize(target.wrist - target.elbow)
    q = self._solve_axis_group(
      q,
      slice(start_index, start_index + 2),
      self._axis_joint_ids[side]["upper"],
      upper_arm,
    )
    q = self._solve_axis_group(
      q,
      slice(start_index + 2, start_index + 4),
      self._axis_joint_ids[side]["lower"],
      lower_arm,
    )
    return self._solve_wrist_group(
      q,
      slice(start_index + 4, start_index + 7),
      self._palm_site_ids[side],
      target.hand_orientation,
    )

  def _solve_orientation_group(
    self,
    q: np.ndarray,
    group: slice,
    body_id: int,
    desired_orientation: np.ndarray,
  ) -> np.ndarray:
    indexes = np.arange(group.start or 0, group.stop or len(q))
    desired = np.asarray(desired_orientation, dtype=float).reshape(3, 3)

    def residual(values: np.ndarray) -> np.ndarray:
      candidate = q.copy()
      candidate[indexes] = values
      self._set_upper_body_joint_angles(candidate)
      current = self.data.xmat[body_id].reshape(3, 3)
      return rotation_vector_from_matrix(desired @ current.T)

    result = least_squares(
      residual,
      q[indexes],
      **self._least_squares_limit_kwargs(indexes),
      xtol=1e-11,
      ftol=1e-11,
      gtol=1e-11,
      max_nfev=300,
    )
    solved = q.copy()
    solved[indexes] = result.x
    return self._clip_joint_angles(solved)

  def _solve_axis_group(
    self,
    q: np.ndarray,
    group: slice,
    joint_id: int,
    target_axis: np.ndarray,
  ) -> np.ndarray:
    indexes = np.arange(group.start or 0, group.stop or len(q))
    base = q.copy()
    base[indexes] = 0.0
    self._set_upper_body_joint_angles(base)
    first_axis = normalize(self.data.xaxis[self.controlled_joint_ids[indexes[0]]])
    second_axis = normalize(self.data.xaxis[self.controlled_joint_ids[indexes[1]]])
    initial_axis = normalize(self.data.xaxis[joint_id])
    candidates = solve_two_axis_rotation(
      initial_axis,
      target_axis,
      first_axis,
      second_axis,
    )
    return self._select_axis_group_candidate(q, indexes, candidates, joint_id, target_axis)

  def _select_axis_group_candidate(
    self,
    q: np.ndarray,
    indexes: np.ndarray,
    candidates: list[tuple[float, float]],
    joint_id: int,
    target_axis: np.ndarray,
  ) -> np.ndarray:
    scored_candidates: list[tuple[np.ndarray, tuple[float, float], float, bool]] = []
    for first, second in candidates:
      for values, clipped in self._bounded_angle_pair_candidates(indexes, first, second):
        candidate_q = q.copy()
        candidate_q[indexes] = values
        self._set_upper_body_joint_angles(candidate_q)
        joint_distance = float(np.linalg.norm(candidate_q[indexes] - q[indexes]))
        score = candidate_sort_key(
          axis_error=orientation_error(self.data.xaxis[joint_id], target_axis),
          joint_distance=joint_distance,
          error_tolerance=self.algorithm_config.error_tolerance,
        )
        scored_candidates.append((candidate_q, score, joint_distance, clipped))
    best_q = select_limit_projected_sew_candidate(scored_candidates, config=self.algorithm_config)
    if best_q is not None:
      return self._clip_joint_angles(best_q)

    fallback = q.copy()
    if candidates:
      fallback[indexes] = np.array(candidates[0], dtype=float)
    return self._clip_joint_angles(fallback)

  def _bounded_angle_pairs(
    self,
    indexes: np.ndarray,
    first: float,
    second: float,
  ) -> list[np.ndarray]:
    return [values for values, _ in self._bounded_angle_pair_candidates(indexes, first, second)]

  def _bounded_angle_pair_candidates(
    self,
    indexes: np.ndarray,
    first: float,
    second: float,
  ) -> list[tuple[np.ndarray, bool]]:
    first_values = self._bounded_equivalent_angle_candidates(first, indexes[0])
    second_values = self._bounded_equivalent_angle_candidates(second, indexes[1])
    return [
      (np.array([a.angle, b.angle], dtype=float), a.clipped or b.clipped)
      for a in first_values
      for b in second_values
    ]

  def _bounded_angle_triplet_candidates(
    self,
    indexes: np.ndarray,
    values: Sequence[float],
  ) -> list[tuple[np.ndarray, bool]]:
    candidates = [
      self._bounded_equivalent_angle_candidates(float(value), int(index))
      for value, index in zip(values, indexes)
    ]
    return [
      (
        np.array([first.angle, second.angle, third.angle], dtype=float),
        first.clipped or second.clipped or third.clipped,
      )
      for first in candidates[0]
      for second in candidates[1]
      for third in candidates[2]
    ]

  def _bounded_equivalent_angle_candidates(self, angle: float, joint_index: int):
    return bounded_equivalent_angle_candidates(angle, self.joint_limits, joint_index, config=self.algorithm_config)

  def _bounded_equivalent_angles(self, angle: float, joint_index: int) -> list[float]:
    return bounded_equivalent_angles(angle, self.joint_limits, joint_index, config=self.algorithm_config)

  def _solve_wrist_group(
    self,
    q: np.ndarray,
    group: slice,
    site_id: int,
    desired_hand_orientation: np.ndarray,
  ) -> np.ndarray:
    indexes = np.arange(group.start or 0, group.stop or len(q))
    desired = np.asarray(desired_hand_orientation, dtype=float).reshape(3, 3)
    wrist_zero = q.copy()
    wrist_zero[indexes] = 0.0
    self._set_upper_body_joint_angles(wrist_zero)
    zero_orientation = self.data.site_xmat[site_id].reshape(3, 3).copy()
    relative_desired = zero_orientation.T @ desired

    scored_candidates: list[tuple[np.ndarray, tuple[float, float], float, bool]] = []
    for euler_values in solve_g1_perpendicular_wrist_angles(relative_desired):
      for values, clipped in self._bounded_angle_triplet_candidates(indexes, euler_values):
        candidate_q = q.copy()
        candidate_q[indexes] = values
        self._set_upper_body_joint_angles(candidate_q)
        current = self.data.site_xmat[site_id].reshape(3, 3)
        joint_distance = float(np.linalg.norm(candidate_q[indexes] - q[indexes]))
        score = candidate_sort_key(
          axis_error=matrix_orientation_error(current, desired),
          joint_distance=joint_distance,
          error_tolerance=self.algorithm_config.error_tolerance,
        )
        scored_candidates.append((candidate_q, score, joint_distance, clipped))

    best_q = select_limit_projected_sew_candidate(scored_candidates, config=self.algorithm_config)
    if best_q is not None:
      return self._clip_joint_angles(best_q)
    return self._clip_joint_angles(q)

  def _least_squares_limit_kwargs(self, indexes: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    if not self.algorithm_config.respect_joint_limits:
      return {}
    return {
      "bounds": (
        self.joint_limits[indexes, 0],
        self.joint_limits[indexes, 1],
      )
    }

  def _diagnostics(self, q: np.ndarray, target: UpperBodyTarget) -> dict[str, float]:
    self._set_upper_body_joint_angles(q)
    current_chest = self.data.xmat[self._torso_body_id].reshape(3, 3)
    errors = {
      "waist": matrix_orientation_error(current_chest, target.chest_orientation),
    }
    errors.update(self._arm_diagnostics("left", target.left_arm))
    errors.update(self._arm_diagnostics("right", target.right_arm))
    return errors

  def _arm_diagnostics(self, side: str, target: ArmKeypointTarget) -> dict[str, float]:
    upper_arm = normalize(target.elbow - target.shoulder)
    lower_arm = normalize(target.wrist - target.elbow)
    current_upper = normalize(self.data.xaxis[self._axis_joint_ids[side]["upper"]])
    current_lower = normalize(self.data.xaxis[self._axis_joint_ids[side]["lower"]])
    current_hand = self.data.site_xmat[self._palm_site_ids[side]].reshape(3, 3)
    return {
      f"{side}_upper_arm": orientation_error(current_upper, upper_arm),
      f"{side}_lower_arm": orientation_error(current_lower, lower_arm),
      f"{side}_wrist": matrix_orientation_error(current_hand, target.hand_orientation),
    }

  def _target_arm_from_current_configuration(self, side: str) -> ArmKeypointTarget:
    keypoint_ids = self._keypoint_joint_ids[side]
    shoulder = self.data.xanchor[keypoint_ids["shoulder"]].copy()
    raw_elbow = self.data.xanchor[keypoint_ids["elbow"]].copy()
    raw_wrist = self.data.xanchor[keypoint_ids["wrist"]].copy()
    upper_length = float(np.linalg.norm(raw_elbow - shoulder))
    lower_length = float(np.linalg.norm(raw_wrist - raw_elbow))
    upper_axis = normalize(self.data.xaxis[self._axis_joint_ids[side]["upper"]])
    lower_axis = normalize(self.data.xaxis[self._axis_joint_ids[side]["lower"]])
    elbow = shoulder + upper_length * upper_axis
    wrist = elbow + lower_length * lower_axis
    return ArmKeypointTarget(
      shoulder=shoulder,
      elbow=elbow,
      wrist=wrist,
      hand_orientation=self.data.site_xmat[self._palm_site_ids[side]].reshape(3, 3).copy(),
    )

  def _set_upper_body_joint_angles(self, joint_angles: np.ndarray) -> None:
    self.data.qpos[:] = self._qpos0
    self.data.qpos[self.controlled_qpos_addresses] = joint_angles
    mujoco.mj_forward(self.model, self.data)

  def _clip_joint_angles(self, joint_angles: np.ndarray) -> np.ndarray:
    q = np.asarray(joint_angles, dtype=float)
    if not self.algorithm_config.respect_joint_limits:
      return q.copy()
    return np.clip(q, self.joint_limits[:, 0], self.joint_limits[:, 1])

  def _postprocess_branch_v2_joint_angles(self, joint_angles: np.ndarray) -> np.ndarray:
    q = np.asarray(joint_angles, dtype=float).copy()
    for joint_index in (3, 5, 10, 12):
      q[joint_index] = self._reflect_angle_through_nearest_pi_boundary(
        q[joint_index],
        self.joint_limits[joint_index],
      )
    return np.clip(q, self.joint_limits[:, 0], self.joint_limits[:, 1])

  @staticmethod
  def _reflect_angle_through_nearest_pi_boundary(angle: float, joint_limit: np.ndarray) -> float:
    low, high = np.asarray(joint_limit, dtype=float)
    reflected = float(angle)
    for _ in range(4):
      if reflected < low:
        reflected = -2.0 * np.pi - reflected
      elif reflected > high:
        reflected = 2.0 * np.pi - reflected
      else:
        break
    return float(np.clip(reflected, low, high))

  def _named_id(self, obj_type: mujoco.mjtObj, name: str) -> int:
    for candidate in (f"robot/{name}", name):
      obj_id = mujoco.mj_name2id(self.model, obj_type, candidate)
      if obj_id >= 0:
        return int(obj_id)
    raise ValueError(f"Could not find {obj_type.name} named '{name}'")

  @staticmethod
  def _arm_joint_names(side: str) -> tuple[str, ...]:
    return tuple(
      f"{side}_{suffix}"
      for suffix in (
        "shoulder_pitch_joint",
        "shoulder_roll_joint",
        "shoulder_yaw_joint",
        "elbow_joint",
        "wrist_roll_joint",
        "wrist_pitch_joint",
        "wrist_yaw_joint",
      )
    )


def retarget_upper_body_targets(
  targets: Sequence[UpperBodyTarget],
  *,
  q_init: Sequence[float] | None = None,
  retargeter: G1UpperBodySEWRetargeter | None = None,
) -> list[UpperBodyRetargetResult]:
  """Retarget a sequence of upper-body targets with previous-frame warm starts."""
  adapter = retargeter or G1UpperBodySEWRetargeter()
  q_previous = np.zeros(17) if q_init is None else np.asarray(q_init, dtype=float)
  results: list[UpperBodyRetargetResult] = []
  for target in targets:
    result = adapter.retarget(q_previous, target)
    results.append(result)
    q_previous = result.solver_joint_angles if result.solver_joint_angles is not None else result.joint_angles
  return results
