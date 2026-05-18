from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from src import SRC_PATH
from src.motion.seed_bones import G1_29DOF_JOINT_COLUMNS


ArrayLike3 = Sequence[float] | np.ndarray
BRANCH_PRESERVATION_MAX_DISTANCE = np.deg2rad(45.0)


@dataclass(frozen=True)
class SEWAlgorithmConfig:
  """Candidate-selection semantics for the SEW closed-form solver."""

  name: str
  error_tolerance: float
  clip_limit_candidates: bool
  branch_preserving: bool
  respect_joint_limits: bool = True
  max_branch_distance: float = BRANCH_PRESERVATION_MAX_DISTANCE
  max_local_axis_error: float = 2e-2
  limit_clip_margin: float = 0.0


@dataclass(frozen=True)
class BoundedAngleCandidate:
  """A joint-limit-aware equivalent angle and whether it was limit-projected."""

  angle: float
  clipped: bool


PAPER_V1_ALGORITHM = SEWAlgorithmConfig(
  name="paper_v1",
  error_tolerance=1e-9,
  clip_limit_candidates=False,
  branch_preserving=False,
  respect_joint_limits=False,
)
"""Closed-form branch selection that matches the paper's geometric solver.

``paper_v1`` describes the candidate-selection rule used by Subproblem
1/2/4 axis solves. For G1, the final three wrist joints use the paper
appendix's perpendicular-wrist Euler decomposition rather than a numerical
optimizer.
"""
BRANCH_V2_ALGORITHM = SEWAlgorithmConfig(
  name="branch_v2",
  error_tolerance=1e-3,
  clip_limit_candidates=True,
  branch_preserving=True,
  respect_joint_limits=True,
  limit_clip_margin=np.pi,
)
SEW_ALGORITHM_CONFIGS = {
  PAPER_V1_ALGORITHM.name: PAPER_V1_ALGORITHM,
  BRANCH_V2_ALGORITHM.name: BRANCH_V2_ALGORITHM,
}


def resolve_sew_algorithm_config(config: str | SEWAlgorithmConfig = BRANCH_V2_ALGORITHM) -> SEWAlgorithmConfig:
  if isinstance(config, SEWAlgorithmConfig):
    return config
  try:
    return SEW_ALGORITHM_CONFIGS[config]
  except KeyError as exc:
    options = ", ".join(sorted(SEW_ALGORITHM_CONFIGS))
    raise ValueError(f"Unknown SEW algorithm version '{config}'. Expected one of: {options}") from exc


def normalize(vector: ArrayLike3, eps: float = 1e-12) -> np.ndarray:
  """Return a unit vector and reject zero-length inputs."""
  array = np.asarray(vector, dtype=float)
  norm = float(np.linalg.norm(array))
  if norm <= eps:
    raise ValueError("Cannot normalize a zero-length vector")
  return array / norm


def skew(vector: ArrayLike3) -> np.ndarray:
  """Return the skew-symmetric cross-product matrix."""
  x, y, z = np.asarray(vector, dtype=float)
  return np.array(
    [
      [0.0, -z, y],
      [z, 0.0, -x],
      [-y, x, 0.0],
    ],
    dtype=float,
  )


def rotation_about_axis(axis: ArrayLike3, theta: float) -> np.ndarray:
  """Rodrigues rotation matrix for a unit or non-unit axis."""
  unit_axis = normalize(axis)
  axis_hat = skew(unit_axis)
  return (
    np.eye(3)
    + math_sin(theta) * axis_hat
    + (1.0 - math_cos(theta)) * (axis_hat @ axis_hat)
  )


def orientation_error(lhs: ArrayLike3, rhs: ArrayLike3) -> float:
  """Paper Eq. 1 vector orientation error in [0, 1]."""
  lhs_unit = normalize(lhs)
  rhs_unit = normalize(rhs)
  cosine = float(np.clip(np.dot(lhs_unit, rhs_unit), -1.0, 1.0))
  return 0.5 - 0.5 * cosine


def candidate_sort_key(
  *,
  axis_error: float,
  joint_distance: float,
  error_tolerance: float = 1e-3,
) -> tuple[float, float]:
  """Rank closed-form candidates while ignoring numerical error ties."""
  comparable_error = 0.0 if abs(float(axis_error)) <= error_tolerance else float(axis_error)
  return comparable_error, float(joint_distance)


def select_sew_candidate(
  candidates: Iterable[tuple[np.ndarray, tuple[float, float], float]],
  *,
  config: str | SEWAlgorithmConfig = BRANCH_V2_ALGORITHM,
) -> np.ndarray | None:
  """Choose a closed-form candidate according to the requested algorithm version."""
  algorithm = resolve_sew_algorithm_config(config)
  candidate_list = list(candidates)
  if not candidate_list:
    return None
  if not algorithm.branch_preserving:
    return min(candidate_list, key=lambda candidate: candidate[1])[0]
  local_candidates = [
    candidate
    for candidate in candidate_list
    if candidate[2] <= algorithm.max_branch_distance and candidate[1][0] <= algorithm.max_local_axis_error
  ]
  search_space = local_candidates or candidate_list
  return min(search_space, key=lambda candidate: candidate[1])[0]


def select_limit_projected_sew_candidate(
  candidates: Iterable[tuple[np.ndarray, tuple[float, float], float, bool]],
  *,
  config: str | SEWAlgorithmConfig = BRANCH_V2_ALGORITHM,
) -> np.ndarray | None:
  """Choose a SEW candidate after joint-limit projection."""
  algorithm = resolve_sew_algorithm_config(config)
  candidate_list = list(candidates)
  if not candidate_list:
    return None
  if algorithm.branch_preserving and algorithm.respect_joint_limits and algorithm.clip_limit_candidates:
    local_exact = [
      (candidate_q, score, joint_distance)
      for candidate_q, score, joint_distance, clipped in candidate_list
      if (
        not clipped
        and joint_distance <= algorithm.max_branch_distance
        and score[0] <= algorithm.max_local_axis_error
      )
    ]
    if local_exact:
      return select_sew_candidate(local_exact, config=algorithm)
    local_projected = [
      (candidate_q, score, joint_distance)
      for candidate_q, score, joint_distance, clipped in candidate_list
      if clipped and joint_distance <= algorithm.max_branch_distance
    ]
    if local_projected:
      return min(local_projected, key=lambda candidate: (candidate[2], candidate[1]))[0]
  return select_sew_candidate(
    (
      (candidate_q, score, joint_distance)
      for candidate_q, score, joint_distance, _ in candidate_list
    ),
    config=algorithm,
  )


def select_branch_preserving_candidate(
  candidates: Iterable[tuple[np.ndarray, tuple[float, float], float]],
  *,
  max_joint_distance: float = BRANCH_PRESERVATION_MAX_DISTANCE,
  max_local_axis_error: float = 2e-2,
) -> np.ndarray | None:
  """Choose the best candidate without leaving a nearby saturated branch."""
  return select_sew_candidate(
    candidates,
    config=SEWAlgorithmConfig(
      name="branch_preserving_custom",
      error_tolerance=1e-3,
      clip_limit_candidates=True,
      branch_preserving=True,
      respect_joint_limits=True,
      max_branch_distance=max_joint_distance,
      max_local_axis_error=max_local_axis_error,
      limit_clip_margin=np.pi,
    ),
  )


def bounded_equivalent_angles(
  angle: float,
  joint_limits: np.ndarray,
  joint_index: int,
  *,
  offsets: Iterable[int] = (-1, 0, 1),
  config: str | SEWAlgorithmConfig = BRANCH_V2_ALGORITHM,
) -> list[float]:
  """Return bounded angle equivalents, including branch-preserving limit clips."""
  return [
    candidate.angle
    for candidate in bounded_equivalent_angle_candidates(
      angle,
      joint_limits,
      joint_index,
      offsets=offsets,
      config=config,
    )
  ]


def bounded_equivalent_angle_candidates(
  angle: float,
  joint_limits: np.ndarray,
  joint_index: int,
  *,
  offsets: Iterable[int] = (-1, 0, 1),
  config: str | SEWAlgorithmConfig = BRANCH_V2_ALGORITHM,
) -> list[BoundedAngleCandidate]:
  """Return equivalent angles annotated with whether joint limits clipped them."""
  algorithm = resolve_sew_algorithm_config(config)
  candidates_by_key: dict[float, BoundedAngleCandidate] = {}
  order: list[float] = []

  def add_candidate(value: float, *, clipped: bool) -> None:
    key = round(float(value), 12)
    candidate = BoundedAngleCandidate(angle=float(value), clipped=clipped)
    existing = candidates_by_key.get(key)
    if existing is None:
      order.append(key)
      candidates_by_key[key] = candidate
    elif existing.clipped and not clipped:
      candidates_by_key[key] = candidate

  if not algorithm.respect_joint_limits:
    for offset in offsets:
      add_candidate(float(angle + 2.0 * np.pi * offset), clipped=False)
    return [candidates_by_key[key] for key in order]

  low, high = joint_limits[joint_index]
  for offset in offsets:
    candidate = float(angle + 2.0 * np.pi * offset)
    if low - 1e-9 <= candidate <= high + 1e-9:
      add_candidate(float(np.clip(candidate, low, high)), clipped=False)
    elif (
      algorithm.clip_limit_candidates
      and low - algorithm.limit_clip_margin <= candidate <= high + algorithm.limit_clip_margin
    ):
      add_candidate(float(np.clip(candidate, low, high)), clipped=True)
    else:
      continue
  return [candidates_by_key[key] for key in order]


def rotation_vector_from_matrix(rotation: np.ndarray, eps: float = 1e-12) -> np.ndarray:
  """Return the SO(3) logarithm as a rotation vector."""
  matrix = np.asarray(rotation, dtype=float).reshape(3, 3)
  cosine = np.clip((np.trace(matrix) - 1.0) * 0.5, -1.0, 1.0)
  angle = float(np.arccos(cosine))
  if angle <= eps:
    return np.zeros(3)
  if abs(np.pi - angle) <= 1e-6:
    axis = np.sqrt(np.maximum(np.diag(matrix) + 1.0, 0.0) * 0.5)
    if axis[0] >= axis[1] and axis[0] >= axis[2] and axis[0] > eps:
      axis[1] = matrix[0, 1] / (2.0 * axis[0])
      axis[2] = matrix[0, 2] / (2.0 * axis[0])
    elif axis[1] >= axis[2] and axis[1] > eps:
      axis[0] = matrix[0, 1] / (2.0 * axis[1])
      axis[2] = matrix[1, 2] / (2.0 * axis[1])
    elif axis[2] > eps:
      axis[0] = matrix[0, 2] / (2.0 * axis[2])
      axis[1] = matrix[1, 2] / (2.0 * axis[2])
    return normalize(axis) * angle
  axis = np.array(
    [
      matrix[2, 1] - matrix[1, 2],
      matrix[0, 2] - matrix[2, 0],
      matrix[1, 0] - matrix[0, 1],
    ],
    dtype=float,
  ) / (2.0 * np.sin(angle))
  return axis * angle


def matrix_orientation_error(lhs: np.ndarray, rhs: np.ndarray) -> float:
  """Chordal-like matrix orientation error used as a scalar diagnostic."""
  delta = np.asarray(lhs, dtype=float).reshape(3, 3).T @ np.asarray(rhs, dtype=float).reshape(3, 3)
  return float(np.linalg.norm(rotation_vector_from_matrix(delta)))


def subproblem1(p1: ArrayLike3, p2: ArrayLike3, k: ArrayLike3) -> float:
  """Paper Subproblem 1: rotate p1 about k to align with p2."""
  axis = normalize(k)
  p1_perp = normalize(np.asarray(p1, dtype=float) - np.dot(p1, axis) * axis)
  p2_perp = normalize(np.asarray(p2, dtype=float) - np.dot(p2, axis) * axis)
  theta = 2.0 * np.arctan2(
    np.linalg.norm(p1_perp - p2_perp),
    np.linalg.norm(p1_perp + p2_perp),
  )
  if float(axis @ np.cross(p1_perp, p2_perp)) < 0.0:
    theta = -theta
  return float(theta)


def subproblem4(p: ArrayLike3, h: ArrayLike3, k: ArrayLike3, d: float) -> list[float]:
  """Paper Subproblem 4: rotate p about k until h dot R(k, theta) p = d."""
  point = np.asarray(p, dtype=float)
  if np.linalg.norm(point) <= 1e-12:
    raise ValueError("Subproblem 4 requires a non-zero point vector")
  plane_normal = np.asarray(h, dtype=float)
  if np.linalg.norm(plane_normal) <= 1e-12:
    raise ValueError("Subproblem 4 requires a non-zero plane normal")
  axis = normalize(k)
  axis_hat = skew(axis)
  basis = np.column_stack((axis_hat @ point, -(axis_hat @ axis_hat) @ point))
  a = plane_normal @ basis
  b = float(d - (plane_normal @ axis) * (axis @ point))
  norm_sq = float(a @ a)
  if norm_sq <= 1e-12:
    return [0.0]

  least_norm = a * (b / norm_sq)
  if norm_sq > b * b:
    null_direction = np.array([a[1], -a[0]], dtype=float) / np.sqrt(norm_sq)
    offset = np.sqrt(max(0.0, 1.0 - float(least_norm @ least_norm)))
    x_plus = least_norm + offset * null_direction
    x_minus = least_norm - offset * null_direction
    return [
      float(np.arctan2(x_plus[0], x_plus[1])),
      float(np.arctan2(x_minus[0], x_minus[1])),
    ]
  return [float(np.arctan2(least_norm[0], least_norm[1]))]


def subproblem2(
  p1: ArrayLike3,
  p2: ArrayLike3,
  k1: ArrayLike3,
  k2: ArrayLike3,
) -> list[tuple[float, float]]:
  """Paper Subproblem 2 candidate pairs built from Subproblem 4."""
  p1_unit = normalize(p1)
  p2_unit = normalize(p2)
  k1_unit = normalize(k1)
  k2_unit = normalize(k2)

  theta1 = subproblem4(k2_unit, p1_unit, k1_unit, float(k2_unit @ p2_unit))
  theta2 = subproblem4(k1_unit, p2_unit, k2_unit, float(k1_unit @ p1_unit))
  if len(theta1) == 1 or len(theta2) == 1:
    return [(theta1[0], theta2[0])]
  return [(theta1[0], theta2[1]), (theta1[1], theta2[0])]


def solve_two_axis_rotation(
  initial_axis: ArrayLike3,
  target_axis: ArrayLike3,
  first_axis: ArrayLike3,
  second_axis: ArrayLike3,
) -> list[tuple[float, float]]:
  """Solve R(k1, q1) R(k2, q2) p = target using paper Subproblem 2."""
  return subproblem2(
    normalize(target_axis),
    normalize(initial_axis),
    normalize(first_axis),
    -normalize(second_axis),
  )


def solve_g1_perpendicular_wrist_angles(desired_relative_orientation: np.ndarray) -> list[np.ndarray]:
  """Return G1 wrist XYZ Euler candidates for the paper's perpendicular-wrist case.

  G1's final three wrist joints are X/Y/Z hinges. The paper appendix states
  perpendicular wrists should be solved by Euler decomposition instead of the
  Subproblem 1/2 parallel-wrist routine.
  """
  euler = Rotation.from_matrix(np.asarray(desired_relative_orientation, dtype=float).reshape(3, 3)).as_euler("XYZ")
  equivalent = np.array([euler[0] + np.pi, np.pi - euler[1], euler[2] + np.pi], dtype=float)
  return [np.asarray(euler, dtype=float), equivalent]


@dataclass(frozen=True)
class ArmKeypointTarget:
  """SEW-Mimic input target for one arm.

  The shoulder/elbow/wrist fields carry one of two semantically distinct
  payloads depending on the producer:

  * Human keypoint target: real shoulder/elbow/wrist positions extracted from
    BVH or MediaPipe. ``elbow - shoulder`` matches the human's upper-arm
    direction. This is the paper-defined input.
  * G1 axis proxy target: synthesized via
    ``synthesize_g1_axis_proxy_arm_target`` (BVH/webcam) or
    ``target_from_configuration`` (G1 FK). ``elbow - shoulder`` matches the
    G1 third joint axis direction, not the human upper-arm direction. This
    is an engineering adapter that lets the closed-form solver hit zero
    residual against a reachable robot pose; it is NOT the paper input.

  Consumers of this dataclass must decide which semantics they expect. The
  closed-form solver in ``G1SEWMimicRetargeter`` treats it as a direction
  target only and does not distinguish between the two.
  """

  shoulder: np.ndarray
  elbow: np.ndarray
  wrist: np.ndarray
  hand_orientation: np.ndarray


@dataclass(frozen=True)
class SEWMimicResult:
  joint_angles: np.ndarray
  success: bool
  upper_arm_error: float
  lower_arm_error: float
  wrist_error: float
  message: str


class G1SEWMimicRetargeter:
  """SEW-Mimic retargeting adapter for a Unitree G1 7-DOF arm.

  The public input is deliberately paper-shaped: shoulder, elbow, wrist,
  and hand orientation. A later bones-seed or mocap parser can produce
  `ArmKeypointTarget` values and feed this adapter without depending on MuJoCo.
  """

  def __init__(
    self,
    side: str = "left",
    xml_path: str | Path | None = None,
    algorithm_version: str | SEWAlgorithmConfig = BRANCH_V2_ALGORITHM,
  ):
    if side not in {"left", "right"}:
      raise ValueError("side must be 'left' or 'right'")
    self.side = side
    self.algorithm_config = resolve_sew_algorithm_config(algorithm_version)
    self.xml_path = Path(xml_path) if xml_path is not None else (
      SRC_PATH / "assets" / "robots" / "unitree_g1" / "xmls" / "g1.xml"
    )
    self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))
    self.data = mujoco.MjData(self.model)
    self.joint_names = tuple(
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
    self.joint_ids = tuple(self._named_id(mujoco.mjtObj.mjOBJ_JOINT, name) for name in self.joint_names)
    self.joint_qpos_addresses = np.array(
      [self.model.jnt_qposadr[joint_id] for joint_id in self.joint_ids],
      dtype=int,
    )
    self.joint_limits = np.array(
      [self.model.jnt_range[joint_id] for joint_id in self.joint_ids],
      dtype=float,
    )
    self._axis_joint_ids = {
      "upper": self.joint_ids[2],
      "lower": self.joint_ids[4],
    }
    self._keypoint_joint_ids = {
      "shoulder": self.joint_ids[0],
      "elbow": self.joint_ids[3],
      "wrist": self.joint_ids[5],
    }
    self._palm_site_id = self._named_id(mujoco.mjtObj.mjOBJ_SITE, f"{side}_palm")
    self._qpos0 = self.model.qpos0.copy()

  def target_from_configuration(self, joint_angles: Sequence[float]) -> ArmKeypointTarget:
    """Return paper-shaped SEW targets produced by a reachable G1 arm pose.

    SEW-Mimic aligns human upper/lower arm vectors to the robot's 3rd and 5th
    joint-axis proxies. G1's physical shoulder-to-elbow displacement is not the
    same direction as the 3rd joint axis, so this helper synthesizes keypoints
    from the proxy axes rather than returning raw robot joint-anchor positions.
    """
    self._set_arm_joint_angles(np.asarray(joint_angles, dtype=float))
    shoulder = self.data.xanchor[self._keypoint_joint_ids["shoulder"]].copy()
    raw_elbow = self.data.xanchor[self._keypoint_joint_ids["elbow"]].copy()
    raw_wrist = self.data.xanchor[self._keypoint_joint_ids["wrist"]].copy()
    upper_length = float(np.linalg.norm(raw_elbow - shoulder))
    lower_length = float(np.linalg.norm(raw_wrist - raw_elbow))
    upper_axis = normalize(self.data.xaxis[self._axis_joint_ids["upper"]])
    lower_axis = normalize(self.data.xaxis[self._axis_joint_ids["lower"]])
    elbow = shoulder + upper_length * upper_axis
    wrist = elbow + lower_length * lower_axis
    return ArmKeypointTarget(
      shoulder=shoulder,
      elbow=elbow,
      wrist=wrist,
      hand_orientation=self.data.site_xmat[self._palm_site_id].reshape(3, 3).copy(),
    )

  def retarget(
    self,
    q_init: Sequence[float],
    shoulder: ArrayLike3,
    elbow: ArrayLike3,
    wrist: ArrayLike3,
    hand_orientation: np.ndarray,
  ) -> SEWMimicResult:
    """Retarget one G1 arm from SEW keypoints and desired hand orientation.

    Note on scope: this solver matches the paper for the two axis groups
    (Subproblem 2 closed-form selection between upper/lower arm direction
    candidates) and uses the paper appendix's G1 perpendicular-wrist Euler
    decomposition for the final three wrist joints.

    The ``shoulder``/``elbow``/``wrist`` inputs are interpreted as a
    direction pair: only ``elbow - shoulder`` and ``wrist - elbow``
    matter. Pass either human keypoints or a G1 axis proxy target depending
    on whether you want paper semantics or zero-residual reachability (see
    ``ArmKeypointTarget``).
    """
    q = self._clip_joint_angles(np.asarray(q_init, dtype=float))
    if q.shape != (7,):
      raise ValueError(f"q_init must have shape (7,), got {q.shape}")

    upper_arm = normalize(np.asarray(elbow, dtype=float) - np.asarray(shoulder, dtype=float))
    lower_arm = normalize(np.asarray(wrist, dtype=float) - np.asarray(elbow, dtype=float))
    desired_hand_orientation = np.asarray(hand_orientation, dtype=float).reshape(3, 3)

    q = self._solve_axis_group(q, slice(0, 2), self._axis_joint_ids["upper"], upper_arm)
    q = self._solve_axis_group(q, slice(2, 4), self._axis_joint_ids["lower"], lower_arm)
    q = self._solve_wrist_group(q, slice(4, 7), desired_hand_orientation)
    diagnostics = self._diagnostics(q, upper_arm, lower_arm, desired_hand_orientation)
    success = (
      diagnostics["upper_arm_error"] <= 1e-3
      and diagnostics["lower_arm_error"] <= 1e-3
      and diagnostics["wrist_error"] <= 1e-3
    )
    return SEWMimicResult(
      joint_angles=q,
      success=success,
      upper_arm_error=diagnostics["upper_arm_error"],
      lower_arm_error=diagnostics["lower_arm_error"],
      wrist_error=diagnostics["wrist_error"],
      message="converged" if success else "retargeting residual above tolerance",
    )

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
    self._set_arm_joint_angles(base)
    first_axis = normalize(self.data.xaxis[self.joint_ids[indexes[0]]])
    second_axis = normalize(self.data.xaxis[self.joint_ids[indexes[1]]])
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
        self._set_arm_joint_angles(candidate_q)
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

  def _bounded_equivalent_angle_candidates(self, angle: float, joint_index: int) -> list[BoundedAngleCandidate]:
    return bounded_equivalent_angle_candidates(angle, self.joint_limits, joint_index, config=self.algorithm_config)

  def _bounded_equivalent_angles(self, angle: float, joint_index: int) -> list[float]:
    return bounded_equivalent_angles(angle, self.joint_limits, joint_index, config=self.algorithm_config)

  def _solve_wrist_group(
    self,
    q: np.ndarray,
    group: slice,
    desired_hand_orientation: np.ndarray,
  ) -> np.ndarray:
    indexes = np.arange(group.start or 0, group.stop or len(q))
    desired = np.asarray(desired_hand_orientation, dtype=float).reshape(3, 3)
    wrist_zero = q.copy()
    wrist_zero[indexes] = 0.0
    self._set_arm_joint_angles(wrist_zero)
    zero_orientation = self.data.site_xmat[self._palm_site_id].reshape(3, 3).copy()
    relative_desired = zero_orientation.T @ desired

    scored_candidates: list[tuple[np.ndarray, tuple[float, float], float, bool]] = []
    for euler_values in solve_g1_perpendicular_wrist_angles(relative_desired):
      for values, clipped in self._bounded_angle_triplet_candidates(indexes, euler_values):
        candidate_q = q.copy()
        candidate_q[indexes] = values
        self._set_arm_joint_angles(candidate_q)
        current = self.data.site_xmat[self._palm_site_id].reshape(3, 3)
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

  def _diagnostics(
    self,
    q: np.ndarray,
    upper_arm: np.ndarray,
    lower_arm: np.ndarray,
    desired_hand_orientation: np.ndarray,
  ) -> dict[str, float]:
    self._set_arm_joint_angles(q)
    current_upper = normalize(self.data.xaxis[self._axis_joint_ids["upper"]])
    current_lower = normalize(self.data.xaxis[self._axis_joint_ids["lower"]])
    current_hand_orientation = self.data.site_xmat[self._palm_site_id].reshape(3, 3)
    return {
      "upper_arm_error": orientation_error(current_upper, upper_arm),
      "lower_arm_error": orientation_error(current_lower, lower_arm),
      "wrist_error": matrix_orientation_error(current_hand_orientation, desired_hand_orientation),
    }

  def _set_arm_joint_angles(self, joint_angles: np.ndarray) -> None:
    self.data.qpos[:] = self._qpos0
    self.data.qpos[self.joint_qpos_addresses] = joint_angles
    mujoco.mj_forward(self.model, self.data)

  def _clip_joint_angles(self, joint_angles: np.ndarray) -> np.ndarray:
    q = np.asarray(joint_angles, dtype=float)
    if not self.algorithm_config.respect_joint_limits:
      return q.copy()
    return np.clip(q, self.joint_limits[:, 0], self.joint_limits[:, 1])

  def _named_id(self, obj_type: mujoco.mjtObj, name: str) -> int:
    for candidate in (f"robot/{name}", name):
      obj_id = mujoco.mj_name2id(self.model, obj_type, candidate)
      if obj_id >= 0:
        return int(obj_id)
    raise ValueError(f"Could not find {obj_type.name} named '{name}'")


def retarget_motion_rows_from_targets(
  rows: Sequence[Sequence[float]],
  *,
  side: str,
  targets: Sequence[ArmKeypointTarget],
  q_init: Sequence[float] | None = None,
  retargeter: G1SEWMimicRetargeter | None = None,
) -> list[list[float]]:
  """Return G1 29-DOF motion rows with one arm replaced by SEW-Mimic output.

  This is the handoff point for a future bones-seed/mocap parser: that parser
  should convert each mocap frame to `ArmKeypointTarget`, then call this helper.
  """
  if len(rows) != len(targets):
    raise ValueError(f"rows and targets must have same length, got {len(rows)} and {len(targets)}")
  adapter = retargeter or G1SEWMimicRetargeter(side=side)
  initial = np.zeros(7) if q_init is None else np.asarray(q_init, dtype=float)
  arm_column_names = [f"{name}_dof" for name in adapter.joint_names]
  arm_indices = [7 + G1_29DOF_JOINT_COLUMNS.index(column) for column in arm_column_names]

  output: list[list[float]] = []
  q_previous = initial
  for row, target in zip(rows, targets):
    result = adapter.retarget(
      q_previous,
      shoulder=target.shoulder,
      elbow=target.elbow,
      wrist=target.wrist,
      hand_orientation=target.hand_orientation,
    )
    updated = list(float(value) for value in row)
    for row_index, joint_angle in zip(arm_indices, result.joint_angles):
      updated[row_index] = float(joint_angle)
    output.append(updated)
    q_previous = result.joint_angles
  return output


def targets_from_iterable(frames: Iterable[dict[str, np.ndarray]]) -> list[ArmKeypointTarget]:
  """Convert generic mocap frame dictionaries to SEW targets.

  Expected keys per frame are `shoulder`, `elbow`, `wrist`, and
  `hand_orientation`. This intentionally avoids assuming a bones-seed CSV
  schema until the concrete mocap export format is available.
  """
  targets: list[ArmKeypointTarget] = []
  for frame in frames:
    targets.append(
      ArmKeypointTarget(
        shoulder=np.asarray(frame["shoulder"], dtype=float),
        elbow=np.asarray(frame["elbow"], dtype=float),
        wrist=np.asarray(frame["wrist"], dtype=float),
        hand_orientation=np.asarray(frame["hand_orientation"], dtype=float).reshape(3, 3),
      )
    )
  return targets


def math_sin(theta: float) -> float:
  return float(np.sin(theta))


def math_cos(theta: float) -> float:
  return float(np.cos(theta))
