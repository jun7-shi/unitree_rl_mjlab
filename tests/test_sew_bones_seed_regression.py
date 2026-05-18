from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pytest

from scripts.compare_sew_with_seed_csv import (
  G1_UPPER_BODY_JOINT_COLUMNS,
  load_seed_upper_body_joint_angles,
)
from src.motion.bvh_upper_body import load_soma_bvh_upper_body_targets
from src.motion.sew_mimic import PAPER_V1_ALGORITHM, matrix_orientation_error, orientation_error, normalize
from src.motion.sew_upper_body import G1UpperBodySEWRetargeter, retarget_upper_body_targets

DATA_ROOT = Path("/data/jun7.shi/datasets/bones-seed")


@dataclass(frozen=True)
class RegressionThresholds:
  expected_success_count: int
  max_target_error: float
  max_joint_diff_deg: float
  max_frame_delta_deg: float
  expected_joint_limit_violations: int = 0


@dataclass(frozen=True)
class RegressionFixture:
  name: str
  bvh_path: Path
  csv_path: Path
  start_frame: int
  max_frames: int
  jitter_joint_names: tuple[str, ...]
  thresholds: RegressionThresholds
  algorithm_version: str = PAPER_V1_ALGORITHM.name
  apply_orientation_offsets: bool = True
  align_upper_arm_axes_to_g1: bool = True
  apply_lower_body_offsets: bool = True
  remove_initial_heading: bool = True
  localize_to_body_frame: bool = True


@dataclass(frozen=True)
class RegressionMetrics:
  frame_count: int
  success_count: int
  max_target_error: float
  mean_abs_joint_diff_deg: float
  max_abs_joint_diff_deg: float
  max_frame_delta_deg: float
  finite: bool
  joint_limit_violations: int


@dataclass(frozen=True)
class TargetAgreementMetrics:
  frame_count: int
  max_chest_error: float
  max_upper_arm_error: float
  max_lower_arm_error: float
  max_wrist_error: float


def _frame_slice(fixture: RegressionFixture) -> slice:
  return slice(fixture.start_frame, fixture.start_frame + fixture.max_frames)


def _skip_if_missing(fixture: RegressionFixture) -> None:
  missing = [path for path in (fixture.bvh_path, fixture.csv_path) if not path.exists()]
  if missing:
    formatted = ", ".join(str(path) for path in missing)
    pytest.skip(f"local bones-seed fixture is unavailable for {fixture.name}: {formatted}")


def _max_error(results: Sequence[object]) -> float:
  if not results:
    return 0.0
  return max(
    float(value)
    for result in results
    for value in result.errors.values()
  )


def _joint_limit_violation_count(q: np.ndarray, limits: np.ndarray, tolerance: float = 1e-8) -> int:
  below = q < (limits[:, 0] - tolerance)
  above = q > (limits[:, 1] + tolerance)
  return int(np.count_nonzero(below | above))


def _joint_column_indexes(joint_names: Sequence[str], all_joint_names: Sequence[str]) -> list[int]:
  indexes: list[int] = []
  for name in joint_names:
    candidates = (name, f"{name}_dof", name.removesuffix("_dof"))
    for candidate in candidates:
      if candidate in all_joint_names:
        indexes.append(all_joint_names.index(candidate))
        break
    else:
      raise ValueError(f"Unknown joint name {name!r}; available joints: {', '.join(all_joint_names)}")
  return indexes


def _max_named_frame_delta_deg(q: np.ndarray, joint_names: Sequence[str], all_joint_names: Sequence[str]) -> float:
  if len(q) <= 1 or not joint_names:
    return 0.0
  indexes = _joint_column_indexes(joint_names, all_joint_names)
  deltas = np.abs(np.diff(np.rad2deg(q[:, indexes]), axis=0))
  return float(np.max(deltas))


def test_joint_limit_violation_count_flags_out_of_range_values():
  q = np.array([
    [0.0, 0.0],
    [1.2, -1.3],
  ])
  limits = np.array([
    [-1.0, 1.0],
    [-1.0, 1.0],
  ])

  assert _joint_limit_violation_count(q, limits) == 2


def _arm_target_errors(seed_arm, bvh_arm) -> tuple[float, float, float]:
  seed_upper = normalize(seed_arm.elbow - seed_arm.shoulder)
  bvh_upper = normalize(bvh_arm.elbow - bvh_arm.shoulder)
  seed_lower = normalize(seed_arm.wrist - seed_arm.elbow)
  bvh_lower = normalize(bvh_arm.wrist - bvh_arm.elbow)
  return (
    orientation_error(seed_upper, bvh_upper),
    orientation_error(seed_lower, bvh_lower),
    matrix_orientation_error(seed_arm.hand_orientation, bvh_arm.hand_orientation),
  )


def _run_target_agreement_fixture(fixture: RegressionFixture) -> TargetAgreementMetrics:
  _skip_if_missing(fixture)
  frame_slice = _frame_slice(fixture)
  seed_q = load_seed_upper_body_joint_angles(fixture.csv_path)[frame_slice]
  bvh_targets = load_soma_bvh_upper_body_targets(
    fixture.bvh_path,
    frame_slice=frame_slice,
    apply_orientation_offsets=fixture.apply_orientation_offsets,
    align_upper_arm_axes_to_g1=fixture.align_upper_arm_axes_to_g1,
    remove_initial_heading=fixture.remove_initial_heading,
  )
  assert len(bvh_targets) == len(seed_q), f"{fixture.name}: target/CSV frame mismatch"

  retargeter = G1UpperBodySEWRetargeter(algorithm_version=fixture.algorithm_version)
  chest_errors: list[float] = []
  upper_errors: list[float] = []
  lower_errors: list[float] = []
  wrist_errors: list[float] = []
  for q, bvh_target in zip(seed_q, bvh_targets):
    seed_target = retargeter.target_from_configuration(q)
    chest_errors.append(matrix_orientation_error(seed_target.chest_orientation, bvh_target.chest_orientation))
    for side in ("left", "right"):
      upper_error, lower_error, wrist_error = _arm_target_errors(
        getattr(seed_target, f"{side}_arm"),
        getattr(bvh_target, f"{side}_arm"),
      )
      upper_errors.append(upper_error)
      lower_errors.append(lower_error)
      wrist_errors.append(wrist_error)
  return TargetAgreementMetrics(
    frame_count=len(bvh_targets),
    max_chest_error=float(np.max(chest_errors)),
    max_upper_arm_error=float(np.max(upper_errors)),
    max_lower_arm_error=float(np.max(lower_errors)),
    max_wrist_error=float(np.max(wrist_errors)),
  )


UPPER_BODY_FIXTURES = (
  RegressionFixture(
    name="a464_left_shoulder_branch_upper",
    bvh_path=DATA_ROOT / "soma_uniform/bvh/231006/dance_blinding_lights_004__A464.bvh",
    csv_path=DATA_ROOT / "g1/csv/231006/dance_blinding_lights_004__A464.csv",
    start_frame=425,
    max_frames=101,
    jitter_joint_names=(
      "left_shoulder_pitch_joint",
      "left_shoulder_roll_joint",
      "left_shoulder_yaw_joint",
      "left_wrist_roll_joint",
      "left_wrist_yaw_joint",
    ),
    thresholds=RegressionThresholds(
      expected_success_count=101,
      max_target_error=1e-3,
      max_joint_diff_deg=380.0,
      max_frame_delta_deg=45.0,
      expected_joint_limit_violations=203,
    ),
  ),
)


def _run_upper_body_fixture(fixture: RegressionFixture) -> RegressionMetrics:
  _skip_if_missing(fixture)
  frame_slice = _frame_slice(fixture)
  seed_q = load_seed_upper_body_joint_angles(fixture.csv_path)[frame_slice]
  targets = load_soma_bvh_upper_body_targets(
    fixture.bvh_path,
    frame_slice=frame_slice,
    apply_orientation_offsets=fixture.apply_orientation_offsets,
    align_upper_arm_axes_to_g1=fixture.align_upper_arm_axes_to_g1,
    remove_initial_heading=fixture.remove_initial_heading,
  )
  assert len(targets) == len(seed_q), f"{fixture.name}: target/CSV frame mismatch"

  retargeter = G1UpperBodySEWRetargeter(algorithm_version=fixture.algorithm_version)
  results = retarget_upper_body_targets(targets, retargeter=retargeter)
  q = np.asarray([result.joint_angles for result in results], dtype=float)
  diff_deg = np.abs(np.rad2deg(q - seed_q))
  return RegressionMetrics(
    frame_count=len(results),
    success_count=sum(1 for result in results if result.success),
    max_target_error=_max_error(results),
    mean_abs_joint_diff_deg=float(np.mean(diff_deg)),
    max_abs_joint_diff_deg=float(np.max(diff_deg)),
    max_frame_delta_deg=_max_named_frame_delta_deg(
      q,
      fixture.jitter_joint_names,
      G1_UPPER_BODY_JOINT_COLUMNS,
    ),
    finite=bool(np.all(np.isfinite(q)) and np.all(np.isfinite(diff_deg))),
    joint_limit_violations=_joint_limit_violation_count(q, retargeter.joint_limits),
  )


@pytest.mark.parametrize("fixture", UPPER_BODY_FIXTURES, ids=lambda fixture: fixture.name)
def test_g1_csv_fk_target_matches_bvh_target_definition(fixture: RegressionFixture):
  metrics = _run_target_agreement_fixture(fixture)

  assert metrics.frame_count == fixture.max_frames
  assert metrics.max_chest_error <= 0.55, f"{fixture.name}: chest target mismatch {metrics}"
  assert metrics.max_upper_arm_error <= 0.13, f"{fixture.name}: upper-arm target mismatch {metrics}"
  assert metrics.max_lower_arm_error <= 0.08, f"{fixture.name}: lower-arm target mismatch {metrics}"
  assert metrics.max_wrist_error <= 0.45, f"{fixture.name}: wrist target mismatch {metrics}"


@pytest.mark.parametrize("fixture", UPPER_BODY_FIXTURES, ids=lambda fixture: fixture.name)
def test_upper_body_bones_seed_regression(fixture: RegressionFixture):
  metrics = _run_upper_body_fixture(fixture)

  assert metrics.frame_count == fixture.max_frames
  assert metrics.finite, f"{fixture.name}: non-finite metrics {metrics}"
  assert metrics.joint_limit_violations == fixture.thresholds.expected_joint_limit_violations, f"{fixture.name}: joint limit violations changed {metrics}"
  assert metrics.success_count == fixture.thresholds.expected_success_count, f"{fixture.name}: success count changed {metrics}"
  assert metrics.max_target_error <= fixture.thresholds.max_target_error, f"{fixture.name}: target error too high {metrics}"
  assert metrics.max_abs_joint_diff_deg <= fixture.thresholds.max_joint_diff_deg, f"{fixture.name}: CSV diff too high {metrics}"
  assert metrics.max_frame_delta_deg <= fixture.thresholds.max_frame_delta_deg, f"{fixture.name}: branch jitter too high {metrics}"
