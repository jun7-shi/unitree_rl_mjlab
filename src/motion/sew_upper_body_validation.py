from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from src.motion.sew_upper_body import G1UpperBodySEWRetargeter


@dataclass(frozen=True)
class FKValidationCase:
  """A reachable G1 upper-body pose used to validate FK-to-retarget round trips."""

  name: str
  joint_angles: np.ndarray


@dataclass(frozen=True)
class FKValidationCaseResult:
  name: str
  success: bool
  errors: dict[str, float]


@dataclass(frozen=True)
class FKValidationReport:
  case_count: int
  success_count: int
  max_errors: dict[str, float]
  case_results: list[FKValidationCaseResult]

  @property
  def all_passed(self) -> bool:
    return self.success_count == self.case_count


_NAMED_VALIDATION_POSES: tuple[tuple[str, tuple[float, ...]], ...] = (
  (
    "neutral",
    (
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
    ),
  ),
  (
    "small_bilateral_reach",
    (
      0.05,
      -0.04,
      0.03,
      0.10,
      0.20,
      -0.20,
      0.50,
      0.10,
      -0.10,
      0.20,
      -0.10,
      -0.20,
      0.20,
      0.50,
      -0.10,
      0.10,
      -0.20,
    ),
  ),
  (
    "asymmetric_upper_body",
    (
      0.12,
      0.08,
      -0.10,
      0.20,
      0.35,
      -0.45,
      0.90,
      0.25,
      -0.35,
      0.40,
      -0.15,
      -0.30,
      0.35,
      0.70,
      -0.20,
      0.25,
      -0.30,
    ),
  ),
  (
    "crossed_waist_offset",
    (
      -0.10,
      -0.06,
      0.08,
      -0.25,
      0.15,
      0.30,
      0.60,
      -0.20,
      0.25,
      -0.30,
      0.25,
      -0.10,
      -0.25,
      0.80,
      0.15,
      -0.20,
      0.35,
    ),
  ),
  (
    "raised_elbows",
    (
      0.00,
      0.10,
      0.12,
      0.35,
      -0.20,
      0.25,
      1.00,
      0.35,
      0.20,
      -0.25,
      -0.35,
      0.15,
      -0.30,
      0.95,
      -0.30,
      -0.15,
      0.25,
    ),
  ),
)

_RANDOM_SAMPLE_RANGES = np.array(
  [
    [-0.25, 0.25],
    [-0.18, 0.18],
    [-0.18, 0.18],
    [-0.45, 0.45],
    [-0.35, 0.35],
    [-0.55, 0.55],
    [0.35, 1.05],
    [-0.45, 0.45],
    [-0.45, 0.45],
    [-0.45, 0.45],
    [-0.45, 0.45],
    [-0.35, 0.35],
    [-0.55, 0.55],
    [0.35, 1.05],
    [-0.45, 0.45],
    [-0.45, 0.45],
    [-0.45, 0.45],
  ],
  dtype=float,
)


def sample_fk_validation_cases(
  retargeter: G1UpperBodySEWRetargeter,
  *,
  count: int,
  seed: int = 0,
) -> list[FKValidationCase]:
  """Build deterministic reachable validation poses from G1 joint space.

  The first samples are named poses that exercise waist, shoulder, elbow, and
  wrist coupling. Additional samples are conservative random joint-space poses
  away from joint limits, so the validation set can be expanded without making
  solver failures indistinguishable from unreachable target data.
  """
  if count < 0:
    raise ValueError(f"count must be non-negative, got {count}")

  cases: list[FKValidationCase] = []
  for name, joint_angles in _NAMED_VALIDATION_POSES[:count]:
    cases.append(
      FKValidationCase(
        name=name,
        joint_angles=_inside_joint_limits(np.asarray(joint_angles, dtype=float), retargeter),
      )
    )

  rng = np.random.default_rng(seed)
  while len(cases) < count:
    sample_index = len(cases) - len(_NAMED_VALIDATION_POSES)
    joint_angles = rng.uniform(_RANDOM_SAMPLE_RANGES[:, 0], _RANDOM_SAMPLE_RANGES[:, 1])
    cases.append(
      FKValidationCase(
        name=f"sample_{sample_index:03d}",
        joint_angles=_inside_joint_limits(joint_angles, retargeter),
      )
    )
  return cases


def validate_upper_body_retargeter_on_fk_cases(
  cases: Sequence[FKValidationCase],
  *,
  retargeter: G1UpperBodySEWRetargeter | None = None,
  tolerance: float = 1e-3,
) -> FKValidationReport:
  """Retarget robot-FK targets and report geometric round-trip residuals."""
  if tolerance <= 0.0:
    raise ValueError(f"tolerance must be positive, got {tolerance}")

  adapter = retargeter or G1UpperBodySEWRetargeter()
  case_results: list[FKValidationCaseResult] = []
  max_errors: dict[str, float] = {}

  for case in cases:
    target = adapter.target_from_configuration(case.joint_angles)
    result = adapter.retarget(np.zeros(17), target)
    errors = {key: float(value) for key, value in result.errors.items()}
    success = all(value <= tolerance for value in errors.values())
    case_results.append(FKValidationCaseResult(name=case.name, success=success, errors=errors))
    for key, value in errors.items():
      max_errors[key] = max(max_errors.get(key, 0.0), value)

  return FKValidationReport(
    case_count=len(case_results),
    success_count=sum(1 for result in case_results if result.success),
    max_errors=dict(sorted(max_errors.items())),
    case_results=case_results,
  )


def _inside_joint_limits(
  joint_angles: np.ndarray,
  retargeter: G1UpperBodySEWRetargeter,
) -> np.ndarray:
  lower = retargeter.joint_limits[:, 0]
  upper = retargeter.joint_limits[:, 1]
  return np.clip(np.asarray(joint_angles, dtype=float), lower, upper)
