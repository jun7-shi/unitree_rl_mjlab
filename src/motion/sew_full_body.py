from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from src.motion.sew_mimic import BRANCH_V2_ALGORITHM, SEWAlgorithmConfig
from src.motion.sew_lower_body import (
  G1LowerBodySEWRetargeter,
  LowerBodyRetargetResult,
  LowerBodyTarget,
)
from src.motion.sew_upper_body import (
  G1UpperBodySEWRetargeter,
  UpperBodyRetargetResult,
  UpperBodyTarget,
)


@dataclass(frozen=True)
class FullBodyTarget:
  lower: LowerBodyTarget
  upper: UpperBodyTarget


@dataclass(frozen=True)
class FullBodyRetargetResult:
  joint_angles: np.ndarray
  full_qpos: np.ndarray
  success: bool
  errors: dict[str, float]
  message: str
  solver_joint_angles: np.ndarray | None = None


class G1FullBodySEWRetargeter:
  """Combine G1 lower-body and upper-body SEW retargeters into a 29-DOF pose."""

  def __init__(
    self,
    xml_path: str | Path | None = None,
    algorithm_version: str | SEWAlgorithmConfig = BRANCH_V2_ALGORITHM,
  ):
    self.lower = G1LowerBodySEWRetargeter(xml_path=xml_path, algorithm_version=algorithm_version)
    self.upper = G1UpperBodySEWRetargeter(xml_path=xml_path, algorithm_version=algorithm_version)
    self.model = self.upper.model
    self.controlled_joint_names = (
      *self.lower.controlled_joint_names,
      *self.upper.controlled_joint_names,
    )
    self.controlled_qpos_addresses = np.concatenate(
      [self.lower.controlled_qpos_addresses, self.upper.controlled_qpos_addresses]
    )

  def target_from_configuration(self, joint_angles: Sequence[float]) -> FullBodyTarget:
    q = np.asarray(joint_angles, dtype=float)
    if q.shape != (29,):
      raise ValueError(f"joint_angles must have shape (29,), got {q.shape}")
    return FullBodyTarget(
      lower=self.lower.target_from_configuration(q[:12]),
      upper=self.upper.target_from_configuration(q[12:]),
    )

  def retarget(
    self,
    q_init: Sequence[float],
    target: FullBodyTarget,
  ) -> FullBodyRetargetResult:
    q = np.asarray(q_init, dtype=float)
    if q.shape != (29,):
      raise ValueError(f"q_init must have shape (29,), got {q.shape}")
    lower_result = self.lower.retarget(q[:12], target.lower)
    upper_result = self.upper.retarget(q[12:], target.upper)
    return self._combine_results(lower_result, upper_result)

  def _combine_results(
    self,
    lower_result: LowerBodyRetargetResult,
    upper_result: UpperBodyRetargetResult,
  ) -> FullBodyRetargetResult:
    full_qpos = upper_result.full_qpos.copy()
    full_qpos[self.lower.controlled_qpos_addresses] = lower_result.joint_angles
    errors = {**lower_result.errors, **upper_result.errors}
    success = lower_result.success and upper_result.success
    upper_solver_q = (
      upper_result.solver_joint_angles
      if upper_result.solver_joint_angles is not None
      else upper_result.joint_angles
    )
    return FullBodyRetargetResult(
      joint_angles=np.concatenate([lower_result.joint_angles, upper_result.joint_angles]),
      full_qpos=full_qpos,
      success=success,
      errors=errors,
      message="converged" if success else "retargeting residual above tolerance",
      solver_joint_angles=np.concatenate([lower_result.joint_angles, upper_solver_q]),
    )


def retarget_full_body_targets(
  targets: Sequence[FullBodyTarget],
  *,
  q_init: Sequence[float] | None = None,
  retargeter: G1FullBodySEWRetargeter | None = None,
) -> list[FullBodyRetargetResult]:
  adapter = retargeter or G1FullBodySEWRetargeter()
  q_previous = np.zeros(29) if q_init is None else np.asarray(q_init, dtype=float)
  results: list[FullBodyRetargetResult] = []
  for target in targets:
    result = adapter.retarget(q_previous, target)
    results.append(result)
    q_previous = result.solver_joint_angles if result.solver_joint_angles is not None else result.joint_angles
  return results
