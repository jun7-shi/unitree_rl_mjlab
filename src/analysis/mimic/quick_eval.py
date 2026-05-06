"""Quick baseline metrics for fast mimic experiment iteration."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from .gait_metrics import (
  clearance_summary,
  compare_contact_sequences,
  gait_phase_summary,
)
from .phase_diagnostics import aggregate_swing_run_rows, swing_run_rows


DEFAULT_BASELINE_METRICS: tuple[dict[str, str], ...] = (
  {
    "name": "reference_single_support_ratio",
    "category": "reference",
    "reason": "Shows how much of the clip requires one-leg balance.",
  },
  {
    "name": "reference_double_support_ratio",
    "category": "reference",
    "reason": "Baseline for expected two-foot support duration.",
  },
  {
    "name": "reference_left_duty_factor",
    "category": "reference",
    "reason": "Left-foot stance ratio used to detect asymmetric clips.",
  },
  {
    "name": "reference_right_duty_factor",
    "category": "reference",
    "reason": "Right-foot stance ratio used to detect asymmetric clips.",
  },
  {
    "name": "reference_min_swing_clearance_m",
    "category": "reference",
    "reason": "Indicates how much clearance margin the source motion provides.",
  },
  {
    "name": "reference_true_swing_run_count",
    "category": "reference",
    "reason": "Counts non-micro reference swing runs that should require real foot lift.",
  },
  {
    "name": "reference_micro_swing_run_ratio",
    "category": "reference",
    "reason": "Separates likely contact-boundary artifacts from true swing phases.",
  },
  {
    "name": "Train/mean_reward",
    "category": "training",
    "reason": "Detects gross learning progress but is not sufficient for gait quality.",
  },
  {
    "name": "Train/mean_episode_length",
    "category": "training",
    "reason": "Smoke-test signal for falls and early terminations.",
  },
  {
    "name": "Episode_Termination/anchor_pos",
    "category": "training",
    "reason": "Detects root-position tracking failures and falls.",
  },
  {
    "name": "Episode_Termination/anchor_ori",
    "category": "training",
    "reason": "Detects root-orientation tracking failures and falls.",
  },
  {
    "name": "Episode_Termination/ee_body_pos",
    "category": "training",
    "reason": "Detects end-effector tracking failures.",
  },
  {
    "name": "Metrics/motion/error_body_pos",
    "category": "tracking",
    "reason": "Tracks body-position fidelity against the reference motion.",
  },
  {
    "name": "Metrics/motion/error_body_rot",
    "category": "tracking",
    "reason": "Tracks body-orientation fidelity against the reference motion.",
  },
  {
    "name": "Metrics/motion/error_joint_pos",
    "category": "tracking",
    "reason": "Detects joint-pose drift that reward-only contact metrics can hide.",
  },
  {
    "name": "Metrics/motion/error_joint_vel",
    "category": "tracking",
    "reason": "Detects noisy or dynamically inconsistent imitation.",
  },
  {
    "name": "Metrics/motion/error_anchor_pos",
    "category": "tracking",
    "reason": "Tracks global root-position imitation quality.",
  },
  {
    "name": "Metrics/motion/error_anchor_rot",
    "category": "tracking",
    "reason": "Tracks global root-orientation imitation quality.",
  },
  {
    "name": "Metrics/motion/error_body_lin_vel",
    "category": "tracking",
    "reason": "Detects velocity mismatch even when pose looks close.",
  },
  {
    "name": "Metrics/motion/error_body_ang_vel",
    "category": "tracking",
    "reason": "Detects angular-velocity mismatch and torso/limb jitter.",
  },
  {
    "name": "swing_contact_ratio",
    "category": "contact",
    "reason": "Primary padding-step indicator: swing foot touches during reference swing.",
  },
  {
    "name": "single_support_survival",
    "category": "contact",
    "reason": "Measures whether policy preserves reference one-leg support phases.",
  },
  {
    "name": "extra_touchdown_count",
    "category": "contact",
    "reason": "Counts extra foot touchdown events relative to the reference.",
  },
  {
    "name": "left_extra_touchdown_count",
    "category": "asymmetry",
    "reason": "Localizes padding-step failures to the left foot.",
  },
  {
    "name": "right_extra_touchdown_count",
    "category": "asymmetry",
    "reason": "Localizes padding-step failures to the right foot.",
  },
  {
    "name": "touchdown_timing_error_s_abs_mean",
    "category": "contact",
    "reason": "Measures touchdown phase error relative to the reference.",
  },
  {
    "name": "policy_single_support_ratio",
    "category": "contact",
    "reason": "Actual one-foot support ratio produced by the policy.",
  },
  {
    "name": "policy_double_support_ratio",
    "category": "contact",
    "reason": "Actual two-foot support ratio; high values indicate shuffling.",
  },
  {
    "name": "policy_no_support_ratio",
    "category": "contact",
    "reason": "Detects flight/no-contact samples that should be rare for quiet indoor walking.",
  },
  {
    "name": "policy_left_duty_factor",
    "category": "asymmetry",
    "reason": "Left-foot stance ratio produced by the policy.",
  },
  {
    "name": "policy_right_duty_factor",
    "category": "asymmetry",
    "reason": "Right-foot stance ratio produced by the policy.",
  },
  {
    "name": "policy_min_swing_clearance_m",
    "category": "contact",
    "reason": "Detects near-ground swing trajectories before contact occurs.",
  },
  {
    "name": "policy_mean_swing_clearance_m",
    "category": "contact",
    "reason": "Measures average swing-foot height margin.",
  },
  {
    "name": "policy_p05_swing_clearance_m",
    "category": "contact",
    "reason": "Low-percentile swing-foot clearance, less brittle than the absolute minimum.",
  },
  {
    "name": "policy_p10_swing_clearance_m",
    "category": "contact",
    "reason": "Tracks whether most swing samples keep a useful ground margin.",
  },
  {
    "name": "policy_low_swing_clearance_ratio_2cm",
    "category": "contact",
    "reason": "Fraction of reference-swing samples below 2 cm clearance.",
  },
  {
    "name": "policy_left_min_swing_clearance_m",
    "category": "asymmetry",
    "reason": "Left-foot swing clearance floor.",
  },
  {
    "name": "policy_right_min_swing_clearance_m",
    "category": "asymmetry",
    "reason": "Right-foot swing clearance floor.",
  },
  {
    "name": "policy_left_low_swing_clearance_ratio_2cm",
    "category": "asymmetry",
    "reason": "Left-foot fraction of swing samples below 2 cm clearance.",
  },
  {
    "name": "policy_right_low_swing_clearance_ratio_2cm",
    "category": "asymmetry",
    "reason": "Right-foot fraction of swing samples below 2 cm clearance.",
  },
  {
    "name": "policy_true_swing_lift_success_ratio_5cm",
    "category": "contact",
    "reason": "Fraction of true reference swing runs where the policy lifts above 5 cm.",
  },
  {
    "name": "policy_true_swing_mid_contact_ratio",
    "category": "contact",
    "reason": "Contact ratio in the central part of true swing runs.",
  },
  {
    "name": "policy_true_swing_mid_low_clearance_ratio_2cm",
    "category": "contact",
    "reason": "Low-clearance ratio in the central part of true swing runs.",
  },
  {
    "name": "policy_true_swing_peak_clearance_ratio_mean",
    "category": "contact",
    "reason": "Policy peak clearance relative to reference peak clearance on true swing runs.",
  },
  {
    "name": "policy_true_swing_touchdown_count_mean",
    "category": "contact",
    "reason": "Average number of policy touchdown transitions inside true reference swing runs.",
  },
  {
    "name": "policy_true_swing_late_downward_velocity_p95_m_s",
    "category": "quietness",
    "reason": "Late-swing downward foot speed proxy on true reference swing runs.",
  },
  {
    "name": "policy_mean_action_rate_l2",
    "category": "action",
    "reason": "Proxy for motor smoothness and policy chatter.",
  },
  {
    "name": "policy_p95_action_rate_l2",
    "category": "action",
    "reason": "Robust high-percentile action-rate proxy for rare jerks.",
  },
  {
    "name": "policy_max_abs_swing_foot_vertical_velocity_m_s",
    "category": "quietness",
    "reason": "Proxy for aggressive swing-foot vertical motion near touchdown.",
  },
  {
    "name": "Metrics/tracking_swing_contact_mean",
    "category": "contact",
    "reason": "Train-time average swing-contact violation when contact-aware rewards are enabled.",
  },
  {
    "name": "Metrics/tracking_landing_force_mean",
    "category": "quietness",
    "reason": "Train-time proxy for landing impact when contact-aware rewards are enabled.",
  },
  {
    "name": "Episode_Reward/action_rate_l2",
    "category": "action",
    "reason": "Weighted train-time action-rate penalty contribution.",
  },
  {
    "name": "Episode_Reward/motion_swing_contact",
    "category": "contact",
    "reason": "Weighted train-time swing-contact penalty contribution.",
  },
  {
    "name": "Episode_Reward/motion_soft_landing",
    "category": "quietness",
    "reason": "Weighted train-time soft-landing penalty contribution.",
  },
)

DEFAULT_SCALAR_TAGS: tuple[str, ...] = tuple(
  metric["name"]
  for metric in DEFAULT_BASELINE_METRICS
  if "/" in metric["name"]
)

DEFAULT_WINDOWS: tuple[tuple[int, int], ...] = (
  (0, 1000),
  (1000, 2000),
  (2000, 3000),
  (3000, 5000),
  (5000, 10000),
  (10000, 15000),
)


@dataclass(frozen=True, slots=True)
class QuickEvalThresholds:
  """Conservative checkpoint promotion thresholds for A120-style gait screening."""

  min_screen_iteration: int = 3000
  promote_iteration: int = 5000
  stop_swing_contact_ratio: float = 0.35
  stop_single_support_survival: float = 0.65
  promote_swing_contact_ratio: float = 0.20
  promote_single_support_survival: float = 0.80
  max_extra_touchdown_count_for_promote: int = 18
  max_action_rate_l2_for_promote: float = 1.0


def _validate_foot_matrix(values: np.ndarray, name: str) -> np.ndarray:
  arr = np.asarray(values)
  if arr.ndim != 2 or arr.shape[1] != 2:
    raise ValueError(f"{name} must have shape (frames, 2)")
  return arr


def _finite_mean(values: Sequence[float]) -> float:
  finite = [float(value) for value in values if math.isfinite(float(value))]
  return float(np.mean(finite)) if finite else float("nan")


def _finite_max_abs(values: np.ndarray) -> float:
  arr = np.asarray(values, dtype=float)
  finite = arr[np.isfinite(arr)]
  return float(np.max(np.abs(finite))) if finite.size else float("nan")


def quick_policy_metrics(
  *,
  reference_contact: np.ndarray,
  reference_clearance_m: np.ndarray | None = None,
  policy_contact: np.ndarray,
  policy_clearance_m: np.ndarray,
  policy_vertical_velocity_m_s: np.ndarray,
  action_rate_l2: np.ndarray,
  fps: float,
) -> dict[str, float | int]:
  """Compute rollout metrics used for fast checkpoint screening."""

  reference = _validate_foot_matrix(reference_contact, "reference_contact").astype(bool)
  reference_clearance = (
    _validate_foot_matrix(reference_clearance_m, "reference_clearance_m").astype(float)
    if reference_clearance_m is not None
    else None
  )
  policy = _validate_foot_matrix(policy_contact, "policy_contact").astype(bool)
  clearance = _validate_foot_matrix(policy_clearance_m, "policy_clearance_m").astype(float)
  vertical_velocity = _validate_foot_matrix(
    policy_vertical_velocity_m_s,
    "policy_vertical_velocity_m_s",
  ).astype(float)
  frame_count = min(
    reference.shape[0],
    *(arr.shape[0] for arr in (reference_clearance,) if arr is not None),
    policy.shape[0],
    clearance.shape[0],
    vertical_velocity.shape[0],
  )
  reference = reference[:frame_count]
  if reference_clearance is not None:
    reference_clearance = reference_clearance[:frame_count]
  policy = policy[:frame_count]
  clearance = clearance[:frame_count]
  vertical_velocity = vertical_velocity[:frame_count]

  comparison = compare_contact_sequences(reference, policy, fps=fps).to_dict()
  policy_phase = gait_phase_summary(policy, fps=fps).to_dict()
  clearance_metrics = clearance_summary(clearance, reference).to_dict()
  swing_velocity = vertical_velocity[~reference]
  action_rate = np.asarray(action_rate_l2, dtype=float).reshape(-1)
  metrics: dict[str, float | int] = {}
  metrics.update(comparison)
  metrics.update({f"policy_{key}": value for key, value in policy_phase.items()})
  metrics.update({f"policy_{key}": value for key, value in clearance_metrics.items()})
  if reference_clearance is not None:
    metrics.update(
      aggregate_swing_run_rows(
        swing_run_rows(
          reference_contact=reference,
          clearance_m=reference_clearance,
          policy_contact=policy,
          policy_clearance_m=clearance,
          policy_vertical_velocity_m_s=vertical_velocity,
          fps=fps,
        )
      )
    )
  metrics["policy_mean_action_rate_l2"] = _finite_mean(action_rate.tolist())
  metrics["policy_p95_action_rate_l2"] = float(
    np.percentile(action_rate[np.isfinite(action_rate)], 95)
  ) if np.isfinite(action_rate).any() else float("nan")
  metrics["policy_max_abs_swing_foot_vertical_velocity_m_s"] = _finite_max_abs(
    swing_velocity
  )
  return metrics


def summarize_scalar_windows(
  rows: Iterable[dict[str, Any]],
  *,
  tags: Sequence[str] = DEFAULT_SCALAR_TAGS,
  windows: Sequence[tuple[int, int]] = DEFAULT_WINDOWS,
) -> dict[str, float | int]:
  """Summarize selected TensorBoard scalar tags in fixed iteration windows."""

  selected_tags = set(tags)
  by_tag: dict[str, list[tuple[int, float]]] = {tag: [] for tag in tags}
  for row in rows:
    tag = str(row.get("tag", ""))
    if tag not in selected_tags:
      continue
    try:
      step = int(row["step"])
      value = float(row["value"])
    except (KeyError, TypeError, ValueError):
      continue
    if math.isfinite(value):
      by_tag.setdefault(tag, []).append((step, value))

  summary: dict[str, float | int] = {}
  for tag in tags:
    samples = by_tag.get(tag, [])
    for start, end in windows:
      values = [value for step, value in samples if start <= step < end]
      prefix = f"{tag}@{start}-{end}"
      summary[f"{prefix}_count"] = len(values)
      if values:
        summary[f"{prefix}_mean"] = float(np.mean(values))
        summary[f"{prefix}_latest"] = float(values[-1])
      else:
        summary[f"{prefix}_mean"] = float("nan")
        summary[f"{prefix}_latest"] = float("nan")
  return summary


def classify_quick_eval(
  metrics: dict[str, float | int],
  *,
  checkpoint_iteration: int,
  thresholds: QuickEvalThresholds = QuickEvalThresholds(),
) -> dict[str, str]:
  """Classify a checkpoint as stop, wait, or promote for the next experiment stage."""

  swing_contact = float(metrics.get("swing_contact_ratio", float("nan")))
  single_support = float(metrics.get("single_support_survival", float("nan")))
  extra_touchdowns = int(metrics.get("extra_touchdown_count", 0))
  action_rate = float(metrics.get("policy_mean_action_rate_l2", float("nan")))

  if checkpoint_iteration < thresholds.min_screen_iteration:
    return {
      "decision": "wait",
      "reason": f"checkpoint {checkpoint_iteration} is before min screen iteration {thresholds.min_screen_iteration}",
    }

  if (
    math.isfinite(swing_contact)
    and swing_contact > thresholds.stop_swing_contact_ratio
  ) or (
    math.isfinite(single_support)
    and single_support < thresholds.stop_single_support_survival
  ):
    return {
      "decision": "stop",
      "reason": "contact quality is below quick-screen threshold",
    }

  if checkpoint_iteration < thresholds.promote_iteration:
    return {
      "decision": "wait",
      "reason": f"checkpoint {checkpoint_iteration} passed stop thresholds but is before promote iteration {thresholds.promote_iteration}",
    }

  if (
    math.isfinite(swing_contact)
    and swing_contact <= thresholds.promote_swing_contact_ratio
    and math.isfinite(single_support)
    and single_support >= thresholds.promote_single_support_survival
    and extra_touchdowns <= thresholds.max_extra_touchdown_count_for_promote
    and (
      not math.isfinite(action_rate)
      or action_rate <= thresholds.max_action_rate_l2_for_promote
    )
  ):
    return {
      "decision": "promote",
      "reason": "checkpoint passed quick validation thresholds",
    }

  return {
    "decision": "continue",
    "reason": "checkpoint is viable but not yet strong enough to promote",
  }


def baseline_metric_rows() -> list[dict[str, str]]:
  """Return baseline metric definitions as CSV-friendly rows."""

  return [dict(metric) for metric in DEFAULT_BASELINE_METRICS]
