"""Pure gait metrics for mimic failure analysis."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class ContactEvents:
  """Touchdown and liftoff sample indices for one contact channel."""

  touchdown_indices: list[int]
  liftoff_indices: list[int]


@dataclass(frozen=True, slots=True)
class GaitPhaseSummary:
  """Episode-level contact phase ratios and event counts."""

  frame_count: int
  duration_s: float
  single_support_ratio: float
  double_support_ratio: float
  no_support_ratio: float
  left_duty_factor: float
  right_duty_factor: float
  left_touchdown_count: int
  right_touchdown_count: int

  def to_dict(self) -> dict[str, float | int]:
    return asdict(self)


@dataclass(frozen=True, slots=True)
class ContactComparison:
  """Reference-versus-policy contact failure metrics."""

  frame_count: int
  swing_contact_ratio: float
  extra_touchdown_count: int
  single_support_survival: float
  touchdown_timing_error_s_mean: float
  touchdown_timing_error_s_abs_mean: float
  left_extra_touchdown_count: int
  right_extra_touchdown_count: int

  def to_dict(self) -> dict[str, float | int]:
    return asdict(self)


@dataclass(frozen=True, slots=True)
class ClearanceSummary:
  """Foot clearance statistics in swing windows."""

  min_swing_clearance_m: float
  mean_swing_clearance_m: float
  p05_swing_clearance_m: float
  p10_swing_clearance_m: float
  low_swing_clearance_ratio_2cm: float
  left_min_swing_clearance_m: float
  right_min_swing_clearance_m: float
  left_low_swing_clearance_ratio_2cm: float
  right_low_swing_clearance_ratio_2cm: float

  def to_dict(self) -> dict[str, float]:
    return asdict(self)


def _as_bool_array(values: np.ndarray) -> np.ndarray:
  arr = np.asarray(values, dtype=bool)
  if arr.ndim == 0:
    raise ValueError("contact values must contain at least one sample")
  return arr


def _validate_contact_matrix(contact_flags: np.ndarray, name: str) -> np.ndarray:
  arr = _as_bool_array(contact_flags)
  if arr.ndim != 2 or arr.shape[1] != 2:
    raise ValueError(f"{name} must have shape (frames, 2)")
  return arr


def contact_events(contact_flag: np.ndarray) -> ContactEvents:
  """Return touchdown and liftoff indices for one boolean contact sequence."""

  contact = _as_bool_array(contact_flag).reshape(-1)
  previous = np.concatenate(([False], contact[:-1]))
  touchdown = np.flatnonzero(contact & ~previous).tolist()
  liftoff = np.flatnonzero(~contact & previous).tolist()
  return ContactEvents(touchdown_indices=touchdown, liftoff_indices=liftoff)


def gait_phase_summary(contact_flags: np.ndarray, fps: float) -> GaitPhaseSummary:
  """Summarize left/right stance phases from contact flags."""

  if fps <= 0:
    raise ValueError("fps must be positive")
  contacts = _validate_contact_matrix(contact_flags, "contact_flags")
  contact_count = contacts.sum(axis=1)
  left_events = contact_events(contacts[:, 0])
  right_events = contact_events(contacts[:, 1])
  frame_count = int(contacts.shape[0])
  return GaitPhaseSummary(
    frame_count=frame_count,
    duration_s=frame_count / fps,
    single_support_ratio=float(np.mean(contact_count == 1)),
    double_support_ratio=float(np.mean(contact_count == 2)),
    no_support_ratio=float(np.mean(contact_count == 0)),
    left_duty_factor=float(np.mean(contacts[:, 0])),
    right_duty_factor=float(np.mean(contacts[:, 1])),
    left_touchdown_count=len(left_events.touchdown_indices),
    right_touchdown_count=len(right_events.touchdown_indices),
  )


def _mean_nearest_timing_error_s(
  reference_indices: list[int], policy_indices: list[int], fps: float
) -> tuple[float, float]:
  if not reference_indices or not policy_indices:
    return 0.0, 0.0
  errors: list[float] = []
  policy = np.asarray(policy_indices)
  for ref_idx in reference_indices:
    nearest = int(policy[np.argmin(np.abs(policy - ref_idx))])
    errors.append((nearest - ref_idx) / fps)
  error_arr = np.asarray(errors, dtype=float)
  return float(np.mean(error_arr)), float(np.mean(np.abs(error_arr)))


def compare_contact_sequences(
  reference_contact: np.ndarray,
  policy_contact: np.ndarray,
  fps: float,
) -> ContactComparison:
  """Compare policy foot contacts against reference gait phases."""

  if fps <= 0:
    raise ValueError("fps must be positive")
  reference = _validate_contact_matrix(reference_contact, "reference_contact")
  policy = _validate_contact_matrix(policy_contact, "policy_contact")
  frame_count = min(reference.shape[0], policy.shape[0])
  reference = reference[:frame_count]
  policy = policy[:frame_count]

  swing_mask = ~reference
  swing_sample_count = int(swing_mask.sum())
  swing_contact_ratio = (
    float((policy & swing_mask).sum() / swing_sample_count)
    if swing_sample_count
    else 0.0
  )

  ref_events = [contact_events(reference[:, foot]) for foot in range(2)]
  policy_events = [contact_events(policy[:, foot]) for foot in range(2)]
  ref_touchdowns = sum(len(events.touchdown_indices) for events in ref_events)
  policy_touchdowns = sum(len(events.touchdown_indices) for events in policy_events)
  left_extra = max(
    0,
    len(policy_events[0].touchdown_indices) - len(ref_events[0].touchdown_indices),
  )
  right_extra = max(
    0,
    len(policy_events[1].touchdown_indices) - len(ref_events[1].touchdown_indices),
  )

  single_support_mask = reference.sum(axis=1) == 1
  if single_support_mask.any():
    single_support_survival = float(
      np.mean(np.all(policy[single_support_mask] == reference[single_support_mask], axis=1))
    )
  else:
    single_support_survival = 1.0

  timing_errors = [
    _mean_nearest_timing_error_s(
      ref_events[foot].touchdown_indices,
      policy_events[foot].touchdown_indices,
      fps,
    )
    for foot in range(2)
  ]
  return ContactComparison(
    frame_count=frame_count,
    swing_contact_ratio=swing_contact_ratio,
    extra_touchdown_count=max(0, policy_touchdowns - ref_touchdowns),
    single_support_survival=single_support_survival,
    touchdown_timing_error_s_mean=float(np.mean([item[0] for item in timing_errors])),
    touchdown_timing_error_s_abs_mean=float(np.mean([item[1] for item in timing_errors])),
    left_extra_touchdown_count=left_extra,
    right_extra_touchdown_count=right_extra,
  )


def clearance_summary(clearance_m: np.ndarray, reference_contact: np.ndarray) -> ClearanceSummary:
  """Summarize foot clearance during reference swing phases."""

  clearance = np.asarray(clearance_m, dtype=float)
  if clearance.ndim != 2 or clearance.shape[1] != 2:
    raise ValueError("clearance_m must have shape (frames, 2)")
  reference = _validate_contact_matrix(reference_contact, "reference_contact")
  frame_count = min(clearance.shape[0], reference.shape[0])
  clearance = clearance[:frame_count]
  reference = reference[:frame_count]
  swing = ~reference

  foot_min: list[float] = []
  foot_low_ratio: list[float] = []
  for foot in range(2):
    values = clearance[swing[:, foot], foot]
    foot_min.append(float(np.min(values)) if values.size else 0.0)
    foot_low_ratio.append(float(np.mean(values < 0.02)) if values.size else 0.0)

  all_values = clearance[swing]
  return ClearanceSummary(
    min_swing_clearance_m=float(np.min(all_values)) if all_values.size else 0.0,
    mean_swing_clearance_m=float(np.mean(all_values)) if all_values.size else 0.0,
    p05_swing_clearance_m=float(np.percentile(all_values, 5)) if all_values.size else 0.0,
    p10_swing_clearance_m=float(np.percentile(all_values, 10)) if all_values.size else 0.0,
    low_swing_clearance_ratio_2cm=(
      float(np.mean(all_values < 0.02)) if all_values.size else 0.0
    ),
    left_min_swing_clearance_m=foot_min[0],
    right_min_swing_clearance_m=foot_min[1],
    left_low_swing_clearance_ratio_2cm=foot_low_ratio[0],
    right_low_swing_clearance_ratio_2cm=foot_low_ratio[1],
  )
