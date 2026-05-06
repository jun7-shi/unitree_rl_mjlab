"""Phase-binned clearance diagnostics for mimic gait analysis."""

from __future__ import annotations

from collections.abc import Sequence
import math

import numpy as np


FOOT_NAMES: tuple[str, str] = ("left", "right")
DEFAULT_PHASE_BINS: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


def _validate_foot_matrix(values: np.ndarray, name: str, dtype: type) -> np.ndarray:
  arr = np.asarray(values, dtype=dtype)
  if arr.ndim != 2 or arr.shape[1] != 2:
    raise ValueError(f"{name} must have shape (frames, 2)")
  return arr


def _safe_mean(values: np.ndarray) -> float:
  finite = values[np.isfinite(values)]
  return float(np.mean(finite)) if finite.size else float("nan")


def _safe_percentile(values: np.ndarray, percentile: float) -> float:
  finite = values[np.isfinite(values)]
  return float(np.percentile(finite, percentile)) if finite.size else float("nan")


def _safe_min(values: np.ndarray) -> float:
  finite = values[np.isfinite(values)]
  return float(np.min(finite)) if finite.size else float("nan")


def _safe_max(values: np.ndarray) -> float:
  finite = values[np.isfinite(values)]
  return float(np.max(finite)) if finite.size else float("nan")


def _count_touchdowns(contact_flag: np.ndarray) -> int:
  contact = np.asarray(contact_flag, dtype=bool).reshape(-1)
  previous = np.concatenate(([False], contact[:-1]))
  return int(np.count_nonzero(contact & ~previous))


def swing_phase_from_contact(reference_contact: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  """Return normalized phase and run id for each contiguous reference swing run."""
  contact = _validate_foot_matrix(reference_contact, "reference_contact", bool)
  phase = np.zeros(contact.shape, dtype=float)
  run_id = np.full(contact.shape, -1, dtype=int)
  for foot in range(2):
    swing_indices = np.flatnonzero(~contact[:, foot])
    if swing_indices.size == 0:
      continue

    current_run = 0
    start = int(swing_indices[0])
    previous = start
    for index in swing_indices[1:].tolist():
      if index != previous + 1:
        _fill_run_phase(phase[:, foot], run_id[:, foot], start, previous, current_run)
        current_run += 1
        start = int(index)
      previous = int(index)
    _fill_run_phase(phase[:, foot], run_id[:, foot], start, previous, current_run)
  return phase, run_id


def _fill_run_phase(
  phase: np.ndarray,
  run_id: np.ndarray,
  start: int,
  end: int,
  current_run: int,
) -> None:
  sample_count = end - start + 1
  if sample_count == 1:
    phase[start] = 0.0
  else:
    phase[start : end + 1] = np.linspace(0.0, 1.0, sample_count)
  run_id[start : end + 1] = current_run


def swing_run_rows(
  *,
  reference_contact: np.ndarray,
  clearance_m: np.ndarray,
  fps: float,
  policy_contact: np.ndarray | None = None,
  policy_clearance_m: np.ndarray | None = None,
  policy_vertical_velocity_m_s: np.ndarray | None = None,
  low_clearance_m: float = 0.02,
  true_swing_min_duration_s: float = 0.35,
  true_swing_min_peak_clearance_m: float = 0.04,
  lift_success_clearance_m: float = 0.05,
) -> list[dict[str, float | int | str]]:
  """Summarize clearance distribution for each contiguous reference swing run."""
  if fps <= 0:
    raise ValueError("fps must be positive")
  contact = _validate_foot_matrix(reference_contact, "reference_contact", bool)
  clearance = _validate_foot_matrix(clearance_m, "clearance_m", float)
  policy_contact_arr = (
    _validate_foot_matrix(policy_contact, "policy_contact", bool)
    if policy_contact is not None
    else None
  )
  policy_clearance_arr = (
    _validate_foot_matrix(policy_clearance_m, "policy_clearance_m", float)
    if policy_clearance_m is not None
    else None
  )
  velocity_arr = (
    _validate_foot_matrix(
      policy_vertical_velocity_m_s,
      "policy_vertical_velocity_m_s",
      float,
    )
    if policy_vertical_velocity_m_s is not None
    else None
  )

  frame_count = min(
    contact.shape[0],
    clearance.shape[0],
    *(arr.shape[0] for arr in (policy_contact_arr, policy_clearance_arr, velocity_arr) if arr is not None),
  )
  contact = contact[:frame_count]
  clearance = clearance[:frame_count]
  if policy_contact_arr is not None:
    policy_contact_arr = policy_contact_arr[:frame_count]
  if policy_clearance_arr is not None:
    policy_clearance_arr = policy_clearance_arr[:frame_count]
  if velocity_arr is not None:
    velocity_arr = velocity_arr[:frame_count]
  phase, run_id = swing_phase_from_contact(contact)

  rows: list[dict[str, float | int | str]] = []
  for foot in range(2):
    for current_run in sorted(set(run_id[:, foot].tolist()) - {-1}):
      mask = run_id[:, foot] == current_run
      indices = np.flatnonzero(mask)
      values = clearance[mask, foot]
      duration_s = float(values.size / fps)
      max_clearance = _safe_max(values)
      row: dict[str, float | int | str] = {
        "foot": FOOT_NAMES[foot],
        "run_index": int(current_run),
        "start_frame": int(indices[0]),
        "end_frame": int(indices[-1]),
        "sample_count": int(values.size),
        "duration_s": duration_s,
        "min_clearance_m": _safe_min(values),
        "p05_clearance_m": _safe_percentile(values, 5),
        "p10_clearance_m": _safe_percentile(values, 10),
        "mean_clearance_m": _safe_mean(values),
        "max_clearance_m": max_clearance,
        "low_clearance_ratio_2cm": float(np.mean(values < low_clearance_m))
        if values.size
        else 0.0,
        "reference_is_micro_swing": bool(
          duration_s < true_swing_min_duration_s
          or max_clearance < true_swing_min_peak_clearance_m
        ),
      }
      run_phase = phase[mask, foot]
      mid_mask = (run_phase >= 0.2) & (run_phase <= 0.8)
      if policy_contact_arr is not None:
        policy_contacts = policy_contact_arr[mask, foot]
        policy_mid_contacts = policy_contacts[mid_mask]
        row["policy_swing_contact_ratio"] = (
          float(np.mean(policy_contacts)) if policy_contacts.size else float("nan")
        )
        row["policy_mid_phase_contact_ratio"] = (
          float(np.mean(policy_mid_contacts))
          if policy_mid_contacts.size
          else float("nan")
        )
        row["policy_touchdown_count_during_reference_swing"] = _count_touchdowns(
          policy_contacts
        )
      if policy_clearance_arr is not None:
        policy_values = policy_clearance_arr[mask, foot]
        policy_mid_values = policy_values[mid_mask]
        policy_peak = _safe_max(policy_values)
        row["policy_min_clearance_m"] = _safe_min(policy_values)
        row["policy_mean_clearance_m"] = _safe_mean(policy_values)
        row["policy_max_clearance_m"] = policy_peak
        row["policy_low_clearance_ratio_2cm"] = _low_ratio(
          policy_values,
          low_clearance_m,
        )
        row["policy_mid_phase_low_clearance_ratio_2cm"] = _low_ratio(
          policy_mid_values,
          low_clearance_m,
        )
        row["policy_peak_clearance_ratio_to_reference"] = (
          float(policy_peak / max_clearance)
          if math.isfinite(policy_peak) and math.isfinite(max_clearance) and max_clearance > 0
          else float("nan")
        )
        row["policy_lift_success_5cm"] = bool(
          math.isfinite(policy_peak) and policy_peak >= lift_success_clearance_m
        )
      if velocity_arr is not None:
        velocity_values = velocity_arr[mask, foot]
        late_values = velocity_values[run_phase >= 0.8]
        row["policy_max_downward_velocity_m_s"] = _safe_max(
          np.maximum(-velocity_values, 0.0)
        )
        row["policy_late_phase_downward_velocity_p95_m_s"] = _safe_percentile(
          np.maximum(-late_values, 0.0),
          95,
        )
      rows.append(row)
  return rows


def aggregate_swing_run_rows(
  rows: Sequence[dict[str, float | int | str]],
) -> dict[str, float | int]:
  """Aggregate contiguous-swing-run diagnostics into report-level metrics."""
  total = len(rows)
  if total == 0:
    return {
      "reference_swing_run_count": 0,
      "reference_true_swing_run_count": 0,
      "reference_micro_swing_run_count": 0,
      "reference_micro_swing_run_ratio": float("nan"),
    }

  micro_rows = [row for row in rows if bool(row.get("reference_is_micro_swing"))]
  true_rows = [row for row in rows if not bool(row.get("reference_is_micro_swing"))]

  def weighted_mean(selected: Sequence[dict[str, float | int | str]], field: str) -> float:
    numerator = 0.0
    denominator = 0
    for row in selected:
      value = float(row.get(field, float("nan")))
      count = int(row.get("sample_count", 0))
      if math.isfinite(value) and count > 0:
        numerator += value * count
        denominator += count
    return numerator / denominator if denominator else float("nan")

  def simple_mean(selected: Sequence[dict[str, float | int | str]], field: str) -> float:
    values = []
    for row in selected:
      value = row.get(field, float("nan"))
      if isinstance(value, bool):
        values.append(float(value))
      else:
        numeric = float(value)
        if math.isfinite(numeric):
          values.append(numeric)
    return float(np.mean(values)) if values else float("nan")

  return {
    "reference_swing_run_count": total,
    "reference_true_swing_run_count": len(true_rows),
    "reference_micro_swing_run_count": len(micro_rows),
    "reference_micro_swing_run_ratio": float(len(micro_rows) / total),
    "policy_run_lift_success_ratio_5cm": simple_mean(
      rows,
      "policy_lift_success_5cm",
    ),
    "policy_true_swing_lift_success_ratio_5cm": simple_mean(
      true_rows,
      "policy_lift_success_5cm",
    ),
    "policy_true_swing_mid_contact_ratio": weighted_mean(
      true_rows,
      "policy_mid_phase_contact_ratio",
    ),
    "policy_true_swing_mid_low_clearance_ratio_2cm": weighted_mean(
      true_rows,
      "policy_mid_phase_low_clearance_ratio_2cm",
    ),
    "policy_true_swing_peak_clearance_ratio_mean": weighted_mean(
      true_rows,
      "policy_peak_clearance_ratio_to_reference",
    ),
    "policy_true_swing_touchdown_count_mean": simple_mean(
      true_rows,
      "policy_touchdown_count_during_reference_swing",
    ),
    "policy_true_swing_late_downward_velocity_p95_m_s": _safe_max(
      np.asarray(
        [
          float(
            row.get("policy_late_phase_downward_velocity_p95_m_s", float("nan"))
          )
          for row in true_rows
        ],
        dtype=float,
      )
    ),
  }


def phase_bin_rows(
  *,
  reference_contact: np.ndarray,
  reference_clearance_m: np.ndarray,
  policy_contact: np.ndarray | None = None,
  policy_clearance_m: np.ndarray | None = None,
  policy_vertical_velocity_m_s: np.ndarray | None = None,
  bins: Sequence[float] = DEFAULT_PHASE_BINS,
  low_clearance_m: float = 0.02,
) -> list[dict[str, float | int | str]]:
  """Summarize reference and policy behavior by reference swing phase bins."""
  contact = _validate_foot_matrix(reference_contact, "reference_contact", bool)
  reference_clearance = _validate_foot_matrix(
    reference_clearance_m,
    "reference_clearance_m",
    float,
  )
  frame_count = min(contact.shape[0], reference_clearance.shape[0])
  contact = contact[:frame_count]
  reference_clearance = reference_clearance[:frame_count]

  policy_contact_arr = (
    _validate_foot_matrix(policy_contact, "policy_contact", bool)[:frame_count]
    if policy_contact is not None
    else None
  )
  policy_clearance_arr = (
    _validate_foot_matrix(policy_clearance_m, "policy_clearance_m", float)[
      :frame_count
    ]
    if policy_clearance_m is not None
    else None
  )
  velocity_arr = (
    _validate_foot_matrix(
      policy_vertical_velocity_m_s,
      "policy_vertical_velocity_m_s",
      float,
    )[:frame_count]
    if policy_vertical_velocity_m_s is not None
    else None
  )

  phase, _ = swing_phase_from_contact(contact)
  edges = [float(edge) for edge in bins]
  if len(edges) < 2 or edges[0] != 0.0 or edges[-1] != 1.0:
    raise ValueError("bins must start at 0.0 and end at 1.0")
  if any(b <= a for a, b in zip(edges, edges[1:])):
    raise ValueError("bins must be strictly increasing")

  rows: list[dict[str, float | int | str]] = []
  for foot in range(2):
    swing = ~contact[:, foot]
    for bin_index, (start, end) in enumerate(zip(edges, edges[1:])):
      if bin_index == len(edges) - 2:
        bin_mask = swing & (phase[:, foot] >= start) & (phase[:, foot] <= end)
      else:
        bin_mask = swing & (phase[:, foot] >= start) & (phase[:, foot] < end)
      ref_values = reference_clearance[bin_mask, foot]
      row: dict[str, float | int | str] = {
        "foot": FOOT_NAMES[foot],
        "phase_bin": f"{start:.2f}-{end:.2f}",
        "phase_start": start,
        "phase_end": end,
        "sample_count": int(ref_values.size),
        "reference_clearance_min_m": _safe_min(ref_values),
        "reference_clearance_p05_m": _safe_percentile(ref_values, 5),
        "reference_clearance_p10_m": _safe_percentile(ref_values, 10),
        "reference_clearance_mean_m": _safe_mean(ref_values),
        "reference_clearance_max_m": _safe_max(ref_values),
        "reference_low_clearance_ratio_2cm": _low_ratio(
          ref_values,
          low_clearance_m,
        ),
      }
      if policy_contact_arr is not None:
        policy_contacts = policy_contact_arr[bin_mask, foot]
        row["policy_swing_contact_ratio"] = (
          float(np.mean(policy_contacts)) if policy_contacts.size else float("nan")
        )
      if policy_clearance_arr is not None:
        policy_values = policy_clearance_arr[bin_mask, foot]
        row["policy_clearance_min_m"] = _safe_min(policy_values)
        row["policy_clearance_p05_m"] = _safe_percentile(policy_values, 5)
        row["policy_clearance_p10_m"] = _safe_percentile(policy_values, 10)
        row["policy_clearance_mean_m"] = _safe_mean(policy_values)
        row["policy_low_clearance_ratio_2cm"] = _low_ratio(
          policy_values,
          low_clearance_m,
        )
        if ref_values.size and policy_values.size:
          deficit = np.maximum(ref_values - policy_values, 0.0)
          row["policy_clearance_deficit_mean_m"] = _safe_mean(deficit)
        else:
          row["policy_clearance_deficit_mean_m"] = float("nan")
      if velocity_arr is not None:
        velocity_values = velocity_arr[bin_mask, foot]
        downward_speed = np.maximum(-velocity_values, 0.0)
        row["policy_downward_velocity_mean_m_s"] = _safe_mean(downward_speed)
        row["policy_downward_velocity_p95_m_s"] = _safe_percentile(
          downward_speed,
          95,
        )
      rows.append(row)
  return rows


def _low_ratio(values: np.ndarray, low_clearance_m: float) -> float:
  if values.size == 0:
    return float("nan")
  finite = values[np.isfinite(values)]
  return float(np.mean(finite < low_clearance_m)) if finite.size else float("nan")


def aggregate_phase_rows(
  rows: Sequence[dict[str, float | int | str]],
) -> dict[str, float | int]:
  """Aggregate selected phase-bin rows into report-level diagnostics."""
  sample_count = sum(int(row.get("sample_count", 0)) for row in rows)
  if sample_count == 0:
    return {"sample_count": 0}

  def weighted(field: str) -> float:
    numerator = 0.0
    denominator = 0
    for row in rows:
      value = float(row.get(field, float("nan")))
      count = int(row.get("sample_count", 0))
      if math.isfinite(value) and count > 0:
        numerator += value * count
        denominator += count
    return numerator / denominator if denominator else float("nan")

  return {
    "sample_count": sample_count,
    "reference_low_clearance_ratio_2cm": weighted(
      "reference_low_clearance_ratio_2cm"
    ),
    "policy_swing_contact_ratio": weighted("policy_swing_contact_ratio"),
    "policy_low_clearance_ratio_2cm": weighted("policy_low_clearance_ratio_2cm"),
    "policy_clearance_deficit_mean_m": weighted("policy_clearance_deficit_mean_m"),
    "policy_downward_velocity_p95_m_s": _safe_max(
      np.asarray(
        [
          float(row.get("policy_downward_velocity_p95_m_s", float("nan")))
          for row in rows
        ],
        dtype=float,
      )
    ),
  }
