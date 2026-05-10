"""Post-event corner dynamics for silent-walking contact analysis."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .types import EpisodeMetricTrace


@dataclass(frozen=True, slots=True)
class FourCornerPostEventDynamicsRow:
  """Aggregated post-event dynamics for one event type and corner."""

  event_type: str
  corner_name: str
  event_count: int
  event_down_speed_mean_m_s: float
  post_1_step_down_speed_mean_m_s: float
  post_3_step_down_speed_mean_m_s: float
  post_3_step_min_height_mean_m: float
  post_3_step_region_contact_ratio_mean: float


def summarize_four_corner_post_event_dynamics(
  trace: EpisodeMetricTrace,
) -> list[FourCornerPostEventDynamicsRow]:
  """Summarize corner speed/height/contact in the first 60 ms after region events."""

  if (
    not trace.region_event_type
    or trace.region_event_step is None
    or trace.region_event_foot_index is None
    or trace.foot_corner_vertical_velocity_m_s is None
    or trace.foot_corner_height_m is None
  ):
    return []

  rows: list[FourCornerPostEventDynamicsRow] = []
  for event_type in ("touchdown", "secondary"):
    event_indices = [
      event_idx
      for event_idx, current_event_type in enumerate(trace.region_event_type)
      if current_event_type == event_type
    ]
    if not event_indices:
      continue
    for corner_idx, corner_name in enumerate(trace.foot_corner_names):
      event_down_speeds: list[float] = []
      post_1_speeds: list[float] = []
      post_3_speeds: list[float] = []
      post_3_min_heights: list[float] = []
      post_3_contact_ratios: list[float] = []
      for event_idx in event_indices:
        foot_idx = _event_int(trace.region_event_foot_index, event_idx)
        step = _event_int(trace.region_event_step, event_idx)
        if foot_idx < 0 or step < 0:
          continue
        event_down_speeds.append(
          _event_corner_down_speed(trace, event_idx, step, foot_idx, corner_idx)
        )
        post_1 = _corner_down_speed_at(trace, step + 1, foot_idx, corner_idx)
        if post_1 is not None:
          post_1_speeds.append(post_1)
        post_3 = _corner_down_speed_at(trace, step + 3, foot_idx, corner_idx)
        if post_3 is not None:
          post_3_speeds.append(post_3)

        window = _post_window(trace.foot_corner_height_m, step, foot_idx, corner_idx, steps=3)
        if window.numel() > 0:
          post_3_min_heights.append(float(window.min().item()))
        contact_ratio = _post_region_contact_ratio(trace, step, foot_idx, corner_name, steps=3)
        if contact_ratio is not None:
          post_3_contact_ratios.append(contact_ratio)

      rows.append(
        FourCornerPostEventDynamicsRow(
          event_type=event_type,
          corner_name=corner_name,
          event_count=len(event_down_speeds),
          event_down_speed_mean_m_s=_mean(event_down_speeds),
          post_1_step_down_speed_mean_m_s=_mean(post_1_speeds),
          post_3_step_down_speed_mean_m_s=_mean(post_3_speeds),
          post_3_step_min_height_mean_m=_mean(post_3_min_heights),
          post_3_step_region_contact_ratio_mean=_mean(post_3_contact_ratios),
        )
      )
  return rows


def summarize_four_corner_post_event_dynamics_metrics(
  trace: EpisodeMetricTrace,
) -> dict[str, float | int]:
  """Return flat raw-metric keys for summary JSON."""

  metrics: dict[str, float | int] = {}
  for row in summarize_four_corner_post_event_dynamics(trace):
    prefix = f"{row.event_type}_{row.corner_name}"
    metrics[f"{prefix}_post_event_count"] = row.event_count
    metrics[f"{prefix}_event_down_speed_mean_m_s"] = row.event_down_speed_mean_m_s
    metrics[f"{prefix}_post_1_step_down_speed_mean_m_s"] = (
      row.post_1_step_down_speed_mean_m_s
    )
    metrics[f"{prefix}_post_3_step_down_speed_mean_m_s"] = (
      row.post_3_step_down_speed_mean_m_s
    )
    metrics[f"{prefix}_post_3_step_min_height_mean_m"] = (
      row.post_3_step_min_height_mean_m
    )
    metrics[f"{prefix}_post_3_step_region_contact_ratio_mean"] = (
      row.post_3_step_region_contact_ratio_mean
    )
  return metrics


def _event_corner_down_speed(
  trace: EpisodeMetricTrace,
  event_idx: int,
  step: int,
  foot_idx: int,
  corner_idx: int,
) -> float:
  if (
    trace.region_event_corner_downward_speeds_m_s is not None
    and trace.region_event_corner_downward_speeds_m_s.ndim == 2
    and event_idx < trace.region_event_corner_downward_speeds_m_s.shape[0]
    and corner_idx < trace.region_event_corner_downward_speeds_m_s.shape[1]
  ):
    return float(trace.region_event_corner_downward_speeds_m_s[event_idx, corner_idx].item())
  value = _corner_down_speed_at(trace, step, foot_idx, corner_idx)
  return 0.0 if value is None else value


def _corner_down_speed_at(
  trace: EpisodeMetricTrace,
  step: int,
  foot_idx: int,
  corner_idx: int,
) -> float | None:
  corner_vz = trace.foot_corner_vertical_velocity_m_s
  if corner_vz is None:
    return None
  if (
    step < 0
    or foot_idx < 0
    or corner_idx < 0
    or step >= corner_vz.shape[0]
    or foot_idx >= corner_vz.shape[1]
    or corner_idx >= corner_vz.shape[2]
  ):
    return None
  return float(torch.clamp(-corner_vz[step, foot_idx, corner_idx], min=0.0).item())


def _post_window(
  value: torch.Tensor,
  step: int,
  foot_idx: int,
  corner_idx: int,
  *,
  steps: int,
) -> torch.Tensor:
  start = step + 1
  stop = min(step + steps + 1, value.shape[0])
  if (
    start >= stop
    or foot_idx < 0
    or corner_idx < 0
    or foot_idx >= value.shape[1]
    or corner_idx >= value.shape[2]
  ):
    return torch.zeros(0, dtype=value.dtype, device=value.device)
  return value[start:stop, foot_idx, corner_idx]


def _post_region_contact_ratio(
  trace: EpisodeMetricTrace,
  step: int,
  foot_idx: int,
  corner_name: str,
  *,
  steps: int,
) -> float | None:
  region_contact = trace.foot_region_contact_flag
  if region_contact is None:
    return None
  region_idx = _corner_region_index(trace.foot_region_names, corner_name)
  if region_idx is None:
    return None
  start = step + 1
  stop = min(step + steps + 1, region_contact.shape[0])
  if start >= stop or foot_idx < 0 or foot_idx >= region_contact.shape[1]:
    return None
  return float(region_contact[start:stop, foot_idx, region_idx].float().mean().item())


def _corner_region_index(region_names: tuple[str, ...], corner_name: str) -> int | None:
  if corner_name.startswith("rear_"):
    target = "heel"
  elif corner_name.startswith("front_"):
    target = "toe"
  else:
    return None
  try:
    return region_names.index(target)
  except ValueError:
    return None


def _event_int(value: torch.Tensor | None, idx: int) -> int:
  if value is None or idx >= value.numel():
    return -1
  return int(value.detach().cpu().flatten()[idx].item())


def _mean(values: list[float]) -> float:
  if not values:
    return 0.0
  return sum(values) / len(values)
