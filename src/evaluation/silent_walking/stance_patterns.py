"""Stance-level contact pattern summaries for silent-walking evaluation."""

from __future__ import annotations

from dataclasses import dataclass

from .types import EpisodeMetricTrace


@dataclass(frozen=True, slots=True)
class StanceContactPattern:
  """Region-event pattern aggregated over one foot stance."""

  stance_id: int
  foot_index: int
  foot: str
  first_step: int
  last_step: int
  first_regions: str
  all_regions: str
  event_count: int
  secondary_event_count: int
  multi_region_first_touchdown: bool
  rollover: bool
  heel_to_toe_rollover: bool
  toe_first: bool


def extract_stance_contact_patterns(trace: EpisodeMetricTrace) -> list[StanceContactPattern]:
  """Aggregate region-level events into one row per stance."""

  if trace.region_event_stance_id is None or trace.region_event_stance_id.numel() == 0:
    return []

  events_by_stance: dict[int, list[int]] = {}
  stance_ids = trace.region_event_stance_id.detach().cpu().flatten().tolist()
  for event_idx, stance_id in enumerate(stance_ids):
    events_by_stance.setdefault(int(stance_id), []).append(event_idx)

  patterns: list[StanceContactPattern] = []
  for stance_id in sorted(events_by_stance):
    event_indices = events_by_stance[stance_id]
    touchdown_indices = [
      idx
      for idx in event_indices
      if idx < len(trace.region_event_type) and trace.region_event_type[idx] == "touchdown"
    ]
    first_idx = touchdown_indices[0] if touchdown_indices else event_indices[0]
    first_regions = _regions_at(trace, first_idx)
    all_region_set: set[str] = set()
    for idx in event_indices:
      all_region_set.update(_split_regions(_regions_at(trace, idx)))
    first_region_set = _split_regions(first_regions)
    secondary_event_count = sum(
      1
      for idx in event_indices
      if idx < len(trace.region_event_type) and trace.region_event_type[idx] == "secondary"
    )
    foot_index = _event_int(trace.region_event_foot_index, first_idx)
    first_step = min(_event_int(trace.region_event_step, idx) for idx in event_indices)
    last_step = max(_event_int(trace.region_event_step, idx) for idx in event_indices)
    patterns.append(
      StanceContactPattern(
        stance_id=stance_id,
        foot_index=foot_index,
        foot=trace.region_event_foot[first_idx] if first_idx < len(trace.region_event_foot) else "",
        first_step=first_step,
        last_step=last_step,
        first_regions=first_regions,
        all_regions="+".join(region for region in ("heel", "midfoot", "toe") if region in all_region_set),
        event_count=len(event_indices),
        secondary_event_count=secondary_event_count,
        multi_region_first_touchdown=len(first_region_set) > 1,
        rollover=secondary_event_count > 0,
        heel_to_toe_rollover=(
          "heel" in first_region_set
          and "toe" not in first_region_set
          and "toe" in all_region_set
        ),
        toe_first=("toe" in first_region_set and "heel" not in first_region_set),
      )
    )
  return patterns


def summarize_stance_contact_patterns(trace: EpisodeMetricTrace) -> dict[str, float | int]:
  """Return compact stance-level counters for reports and JSON summaries."""

  patterns = extract_stance_contact_patterns(trace)
  stance_count = len(patterns)
  if stance_count == 0:
    return {
      "stance_count": 0,
      "single_region_first_touchdown_stance_count": 0,
      "multi_region_first_touchdown_stance_count": 0,
      "rollover_stance_count": 0,
      "heel_to_toe_rollover_stance_count": 0,
      "toe_first_stance_count": 0,
      "mean_secondary_events_per_stance": 0.0,
    }

  multi_region_first = sum(1 for pattern in patterns if pattern.multi_region_first_touchdown)
  rollover = sum(1 for pattern in patterns if pattern.rollover)
  heel_to_toe = sum(1 for pattern in patterns if pattern.heel_to_toe_rollover)
  toe_first = sum(1 for pattern in patterns if pattern.toe_first)
  secondary_events = sum(pattern.secondary_event_count for pattern in patterns)
  return {
    "stance_count": stance_count,
    "single_region_first_touchdown_stance_count": stance_count - multi_region_first,
    "multi_region_first_touchdown_stance_count": multi_region_first,
    "rollover_stance_count": rollover,
    "heel_to_toe_rollover_stance_count": heel_to_toe,
    "toe_first_stance_count": toe_first,
    "mean_secondary_events_per_stance": secondary_events / stance_count,
  }


def _regions_at(trace: EpisodeMetricTrace, event_idx: int) -> str:
  if event_idx >= len(trace.region_event_regions):
    return ""
  return trace.region_event_regions[event_idx]


def _split_regions(regions: str) -> set[str]:
  return {region for region in regions.split("+") if region}


def _event_int(value, idx: int) -> int:
  if value is None or idx >= value.numel():
    return -1
  return int(value.detach().cpu().flatten()[idx].item())
