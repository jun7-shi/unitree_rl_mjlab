"""Markdown reporting helpers for silent walking evaluation."""

from __future__ import annotations

from collections import Counter

import torch

from .foot_grid import FootGridHeatmapRow, summarize_post_contact_foot_grid
from .post_event_dynamics import summarize_four_corner_post_event_dynamics
from .stance_patterns import summarize_stance_contact_patterns
from .types import EpisodeMetricSummary, EpisodeMetricTrace


def _escape_inline_code(value: str) -> str:
  """Wrap values in a safe Markdown code span fence."""

  max_run = 0
  current_run = 0
  for char in value:
    if char == "`":
      current_run += 1
      max_run = max(max_run, current_run)
    else:
      current_run = 0

  fence = "`" * (max_run + 1)
  return f"{fence}{value}{fence}"


def render_markdown_report(
  robot_name: str,
  policy_label: str,
  summary: EpisodeMetricSummary,
  trace: EpisodeMetricTrace | None = None,
) -> str:
  """Render a compact markdown report for one evaluated policy."""

  report = f"""# Silent Walking Report

- Robot: {_escape_inline_code(robot_name)}
- Policy: {_escape_inline_code(policy_label)}
- Contact Quietness: `{summary.contact_quietness:.2f}`
- Body Smoothness: `{summary.body_smoothness:.2f}`
- Task Compliance: `{summary.task_compliance:.2f}`
- Total Score: `{summary.total_score:.2f}`
"""

  if trace is None:
    return report

  extended_rows: list[str] = []
  if trace.foot_loading_rate_bw_s is not None:
    extended_rows.append(
      f"- Mean Foot Loading Rate / BW/s: `{_mean(trace.foot_loading_rate_bw_s):.2f}`"
    )
    extended_rows.append(
      f"- Max Foot Loading Rate / BW/s: `{_max(trace.foot_loading_rate_bw_s):.2f}`"
    )
  if trace.heel_vertical_velocity_m_s is not None:
    extended_rows.append(
      f"- Mean Heel Vertical Velocity / m/s: `{_mean(trace.heel_vertical_velocity_m_s):.2f}`"
    )
  if trace.toe_vertical_velocity_m_s is not None:
    extended_rows.append(
      f"- Mean Toe Vertical Velocity / m/s: `{_mean(trace.toe_vertical_velocity_m_s):.2f}`"
    )
  if trace.touchdown_heel_vz_m_s is not None:
    extended_rows.append(
      f"- Touchdown Heel Vz / m/s: `{_mean(trace.touchdown_heel_vz_m_s):.2f}`"
    )
  if trace.touchdown_toe_vz_m_s is not None:
    extended_rows.append(
      f"- Touchdown Toe Vz / m/s: `{_mean(trace.touchdown_toe_vz_m_s):.2f}`"
    )
  if trace.touchdown_roll_angle_rad is not None:
    extended_rows.append(
      f"- Touchdown Roll Angle / rad: `{_mean(trace.touchdown_roll_angle_rad):.2f}`"
    )
  if trace.touchdown_first_contact_region:
    region_counts = Counter(trace.touchdown_first_contact_region)
    extended_rows.append(
      "- Touchdown First Contact Regions: "
      + ", ".join(f"`{name}={count}`" for name, count in sorted(region_counts.items()))
    )
  if trace.region_event_type:
    event_counts = Counter(trace.region_event_type)
    region_counts = Counter(trace.region_event_regions)
    extended_rows.append(
      "- Region Contact Events: "
      + ", ".join(f"`{name}={count}`" for name, count in sorted(event_counts.items()))
    )
    extended_rows.append(
      "- Region Event Regions: "
      + ", ".join(f"`{name}={count}`" for name, count in sorted(region_counts.items()))
    )
    if trace.region_event_corner_downward_speed_m_s is not None:
      extended_rows.append(
        "- Region Event Corner Downward Speed / m/s: "
        f"`{_mean(trace.region_event_corner_downward_speed_m_s):.2f}`"
      )
  extended_metrics = "\n".join(extended_rows)
  stance_pattern_table = _render_stance_pattern_table(trace)
  regional_distribution_table = _render_regional_event_distribution_table(trace)
  region_event_speed_table = _render_region_event_speed_table(trace)
  post_event_dynamics_table = _render_post_event_dynamics_table(trace)
  foot_grid_table = _render_foot_grid_table(trace)

  capsule_rows = "\n".join(
    (
      f"| {_escape_inline_code(name)} | `{int(count.item())}` | "
      f"`{peak.item():.2f}` | `{rate.item():.2f}` | `{speed.item():.2f}` |"
    )
    for name, count, peak, rate, speed in zip(
      trace.capsule_names,
      trace.capsule_touchdown_count,
      trace.capsule_touchdown_peak_force_bw,
      trace.capsule_touchdown_loading_rate_bw_s,
      trace.capsule_touchdown_vertical_speed_m_s,
      strict=True,
    )
  )

  return (
    report
    + f"""
## Raw Metrics

Foot-level touchdown metrics below are legacy control-step summaries. Use
`Region Event Speeds` for physics-substep touchdown/secondary velocity analysis.

- Peak Force / BW: `{_mean(trace.peak_force_bw):.2f}`
- Loading Rate / BW/s: `{_mean(trace.loading_rate_bw_s):.2f}`
- Touchdown Peak Force / BW: `{_mean(trace.touchdown_peak_force_bw):.2f}`
- Touchdown Loading Rate / BW/s: `{_mean(trace.touchdown_loading_rate_bw_s):.2f}`
- Touchdown Site Vz / m/s: `{_mean(trace.touchdown_vertical_speed_m_s):.2f}`
- Left Touchdown Peak / BW: `{_mean(trace.left_touchdown_peak_force_bw):.2f}`
- Right Touchdown Peak / BW: `{_mean(trace.right_touchdown_peak_force_bw):.2f}`
- Left Touchdown Loading Rate / BW/s: `{_mean(trace.left_touchdown_loading_rate_bw_s):.2f}`
- Right Touchdown Loading Rate / BW/s: `{_mean(trace.right_touchdown_loading_rate_bw_s):.2f}`
- Left Touchdown Site Vz / m/s: `{_mean(trace.left_touchdown_vertical_speed_m_s):.2f}`
- Right Touchdown Site Vz / m/s: `{_mean(trace.right_touchdown_vertical_speed_m_s):.2f}`
- Touchdown Peak Asymmetry / BW: `{_mean(trace.touchdown_peak_asymmetry_bw):.2f}`
- Action Rate L2: `{_mean(trace.action_rate_l2):.2f}`
- Linear Velocity Error: `{_mean(trace.linear_velocity_error):.2f}`
- Yaw Rate Error: `{_mean(trace.yaw_rate_error):.2f}`
{extended_metrics}
{stance_pattern_table}
{regional_distribution_table}
{region_event_speed_table}
{post_event_dynamics_table}
{foot_grid_table}

## Per-Capsule Touchdown Metrics

| Capsule | Count | Peak / BW | Loading Rate / BW/s | Vertical Speed / m/s |
| --- | ---: | ---: | ---: | ---: |
{capsule_rows}
"""
  )


def _render_foot_grid_table(trace: EpisodeMetricTrace) -> str:
  rows = summarize_post_contact_foot_grid(trace)
  if not rows:
    return ""

  foot_rows: list[str] = []
  for foot in tuple(dict.fromkeys(row.foot for row in rows)):
    selected = [row for row in rows if row.foot == foot]
    event_count = max((row.event_count for row in selected), default=0)
    mean_event_down = _mean_grid_value(selected, post_step=0)
    max_event_down = _max_grid_value(selected, post_step=0)
    max_post_1_down = _max_grid_value(selected, post_step=1)
    max_post_3_down = _max_grid_value(selected, post_step=3)
    foot_rows.append(
      f"| {foot} | `{event_count}` | `{mean_event_down:.2f}` | "
      f"`{max_event_down:.2f}` | `{max_post_1_down:.2f}` | `{max_post_3_down:.2f}` |"
    )

  return (
    "\n## Foot Grid Post-Contact Heatmap\n\n"
    "| Foot | Events | Mean Event Down Speed | Max Event Down Speed | "
    "Max +1 Down Speed | Max +3 Down Speed |\n"
    "| --- | ---: | ---: | ---: | ---: | ---: |\n"
    + "\n".join(foot_rows)
  )


def _render_regional_event_distribution_table(trace: EpisodeMetricTrace) -> str:
  if not trace.region_event_type:
    return ""

  default_region_columns = (
    "heel",
    "midfoot",
    "toe",
    "heel+midfoot",
    "midfoot+toe",
    "heel+midfoot+toe",
  )
  extra_region_columns = tuple(
    sorted(
      {
        regions
        for regions in trace.region_event_regions
        if regions and regions not in default_region_columns
      }
    )
  )
  region_columns = (*default_region_columns, *extra_region_columns)
  headers = ("Event Type", "Events", *region_columns)
  rows: list[list[str]] = []
  for event_type in ("touchdown", "secondary"):
    event_regions = [
      regions
      for candidate_event_type, regions in zip(
        trace.region_event_type,
        trace.region_event_regions,
        strict=False,
      )
      if candidate_event_type == event_type
    ]
    counts = Counter(event_regions)
    rows.append(
      [
        f"`{event_type}`",
        f"`{len(event_regions)}`",
        *[f"`{counts.get(region, 0)}`" for region in region_columns],
      ]
    )

  header_row = "| " + " | ".join(headers) + " |"
  align_row = "| " + " | ".join(["---"] + ["---:"] * (len(headers) - 1)) + " |"
  body_rows = ["| " + " | ".join(row) + " |" for row in rows]
  return (
    "\n## Regional Event Distribution\n\n"
    "这个表按 `touchdown` 与 `secondary` 拆分区域组合，避免把首次落地和后续补接触混在一起。\n\n"
    + "\n".join([header_row, align_row, *body_rows])
  )


def _render_post_event_dynamics_table(trace: EpisodeMetricTrace) -> str:
  rows = summarize_four_corner_post_event_dynamics(trace)
  if not rows:
    return ""

  body_rows = [
    (
      f"| `{row.event_type}` | `{row.corner_name}` | `{row.event_count}` | "
      f"`{row.event_down_speed_mean_m_s:.2f}` | "
      f"`{row.post_1_step_down_speed_mean_m_s:.2f}` | "
      f"`{row.post_3_step_down_speed_mean_m_s:.2f}` | "
      f"`{row.post_3_step_min_height_mean_m:.3f}` | "
      f"`{row.post_3_step_region_contact_ratio_mean:.2f}` |"
    )
    for row in rows
  ]
  return (
    "\n## Four-Corner Post-Event Dynamics\n\n"
    "`+1` 是 event 后一个 control step，`+3` 是约 60 ms 后；rear corners 映射到 heel contact，"
    "front corners 映射到 toe contact。\n\n"
    "| Event Type | Corner | Events | Event Down | +1 Down | +3 Down | +3 Min Height | +3 Region Contact |\n"
    "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |\n"
    + "\n".join(body_rows)
  )


def _render_stance_pattern_table(trace: EpisodeMetricTrace) -> str:
  stance_summary = summarize_stance_contact_patterns(trace)
  stance_count = int(stance_summary["stance_count"])
  if stance_count == 0:
    return ""

  rows = [
    (
      "Single-region first touchdown",
      int(stance_summary["single_region_first_touchdown_stance_count"]),
    ),
    (
      "Multi-region first touchdown",
      int(stance_summary["multi_region_first_touchdown_stance_count"]),
    ),
    ("Rollover stance", int(stance_summary["rollover_stance_count"])),
    (
      "Heel-to-toe rollover stance",
      int(stance_summary["heel_to_toe_rollover_stance_count"]),
    ),
    ("Toe-first stance", int(stance_summary["toe_first_stance_count"])),
  ]
  body_rows = [
    f"| {name} | `{count}` | `{_ratio(count, stance_count):.2f}` |"
    for name, count in rows
  ]
  return (
    "\n## Stance-Level Contact Patterns\n\n"
    "`stance_count` 是这个表的 denominator；rollover 只有在同一 stance 内出现 secondary region event 时才计数。\n\n"
    f"- Stance Count: `{stance_count}`\n"
    f"- Mean Secondary Events / Stance: `{stance_summary['mean_secondary_events_per_stance']:.2f}`\n\n"
    "| Pattern | Count | Ratio |\n"
    "| --- | ---: | ---: |\n"
    + "\n".join(body_rows)
  )


def _render_region_event_speed_table(trace: EpisodeMetricTrace) -> str:
  if not trace.region_event_type:
    return ""

  corner_headers = [
    _corner_label(corner_name)
    for corner_name in trace.foot_corner_names
  ]
  headers = [
    "Event Type",
    "Count",
    "Mean Offset ms",
    "Foot Site Down Vz Mean",
    "Corner Max Down Mean",
    "Corner Max Down P95",
    *corner_headers,
  ]
  rows = []
  for event_type in ("touchdown", "secondary"):
    site_speed = _event_values(
      trace.region_event_vertical_speed_m_s,
      trace.region_event_type,
      event_type,
    )
    corner_max_speed = _event_values(
      trace.region_event_corner_downward_speed_m_s,
      trace.region_event_type,
      event_type,
    )
    corner_speeds = _event_values(
      trace.region_event_corner_downward_speeds_m_s,
      trace.region_event_type,
      event_type,
    )
    time_offsets = _event_values(
      trace.region_event_time_offset_s,
      trace.region_event_type,
      event_type,
    )
    row = [
      f"`{event_type}`",
      str(int(corner_max_speed.shape[0])),
      f"`{_mean(time_offsets) * 1000.0:.1f}`",
      f"`{_mean(site_speed):.2f}`",
      f"`{_mean(corner_max_speed):.2f}`",
      f"`{_p95(corner_max_speed):.2f}`",
    ]
    for corner_idx in range(len(trace.foot_corner_names)):
      if corner_speeds.ndim != 2 or corner_idx >= corner_speeds.shape[1]:
        row.append("`0.00`")
      else:
        row.append(f"`{_mean(corner_speeds[:, corner_idx]):.2f}`")
    rows.append(row)

  header_row = "| " + " | ".join(headers) + " |"
  align_row = "| " + " | ".join(["---"] + ["---:"] * (len(headers) - 1)) + " |"
  body_rows = ["| " + " | ".join(row) + " |" for row in rows]
  return (
    "\n## Region Event Speeds\n\n"
    "Region event 使用 physics substep 级 contact/velocity；`Mean Offset ms` 是事件发生在当前 "
    "control step 内的平均时间偏移。TD Site Vz 使用 `left_foot/right_foot` site 的向下速度；"
    "corner 指标使用 evaluator-local `rear_left/rear_right/front_left/front_right` 四个虚拟点。\n\n"
    + "\n".join([header_row, align_row, *body_rows])
  )


def _event_values(value, event_types: tuple[str, ...], target_event_type: str):
  if value is None or value.numel() == 0:
    return torch.zeros(0, dtype=torch.float32)
  count = min(len(event_types), int(value.shape[0]))
  if count == 0:
    return value[:0]
  mask = torch.tensor(
    [event_type == target_event_type for event_type in event_types[:count]],
    dtype=torch.bool,
    device=value.device,
  )
  return value[:count][mask]


def _corner_label(corner_name: str) -> str:
  return corner_name.replace("_", " ").title() + " Down Mean"


def _mean(value) -> float:
  if value is None or value.numel() == 0:
    return 0.0
  return float(value.float().mean().item())


def _max(value) -> float:
  if value is None or value.numel() == 0:
    return 0.0
  return float(value.float().max().item())


def _p95(value) -> float:
  if value is None or value.numel() == 0:
    return 0.0
  flattened = value.float().flatten()
  if flattened.numel() == 1:
    return float(flattened.item())
  return float(flattened.quantile(0.95).item())


def _mean_grid_value(rows: list[FootGridHeatmapRow], *, post_step: int) -> float:
  values = [row.down_speed_mean_by_step_m_s.get(post_step, 0.0) for row in rows]
  if not values:
    return 0.0
  return sum(values) / len(values)


def _max_grid_value(rows: list[FootGridHeatmapRow], *, post_step: int) -> float:
  values = [row.down_speed_mean_by_step_m_s.get(post_step, 0.0) for row in rows]
  return max(values) if values else 0.0


def _ratio(count: int, denominator: int) -> float:
  if denominator <= 0:
    return 0.0
  return count / denominator
