"""Artifact writers for silent walking evaluation."""

from __future__ import annotations

import csv
from dataclasses import asdict
import json
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from .reporting import render_markdown_report
from .post_event_dynamics import summarize_four_corner_post_event_dynamics_metrics
from .stance_patterns import (
  extract_stance_contact_patterns,
  summarize_stance_contact_patterns,
)
from .types import EpisodeMetricTrace

if TYPE_CHECKING:
  from .runner import SilentEvalResult


def write_eval_artifacts(
  result: "SilentEvalResult",
  *,
  policy_label: str,
  output_dir: str | Path,
) -> dict[str, Path]:
  """Write JSON, Markdown, and CSV artifacts for one silent-walking rollout."""

  out_dir = Path(output_dir)
  out_dir.mkdir(parents=True, exist_ok=True)

  summary_path = out_dir / "summary.json"
  report_path = out_dir / "report.md"
  trace_path = out_dir / "trace.csv"
  touchdown_path = out_dir / "touchdowns.csv"
  regional_events_path = out_dir / "regional_events.csv"
  stance_patterns_path = out_dir / "stance_patterns.csv"

  summary_payload = {
    "robot": result.robot_name,
    "task_id": result.task_id,
    "policy": policy_label,
    "steps": result.steps,
    "step_dt": result.step_dt,
    "summary": asdict(result.summary),
    "raw_metrics": _raw_metric_summary(result.trace),
  }
  summary_path.write_text(
    json.dumps(summary_payload, indent=2, sort_keys=True),
    encoding="utf-8",
  )
  report_path.write_text(
    render_markdown_report(
      result.robot_name,
      policy_label,
      result.summary,
      trace=result.trace,
    ),
    encoding="utf-8",
  )
  _write_trace_csv(trace_path, result.trace, result.step_dt)
  _write_touchdown_csv(touchdown_path, result.trace)
  _write_regional_events_csv(regional_events_path, result.trace)
  _write_stance_patterns_csv(stance_patterns_path, result.trace)

  return {
    "summary": summary_path,
    "report": report_path,
    "trace": trace_path,
    "touchdowns": touchdown_path,
    "regional_events": regional_events_path,
    "stance_patterns": stance_patterns_path,
  }


def _raw_metric_summary(trace: EpisodeMetricTrace) -> dict[str, float | int]:
  raw_metrics: dict[str, float | int] = {
    "mean_peak_force_bw": _mean_value(trace.touchdown_peak_force_bw),
    "mean_loading_rate_bw_s": _mean_value(trace.touchdown_loading_rate_bw_s),
    "mean_touchdown_vertical_speed_m_s": _mean_value(trace.touchdown_vertical_speed_m_s),
    "mean_action_rate_l2": _mean_value(trace.action_rate_l2),
    "mean_linear_velocity_error": _mean_value(trace.linear_velocity_error),
    "mean_yaw_rate_error": _mean_value(trace.yaw_rate_error),
    "touchdown_count": int(trace.touchdown_peak_force_bw.numel()),
    "mean_foot_loading_rate_bw_s": _mean_value(trace.foot_loading_rate_bw_s),
    "mean_touchdown_heel_vz_m_s": _mean_value(trace.touchdown_heel_vz_m_s),
    "mean_touchdown_toe_vz_m_s": _mean_value(trace.touchdown_toe_vz_m_s),
    "mean_touchdown_roll_angle_rad": _mean_value(trace.touchdown_roll_angle_rad),
    "region_event_count": len(trace.region_event_type),
    "secondary_region_event_count": sum(
      1 for event_type in trace.region_event_type if event_type == "secondary"
    ),
    "mean_region_event_corner_downward_speed_m_s": _mean_value(
      trace.region_event_corner_downward_speed_m_s
    ),
  }
  raw_metrics.update(summarize_stance_contact_patterns(trace))
  raw_metrics.update(summarize_four_corner_post_event_dynamics_metrics(trace))
  for event_type in ("touchdown", "secondary"):
    prefix = f"{event_type}_region_event"
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
    time_offsets = _event_values(
      trace.region_event_time_offset_s,
      trace.region_event_type,
      event_type,
    )
    raw_metrics[f"{prefix}_count"] = int(corner_max_speed.shape[0])
    raw_metrics[f"{event_type}_time_offset_ms_mean"] = _mean_value(time_offsets) * 1000.0
    raw_metrics[f"{event_type}_site_vertical_speed_mean_m_s"] = _mean_value(site_speed)
    raw_metrics[f"{event_type}_corner_downward_speed_mean_m_s"] = _mean_value(corner_max_speed)
    raw_metrics[f"{event_type}_corner_downward_speed_p95_m_s"] = _p95_value(corner_max_speed)

    corner_speeds = _event_values(
      trace.region_event_corner_downward_speeds_m_s,
      trace.region_event_type,
      event_type,
    )
    if corner_speeds.ndim == 2:
      for corner_idx, corner_name in enumerate(trace.foot_corner_names):
        if corner_idx >= corner_speeds.shape[1]:
          continue
        values = corner_speeds[:, corner_idx]
        raw_metrics[f"{event_type}_{corner_name}_down_speed_mean_m_s"] = _mean_value(values)
        raw_metrics[f"{event_type}_{corner_name}_down_speed_p95_m_s"] = _p95_value(values)
  return raw_metrics


def _write_trace_csv(path: Path, trace: EpisodeMetricTrace, step_dt: float) -> None:
  num_steps = int(trace.foot_z_force_n.shape[0])
  num_feet = int(trace.foot_z_force_n.shape[1])
  headers = ["step", "time_s"]
  for foot_idx in range(num_feet):
    headers.extend(
      [
        f"foot{foot_idx}_force_n",
        f"foot{foot_idx}_contact",
        f"foot{foot_idx}_vz_m_s",
      ]
    )
    if trace.foot_loading_rate_bw_s is not None:
      headers.append(f"foot{foot_idx}_loading_rate_bw_s")
    if trace.heel_vertical_velocity_m_s is not None:
      headers.append(f"foot{foot_idx}_heel_vz_m_s")
    if trace.toe_vertical_velocity_m_s is not None:
      headers.append(f"foot{foot_idx}_toe_vz_m_s")
    if trace.heel_height_m is not None:
      headers.append(f"foot{foot_idx}_heel_height_m")
    if trace.toe_height_m is not None:
      headers.append(f"foot{foot_idx}_toe_height_m")
    if trace.foot_roll_angle_rad is not None:
      headers.append(f"foot{foot_idx}_roll_angle_rad")
    if trace.foot_region_contact_flag is not None:
      for region_name in trace.foot_region_names:
        headers.append(f"foot{foot_idx}_region_{region_name}_contact")
    if trace.foot_corner_vertical_velocity_m_s is not None:
      for corner_name in trace.foot_corner_names:
        headers.append(f"foot{foot_idx}_{corner_name}_vz_m_s")
    if trace.foot_corner_height_m is not None:
      for corner_name in trace.foot_corner_names:
        headers.append(f"foot{foot_idx}_{corner_name}_height_m")
  headers.extend(
    [
      "action_rate_l2",
      "linear_velocity_error",
      "yaw_rate_error",
    ]
  )

  with path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.writer(handle)
    writer.writerow(headers)
    for step in range(num_steps):
      row: list[float | int] = [step, step * step_dt]
      for foot_idx in range(num_feet):
        row.extend(
          [
            _scalar(trace.foot_z_force_n[step, foot_idx]),
            _scalar(trace.foot_contact_flag[step, foot_idx]),
            _scalar(trace.foot_vertical_velocity_m_s[step, foot_idx]),
          ]
        )
        if trace.foot_loading_rate_bw_s is not None:
          row.append(_scalar(trace.foot_loading_rate_bw_s[step, foot_idx]))
        if trace.heel_vertical_velocity_m_s is not None:
          row.append(_scalar(trace.heel_vertical_velocity_m_s[step, foot_idx]))
        if trace.toe_vertical_velocity_m_s is not None:
          row.append(_scalar(trace.toe_vertical_velocity_m_s[step, foot_idx]))
        if trace.heel_height_m is not None:
          row.append(_scalar(trace.heel_height_m[step, foot_idx]))
        if trace.toe_height_m is not None:
          row.append(_scalar(trace.toe_height_m[step, foot_idx]))
        if trace.foot_roll_angle_rad is not None:
          row.append(_scalar(trace.foot_roll_angle_rad[step, foot_idx]))
        if trace.foot_region_contact_flag is not None:
          for region_idx in range(len(trace.foot_region_names)):
            row.append(_scalar(trace.foot_region_contact_flag[step, foot_idx, region_idx]))
        if trace.foot_corner_vertical_velocity_m_s is not None:
          for corner_idx in range(len(trace.foot_corner_names)):
            row.append(_scalar(trace.foot_corner_vertical_velocity_m_s[step, foot_idx, corner_idx]))
        if trace.foot_corner_height_m is not None:
          for corner_idx in range(len(trace.foot_corner_names)):
            row.append(_scalar(trace.foot_corner_height_m[step, foot_idx, corner_idx]))
      row.extend(
        [
          _scalar(trace.action_rate_l2[min(step, trace.action_rate_l2.numel() - 1)]),
          _scalar(trace.linear_velocity_error[min(step, trace.linear_velocity_error.numel() - 1)]),
          _scalar(trace.yaw_rate_error[min(step, trace.yaw_rate_error.numel() - 1)]),
        ]
      )
      writer.writerow(row)


def _write_touchdown_csv(path: Path, trace: EpisodeMetricTrace) -> None:
  headers = [
    "event_index",
    "peak_force_bw",
    "loading_rate_bw_s",
    "vertical_speed_m_s",
    "heel_vz_m_s",
    "toe_vz_m_s",
    "roll_angle_rad",
    "first_contact_region",
  ]
  count = int(trace.touchdown_peak_force_bw.numel())

  with path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.writer(handle)
    writer.writerow(headers)
    for idx in range(count):
      writer.writerow(
        [
          idx,
          _scalar(trace.touchdown_peak_force_bw[idx]),
          _scalar(trace.touchdown_loading_rate_bw_s[idx]),
          _scalar(trace.touchdown_vertical_speed_m_s[idx]),
          _optional_event_scalar(trace.touchdown_heel_vz_m_s, idx),
          _optional_event_scalar(trace.touchdown_toe_vz_m_s, idx),
          _optional_event_scalar(trace.touchdown_roll_angle_rad, idx),
          (
            trace.touchdown_first_contact_region[idx]
            if idx < len(trace.touchdown_first_contact_region)
            else ""
          ),
        ]
      )


def _write_regional_events_csv(path: Path, trace: EpisodeMetricTrace) -> None:
  headers = [
    "event_index",
    "stance_id",
    "step",
    "substep_index",
    "time_offset_s",
    "event_type",
    "foot_index",
    "foot",
    "regions",
    "peak_force_bw",
    "loading_rate_bw_s",
    "vertical_speed_m_s",
    "corner_downward_speed_m_s",
  ]
  for corner_name in trace.foot_corner_names:
    headers.append(f"{corner_name}_down_speed_m_s")
  count = len(trace.region_event_type)

  with path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.writer(handle)
    writer.writerow(headers)
    for idx in range(count):
      writer.writerow(
        [
          idx,
          _optional_event_int(trace.region_event_stance_id, idx),
          _optional_event_int(trace.region_event_step, idx),
          _optional_event_int(trace.region_event_substep_index, idx),
          _optional_event_scalar(trace.region_event_time_offset_s, idx),
          trace.region_event_type[idx],
          _optional_event_int(trace.region_event_foot_index, idx),
          trace.region_event_foot[idx] if idx < len(trace.region_event_foot) else "",
          trace.region_event_regions[idx] if idx < len(trace.region_event_regions) else "",
          _optional_event_scalar(trace.region_event_peak_force_bw, idx),
          _optional_event_scalar(trace.region_event_loading_rate_bw_s, idx),
          _optional_event_scalar(trace.region_event_vertical_speed_m_s, idx),
          _optional_event_scalar(trace.region_event_corner_downward_speed_m_s, idx),
          *[
            _optional_event_matrix_scalar(
              trace.region_event_corner_downward_speeds_m_s,
              idx,
              corner_idx,
            )
            for corner_idx in range(len(trace.foot_corner_names))
          ],
        ]
      )


def _write_stance_patterns_csv(path: Path, trace: EpisodeMetricTrace) -> None:
  headers = [
    "stance_id",
    "foot",
    "first_step",
    "last_step",
    "first_regions",
    "all_regions",
    "event_count",
    "secondary_event_count",
    "multi_region_first_touchdown",
    "rollover",
    "heel_to_toe_rollover",
    "toe_first",
  ]
  patterns = extract_stance_contact_patterns(trace)

  with path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.writer(handle)
    writer.writerow(headers)
    for pattern in patterns:
      writer.writerow(
        [
          pattern.stance_id,
          pattern.foot,
          pattern.first_step,
          pattern.last_step,
          pattern.first_regions,
          pattern.all_regions,
          pattern.event_count,
          pattern.secondary_event_count,
          int(pattern.multi_region_first_touchdown),
          int(pattern.rollover),
          int(pattern.heel_to_toe_rollover),
          int(pattern.toe_first),
        ]
      )


def _mean_value(value: torch.Tensor | None) -> float:
  if value is None or value.numel() == 0:
    return 0.0
  return float(value.float().mean().item())


def _p95_value(value: torch.Tensor | None) -> float:
  if value is None or value.numel() == 0:
    return 0.0
  flattened = value.float().flatten()
  if flattened.numel() == 1:
    return float(flattened.item())
  return float(torch.quantile(flattened, 0.95).item())


def _event_values(
  value: torch.Tensor | None,
  event_types: tuple[str, ...],
  target_event_type: str,
) -> torch.Tensor:
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


def _optional_event_scalar(value: torch.Tensor | None, idx: int) -> float:
  if value is None or idx >= value.numel():
    return 0.0
  return _scalar(value[idx])


def _optional_event_matrix_scalar(value: torch.Tensor | None, row: int, col: int) -> float:
  if value is None or value.ndim != 2 or row >= value.shape[0] or col >= value.shape[1]:
    return 0.0
  return _scalar(value[row, col])


def _optional_event_int(value: torch.Tensor | None, idx: int) -> int | str:
  if value is None or idx >= value.numel():
    return ""
  return int(value.detach().cpu().flatten()[idx].item())


def _scalar(value: torch.Tensor) -> float:
  return float(value.detach().cpu().reshape(()).item())
