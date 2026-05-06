"""Generate phase-binned foot-clearance diagnostics for mimic policies."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
  sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.mimic.gait_metrics import compare_contact_sequences
from src.analysis.mimic.phase_diagnostics import (
  aggregate_phase_rows,
  aggregate_swing_run_rows,
  phase_bin_rows,
  swing_run_rows,
)
from src.analysis.mimic.quick_eval import quick_policy_metrics
from src.analysis.mimic.reference import analyze_reference_motion
from src.analysis.mimic.reporting import write_csv, write_json


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
    description="Analyze foot clearance by reference swing phase."
  )
  parser.add_argument("--motion-file", required=True)
  parser.add_argument("--checkpoint")
  parser.add_argument("--task-id", default="Unitree-G1-Tracking-No-State-Estimation")
  parser.add_argument("--device", default="cpu")
  parser.add_argument("--steps", type=int)
  parser.add_argument("--output-dir", required=True)
  parser.add_argument("--foot-body-indices", nargs=2, type=int, default=(6, 12))
  parser.add_argument("--height-threshold-m", type=float, default=0.025)
  parser.add_argument("--vertical-speed-threshold-m-s", type=float, default=0.35)
  parser.add_argument("--low-clearance-m", type=float, default=0.02)
  parser.add_argument(
    "--disable-foot-phase-observation",
    action="store_true",
    help="Use when analyzing pre-V4 checkpoints with 154-dim no-state actor observations.",
  )
  return parser


def _rows_to_markdown_table(
  rows: list[dict[str, Any]],
  fields: list[str],
  max_rows: int = 12,
) -> str:
  if not rows:
    return ""
  selected = rows[:max_rows]
  lines = [
    "| " + " | ".join(fields) + " |",
    "| " + " | ".join("---" for _ in fields) + " |",
  ]
  for row in selected:
    values = [_format_value(row.get(field, "")) for field in fields]
    lines.append("| " + " | ".join(values) + " |")
  return "\n".join(lines)


def _format_value(value: Any) -> str:
  if isinstance(value, float):
    return f"{value:.4f}" if abs(value) < 10 else f"{value:.2f}"
  return str(value)


def _render_report(
  *,
  reference_summary: dict[str, Any],
  reference_phase_summary: dict[str, Any],
  reference_run_summary: dict[str, Any],
  policy_metrics: dict[str, Any] | None,
  policy_phase_summary: dict[str, Any] | None,
  reference_runs: list[dict[str, Any]],
  phase_rows: list[dict[str, Any]],
) -> str:
  lines = ["# Mimic Phase/Clearance Diagnostics", ""]
  lines.extend(
    [
      "## Reference Summary",
      "",
      f"- frame_count: `{reference_summary['frame_count']}`",
      f"- reference_single_support_ratio: `{reference_summary['reference_single_support_ratio']:.3f}`",
      f"- reference_mean_swing_clearance_m: `{reference_summary['reference_mean_swing_clearance_m']:.3f}`",
      f"- reference_min_swing_clearance_m: `{reference_summary['reference_min_swing_clearance_m']:.3f}`",
      f"- reference_low_clearance_ratio_2cm: `{reference_phase_summary.get('reference_low_clearance_ratio_2cm', float('nan')):.3f}`",
      f"- reference_true_swing_run_count: `{reference_run_summary.get('reference_true_swing_run_count', 0)}`",
      f"- reference_micro_swing_run_ratio: `{reference_run_summary.get('reference_micro_swing_run_ratio', float('nan')):.3f}`",
      "",
    ]
  )
  if policy_metrics is not None and policy_phase_summary is not None:
    lines.extend(
      [
        "## Policy Summary",
        "",
        f"- swing_contact_ratio: `{policy_metrics['swing_contact_ratio']:.3f}`",
        f"- single_support_survival: `{policy_metrics['single_support_survival']:.3f}`",
        f"- extra_touchdown_count: `{policy_metrics['extra_touchdown_count']}`",
        f"- policy_low_clearance_ratio_2cm: `{policy_metrics['policy_low_swing_clearance_ratio_2cm']:.3f}`",
        f"- policy_p05_swing_clearance_m: `{policy_metrics['policy_p05_swing_clearance_m']:.3f}`",
        f"- policy_true_swing_lift_success_ratio_5cm: `{policy_metrics.get('policy_true_swing_lift_success_ratio_5cm', float('nan')):.3f}`",
        f"- policy_true_swing_mid_contact_ratio: `{policy_metrics.get('policy_true_swing_mid_contact_ratio', float('nan')):.3f}`",
        f"- policy_true_swing_mid_low_clearance_ratio_2cm: `{policy_metrics.get('policy_true_swing_mid_low_clearance_ratio_2cm', float('nan')):.3f}`",
        f"- phase_weighted_policy_clearance_deficit_mean_m: `{policy_phase_summary.get('policy_clearance_deficit_mean_m', float('nan')):.3f}`",
        f"- phase_max_policy_downward_velocity_p95_m_s: `{policy_phase_summary.get('policy_downward_velocity_p95_m_s', float('nan')):.3f}`",
        "",
      ]
    )

  lines.extend(["## Reference Swing Runs", ""])
  run_fields = [
    "foot",
    "run_index",
    "start_frame",
    "end_frame",
    "duration_s",
    "min_clearance_m",
    "p10_clearance_m",
    "max_clearance_m",
    "reference_is_micro_swing",
    "low_clearance_ratio_2cm",
    "policy_swing_contact_ratio",
    "policy_mid_phase_contact_ratio",
    "policy_mid_phase_low_clearance_ratio_2cm",
    "policy_peak_clearance_ratio_to_reference",
    "policy_touchdown_count_during_reference_swing",
  ]
  lines.append(_rows_to_markdown_table(reference_runs, run_fields, max_rows=24))
  lines.extend(["", "## Phase Bins", ""])
  phase_fields = [
    "foot",
    "phase_bin",
    "sample_count",
    "reference_clearance_mean_m",
    "reference_low_clearance_ratio_2cm",
    "policy_swing_contact_ratio",
    "policy_clearance_mean_m",
    "policy_low_clearance_ratio_2cm",
    "policy_clearance_deficit_mean_m",
    "policy_downward_velocity_p95_m_s",
  ]
  lines.append(_rows_to_markdown_table(phase_rows, phase_fields, max_rows=20))
  lines.append("")
  return "\n".join(lines)


def main() -> None:
  args = build_parser().parse_args()
  output_dir = Path(args.output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)

  reference = analyze_reference_motion(
    args.motion_file,
    foot_body_indices=args.foot_body_indices,
    height_threshold_m=args.height_threshold_m,
    vertical_speed_threshold_m_s=args.vertical_speed_threshold_m_s,
  )
  policy_metrics: dict[str, Any] | None = None
  trace = None
  if args.checkpoint:
    from src.analysis.mimic.rollout import run_checkpoint_gait_trace

    steps = args.steps if args.steps is not None else reference.frame_count
    trace = run_checkpoint_gait_trace(
      task_id=args.task_id,
      motion_file=args.motion_file,
      checkpoint=args.checkpoint,
      steps=steps,
      device=args.device,
      disable_foot_phase_observation=args.disable_foot_phase_observation,
    )
    policy_metrics = quick_policy_metrics(
      reference_contact=reference.contact_flags,
      reference_clearance_m=reference.foot_clearance_m,
      policy_contact=trace.contact_flags,
      policy_clearance_m=trace.foot_clearance_m,
      policy_vertical_velocity_m_s=trace.foot_vertical_velocity_m_s,
      action_rate_l2=trace.action_rate_l2,
      fps=reference.fps,
    )
    policy_metrics.update(
      compare_contact_sequences(
        reference.contact_flags,
        trace.contact_flags,
        fps=reference.fps,
      ).to_dict()
    )
  reference_runs = swing_run_rows(
    reference_contact=reference.contact_flags,
    clearance_m=reference.foot_clearance_m,
    fps=reference.fps,
    policy_contact=trace.contact_flags if trace is not None else None,
    policy_clearance_m=trace.foot_clearance_m if trace is not None else None,
    policy_vertical_velocity_m_s=(
      trace.foot_vertical_velocity_m_s if trace is not None else None
    ),
    low_clearance_m=args.low_clearance_m,
  )
  if trace is not None:
    phase_rows = phase_bin_rows(
      reference_contact=reference.contact_flags,
      reference_clearance_m=reference.foot_clearance_m,
      policy_contact=trace.contact_flags,
      policy_clearance_m=trace.foot_clearance_m,
      policy_vertical_velocity_m_s=trace.foot_vertical_velocity_m_s,
      low_clearance_m=args.low_clearance_m,
    )
  else:
    phase_rows = phase_bin_rows(
      reference_contact=reference.contact_flags,
      reference_clearance_m=reference.foot_clearance_m,
      low_clearance_m=args.low_clearance_m,
    )

  reference_phase_summary = aggregate_phase_rows(phase_rows)
  reference_run_summary = aggregate_swing_run_rows(reference_runs)
  policy_phase_summary = aggregate_phase_rows(phase_rows) if policy_metrics else None

  write_csv(output_dir / "reference_swing_runs.csv", reference_runs)
  write_csv(output_dir / "phase_bins.csv", phase_rows)
  write_json(
    output_dir / "summary.json",
    {
      "reference_summary": reference.summary_dict(),
      "reference_phase_summary": reference_phase_summary,
      "reference_run_summary": reference_run_summary,
      "policy_metrics": policy_metrics,
      "policy_phase_summary": policy_phase_summary,
    },
  )
  report = _render_report(
    reference_summary=reference.summary_dict(),
    reference_phase_summary=reference_phase_summary,
    reference_run_summary=reference_run_summary,
    policy_metrics=policy_metrics,
    policy_phase_summary=policy_phase_summary,
    reference_runs=reference_runs,
    phase_rows=phase_rows,
  )
  (output_dir / "report.md").write_text(report, encoding="utf-8")
  print(f"Report written to: {output_dir / 'report.md'}")


if __name__ == "__main__":
  main()
