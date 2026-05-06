"""Fast checkpoint screening for mimic experiment iteration."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
  sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.mimic.quick_eval import (
  baseline_metric_rows,
  classify_quick_eval,
  quick_policy_metrics,
  summarize_scalar_windows,
)
from src.analysis.mimic.reference import analyze_reference_motion
from src.analysis.mimic.reporting import write_csv, write_json
from src.analysis.mimic.rollout import run_checkpoint_gait_trace
from src.analysis.mimic.training_logs import (
  checkpoint_iteration,
  read_tensorboard_scalars,
)


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description="Quick-evaluate mimic checkpoints.")
  parser.add_argument("--motion-file", required=True)
  parser.add_argument("--run-dir", required=True)
  parser.add_argument("--checkpoint", action="append", required=True)
  parser.add_argument("--task-id", default="Unitree-G1-Tracking-No-State-Estimation")
  parser.add_argument("--device", default="cpu")
  parser.add_argument("--steps", type=int, default=500)
  parser.add_argument("--output-dir", required=True)
  parser.add_argument("--foot-body-indices", nargs=2, type=int, default=(6, 12))
  parser.add_argument("--height-threshold-m", type=float, default=0.025)
  parser.add_argument("--vertical-speed-threshold-m-s", type=float, default=0.35)
  return parser


def _checkpoint_label(path: Path) -> int:
  iteration = checkpoint_iteration(path)
  if iteration is None:
    raise ValueError(f"Checkpoint filename does not match model_<iter>.pt: {path}")
  return iteration


def _render_report(
  *,
  inputs: dict[str, Any],
  baseline_rows: list[dict[str, str]],
  checkpoint_rows: list[dict[str, Any]],
) -> str:
  lines = [
    "# Mimic Quick Evaluation",
    "",
    "## Inputs",
    "",
  ]
  for key, value in inputs.items():
    lines.append(f"- `{key}`: `{value}`")
  lines.extend(["", "## Baseline Metrics", ""])
  for row in baseline_rows:
    lines.append(f"- `{row['category']}` `{row['name']}`: {row['reason']}")
  lines.extend(["", "## Checkpoints", ""])
  for row in checkpoint_rows:
    lines.append(
      "- `{iteration}` decision `{decision}`: swing `{swing:.3f}`, "
      "single-support `{survival:.3f}`, extra touchdowns `{extra}`".format(
        iteration=row["iteration"],
        decision=row["decision"],
        swing=float(row["swing_contact_ratio"]),
        survival=float(row["single_support_survival"]),
        extra=int(row["extra_touchdown_count"]),
      )
    )
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
  scalar_rows, scalar_metadata = read_tensorboard_scalars(args.run_dir)
  scalar_summary = summarize_scalar_windows(scalar_rows)
  checkpoint_rows: list[dict[str, Any]] = []

  for checkpoint_text in args.checkpoint:
    checkpoint = Path(checkpoint_text)
    iteration = _checkpoint_label(checkpoint)
    trace = run_checkpoint_gait_trace(
      task_id=args.task_id,
      motion_file=args.motion_file,
      checkpoint=checkpoint,
      steps=args.steps,
      device=args.device,
    )
    metrics = quick_policy_metrics(
      reference_contact=reference.contact_flags,
      reference_clearance_m=reference.foot_clearance_m,
      policy_contact=trace.contact_flags,
      policy_clearance_m=trace.foot_clearance_m,
      policy_vertical_velocity_m_s=trace.foot_vertical_velocity_m_s,
      action_rate_l2=trace.action_rate_l2,
      fps=reference.fps,
    )
    decision = classify_quick_eval(metrics, checkpoint_iteration=iteration)
    checkpoint_rows.append(
      {
        "checkpoint": str(checkpoint),
        "iteration": iteration,
        **metrics,
        **decision,
      }
    )

  baseline_rows = baseline_metric_rows()
  write_csv(output_dir / "baseline_metrics.csv", baseline_rows)
  write_csv(output_dir / "checkpoint_quick_eval.csv", checkpoint_rows)
  write_json(
    output_dir / "summary.json",
    {
      "inputs": vars(args),
      "reference": reference.summary_dict(),
      "scalar_metadata": scalar_metadata,
      "scalar_window_summary": scalar_summary,
      "baseline_metrics": baseline_rows,
      "checkpoint_quick_eval": checkpoint_rows,
    },
  )
  (output_dir / "report.md").write_text(
    _render_report(
      inputs=vars(args),
      baseline_rows=baseline_rows,
      checkpoint_rows=checkpoint_rows,
    ),
    encoding="utf-8",
  )
  print(f"Quick eval written to: {output_dir / 'report.md'}")


if __name__ == "__main__":
  main()
