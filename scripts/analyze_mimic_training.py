"""Analyze BeyondMimic reference gait, policy rollout, and training logs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
  sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.mimic.gait_metrics import compare_contact_sequences
from src.analysis.mimic.reference import analyze_reference_motion, reference_rows
from src.analysis.mimic.reporting import (
  render_markdown_report,
  save_clearance_plot,
  save_checkpoint_scan_plot,
  save_contact_raster_plot,
  save_training_curve_plot,
  write_csv,
  write_json,
)
from src.analysis.mimic.training_logs import (
  downsample_scalar_rows,
  list_checkpoints,
  read_tensorboard_scalars,
  read_wandb_summaries,
  scalar_tag_summary,
)


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description="Analyze mimic gait training failures.")
  parser.add_argument("--motion-file", required=True)
  parser.add_argument("--checkpoint")
  parser.add_argument("--run-dir")
  parser.add_argument("--task-id", default="Unitree-G1-Tracking-No-State-Estimation")
  parser.add_argument("--device", default="cpu")
  parser.add_argument("--steps", type=int)
  parser.add_argument("--output-dir", required=True)
  parser.add_argument("--wandb-dir", default="wandb")
  parser.add_argument("--scan-checkpoints", action="store_true")
  parser.add_argument("--checkpoint-stride", type=int, default=5000)
  parser.add_argument("--foot-body-indices", nargs=2, type=int, default=(6, 12))
  parser.add_argument("--height-threshold-m", type=float, default=0.025)
  parser.add_argument("--vertical-speed-threshold-m-s", type=float, default=0.35)
  parser.add_argument("--max-scalar-rows-per-tag", type=int, default=1000)
  return parser


def _resolve_steps(requested_steps: int | None, frame_count: int) -> int:
  if requested_steps is not None:
    return requested_steps
  return frame_count


def _checkpoint_scan_selection(run_dir: Path, stride: int) -> list[Path]:
  selected = []
  for checkpoint in list_checkpoints(run_dir):
    iteration = int(checkpoint.stem.split("_")[1])
    if iteration == 0 or iteration % stride == 0:
      selected.append(checkpoint)
  return selected


def main() -> None:
  args = build_parser().parse_args()
  output_dir = Path(args.output_dir)
  plots_dir = output_dir / "plots"
  output_dir.mkdir(parents=True, exist_ok=True)
  plots_dir.mkdir(parents=True, exist_ok=True)

  reference = analyze_reference_motion(
    args.motion_file,
    foot_body_indices=args.foot_body_indices,
    height_threshold_m=args.height_threshold_m,
    vertical_speed_threshold_m_s=args.vertical_speed_threshold_m_s,
  )
  reference_summary = reference.summary_dict()
  write_csv(output_dir / "reference_metrics.csv", reference_rows(reference))
  save_contact_raster_plot(
    plots_dir / "foot_contact_raster.png",
    reference.contact_flags,
    title="Reference foot contact raster",
  )
  save_clearance_plot(
    plots_dir / "foot_clearance.png",
    reference.foot_clearance_m,
    fps=reference.fps,
    title="Reference foot clearance",
  )

  scalar_rows: list[dict[str, Any]] = []
  scalar_metadata: dict[str, Any] = {}
  training_summary: list[dict[str, Any]] = []
  if args.run_dir:
    scalar_rows, scalar_metadata = read_tensorboard_scalars(args.run_dir)
    training_summary = scalar_tag_summary(scalar_rows)
    sampled_scalar_rows = downsample_scalar_rows(
      scalar_rows,
      max_rows_per_tag=args.max_scalar_rows_per_tag,
    )
    write_csv(output_dir / "training_curves.csv", sampled_scalar_rows)
    write_csv(output_dir / "training_curve_summary.csv", training_summary)
    save_training_curve_plot(plots_dir / "training_curves.png", scalar_rows)

  wandb_summaries = read_wandb_summaries(args.wandb_dir)

  contact_comparison: dict[str, Any] | None = None
  rollout_error: str | None = None
  policy_rows: list[dict[str, Any]] = []
  if args.checkpoint:
    try:
      from src.analysis.mimic.rollout import run_checkpoint_gait_trace

      steps = _resolve_steps(args.steps, reference.frame_count)
      policy_trace = run_checkpoint_gait_trace(
        task_id=args.task_id,
        motion_file=args.motion_file,
        checkpoint=args.checkpoint,
        steps=steps,
        device=args.device,
      )
      comparison = compare_contact_sequences(
        reference.contact_flags,
        policy_trace.contact_flags,
        fps=reference.fps,
      )
      contact_comparison = comparison.to_dict()
      save_contact_raster_plot(
        plots_dir / "policy_vs_reference_contact_raster.png",
        reference.contact_flags,
        policy_trace.contact_flags,
        title="Reference vs policy foot contact",
      )
      for idx in range(policy_trace.contact_flags.shape[0]):
        policy_rows.append(
          {
            "frame": idx,
            "time_s": idx / reference.fps,
            "left_contact": int(policy_trace.contact_flags[idx, 0]),
            "right_contact": int(policy_trace.contact_flags[idx, 1]),
            "left_clearance_m": float(policy_trace.foot_clearance_m[idx, 0]),
            "right_clearance_m": float(policy_trace.foot_clearance_m[idx, 1]),
            "action_rate_l2": float(policy_trace.action_rate_l2[idx]),
          }
        )
      write_csv(output_dir / "policy_metrics.csv", policy_rows)
    except Exception as exc:  # noqa: BLE001 - report runtime failures without losing offline analysis.
      rollout_error = f"{type(exc).__name__}: {exc}"

  checkpoint_scan_rows: list[dict[str, Any]] = []
  if args.scan_checkpoints and args.run_dir:
    try:
      from src.analysis.mimic.rollout import run_checkpoint_gait_trace

      for checkpoint in _checkpoint_scan_selection(Path(args.run_dir), args.checkpoint_stride):
        steps = min(_resolve_steps(args.steps, reference.frame_count), 500)
        policy_trace = run_checkpoint_gait_trace(
          task_id=args.task_id,
          motion_file=args.motion_file,
          checkpoint=checkpoint,
          steps=steps,
          device=args.device,
        )
        comparison = compare_contact_sequences(
          reference.contact_flags,
          policy_trace.contact_flags,
          fps=reference.fps,
        )
        row = comparison.to_dict()
        row["checkpoint"] = str(checkpoint)
        row["iteration"] = int(checkpoint.stem.split("_")[1])
        checkpoint_scan_rows.append(row)
      write_csv(output_dir / "checkpoint_scan.csv", checkpoint_scan_rows)
      save_checkpoint_scan_plot(
        plots_dir / "checkpoint_scan.png",
        checkpoint_scan_rows,
      )
    except Exception as exc:  # noqa: BLE001
      if rollout_error:
        rollout_error += f"; checkpoint scan failed: {type(exc).__name__}: {exc}"
      else:
        rollout_error = f"checkpoint scan failed: {type(exc).__name__}: {exc}"

  inputs = {
    "motion_file": args.motion_file,
    "checkpoint": args.checkpoint,
    "run_dir": args.run_dir,
    "task_id": args.task_id,
    "device": args.device,
  }
  report = render_markdown_report(
    title="BeyondMimic Gait Failure Analysis",
    inputs=inputs,
    reference_summary=reference_summary,
    training_summary=training_summary,
    contact_comparison=contact_comparison,
    checkpoint_scan=checkpoint_scan_rows,
    rollout_error=rollout_error,
  )
  (output_dir / "report.md").write_text(report, encoding="utf-8")
  write_json(
    output_dir / "summary.json",
    {
      "inputs": inputs,
      "reference_summary": reference_summary,
      "contact_comparison": contact_comparison,
      "training_scalar_metadata": scalar_metadata,
      "training_curve_summary": training_summary,
      "wandb_summaries": wandb_summaries,
      "checkpoint_scan": checkpoint_scan_rows,
      "rollout_error": rollout_error,
      "outputs": {
        "report": str(output_dir / "report.md"),
        "plots_dir": str(plots_dir),
      },
    },
  )
  print(f"Report written to: {output_dir / 'report.md'}")
  if rollout_error:
    print(f"Rollout warning: {rollout_error}")


if __name__ == "__main__":
  main()
