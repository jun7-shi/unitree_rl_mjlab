"""Watch short mimic experiments and quick-evaluate checkpoints."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import signal
import sys
import time
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
  sys.path.insert(0, str(PROJECT_ROOT))

from scripts.monitor_mimic_training import find_training_pids, latest_run_dir
from src.analysis.mimic.quick_eval import (
  baseline_metric_rows,
  classify_quick_eval,
  quick_policy_metrics,
  summarize_scalar_windows,
)
from src.analysis.mimic.reference import analyze_reference_motion
from src.analysis.mimic.reporting import write_csv, write_json
from src.analysis.mimic.rollout import run_checkpoint_gait_trace
from src.analysis.mimic.training_logs import read_tensorboard_scalars


@dataclass(frozen=True, slots=True)
class PendingCheckpoint:
  iteration: int
  path: Path


def parse_checkpoint_schedule(value: str) -> tuple[int, ...]:
  """Parse comma-separated positive checkpoint iterations."""

  iterations: list[int] = []
  for item in value.split(","):
    stripped = item.strip()
    if not stripped:
      continue
    iteration = int(stripped)
    if iteration <= 0:
      raise ValueError("checkpoint iterations must be positive")
    iterations.append(iteration)
  if not iterations:
    raise ValueError("at least one checkpoint iteration is required")
  return tuple(sorted(set(iterations)))


def evaluated_iterations(output_dir: Path) -> set[int]:
  """Return checkpoint iterations already evaluated under output_dir."""

  if not output_dir.exists():
    return set()
  evaluated: set[int] = set()
  for child in output_dir.iterdir():
    if not child.is_dir() or not child.name.startswith("checkpoint_"):
      continue
    if not (child / "summary.json").is_file():
      continue
    suffix = child.name.removeprefix("checkpoint_")
    if suffix.isdigit():
      evaluated.add(int(suffix))
  return evaluated


def pending_checkpoints(
  run_dir: Path,
  output_dir: Path,
  *,
  schedule: tuple[int, ...],
) -> list[PendingCheckpoint]:
  """Return scheduled checkpoint files that exist and have not been evaluated."""

  already_done = evaluated_iterations(output_dir)
  pending: list[PendingCheckpoint] = []
  for iteration in schedule:
    checkpoint = run_dir / f"model_{iteration}.pt"
    if checkpoint.exists() and iteration not in already_done:
      pending.append(PendingCheckpoint(iteration=iteration, path=checkpoint))
  return pending


def should_stop_for_quick_decision(decision: dict[str, str], iteration: int) -> bool:
  """Return whether a quick-eval decision should stop a training run."""

  return iteration >= 3000 and decision.get("decision") == "stop"


def _render_run_summary(rows: list[dict[str, Any]]) -> str:
  lines = ["# Mimic Quick Watch", "", "## Checkpoints", ""]
  if not rows:
    lines.append("- No checkpoints evaluated yet.")
  for row in rows:
    lines.append(
      "- `{iteration}` `{decision}`: swing `{swing:.3f}`, "
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


def _existing_rows(run_output_dir: Path) -> list[dict[str, Any]]:
  rows: list[dict[str, Any]] = []
  for child in sorted(run_output_dir.glob("checkpoint_*/summary.json")):
    with child.open("r", encoding="utf-8") as handle:
      summary = json.load(handle)
    rows.append(summary["checkpoint_quick_eval"])
  return rows


def evaluate_checkpoint(
  *,
  motion_file: str,
  run_dir: Path,
  checkpoint: PendingCheckpoint,
  run_output_dir: Path,
  task_id: str,
  device: str,
  steps: int,
) -> dict[str, Any]:
  """Evaluate one checkpoint and persist checkpoint-level artifacts."""

  output_dir = run_output_dir / f"checkpoint_{checkpoint.iteration}"
  output_dir.mkdir(parents=True, exist_ok=True)
  reference = analyze_reference_motion(motion_file)
  trace = run_checkpoint_gait_trace(
    task_id=task_id,
    motion_file=motion_file,
    checkpoint=checkpoint.path,
    steps=steps,
    device=device,
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
  decision = classify_quick_eval(metrics, checkpoint_iteration=checkpoint.iteration)
  scalar_rows, scalar_metadata = read_tensorboard_scalars(run_dir)
  row: dict[str, Any] = {
    "checkpoint": str(checkpoint.path),
    "iteration": checkpoint.iteration,
    **metrics,
    **decision,
  }
  write_csv(output_dir / "baseline_metrics.csv", baseline_metric_rows())
  write_csv(output_dir / "checkpoint_quick_eval.csv", [row])
  write_json(
    output_dir / "summary.json",
    {
      "motion_file": motion_file,
      "run_dir": str(run_dir),
      "checkpoint_quick_eval": row,
      "reference": reference.summary_dict(),
      "scalar_metadata": scalar_metadata,
      "scalar_window_summary": summarize_scalar_windows(scalar_rows),
    },
  )
  all_rows = _existing_rows(run_output_dir)
  write_csv(run_output_dir / "checkpoint_quick_eval.csv", all_rows)
  (run_output_dir / "report.md").write_text(
    _render_run_summary(all_rows),
    encoding="utf-8",
  )
  return row


def _stop_run(run_name: str, reason: str) -> None:
  for pid in find_training_pids(run_name):
    print(f"[STOP] run={run_name} pid={pid} reason={reason}", flush=True)
    import os

    os.kill(pid, signal.SIGTERM)


def check_once(args: argparse.Namespace, schedule: tuple[int, ...]) -> int:
  """Evaluate newly available scheduled checkpoints once."""

  any_pending = False
  for run_name in args.run_name:
    run_dir = latest_run_dir(args.log_root, run_name)
    if run_dir is None:
      print(f"[WAIT] run={run_name} run directory not found", flush=True)
      continue
    run_output_dir = args.output_root / run_name
    pending = pending_checkpoints(run_dir, run_output_dir, schedule=schedule)
    if not pending:
      print(f"[WAIT] run={run_name} no new scheduled checkpoints", flush=True)
      continue
    any_pending = True
    for checkpoint in pending:
      print(
        f"[EVAL] run={run_name} checkpoint={checkpoint.iteration}",
        flush=True,
      )
      row = evaluate_checkpoint(
        motion_file=args.motion_file,
        run_dir=run_dir,
        checkpoint=checkpoint,
        run_output_dir=run_output_dir,
        task_id=args.task_id,
        device=args.device,
        steps=args.steps,
      )
      print(
        "[RESULT] "
        f"run={run_name} iter={checkpoint.iteration} decision={row['decision']} "
        f"swing={float(row['swing_contact_ratio']):.3f} "
        f"single_support={float(row['single_support_survival']):.3f}",
        flush=True,
      )
      if should_stop_for_quick_decision(row, checkpoint.iteration):
        _stop_run(run_name, str(row["reason"]))
  return 0 if any_pending else 1


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description="Watch mimic quick experiments.")
  parser.add_argument("--motion-file", required=True)
  parser.add_argument("--run-name", action="append", required=True)
  parser.add_argument("--log-root", type=Path, default=Path("logs/rsl_rl/g1_tracking"))
  parser.add_argument("--output-root", type=Path, required=True)
  parser.add_argument("--task-id", default="Unitree-G1-Tracking-No-State-Estimation")
  parser.add_argument("--device", default="cpu")
  parser.add_argument("--steps", type=int, default=500)
  parser.add_argument("--checkpoints", default="1000,2000,3000,4000,5000")
  parser.add_argument("--interval-sec", type=int, default=60)
  parser.add_argument("--once", action="store_true")
  return parser


def main() -> None:
  args = build_parser().parse_args()
  schedule = parse_checkpoint_schedule(args.checkpoints)
  args.output_root.mkdir(parents=True, exist_ok=True)
  if args.once:
    raise SystemExit(check_once(args, schedule))
  while True:
    check_once(args, schedule)
    time.sleep(args.interval_sec)


if __name__ == "__main__":
  main()
