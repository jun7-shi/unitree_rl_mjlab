from __future__ import annotations

import argparse
import math
import os
import re
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


HARD_FAILURE_PATTERNS = (
  "Traceback (most recent call last)",
  "CUDA out of memory",
  "RuntimeError:",
  "ValueError:",
  "FloatingPointError",
  "NaN guard",
)


@dataclass(frozen=True)
class RunSnapshot:
  run_name: str
  run_dir: Path | None
  latest_iteration: int | None
  latest_checkpoint_iteration: int | None
  seconds_since_event_update: float | None
  output_tail: str
  metrics: dict[str, float]
  pids: tuple[int, ...]


@dataclass(frozen=True)
class PauseDecision:
  should_pause: bool
  reason: str


def latest_run_dir(log_root: Path, run_name: str) -> Path | None:
  candidates = sorted(
    (path for path in log_root.glob(f"*_{run_name}") if path.is_dir()),
    key=lambda path: path.name,
  )
  return candidates[-1] if candidates else None


def latest_checkpoint_iteration(run_dir: Path | None) -> int | None:
  if run_dir is None:
    return None
  latest: int | None = None
  for checkpoint in run_dir.glob("model_*.pt"):
    match = re.fullmatch(r"model_(\d+)\.pt", checkpoint.name)
    if match is None:
      continue
    iteration = int(match.group(1))
    latest = iteration if latest is None else max(latest, iteration)
  return latest


def latest_event_age_seconds(run_dir: Path | None, now: float | None = None) -> float | None:
  if run_dir is None:
    return None
  event_files = list(run_dir.glob("events.out.tfevents.*"))
  if not event_files:
    return None
  newest_mtime = max(path.stat().st_mtime for path in event_files)
  return (time.time() if now is None else now) - newest_mtime


def find_wandb_output_log(wandb_dir: Path, run_name: str) -> Path | None:
  candidates: list[Path] = []
  for output_log in wandb_dir.glob("run-*/files/output.log"):
    try:
      text = output_log.read_text(encoding="utf-8", errors="replace")
    except OSError:
      continue
    if f"Run name: {run_name}" in text:
      candidates.append(output_log)
  if not candidates:
    return None
  return max(candidates, key=lambda path: path.stat().st_mtime)


def read_tail(path: Path | None, max_bytes: int = 256_000) -> str:
  if path is None or not path.exists():
    return ""
  size = path.stat().st_size
  with path.open("rb") as handle:
    if size > max_bytes:
      handle.seek(size - max_bytes)
    return handle.read().decode("utf-8", errors="replace")


def parse_latest_iteration(output_tail: str) -> int | None:
  matches = re.findall(r"Learning iteration\s+(\d+)/(\d+)", output_tail)
  if not matches:
    return None
  return int(matches[-1][0])


def parse_latest_metrics(output_tail: str) -> dict[str, float]:
  metrics: dict[str, float] = {}
  block_start = output_tail.rfind("Learning iteration")
  block = output_tail[block_start:] if block_start >= 0 else output_tail
  for line in block.splitlines():
    if ":" not in line:
      continue
    key, value_text = line.rsplit(":", 1)
    key = key.strip()
    value_text = value_text.strip()
    try:
      value = float(value_text)
    except ValueError:
      continue
    if math.isfinite(value):
      metrics[key] = value
  return metrics


def find_training_pids(run_name: str) -> tuple[int, ...]:
  result = subprocess.run(
    ["ps", "-eo", "pid=,cmd="],
    check=True,
    text=True,
    stdout=subprocess.PIPE,
  )
  pids: list[int] = []
  own_pid = os.getpid()
  for line in result.stdout.splitlines():
    stripped = line.strip()
    if not stripped:
      continue
    pid_text, _, command = stripped.partition(" ")
    if not pid_text.isdigit():
      continue
    pid = int(pid_text)
    if pid == own_pid:
      continue
    if "scripts/train.py" in command and run_name in command:
      pids.append(pid)
  return tuple(pids)


def snapshot_run(log_root: Path, wandb_dir: Path, run_name: str) -> RunSnapshot:
  run_dir = latest_run_dir(log_root, run_name)
  output_log = find_wandb_output_log(wandb_dir, run_name)
  output_tail = read_tail(output_log)
  return RunSnapshot(
    run_name=run_name,
    run_dir=run_dir,
    latest_iteration=parse_latest_iteration(output_tail),
    latest_checkpoint_iteration=latest_checkpoint_iteration(run_dir),
    seconds_since_event_update=latest_event_age_seconds(run_dir),
    output_tail=output_tail,
    metrics=parse_latest_metrics(output_tail),
    pids=find_training_pids(run_name),
  )


def should_pause(snapshot: RunSnapshot, args: argparse.Namespace) -> PauseDecision:
  if snapshot.run_dir is None:
    return PauseDecision(False, "run directory not created yet")

  if any(pattern in snapshot.output_tail for pattern in HARD_FAILURE_PATTERNS):
    return PauseDecision(True, "hard failure pattern found in wandb output.log")

  if re.search(r"\b(?:nan|inf)\b", snapshot.output_tail, flags=re.IGNORECASE):
    return PauseDecision(True, "non-finite token found in wandb output.log")

  if (
    snapshot.seconds_since_event_update is not None
    and snapshot.seconds_since_event_update > args.stale_seconds
    and snapshot.pids
  ):
    return PauseDecision(
      True,
      f"event file stale for {snapshot.seconds_since_event_update:.0f}s",
    )

  iteration = snapshot.latest_iteration
  if iteration is None or iteration < args.quality_min_iteration:
    return PauseDecision(False, "quality checks not active yet")

  mean_episode_length = snapshot.metrics.get("Mean episode length")
  if (
    mean_episode_length is not None
    and mean_episode_length < args.min_mean_episode_length
  ):
    return PauseDecision(
      True,
      f"mean episode length {mean_episode_length:.2f} below "
      f"{args.min_mean_episode_length:.2f} after iter {iteration}",
    )

  swing_contact = snapshot.metrics.get("Metrics/tracking_swing_contact_mean")
  landing_force = snapshot.metrics.get("Metrics/tracking_landing_force_mean")
  if (
    swing_contact is not None
    and landing_force is not None
    and swing_contact > args.max_swing_contact_mean
    and landing_force > args.max_landing_force_mean
  ):
    return PauseDecision(
      True,
      "swing contact and landing force both exceed conservative thresholds "
      f"after iter {iteration}",
    )

  return PauseDecision(False, "within monitor thresholds")


def terminate_pids(pids: tuple[int, ...], reason: str) -> None:
  for pid in pids:
    print(f"[PAUSE] Sending SIGTERM to pid={pid}: {reason}", flush=True)
    os.kill(pid, signal.SIGTERM)


def check_once(args: argparse.Namespace) -> int:
  exit_code = 0
  for run_name in args.run_name:
    snapshot = snapshot_run(args.log_root, args.wandb_dir, run_name)
    decision = should_pause(snapshot, args)
    print(
      "[CHECK] "
      f"run={run_name} iter={snapshot.latest_iteration} "
      f"ckpt={snapshot.latest_checkpoint_iteration} "
      f"event_age_s={snapshot.seconds_since_event_update} "
      f"pids={list(snapshot.pids)} decision={decision.reason}",
      flush=True,
    )
    if decision.should_pause:
      exit_code = 2
      if snapshot.pids:
        terminate_pids(snapshot.pids, decision.reason)
      else:
        print(f"[PAUSE] No matching training pid found for run={run_name}", flush=True)
  return exit_code


def main() -> None:
  parser = argparse.ArgumentParser(description="Monitor contact-aware mimic training.")
  parser.add_argument(
    "--log-root",
    type=Path,
    default=Path("logs/rsl_rl/g1_tracking"),
  )
  parser.add_argument("--wandb-dir", type=Path, default=Path("wandb"))
  parser.add_argument("--run-name", action="append", required=True)
  parser.add_argument("--interval-sec", type=int, default=900)
  parser.add_argument("--once", action="store_true")
  parser.add_argument("--stale-seconds", type=int, default=1800)
  parser.add_argument("--quality-min-iteration", type=int, default=5000)
  parser.add_argument("--min-mean-episode-length", type=float, default=60.0)
  parser.add_argument("--max-swing-contact-mean", type=float, default=0.9)
  parser.add_argument("--max-landing-force-mean", type=float, default=700.0)
  args = parser.parse_args()

  if args.once:
    raise SystemExit(check_once(args))

  while True:
    check_once(args)
    time.sleep(args.interval_sec)


if __name__ == "__main__":
  main()
