"""Training log readers for mimic analysis."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any


_CHECKPOINT_RE = re.compile(r"^model_(\d+)\.pt$")


def checkpoint_iteration(path: str | Path) -> int | None:
  """Return the iteration encoded in a RSL-RL checkpoint filename."""

  match = _CHECKPOINT_RE.match(Path(path).name)
  return int(match.group(1)) if match else None


def list_checkpoints(run_dir: str | Path) -> list[Path]:
  """List model checkpoints sorted by training iteration."""

  root = Path(run_dir)
  checkpoints = [
    path
    for path in root.glob("model_*.pt")
    if checkpoint_iteration(path) is not None
  ]
  return sorted(checkpoints, key=lambda path: checkpoint_iteration(path) or -1)


def read_wandb_summaries(wandb_dir: str | Path) -> list[dict[str, Any]]:
  """Read local W&B summary JSON files."""

  root = Path(wandb_dir)
  if not root.exists():
    return []
  summaries: list[dict[str, Any]] = []
  for summary_path in sorted(root.glob("run-*/files/wandb-summary.json")):
    with summary_path.open("r", encoding="utf-8") as handle:
      summary = json.load(handle)
    summaries.append(
      {
        "run_dir": summary_path.parents[1].name,
        "summary_path": str(summary_path),
        "summary": summary,
      }
    )
  return summaries


def read_tensorboard_scalars(run_dir: str | Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
  """Read scalar rows from TensorBoard event files."""

  root = Path(run_dir)
  event_files = sorted(root.glob("events.out.tfevents*"))
  metadata: dict[str, Any] = {
    "event_files": [str(path) for path in event_files],
    "warning": None,
  }
  if not event_files:
    metadata["warning"] = f"No TensorBoard event files found in {root}"
    return [], metadata

  try:
    from tensorboard.backend.event_processing.event_accumulator import (
      EventAccumulator,
    )
  except ModuleNotFoundError as exc:
    metadata["warning"] = f"TensorBoard is not available: {exc.name}"
    return [], metadata

  rows: list[dict[str, Any]] = []
  for event_file in event_files:
    accumulator = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
    accumulator.Reload()
    for tag in accumulator.Tags().get("scalars", []):
      for event in accumulator.Scalars(tag):
        rows.append(
          {
            "tag": tag,
            "step": int(event.step),
            "wall_time": float(event.wall_time),
            "value": float(event.value),
            "event_file": str(event_file),
          }
        )
  return rows, metadata


def scalar_tag_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
  """Summarize scalar tags by first/last step and latest value."""

  by_tag: dict[str, list[dict[str, Any]]] = {}
  for row in rows:
    by_tag.setdefault(str(row["tag"]), []).append(row)

  summary: list[dict[str, Any]] = []
  for tag, tag_rows in sorted(by_tag.items()):
    ordered = sorted(tag_rows, key=lambda row: int(row["step"]))
    summary.append(
      {
        "tag": tag,
        "count": len(ordered),
        "first_step": int(ordered[0]["step"]),
        "last_step": int(ordered[-1]["step"]),
        "latest_value": float(ordered[-1]["value"]),
      }
    )
  return summary


def downsample_scalar_rows(
  rows: list[dict[str, Any]],
  max_rows_per_tag: int = 1000,
) -> list[dict[str, Any]]:
  """Downsample scalar rows per tag for manageable CSV output."""

  if max_rows_per_tag <= 0:
    raise ValueError("max_rows_per_tag must be positive")
  by_tag: dict[str, list[dict[str, Any]]] = {}
  for row in rows:
    by_tag.setdefault(str(row["tag"]), []).append(row)

  output: list[dict[str, Any]] = []
  for tag_rows in by_tag.values():
    ordered = sorted(tag_rows, key=lambda row: int(row["step"]))
    if len(ordered) <= max_rows_per_tag:
      output.extend(ordered)
      continue
    stride = max(1, len(ordered) // max_rows_per_tag)
    sampled = ordered[::stride]
    if sampled[-1] is not ordered[-1]:
      sampled.append(ordered[-1])
    output.extend(sampled)
  return sorted(output, key=lambda row: (str(row["tag"]), int(row["step"])))
