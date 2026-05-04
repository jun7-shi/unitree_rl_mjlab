"""Reporting utilities for mimic failure analysis."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable


def write_json(path: str | Path, data: dict[str, Any]) -> Path:
  """Write pretty JSON data."""

  output = Path(path)
  output.parent.mkdir(parents=True, exist_ok=True)
  with output.open("w", encoding="utf-8") as handle:
    json.dump(data, handle, ensure_ascii=False, indent=2)
    handle.write("\n")
  return output


def write_csv(
  path: str | Path,
  rows: Iterable[dict[str, Any]],
  fieldnames: list[str] | None = None,
) -> Path:
  """Write dictionaries to CSV."""

  output = Path(path)
  output.parent.mkdir(parents=True, exist_ok=True)
  materialized = list(rows)
  if fieldnames is None:
    fieldnames = sorted({key for row in materialized for key in row})
  with output.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for row in materialized:
      writer.writerow(row)
  return output


def diagnosis_lines(
  reference_summary: dict[str, Any],
  contact_comparison: dict[str, Any] | None,
  rollout_error: str | None = None,
) -> list[str]:
  """Generate simple evidence-driven diagnosis lines."""

  lines: list[str] = []
  single_support = float(reference_summary.get("reference_single_support_ratio", 0.0))
  min_clearance = float(reference_summary.get("reference_min_swing_clearance_m", 0.0))
  if single_support > 0.35:
    lines.append(
      f"Reference single-support ratio is high ({single_support:.3f}); balance-sensitive phase tracking is likely important."
    )
  if min_clearance < 0.03:
    lines.append(
      f"Reference minimum swing clearance is low ({min_clearance:.4f} m); small tracking errors can become early contacts."
    )
  if contact_comparison is not None:
    swing_contact = float(contact_comparison.get("swing_contact_ratio", 0.0))
    extra_touchdowns = int(contact_comparison.get("extra_touchdown_count", 0))
    survival = float(contact_comparison.get("single_support_survival", 1.0))
    if swing_contact > 0.1:
      lines.append(
        f"Policy contacts the ground during reference swing windows ({swing_contact:.3f}); this directly matches the padding-step symptom."
      )
    if extra_touchdowns > 0:
      lines.append(
        f"Policy has {extra_touchdowns} extra touchdown events relative to reference; contact timing should be inspected."
      )
    if survival < 0.8:
      lines.append(
        f"Policy preserves only {survival:.3f} of reference single-support samples; this supports the single-leg support failure hypothesis."
      )
  if rollout_error:
    lines.append(f"Policy rollout was not completed in this run: {rollout_error}")
  if not lines:
    lines.append("No strong gait failure signature was detected by the first-pass metrics.")
  return lines


def render_markdown_report(
  *,
  title: str,
  inputs: dict[str, Any],
  reference_summary: dict[str, Any],
  training_summary: list[dict[str, Any]],
  contact_comparison: dict[str, Any] | None = None,
  checkpoint_scan: list[dict[str, Any]] | None = None,
  rollout_error: str | None = None,
) -> str:
  """Render a compact Markdown analysis report."""

  lines = [f"# {title}", ""]
  lines.extend(["## Inputs", ""])
  for key, value in inputs.items():
    lines.append(f"- `{key}`: `{value}`")
  lines.extend(["", "## Reference Gait", ""])
  for key in sorted(reference_summary):
    lines.append(f"- `{key}`: `{reference_summary[key]}`")
  if contact_comparison is not None:
    lines.extend(["", "## Policy Contact Comparison", ""])
    for key in sorted(contact_comparison):
      lines.append(f"- `{key}`: `{contact_comparison[key]}`")
  if checkpoint_scan:
    lines.extend(["", "## Checkpoint Scan", ""])
    for row in checkpoint_scan:
      lines.append(
        "- iteration `{iteration}`: swing_contact_ratio `{swing:.3f}`, "
        "extra_touchdown_count `{extra}`, single_support_survival `{survival:.3f}`".format(
          iteration=row["iteration"],
          swing=float(row["swing_contact_ratio"]),
          extra=int(row["extra_touchdown_count"]),
          survival=float(row["single_support_survival"]),
        )
      )
  lines.extend(["", "## Training Curves", ""])
  if training_summary:
    for row in training_summary[:40]:
      lines.append(
        f"- `{row['tag']}`: latest `{row['latest_value']}` at step `{row['last_step']}`"
      )
  else:
    lines.append("- No TensorBoard scalar rows were available.")
  lines.extend(["", "## Diagnosis", ""])
  for line in diagnosis_lines(reference_summary, contact_comparison, rollout_error):
    lines.append(f"- {line}")
  lines.append("")
  return "\n".join(lines)


def save_contact_raster_plot(
  path: str | Path,
  reference_contact,
  policy_contact=None,
  title: str = "Foot contact raster",
) -> Path:
  """Save a compact contact raster plot."""

  import matplotlib.pyplot as plt
  import numpy as np

  output = Path(path)
  output.parent.mkdir(parents=True, exist_ok=True)
  panels = [("reference", np.asarray(reference_contact, dtype=float))]
  if policy_contact is not None:
    panels.append(("policy", np.asarray(policy_contact, dtype=float)))
  fig, axes = plt.subplots(len(panels), 1, figsize=(12, 2.2 * len(panels)), sharex=True)
  if len(panels) == 1:
    axes = [axes]
  for ax, (name, contacts) in zip(axes, panels):
    ax.imshow(contacts.T, aspect="auto", interpolation="nearest", cmap="Greys")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["left", "right"])
    ax.set_ylabel(name)
  axes[0].set_title(title)
  axes[-1].set_xlabel("frame")
  fig.tight_layout()
  fig.savefig(output, dpi=150)
  plt.close(fig)
  return output


def save_clearance_plot(
  path: str | Path,
  clearance_m,
  fps: float,
  title: str = "Foot clearance",
) -> Path:
  """Save left/right foot clearance time series."""

  import matplotlib.pyplot as plt
  import numpy as np

  output = Path(path)
  output.parent.mkdir(parents=True, exist_ok=True)
  clearance = np.asarray(clearance_m, dtype=float)
  time = np.arange(clearance.shape[0]) / fps
  fig, ax = plt.subplots(figsize=(12, 4))
  ax.plot(time, clearance[:, 0], label="left")
  ax.plot(time, clearance[:, 1], label="right")
  ax.set_title(title)
  ax.set_xlabel("time (s)")
  ax.set_ylabel("clearance (m)")
  ax.legend()
  fig.tight_layout()
  fig.savefig(output, dpi=150)
  plt.close(fig)
  return output


def save_training_curve_plot(
  path: str | Path,
  scalar_rows: list[dict[str, Any]],
  tag_limit: int = 8,
) -> Path | None:
  """Save a multi-tag TensorBoard scalar plot."""

  if not scalar_rows:
    return None
  import matplotlib.pyplot as plt

  output = Path(path)
  output.parent.mkdir(parents=True, exist_ok=True)
  by_tag: dict[str, list[dict[str, Any]]] = {}
  for row in scalar_rows:
    by_tag.setdefault(str(row["tag"]), []).append(row)
  selected = sorted(by_tag)[:tag_limit]
  fig, ax = plt.subplots(figsize=(12, 5))
  for tag in selected:
    rows = sorted(by_tag[tag], key=lambda row: int(row["step"]))
    ax.plot([row["step"] for row in rows], [row["value"] for row in rows], label=tag)
  ax.set_xlabel("step")
  ax.set_ylabel("value")
  ax.set_title("Training scalar curves")
  ax.legend(fontsize=8)
  fig.tight_layout()
  fig.savefig(output, dpi=150)
  plt.close(fig)
  return output


def save_checkpoint_scan_plot(
  path: str | Path,
  checkpoint_rows: list[dict[str, Any]],
) -> Path | None:
  """Save gait-quality metrics over checkpoint iterations."""

  if not checkpoint_rows:
    return None
  import matplotlib.pyplot as plt

  output = Path(path)
  output.parent.mkdir(parents=True, exist_ok=True)
  ordered = sorted(checkpoint_rows, key=lambda row: int(row["iteration"]))
  iterations = [int(row["iteration"]) for row in ordered]
  fig, ax = plt.subplots(figsize=(10, 4))
  ax.plot(
    iterations,
    [float(row["swing_contact_ratio"]) for row in ordered],
    marker="o",
    label="swing_contact_ratio",
  )
  ax.plot(
    iterations,
    [float(row["single_support_survival"]) for row in ordered],
    marker="o",
    label="single_support_survival",
  )
  ax.set_xlabel("iteration")
  ax.set_ylabel("ratio")
  ax.set_ylim(0.0, 1.05)
  ax.set_title("Checkpoint gait-quality scan")
  ax.legend()
  ax2 = ax.twinx()
  ax2.plot(
    iterations,
    [int(row["extra_touchdown_count"]) for row in ordered],
    color="tab:red",
    marker="x",
    linestyle="--",
    label="extra_touchdown_count",
  )
  ax2.set_ylabel("extra touchdowns")
  fig.tight_layout()
  fig.savefig(output, dpi=150)
  plt.close(fig)
  return output
