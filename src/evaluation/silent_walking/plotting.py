"""Plotting helpers for silent walking evaluation traces."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from .types import EpisodeMetricTrace


def save_trace_plot(
  trace: EpisodeMetricTrace,
  output_path: str | Path,
  title: str,
  dt: float,
) -> Path:
  """Save a compact multi-panel plot for one silent-walking rollout trace."""

  output = Path(output_path)
  output.parent.mkdir(parents=True, exist_ok=True)

  time_axis = torch.arange(trace.foot_z_force_n.shape[0], dtype=torch.float32) * dt
  num_rows = 5

  fig, axes = plt.subplots(num_rows, 1, figsize=(12, 2.4 * num_rows), sharex=True)
  fig.suptitle(title)

  axes[0].plot(time_axis, trace.foot_z_force_n[:, 0].cpu(), label="left foot Fz")
  axes[0].plot(time_axis, trace.foot_z_force_n[:, 1].cpu(), label="right foot Fz")
  axes[0].set_ylabel("Force (N)")
  axes[0].legend(loc="upper right")
  axes[0].grid(True, alpha=0.3)

  axes[1].step(time_axis, trace.foot_contact_flag[:, 0].cpu(), where="post", label="left contact")
  axes[1].step(time_axis, trace.foot_contact_flag[:, 1].cpu(), where="post", label="right contact")
  axes[1].set_ylabel("Contact")
  axes[1].legend(loc="upper right")
  axes[1].grid(True, alpha=0.3)

  axes[2].plot(time_axis, trace.command_velocity[:, 0].cpu(), label="cmd vx")
  axes[2].plot(time_axis, trace.actual_linear_velocity[:, 0].cpu(), label="actual vx")
  axes[2].plot(time_axis, trace.foot_vertical_velocity_m_s[:, 0].cpu(), label="left vz")
  axes[2].plot(time_axis, trace.foot_vertical_velocity_m_s[:, 1].cpu(), label="right vz")
  axes[2].set_ylabel("Foot Vz")
  axes[2].legend(loc="upper right", ncol=3)
  axes[2].grid(True, alpha=0.3)

  axes[3].plot(time_axis, trace.command_velocity[:, 0].cpu(), label="cmd vx")
  axes[3].plot(time_axis, trace.actual_linear_velocity[:, 0].cpu(), label="actual vx")
  axes[3].plot(time_axis, trace.command_velocity[:, 2].cpu(), label="cmd wz")
  axes[3].plot(time_axis, trace.actual_yaw_rate.cpu(), label="actual wz")
  axes[3].set_ylabel("Velocity")
  axes[3].legend(loc="upper right", ncol=2)
  axes[3].grid(True, alpha=0.3)

  axes[4].plot(time_axis, trace.action_rate_l2.cpu(), label="action rate")
  axes[4].plot(time_axis, trace.linear_velocity_error.cpu(), label="lin vel err")
  axes[4].plot(time_axis, trace.yaw_rate_error.cpu(), label="yaw rate err")
  axes[4].set_ylabel("Errors")
  axes[4].legend(loc="upper right", ncol=3)
  axes[4].grid(True, alpha=0.3)

  axes[4].set_xlabel("Time (s)")

  fig.tight_layout()
  fig.savefig(output, dpi=150)
  plt.close(fig)
  return output


def save_capsule_distribution_plot(
  trace: EpisodeMetricTrace,
  output_path: str | Path,
  title: str,
) -> Path:
  """Save a single 2D footprint distribution plot colored by touchdown peak."""

  output = Path(output_path)
  output.parent.mkdir(parents=True, exist_ok=True)

  fig, axis = plt.subplots(1, 1, figsize=(7, 6))
  fig.suptitle(title)

  capsule_xy = trace.capsule_layout_xy_m.cpu().clone()
  capsule_xy = _normalize_capsule_layout(capsule_xy, trace.capsule_names)
  values = trace.capsule_touchdown_peak_force_bw.cpu()
  scatter = axis.scatter(
    capsule_xy[:, 0],
    capsule_xy[:, 1],
    c=values,
    s=130 + 25 * trace.capsule_touchdown_count.cpu(),
    cmap="viridis",
    edgecolors="black",
    linewidths=0.8,
  )
  for idx, (name, xy, value) in enumerate(
    zip(trace.capsule_names, capsule_xy, values, strict=True),
    start=1,
  ):
    capsule_idx = idx if "left" in name else idx - 7
    axis.text(
      xy[0],
      xy[1],
      f"{capsule_idx}\n{value:.2f}",
      ha="center",
      va="center",
      fontsize=8,
      color="white",
    )

  axis.axvline(0.0, color="black", linewidth=0.8, linestyle="--", alpha=0.4)
  axis.set_title("Touchdown Peak / BW")
  axis.set_xlabel("Fore-Aft Offset (m)")
  axis.set_ylabel("Lateral Offset (m)")
  axis.set_aspect("equal", adjustable="box")
  axis.grid(True, alpha=0.2)
  fig.colorbar(scatter, ax=axis, shrink=0.85, label="Peak / BW")

  fig.tight_layout()
  fig.savefig(output, dpi=150)
  plt.close(fig)
  return output


def save_capsule_force_timeseries_plot(
  trace: EpisodeMetricTrace,
  output_path: str | Path,
  title: str,
  dt: float,
) -> Path:
  """Save per-capsule world-z force time series for left and right feet."""

  output = Path(output_path)
  output.parent.mkdir(parents=True, exist_ok=True)

  time_axis = torch.arange(trace.capsule_z_force_n.shape[0], dtype=torch.float32) * dt
  fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, sharey=True)
  fig.suptitle(title)

  capsule_xy = _normalize_capsule_layout(trace.capsule_layout_xy_m.cpu().clone(), trace.capsule_names)
  order = torch.argsort(capsule_xy[:, 0])

  for axis, side in zip(axes, ("left", "right"), strict=True):
    side_indices = [idx for idx in order.tolist() if side in trace.capsule_names[idx]]
    for idx in side_indices:
      name = trace.capsule_names[idx]
      series = trace.capsule_z_force_n[:, idx].cpu()
      axis.plot(time_axis, series, label=name.replace("_collision", ""))
    axis.set_title(f"{side.capitalize()} foot capsule Fz")
    axis.set_ylabel("Force (N)")
    axis.grid(True, alpha=0.3)
    axis.legend(loc="upper right", ncol=2, fontsize=8)

  axes[-1].set_xlabel("Time (s)")
  fig.tight_layout()
  fig.savefig(output, dpi=150)
  plt.close(fig)
  return output


def _normalize_capsule_layout(
  capsule_xy: torch.Tensor,
  capsule_names: tuple[str, ...],
) -> torch.Tensor:
  """Center left/right foot capsule positions and offset them for plotting."""

  if capsule_xy.numel() == 0:
    return capsule_xy

  normalized = capsule_xy.clone()
  for side, offset in (("left", -0.12), ("right", 0.12)):
    indices = [idx for idx, name in enumerate(capsule_names) if side in name]
    if not indices:
      continue
    side_xy = normalized[indices]
    side_xy = side_xy - side_xy.mean(dim=0, keepdim=True)
    side_xy[:, 0] += offset
    normalized[indices] = side_xy
  return normalized
