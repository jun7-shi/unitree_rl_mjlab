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

  fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
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
  axes[2].plot(time_axis, trace.command_velocity[:, 2].cpu(), label="cmd wz")
  axes[2].plot(time_axis, trace.actual_yaw_rate.cpu(), label="actual wz")
  axes[2].set_ylabel("Velocity")
  axes[2].legend(loc="upper right", ncol=2)
  axes[2].grid(True, alpha=0.3)

  axes[3].plot(time_axis, trace.action_rate_l2.cpu(), label="action rate")
  axes[3].plot(time_axis, trace.linear_velocity_error.cpu(), label="lin vel err")
  axes[3].plot(time_axis, trace.yaw_rate_error.cpu(), label="yaw rate err")
  axes[3].set_ylabel("Errors")
  axes[3].set_xlabel("Time (s)")
  axes[3].legend(loc="upper right", ncol=3)
  axes[3].grid(True, alpha=0.3)

  fig.tight_layout()
  fig.savefig(output, dpi=150)
  plt.close(fig)
  return output
