"""Plotting helpers for silent walking evaluation traces."""

from __future__ import annotations

from pathlib import Path

from matplotlib import cm
from matplotlib import colors
from matplotlib.patches import Polygon
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

  capsule_outline = trace.capsule_outline_fromto_xy_m.cpu().clone()
  capsule_outline = _normalize_capsule_outline(capsule_outline, trace.capsule_names)
  capsule_xy = capsule_outline.mean(dim=1)
  values = trace.capsule_touchdown_peak_force_bw.cpu()
  radii = trace.capsule_radius_m.cpu()
  norm = colors.Normalize(
    vmin=float(values.min().item()) if values.numel() else 0.0,
    vmax=float(values.max().item()) if values.numel() else 1.0,
  )
  cmap = cm.get_cmap("viridis")

  for idx, (name, fromto_xy, radius, value) in enumerate(
    zip(trace.capsule_names, capsule_outline, radii, values, strict=True),
    start=1,
  ):
    polygon_xy = _capsule_outline_polygon(fromto_xy[0], fromto_xy[1], float(radius.item()))
    patch = Polygon(
      polygon_xy.tolist(),
      closed=True,
      facecolor=cmap(norm(float(value.item()))),
      edgecolor="black",
      linewidth=0.9,
    )
    axis.add_patch(patch)
    capsule_idx = idx if "left" in name else idx - 7
    center_xy = fromto_xy.mean(dim=0)
    axis.text(
      float(center_xy[0].item()),
      float(center_xy[1].item()),
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
  padding = 0.025
  if capsule_outline.numel():
    axis.set_xlim(
      float(capsule_outline[..., 0].min().item()) - padding,
      float(capsule_outline[..., 0].max().item()) + padding,
    )
    axis.set_ylim(
      float(capsule_outline[..., 1].min().item()) - padding,
      float(capsule_outline[..., 1].max().item()) + padding,
    )
  scalar_mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
  scalar_mappable.set_array(values.numpy() if values.numel() else [])
  fig.colorbar(scalar_mappable, ax=axis, shrink=0.85, label="Peak / BW")

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


def _normalize_capsule_outline(
  capsule_outline_xy: torch.Tensor,
  capsule_names: tuple[str, ...],
) -> torch.Tensor:
  if capsule_outline_xy.numel() == 0:
    return capsule_outline_xy

  normalized = capsule_outline_xy.clone()
  centers = normalized.mean(dim=1)
  centered = _normalize_capsule_layout(centers, capsule_names)
  deltas = centered - centers
  normalized = normalized + deltas[:, None, :]
  return normalized


def _capsule_outline_polygon(
  start_xy: torch.Tensor,
  end_xy: torch.Tensor,
  radius: float,
  arc_points: int = 16,
) -> torch.Tensor:
  axis = end_xy - start_xy
  length = torch.linalg.norm(axis)
  if length <= 1e-8:
    angles = torch.linspace(0.0, 2.0 * torch.pi, arc_points * 2)
    unit_circle = torch.stack((torch.cos(angles), torch.sin(angles)), dim=1)
    return start_xy + radius * unit_circle

  tangent = axis / length
  normal = torch.tensor((-tangent[1].item(), tangent[0].item()), dtype=torch.float32)
  theta = torch.atan2(tangent[1], tangent[0])
  start_arc = torch.stack(
    (
      start_xy[0] + radius * torch.cos(torch.linspace(theta + torch.pi / 2, theta + 3 * torch.pi / 2, arc_points)),
      start_xy[1] + radius * torch.sin(torch.linspace(theta + torch.pi / 2, theta + 3 * torch.pi / 2, arc_points)),
    ),
    dim=1,
  )
  end_arc = torch.stack(
    (
      end_xy[0] + radius * torch.cos(torch.linspace(theta - torch.pi / 2, theta + torch.pi / 2, arc_points)),
      end_xy[1] + radius * torch.sin(torch.linspace(theta - torch.pi / 2, theta + torch.pi / 2, arc_points)),
    ),
    dim=1,
  )
  side_a = torch.stack((start_xy + radius * normal, end_xy + radius * normal))
  side_b = torch.stack((end_xy - radius * normal, start_xy - radius * normal))
  return torch.cat((side_a[:1], end_arc, side_b[:1], start_arc), dim=0)
