"""Plotting helpers for silent walking evaluation traces."""

from __future__ import annotations

from pathlib import Path

from matplotlib import animation
from matplotlib import cm
from matplotlib import colors
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
import torch

from .foot_grid import summarize_post_contact_foot_grid
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


def save_foot_grid_post_contact_heatmap(
  trace: EpisodeMetricTrace,
  output_path: str | Path,
  title: str,
  *,
  post_step: int = 0,
) -> Path:
  """Save per-foot heatmaps of post-contact virtual grid-point downward speed."""

  output = Path(output_path)
  output.parent.mkdir(parents=True, exist_ok=True)
  rows = summarize_post_contact_foot_grid(trace, post_steps=(post_step,))
  foot_names = tuple(dict.fromkeys(row.foot for row in rows))
  if not rows or not foot_names:
    fig, axis = plt.subplots(1, 1, figsize=(6, 4))
    axis.set_title("No foot-grid touchdown data")
    axis.set_axis_off()
    fig.suptitle(title)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output

  grid_rows, grid_cols = trace.foot_grid_shape
  fig, axes = plt.subplots(
    1,
    len(foot_names),
    figsize=(5 * len(foot_names), 4),
    squeeze=False,
    constrained_layout=True,
  )
  fig.suptitle(title)
  values_for_norm = [
    row.down_speed_mean_by_step_m_s.get(post_step, 0.0)
    for row in rows
  ]
  norm = colors.Normalize(
    vmin=0.0,
    vmax=max(values_for_norm) if values_for_norm else 1.0,
  )
  cmap = cm.get_cmap("magma")
  local_xy = trace.foot_grid_local_xy_m.detach().cpu() if trace.foot_grid_local_xy_m is not None else None
  use_rectangular_grid = (
    grid_rows > 0
    and grid_cols > 0
    and grid_rows * grid_cols == len(trace.foot_grid_names)
  )

  for axis, foot in zip(axes[0], foot_names, strict=True):
    foot_rows = sorted(
      (row for row in rows if row.foot == foot),
      key=lambda row: row.point_index,
    )
    values = torch.tensor(
      [row.down_speed_mean_by_step_m_s.get(post_step, 0.0) for row in foot_rows],
      dtype=torch.float32,
    )
    if use_rectangular_grid:
      image = axis.imshow(
        values.reshape(grid_rows, grid_cols).numpy(),
        origin="lower",
        cmap=cmap,
        norm=norm,
        aspect="auto",
      )
    else:
      if local_xy is None:
        raise ValueError("irregular foot-grid heatmap requires local XY point coordinates")
      image = axis.scatter(
        local_xy[:, 0].numpy(),
        local_xy[:, 1].numpy(),
        c=values.numpy(),
        cmap=cmap,
        norm=norm,
        s=90,
        edgecolors="black",
        linewidths=0.4,
      )
      axis.set_aspect("equal", adjustable="box")
    axis.set_title(f"{foot} foot")
    axis.set_xlabel("Local x / fore-aft (m)" if not use_rectangular_grid else "Lateral grid index")
    axis.set_ylabel("Local y / lateral (m)" if not use_rectangular_grid else "Fore-aft grid index")
    if use_rectangular_grid:
      shaped_values = values.reshape(grid_rows, grid_cols)
      for row_index in range(grid_rows):
        for col_index in range(grid_cols):
          axis.text(
            col_index,
            row_index,
            f"{shaped_values[row_index, col_index].item():.2f}",
            ha="center",
            va="center",
            fontsize=7,
            color="white" if shaped_values[row_index, col_index].item() > norm.vmax * 0.45 else "black",
          )
  fig.colorbar(image, ax=axes[0].tolist(), shrink=0.85, label="Downward speed (m/s)")
  fig.savefig(output, dpi=150)
  plt.close(fig)
  return output


def save_foot_grid_velocity_video(
  trace: EpisodeMetricTrace,
  output_path: str | Path,
  title: str,
  dt: float,
  *,
  stride: int = 1,
  fps: int | None = None,
  metric: str = "downward_speed",
  include_pressure: bool = False,
) -> Path:
  """Save an animated foot-grid velocity heatmap over rollout time."""

  if dt <= 0.0:
    raise ValueError("dt must be positive")
  if stride <= 0:
    raise ValueError("stride must be positive")
  if fps is not None and fps <= 0:
    raise ValueError("fps must be positive")
  if metric not in {"downward_speed", "signed_vz", "speed"}:
    raise ValueError("metric must be one of: downward_speed, signed_vz, speed")

  output = Path(output_path)
  output.parent.mkdir(parents=True, exist_ok=True)
  if trace.foot_grid_velocity_m_s is None:
    raise ValueError("trace does not contain foot-grid velocity frames")
  grid_velocity = trace.foot_grid_velocity_m_s.detach().cpu()
  if grid_velocity.ndim != 4 or grid_velocity.shape[-1] != 3:
    raise ValueError("foot_grid_velocity_m_s must have shape [T, F, P, 3]")
  pressure_values = None
  if include_pressure:
    if trace.foot_grid_force_n is None:
      raise ValueError("include_pressure requires foot_grid_force_n in the trace")
    pressure_source = trace.foot_grid_force_n.detach().cpu()
    if pressure_source.shape != grid_velocity.shape[:3]:
      raise ValueError("foot_grid_force_n must have shape [T, F, P]")
    pressure_values = pressure_source[::stride]
  if metric == "downward_speed":
    point_values = torch.clamp(-grid_velocity[..., 2], min=0.0)[::stride]
    cmap = cm.get_cmap("magma")
    vmin = 0.0
    vmax = float(point_values.max().item()) if point_values.numel() else 1.0
    colorbar_label = "Downward speed (m/s)"
  elif metric == "signed_vz":
    point_values = grid_velocity[..., 2][::stride]
    limit = float(point_values.abs().max().item()) if point_values.numel() else 1.0
    cmap = _signed_vz_video_colormap()
    vmin = -max(limit, 1.0e-6)
    vmax = max(limit, 1.0e-6)
    colorbar_label = "Signed Vz (m/s)"
  else:
    point_values = torch.linalg.norm(grid_velocity, dim=-1)[::stride]
    cmap = cm.get_cmap("viridis")
    vmin = 0.0
    vmax = float(point_values.max().item()) if point_values.numel() else 1.0
    colorbar_label = "3D speed (m/s)"

  contact = trace.foot_contact_flag.detach().cpu().bool()[::stride]
  foot_names = ("left", "right")
  vmax = max(vmax, 1.0e-6)
  norm = (
    colors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    if metric == "signed_vz"
    else colors.Normalize(vmin=vmin, vmax=vmax)
  )
  video_fps = int(fps) if fps is not None else max(1, round(1.0 / (dt * stride)))
  grid_rows, grid_cols = trace.foot_grid_shape
  use_rectangular_grid = (
    grid_rows > 0
    and grid_cols > 0
    and grid_rows * grid_cols == point_values.shape[2]
  )
  if use_rectangular_grid:
    frames = point_values.reshape(point_values.shape[0], point_values.shape[1], grid_rows, grid_cols)
    pressure_frames = (
      pressure_values.reshape(pressure_values.shape[0], pressure_values.shape[1], grid_rows, grid_cols)
      if pressure_values is not None
      else None
    )
  else:
    frames = None
    pressure_frames = None
  local_xy = trace.foot_grid_local_xy_m.detach().cpu() if trace.foot_grid_local_xy_m is not None else None
  foot_count = point_values.shape[1]
  row_count = 2 if include_pressure else 1
  pressure_cmap = _pressure_video_colormap()
  pressure_norm = None
  if pressure_values is not None:
    pressure_norm = colors.Normalize(
      vmin=0.0,
      vmax=max(float(pressure_values.max().item()) if pressure_values.numel() else 1.0, 1.0e-6),
    )

  fig, axes = plt.subplots(
    row_count,
    foot_count,
    figsize=(5 * foot_count, 4 * row_count),
    squeeze=False,
    constrained_layout=True,
  )
  fig.suptitle(title)
  artists = []
  pressure_artists = []
  contact_badges = []
  for foot_idx, axis in enumerate(axes[0]):
    if use_rectangular_grid:
      artist = axis.imshow(
        frames[0, foot_idx].numpy(),
        origin="lower",
        cmap=cmap,
        norm=norm,
        aspect="auto",
        animated=True,
      )
      axis.set_xlabel("Lateral grid index")
      axis.set_ylabel("Fore-aft grid index")
    else:
      if local_xy is None:
        raise ValueError("irregular foot-grid video requires local XY point coordinates")
      artist = axis.scatter(
        local_xy[:, 0].numpy(),
        local_xy[:, 1].numpy(),
        c=point_values[0, foot_idx].numpy(),
        cmap=cmap,
        norm=norm,
        s=90,
        edgecolors="black",
        linewidths=0.4,
        animated=True,
      )
      axis.set_aspect("equal", adjustable="box")
      axis.set_xlabel("Local x / fore-aft (m)")
      axis.set_ylabel("Local y / lateral (m)")
    badge = axis.text(
      0.03,
      0.96,
      "",
      transform=axis.transAxes,
      ha="left",
      va="top",
      fontsize=12,
      fontweight="bold",
      color="white",
      bbox={
        "boxstyle": "round,pad=0.35",
        "facecolor": "#6b7280",
        "edgecolor": "white",
        "linewidth": 1.2,
        "alpha": 0.95,
      },
    )
    artists.append(artist)
    contact_badges.append(badge)
  fig.colorbar(artists[0], ax=axes[0].tolist(), shrink=0.85, label=colorbar_label)
  if include_pressure and pressure_values is not None and pressure_norm is not None:
    for foot_idx, axis in enumerate(axes[1]):
      if use_rectangular_grid:
        pressure_artist = axis.imshow(
          pressure_frames[0, foot_idx].numpy(),
          origin="lower",
          cmap=pressure_cmap,
          norm=pressure_norm,
          aspect="auto",
          animated=True,
        )
        axis.set_xlabel("Lateral grid index")
        axis.set_ylabel("Fore-aft grid index")
      else:
        if local_xy is None:
          raise ValueError("irregular foot-grid pressure row requires local XY point coordinates")
        pressure_artist = axis.scatter(
          local_xy[:, 0].numpy(),
          local_xy[:, 1].numpy(),
          c=pressure_values[0, foot_idx].numpy(),
          cmap=pressure_cmap,
          norm=pressure_norm,
          s=90,
          edgecolors="black",
          linewidths=0.4,
          animated=True,
        )
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("Local x / fore-aft (m)")
        axis.set_ylabel("Local y / lateral (m)")
      foot_name = foot_names[foot_idx] if foot_idx < len(foot_names) else f"foot {foot_idx}"
      axis.set_title(f"{foot_name} foot force proxy")
      pressure_artists.append(pressure_artist)
    fig.colorbar(
      pressure_artists[0],
      ax=axes[1].tolist(),
      shrink=0.85,
      label="Normal force proxy (N)",
    )

  def update(frame_idx: int):
    time_s = frame_idx * stride * dt
    for foot_idx, artist in enumerate(artists):
      if use_rectangular_grid:
        artist.set_array(frames[frame_idx, foot_idx].numpy())
      else:
        artist.set_array(point_values[frame_idx, foot_idx].numpy())
      foot_name = foot_names[foot_idx] if foot_idx < len(foot_names) else f"foot {foot_idx}"
      is_contact = bool(contact[frame_idx, foot_idx].item())
      contact_label = "CONTACT" if is_contact else "AIR"
      badge = contact_badges[foot_idx]
      badge.set_text(contact_label)
      badge.get_bbox_patch().set_facecolor("#dc2626" if is_contact else "#6b7280")
      axes[0, foot_idx].set_title(f"{foot_name} foot | t={time_s:.2f}s")
    for foot_idx, pressure_artist in enumerate(pressure_artists):
      if use_rectangular_grid:
        pressure_artist.set_array(pressure_frames[frame_idx, foot_idx].numpy())
      else:
        pressure_artist.set_array(pressure_values[frame_idx, foot_idx].numpy())
    return artists + pressure_artists + contact_badges

  movie = animation.FuncAnimation(
    fig,
    update,
    frames=point_values.shape[0],
    interval=1000.0 / video_fps,
    blit=False,
  )
  suffix = output.suffix.lower()
  if suffix == ".gif":
    writer = animation.PillowWriter(fps=video_fps)
  elif suffix == ".mp4":
    if not animation.writers.is_available("ffmpeg"):
      plt.close(fig)
      raise RuntimeError("Saving MP4 foot-grid video requires ffmpeg; use .gif instead")
    writer = animation.FFMpegWriter(fps=video_fps)
  else:
    plt.close(fig)
    raise ValueError("foot-grid video output must end with .gif or .mp4")
  movie.save(output, writer=writer)
  plt.close(fig)
  return output


def _signed_vz_video_colormap() -> LinearSegmentedColormap:
  """Return a high-contrast signed vertical velocity map: negative red, zero white, positive blue."""

  return LinearSegmentedColormap.from_list(
    "signed_vz_red_white_blue",
    [(0.0, "#b91c1c"), (0.5, "#ffffff"), (1.0, "#1d4ed8")],
    N=257,
  )


def _pressure_video_colormap() -> LinearSegmentedColormap:
  """Return a force-proxy map with exact white at zero force."""

  return LinearSegmentedColormap.from_list(
    "pressure_white_yellow_red",
    [
      (0.0, "#ffffff"),
      (0.25, "#fde68a"),
      (0.65, "#f97316"),
      (1.0, "#991b1b"),
    ],
    N=257,
  )


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
