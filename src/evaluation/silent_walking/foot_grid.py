"""Foot-grid velocity helpers for post-contact silent-walking analysis."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import torch

from mjlab.utils.lab_api.math import quat_apply

from .types import EpisodeMetricTrace

FOOT_NAMES = ("left", "right")


@dataclass(frozen=True, slots=True)
class FootGridHeatmapRow:
  """Post-contact downward-speed aggregate for one virtual foot-grid point."""

  foot: str
  point_name: str
  point_index: int
  local_x_m: float
  local_y_m: float
  event_count: int
  down_speed_mean_by_step_m_s: dict[int, float]
  down_speed_p95_by_step_m_s: dict[int, float]


def make_rectangular_foot_grid_offsets(
  *,
  x_range_m: tuple[float, float],
  y_range_m: tuple[float, float],
  z_m: float,
  shape: tuple[int, int],
) -> torch.Tensor:
  """Return uniformly spaced local foot offsets with shape [rows * cols, 3]."""

  rows, cols = shape
  if rows <= 0 or cols <= 0:
    raise ValueError("shape must contain positive row and column counts")
  xs = torch.linspace(float(x_range_m[0]), float(x_range_m[1]), rows)
  ys = torch.linspace(float(y_range_m[0]), float(y_range_m[1]), cols)
  grid_x = xs.repeat_interleave(cols)
  grid_y = ys.repeat(rows)
  grid_z = torch.full_like(grid_x, float(z_m))
  return torch.stack((grid_x, grid_y, grid_z), dim=1)


def make_capsule_footprint_sample_offsets(
  *,
  capsule_fromto_xy_m: tuple[tuple[tuple[float, float], tuple[float, float]], ...],
  z_m: float,
  target_count: int,
  capsule_radius_m: tuple[float, ...] | float = 0.01,
) -> torch.Tensor:
  """Return local offsets approximately uniformly filling the capsule footprint."""

  if target_count <= 0:
    raise ValueError("target_count must be positive")
  if not capsule_fromto_xy_m:
    return torch.zeros((0, 3), dtype=torch.float32)

  fromto = torch.tensor(capsule_fromto_xy_m, dtype=torch.float32)
  radii = _normalize_capsule_radii(
    capsule_radius_m,
    count=fromto.shape[0],
    dtype=fromto.dtype,
    device=fromto.device,
  )
  candidates = _capsule_footprint_candidates(
    capsule_fromto_xy_m=fromto,
    capsule_radius_m=radii,
    target_count=target_count,
  )
  selected = _farthest_point_sample(candidates, target_count)
  selected = _sort_xy(selected)
  z = torch.full((selected.shape[0], 1), float(z_m), dtype=selected.dtype)
  return torch.cat((selected, z), dim=1)


def make_foot_grid_names(shape: tuple[int, int]) -> tuple[str, ...]:
  """Return stable row-major names for a rectangular foot grid."""

  rows, cols = shape
  if rows <= 0 or cols <= 0:
    return ()
  return tuple(f"r{row:02d}_c{col:02d}" for row in range(rows) for col in range(cols))


def make_indexed_foot_grid_names(count: int) -> tuple[str, ...]:
  """Return stable names for an irregular foot-grid point cloud."""

  return tuple(f"p{idx:02d}" for idx in range(count))


def foot_grid_point_velocities(
  *,
  body_pos_w: torch.Tensor,
  body_quat_w: torch.Tensor,
  body_lin_vel_w: torch.Tensor,
  body_ang_vel_w: torch.Tensor,
  local_offsets_m: torch.Tensor,
) -> torch.Tensor:
  """Compute world-frame velocities for virtual local foot-grid points.

  The points are evaluator-local telemetry probes. They are not MuJoCo geoms and
  do not affect contact solving.
  """

  if body_pos_w.ndim != 3 or body_pos_w.shape[-1] != 3:
    raise ValueError("body_pos_w must have shape [B, F, 3]")
  expected_body_shape = body_pos_w.shape
  for name, value, width in (
    ("body_quat_w", body_quat_w, 4),
    ("body_lin_vel_w", body_lin_vel_w, 3),
    ("body_ang_vel_w", body_ang_vel_w, 3),
  ):
    if value.ndim != 3 or value.shape[:2] != expected_body_shape[:2] or value.shape[-1] != width:
      raise ValueError(f"{name} has incompatible shape")

  offsets = _normalize_offsets(
    local_offsets_m.to(dtype=body_pos_w.dtype, device=body_pos_w.device),
    batch_size=body_pos_w.shape[0],
    num_feet=body_pos_w.shape[1],
  )
  batch_size, num_feet, num_points = offsets.shape[:3]
  expanded_quat = body_quat_w.unsqueeze(2).expand(batch_size, num_feet, num_points, 4)
  rotated_offsets = quat_apply(
    expanded_quat.reshape(-1, 4),
    offsets.reshape(-1, 3),
  ).reshape(batch_size, num_feet, num_points, 3)
  linear_velocity = body_lin_vel_w.unsqueeze(2).expand_as(rotated_offsets)
  angular_velocity = body_ang_vel_w.unsqueeze(2).expand_as(rotated_offsets)
  return linear_velocity + torch.cross(angular_velocity, rotated_offsets, dim=-1)


def distribute_capsule_forces_to_foot_grid(
  *,
  capsule_z_force_n: torch.Tensor,
  capsule_fromto_xy_m: tuple[tuple[tuple[float, float], tuple[float, float]], ...],
  capsule_radius_m: tuple[float, ...],
  local_xy_m: torch.Tensor,
  num_feet: int = 2,
) -> torch.Tensor:
  """Distribute per-capsule normal forces onto virtual foot-grid points.

  The returned values are a geometry-based force proxy for visualization. They
  conserve each foot's total capsule force, but they are not independent contact
  sensor readings at the virtual points.
  """

  if capsule_z_force_n.ndim != 2:
    raise ValueError("capsule_z_force_n must have shape [T, F * C]")
  if num_feet <= 0:
    raise ValueError("num_feet must be positive")
  if capsule_z_force_n.shape[1] % num_feet != 0:
    raise ValueError("capsule_z_force_n column count must be divisible by num_feet")
  if local_xy_m.ndim != 2 or local_xy_m.shape[-1] != 2:
    raise ValueError("local_xy_m must have shape [P, 2]")

  capsules_per_foot = capsule_z_force_n.shape[1] // num_feet
  if len(capsule_fromto_xy_m) < capsules_per_foot:
    raise ValueError("capsule geometry does not cover one foot")
  if len(capsule_radius_m) < capsules_per_foot:
    raise ValueError("capsule radii do not cover one foot")

  dtype = capsule_z_force_n.dtype
  device = capsule_z_force_n.device
  local_xy = local_xy_m.to(dtype=dtype, device=device)
  fromto = torch.tensor(
    capsule_fromto_xy_m[:capsules_per_foot],
    dtype=dtype,
    device=device,
  )
  radii = torch.tensor(
    capsule_radius_m[:capsules_per_foot],
    dtype=dtype,
    device=device,
  ).clamp_min(torch.finfo(dtype).eps)
  weights = _capsule_to_grid_weights(fromto, radii, local_xy)
  force_by_foot = capsule_z_force_n.reshape(
    capsule_z_force_n.shape[0],
    num_feet,
    capsules_per_foot,
  )
  return torch.einsum("tfc,cp->tfp", force_by_foot, weights)


def summarize_post_contact_foot_grid(
  trace: EpisodeMetricTrace,
  *,
  post_steps: tuple[int, ...] = (0, 1, 3),
) -> list[FootGridHeatmapRow]:
  """Aggregate virtual grid-point downward speeds after foot touchdown events."""

  if (
    trace.foot_grid_velocity_m_s is None
    or trace.foot_grid_local_xy_m is None
    or not trace.foot_grid_names
  ):
    return []
  if any(step < 0 for step in post_steps):
    raise ValueError("post_steps must be non-negative")

  contact = trace.foot_contact_flag.detach().cpu().bool()
  velocity = trace.foot_grid_velocity_m_s.detach().cpu()
  local_xy = trace.foot_grid_local_xy_m.detach().cpu()
  if velocity.ndim != 4 or velocity.shape[-1] != 3:
    raise ValueError("foot_grid_velocity_m_s must have shape [T, F, P, 3]")
  if contact.shape[:2] != velocity.shape[:2]:
    raise ValueError("foot contact and foot-grid velocity traces are incompatible")
  if local_xy.shape != velocity.shape[2:3] + (2,):
    raise ValueError("foot_grid_local_xy_m must have shape [P, 2]")

  touchdown = _touchdown_mask(contact)
  rows: list[FootGridHeatmapRow] = []
  num_steps, num_feet, num_points = velocity.shape[:3]
  for foot_idx in range(num_feet):
    foot = FOOT_NAMES[foot_idx] if foot_idx < len(FOOT_NAMES) else f"foot_{foot_idx}"
    event_steps = torch.nonzero(touchdown[:, foot_idx], as_tuple=False).flatten()
    for point_idx in range(num_points):
      means: dict[int, float] = {}
      p95s: dict[int, float] = {}
      for post_step in post_steps:
        target_steps = event_steps + post_step
        target_steps = target_steps[target_steps < num_steps]
        if target_steps.numel() == 0:
          means[post_step] = 0.0
          p95s[post_step] = 0.0
          continue
        down_speed = torch.clamp(
          -velocity[target_steps, foot_idx, point_idx, 2].to(torch.float64),
          min=0.0,
        )
        means[post_step] = _clean_float(down_speed.mean())
        p95s[post_step] = _clean_float(torch.quantile(down_speed, 0.95))
      rows.append(
        FootGridHeatmapRow(
          foot=foot,
          point_name=trace.foot_grid_names[point_idx],
          point_index=point_idx,
          local_x_m=_clean_float(local_xy[point_idx, 0]),
          local_y_m=_clean_float(local_xy[point_idx, 1]),
          event_count=int(event_steps.numel()),
          down_speed_mean_by_step_m_s=means,
          down_speed_p95_by_step_m_s=p95s,
        )
      )
  return rows


def foot_grid_down_speed_frames(trace: EpisodeMetricTrace) -> torch.Tensor:
  """Return time-varying foot-grid downward speeds with shape [T, F, rows, cols]."""

  if trace.foot_grid_velocity_m_s is None:
    return torch.zeros((0, 0, 0, 0), dtype=torch.float32)
  velocity = trace.foot_grid_velocity_m_s.detach().cpu()
  if velocity.ndim != 4 or velocity.shape[-1] != 3:
    raise ValueError("foot_grid_velocity_m_s must have shape [T, F, P, 3]")
  rows, cols = trace.foot_grid_shape
  if rows <= 0 or cols <= 0:
    rows, cols = 1, velocity.shape[2]
  if rows * cols != velocity.shape[2]:
    raise ValueError("foot_grid_shape does not match foot-grid point count")
  return torch.clamp(-velocity[..., 2], min=0.0).reshape(
    velocity.shape[0],
    velocity.shape[1],
    rows,
    cols,
  )


def write_foot_grid_heatmap_csv(
  output_path: str | Path,
  rows: list[FootGridHeatmapRow],
  *,
  post_steps: tuple[int, ...] = (0, 1, 3),
) -> Path:
  """Write post-contact foot-grid heatmap rows to CSV."""

  output = Path(output_path)
  output.parent.mkdir(parents=True, exist_ok=True)
  fieldnames = [
    "foot",
    "point_name",
    "point_index",
    "local_x_m",
    "local_y_m",
    "event_count",
  ]
  for post_step in post_steps:
    fieldnames.append(f"post_{post_step}_down_speed_mean_m_s")
    fieldnames.append(f"post_{post_step}_down_speed_p95_m_s")

  with output.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
      record: dict[str, str | int | float] = {
        "foot": row.foot,
        "point_name": row.point_name,
        "point_index": row.point_index,
        "local_x_m": row.local_x_m,
        "local_y_m": row.local_y_m,
        "event_count": row.event_count,
      }
      for post_step in post_steps:
        record[f"post_{post_step}_down_speed_mean_m_s"] = (
          row.down_speed_mean_by_step_m_s.get(post_step, 0.0)
        )
        record[f"post_{post_step}_down_speed_p95_m_s"] = (
          row.down_speed_p95_by_step_m_s.get(post_step, 0.0)
        )
      writer.writerow(record)
  return output


def write_foot_grid_raw_csv(
  trace: EpisodeMetricTrace,
  output_path: str | Path,
  *,
  dt: float,
) -> Path:
  """Write per-frame, per-foot, per-point foot-grid velocity samples."""

  if dt <= 0.0:
    raise ValueError("dt must be positive")
  if (
    trace.foot_grid_velocity_m_s is None
    or trace.foot_grid_local_xy_m is None
    or not trace.foot_grid_names
  ):
    raise ValueError("trace does not contain foot-grid velocity samples")

  velocity = trace.foot_grid_velocity_m_s.detach().cpu()
  local_xy = trace.foot_grid_local_xy_m.detach().cpu()
  contact = trace.foot_contact_flag.detach().cpu().bool()
  force = trace.foot_grid_force_n.detach().cpu() if trace.foot_grid_force_n is not None else None
  if velocity.ndim != 4 or velocity.shape[-1] != 3:
    raise ValueError("foot_grid_velocity_m_s must have shape [T, F, P, 3]")
  if contact.shape[:2] != velocity.shape[:2]:
    raise ValueError("foot contact and foot-grid velocity traces are incompatible")
  if local_xy.shape != velocity.shape[2:3] + (2,):
    raise ValueError("foot_grid_local_xy_m must have shape [P, 2]")
  if force is not None and force.shape != velocity.shape[:3]:
    raise ValueError("foot_grid_force_n must have shape [T, F, P]")

  output = Path(output_path)
  output.parent.mkdir(parents=True, exist_ok=True)
  fieldnames = [
    "step",
    "time_s",
    "foot",
    "foot_index",
    "foot_contact",
    "point_name",
    "point_index",
    "local_x_m",
    "local_y_m",
    "vx_m_s",
    "vy_m_s",
    "vz_m_s",
    "speed_m_s",
    "downward_speed_m_s",
    "normal_force_proxy_n",
  ]
  with output.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for step_idx in range(velocity.shape[0]):
      for foot_idx in range(velocity.shape[1]):
        foot = FOOT_NAMES[foot_idx] if foot_idx < len(FOOT_NAMES) else f"foot_{foot_idx}"
        for point_idx, point_name in enumerate(trace.foot_grid_names):
          vx, vy, vz = velocity[step_idx, foot_idx, point_idx]
          speed = torch.linalg.norm(velocity[step_idx, foot_idx, point_idx])
          writer.writerow(
            {
              "step": step_idx,
              "time_s": _clean_float(torch.tensor(step_idx * dt)),
              "foot": foot,
              "foot_index": foot_idx,
              "foot_contact": int(contact[step_idx, foot_idx].item()),
              "point_name": point_name,
              "point_index": point_idx,
              "local_x_m": _clean_float(local_xy[point_idx, 0]),
              "local_y_m": _clean_float(local_xy[point_idx, 1]),
              "vx_m_s": _clean_float(vx),
              "vy_m_s": _clean_float(vy),
              "vz_m_s": _clean_float(vz),
              "speed_m_s": _clean_float(speed),
              "downward_speed_m_s": _clean_float(torch.clamp(-vz, min=0.0)),
              "normal_force_proxy_n": _clean_float(
                force[step_idx, foot_idx, point_idx]
                if force is not None
                else torch.tensor(0.0)
              ),
            }
          )
  return output


def _capsule_to_grid_weights(
  capsule_fromto_xy_m: torch.Tensor,
  capsule_radius_m: torch.Tensor,
  local_xy_m: torch.Tensor,
) -> torch.Tensor:
  weights = []
  for capsule_idx in range(capsule_fromto_xy_m.shape[0]):
    start_xy = capsule_fromto_xy_m[capsule_idx, 0]
    end_xy = capsule_fromto_xy_m[capsule_idx, 1]
    radius = capsule_radius_m[capsule_idx]
    distance = _distance_to_segment(local_xy_m, start_xy, end_xy)
    cutoff = radius * 2.0
    raw_weight = torch.clamp(1.0 - distance / cutoff, min=0.0)
    if float(raw_weight.sum().item()) <= 0.0:
      raw_weight = torch.zeros_like(distance)
      raw_weight[int(torch.argmin(distance).item())] = 1.0
    weights.append(raw_weight / raw_weight.sum().clamp_min(torch.finfo(raw_weight.dtype).eps))
  return torch.stack(weights, dim=0)


def _capsule_footprint_candidates(
  *,
  capsule_fromto_xy_m: torch.Tensor,
  capsule_radius_m: torch.Tensor,
  target_count: int,
) -> torch.Tensor:
  max_radius = float(capsule_radius_m.max().item())
  min_xy = capsule_fromto_xy_m.amin(dim=(0, 1)) - max_radius
  max_xy = capsule_fromto_xy_m.amax(dim=(0, 1)) + max_radius
  span = torch.clamp(max_xy - min_xy, min=1.0e-6)

  candidates = torch.empty((0, 2), dtype=capsule_fromto_xy_m.dtype)
  for multiplier in (64, 128, 256, 512, 1024):
    candidate_target = max(target_count * multiplier, target_count + 512)
    points = min_xy + _halton_2d(
      candidate_target,
      dtype=capsule_fromto_xy_m.dtype,
      device=capsule_fromto_xy_m.device,
    ) * span
    inside = _inside_capsule_footprint(
      points,
      capsule_fromto_xy_m=capsule_fromto_xy_m,
      capsule_radius_m=capsule_radius_m,
    )
    candidates = points[inside]
    if candidates.shape[0] >= target_count:
      return candidates

  if candidates.shape[0] == 0:
    centers = capsule_fromto_xy_m.mean(dim=1)
    return centers[:target_count]
  repeats = (target_count + candidates.shape[0] - 1) // candidates.shape[0]
  return candidates.repeat((repeats, 1))[:target_count]


def _inside_capsule_footprint(
  points_xy: torch.Tensor,
  *,
  capsule_fromto_xy_m: torch.Tensor,
  capsule_radius_m: torch.Tensor,
) -> torch.Tensor:
  distances = []
  for capsule_idx in range(capsule_fromto_xy_m.shape[0]):
    distances.append(
      _distance_to_segment(
        points_xy,
        capsule_fromto_xy_m[capsule_idx, 0],
        capsule_fromto_xy_m[capsule_idx, 1],
      )
      <= capsule_radius_m[capsule_idx]
    )
  return torch.stack(distances, dim=1).any(dim=1)


def _farthest_point_sample(points_xy: torch.Tensor, target_count: int) -> torch.Tensor:
  if points_xy.shape[0] <= target_count:
    return points_xy

  centroid = points_xy.mean(dim=0)
  selected_indices = [
    int(torch.argmin(torch.linalg.norm(points_xy - centroid, dim=1)).item())
  ]
  min_distance_sq = torch.full(
    (points_xy.shape[0],),
    torch.inf,
    dtype=points_xy.dtype,
    device=points_xy.device,
  )
  for _ in range(1, target_count):
    selected = points_xy[selected_indices[-1]]
    distance_sq = torch.sum((points_xy - selected) ** 2, dim=1)
    min_distance_sq = torch.minimum(min_distance_sq, distance_sq)
    selected_indices.append(int(torch.argmax(min_distance_sq).item()))
  return points_xy[torch.tensor(selected_indices, dtype=torch.long, device=points_xy.device)]


def _halton_2d(
  count: int,
  *,
  dtype: torch.dtype,
  device: torch.device,
) -> torch.Tensor:
  indices = torch.arange(1, count + 1, dtype=torch.long, device=device)
  return torch.stack(
    (
      _van_der_corput(indices, base=2),
      _van_der_corput(indices, base=3),
    ),
    dim=1,
  ).to(dtype=dtype)


def _van_der_corput(indices: torch.Tensor, *, base: int) -> torch.Tensor:
  values = torch.zeros(indices.shape, dtype=torch.float32, device=indices.device)
  remaining = indices.clone()
  denominator = float(base)
  while bool((remaining > 0).any().item()):
    values += torch.remainder(remaining, base).to(torch.float32) / denominator
    remaining = torch.div(remaining, base, rounding_mode="floor")
    denominator *= float(base)
  return values


def _normalize_capsule_radii(
  capsule_radius_m: tuple[float, ...] | float,
  *,
  count: int,
  dtype: torch.dtype,
  device: torch.device,
) -> torch.Tensor:
  if isinstance(capsule_radius_m, tuple):
    if len(capsule_radius_m) < count:
      raise ValueError("capsule_radius_m does not cover every capsule")
    values = capsule_radius_m[:count]
  else:
    values = (float(capsule_radius_m),) * count
  return torch.tensor(values, dtype=dtype, device=device).clamp_min(
    torch.finfo(dtype).eps
  )


def _sort_xy(points_xy: torch.Tensor) -> torch.Tensor:
  order = sorted(
    range(points_xy.shape[0]),
    key=lambda idx: (float(points_xy[idx, 0].item()), float(points_xy[idx, 1].item())),
  )
  return points_xy[torch.tensor(order, dtype=torch.long, device=points_xy.device)]


def _distance_to_segment(
  points_xy: torch.Tensor,
  start_xy: torch.Tensor,
  end_xy: torch.Tensor,
) -> torch.Tensor:
  segment = end_xy - start_xy
  length_sq = torch.dot(segment, segment).clamp_min(torch.finfo(points_xy.dtype).eps)
  t = torch.sum((points_xy - start_xy) * segment, dim=1) / length_sq
  t = torch.clamp(t, min=0.0, max=1.0)
  projection = start_xy + t[:, None] * segment
  return torch.linalg.norm(points_xy - projection, dim=1)


def _normalize_offsets(
  local_offsets_m: torch.Tensor,
  *,
  batch_size: int,
  num_feet: int,
) -> torch.Tensor:
  if local_offsets_m.ndim == 2 and local_offsets_m.shape[-1] == 3:
    return local_offsets_m.view(1, 1, -1, 3).expand(batch_size, num_feet, -1, 3)
  if local_offsets_m.ndim == 3 and local_offsets_m.shape[-1] == 3:
    if local_offsets_m.shape[0] != num_feet:
      raise ValueError("local_offsets_m foot dimension does not match body tensors")
    return local_offsets_m.unsqueeze(0).expand(batch_size, -1, -1, -1)
  raise ValueError("local_offsets_m must have shape [P, 3] or [F, P, 3]")


def _touchdown_mask(contact: torch.Tensor) -> torch.Tensor:
  if contact.ndim != 2:
    raise ValueError("foot_contact_flag must have shape [T, F]")
  previous = torch.zeros_like(contact)
  if contact.shape[0] > 1:
    previous[1:] = contact[:-1]
  return contact & ~previous


def _clean_float(value: torch.Tensor) -> float:
  return round(float(value.item()), 6)
