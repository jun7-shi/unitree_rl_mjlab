"""Region-level foot contact helpers for silent-walking evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse

FootRegion = Literal["heel", "midfoot", "toe"]
RegionEventType = Literal["touchdown", "secondary"]

REGION_NAMES: tuple[FootRegion, ...] = ("heel", "midfoot", "toe")
CORNER_NAMES: tuple[str, ...] = (
  "rear_left",
  "rear_right",
  "front_left",
  "front_right",
)


@dataclass(frozen=True, slots=True)
class RegionalContactEvent:
  """One region-level contact event inside a foot stance."""

  step: int
  foot_index: int
  foot_name: str
  event_type: RegionEventType
  regions: tuple[FootRegion, ...]
  peak_force_n: torch.Tensor
  loading_rate_n_s: torch.Tensor
  vertical_speed_m_s: torch.Tensor
  corner_downward_speed_m_s: torch.Tensor
  stance_id: int = -1
  substep_index: int = 0
  time_offset_s: float = 0.0
  corner_downward_speeds_m_s: torch.Tensor | None = None


def virtual_foot_corner_points(
  *,
  body_pos_w: torch.Tensor,
  body_quat_w: torch.Tensor,
  corner_local_offsets_m: tuple[tuple[float, float, float], ...],
) -> torch.Tensor:
  """Return world positions for evaluator-local virtual foot corner points.

  These points are metadata for analysis, not MuJoCo geoms in the default G1 asset.
  """

  if body_pos_w.ndim != 3 or body_pos_w.shape[-1] != 3:
    raise ValueError("body_pos_w must have shape [B, F, 3]")
  if body_quat_w.ndim != 3 or body_quat_w.shape[-1] != 4:
    raise ValueError("body_quat_w must have shape [B, F, 4]")

  offsets = torch.tensor(
    corner_local_offsets_m,
    dtype=body_pos_w.dtype,
    device=body_pos_w.device,
  )
  batch_size, num_feet = body_pos_w.shape[:2]
  num_corners = offsets.shape[0]
  expanded_offsets = offsets.view(1, 1, num_corners, 3).expand(
    batch_size,
    num_feet,
    num_corners,
    3,
  )
  expanded_quat = body_quat_w.unsqueeze(2).expand(
    batch_size,
    num_feet,
    num_corners,
    4,
  )
  rotated = quat_apply(
    expanded_quat.reshape(-1, 4),
    expanded_offsets.reshape(-1, 3),
  ).reshape(batch_size, num_feet, num_corners, 3)
  return body_pos_w.unsqueeze(2) + rotated


def world_to_local_foot_points(
  *,
  body_pos_w: torch.Tensor,
  body_quat_w: torch.Tensor,
  points_w: torch.Tensor,
) -> torch.Tensor:
  """Transform world points into the local rigid-foot body frame."""

  if points_w.ndim != 4 or points_w.shape[-1] != 3:
    raise ValueError("points_w must have shape [B, F, C, 3]")
  batch_size, num_feet, num_points = points_w.shape[:3]
  expanded_body_pos = body_pos_w.unsqueeze(2).expand(
    batch_size,
    num_feet,
    num_points,
    3,
  )
  expanded_body_quat = body_quat_w.unsqueeze(2).expand(
    batch_size,
    num_feet,
    num_points,
    4,
  )
  local = quat_apply_inverse(
    expanded_body_quat.reshape(-1, 4),
    (points_w - expanded_body_pos).reshape(-1, 3),
  )
  return local.reshape(batch_size, num_feet, num_points, 3)


def classify_contact_regions(
  *,
  contact_pos_local_m: torch.Tensor,
  contact_mask: torch.Tensor,
  heel_region_max_x_m: float,
  toe_region_min_x_m: float,
) -> torch.Tensor:
  """Classify local contact points into heel/midfoot/toe region flags.

  The classifier intentionally uses contact point local x, not capsule identity,
  because G1 foot collision capsules are long strips rather than heel/toe geoms.
  """

  if contact_pos_local_m.ndim != 4 or contact_pos_local_m.shape[-1] != 3:
    raise ValueError("contact_pos_local_m must have shape [B, F, C, 3]")
  if contact_mask.shape != contact_pos_local_m.shape[:3]:
    raise ValueError("contact_mask must have shape [B, F, C]")

  mask = contact_mask.bool()
  x_pos = contact_pos_local_m[..., 0]
  heel = (x_pos <= heel_region_max_x_m) & mask
  toe = (x_pos >= toe_region_min_x_m) & mask
  midfoot = (x_pos > heel_region_max_x_m) & (x_pos < toe_region_min_x_m) & mask
  return torch.stack(
    [
      heel.any(dim=-1),
      midfoot.any(dim=-1),
      toe.any(dim=-1),
    ],
    dim=-1,
  )


class RegionalContactState:
  """Track first and secondary region impacts within each stance."""

  def __init__(self, *, num_feet: int) -> None:
    self._seen_regions = torch.zeros((num_feet, len(REGION_NAMES)), dtype=torch.bool)
    self._active_stance_ids = torch.full((num_feet,), -1, dtype=torch.int64)
    self._next_stance_id = 0

  @property
  def seen_regions(self) -> torch.Tensor:
    """Return regions already seen in the current stance."""

    return self._seen_regions

  def update(
    self,
    *,
    step: int,
    region_contact: torch.Tensor,
    foot_names: tuple[str, ...],
    foot_force_n: torch.Tensor,
    foot_loading_rate_n_s: torch.Tensor,
    foot_vertical_speed_m_s: torch.Tensor,
    corner_downward_speeds_m_s: torch.Tensor | None = None,
    corner_downward_speed_m_s: torch.Tensor | None = None,
    substep_index: int = 0,
    time_offset_s: float = 0.0,
  ) -> list[RegionalContactEvent]:
    """Update stance state and return newly observed region events."""

    current = region_contact.detach().cpu().bool()
    if current.shape != self._seen_regions.shape:
      raise ValueError(
        "region_contact must have shape "
        f"{tuple(self._seen_regions.shape)}, got {tuple(current.shape)}"
      )
    if corner_downward_speeds_m_s is None:
      if corner_downward_speed_m_s is None:
        corner_downward_speeds_m_s = torch.zeros(
          (current.shape[0], 0),
          dtype=foot_vertical_speed_m_s.dtype,
          device=foot_vertical_speed_m_s.device,
        )
      else:
        corner_downward_speeds_m_s = corner_downward_speed_m_s.reshape(-1, 1)
    if corner_downward_speed_m_s is None:
      if corner_downward_speeds_m_s.shape[-1] == 0:
        corner_downward_speed_m_s = torch.zeros(
          current.shape[0],
          dtype=foot_vertical_speed_m_s.dtype,
          device=foot_vertical_speed_m_s.device,
        )
      else:
        corner_downward_speed_m_s = corner_downward_speeds_m_s.max(dim=-1).values

    events: list[RegionalContactEvent] = []
    for foot_index in range(current.shape[0]):
      current_regions = current[foot_index]
      if not bool(current_regions.any().item()):
        self._seen_regions[foot_index] = False
        self._active_stance_ids[foot_index] = -1
        continue

      seen_regions = self._seen_regions[foot_index]
      if not bool(seen_regions.any().item()):
        self._active_stance_ids[foot_index] = self._next_stance_id
        self._next_stance_id += 1
      event_regions = current_regions & ~seen_regions
      if not bool(event_regions.any().item()):
        continue

      event_type: RegionEventType = (
        "touchdown" if not bool(seen_regions.any().item()) else "secondary"
      )
      regions = tuple(
        name
        for idx, name in enumerate(REGION_NAMES)
        if bool(event_regions[idx].item())
      )
      events.append(
        RegionalContactEvent(
          step=step,
          stance_id=int(self._active_stance_ids[foot_index].item()),
          foot_index=int(foot_index),
          foot_name=foot_names[foot_index],
          event_type=event_type,
          regions=regions,
          peak_force_n=foot_force_n[foot_index].detach().clone(),
          loading_rate_n_s=foot_loading_rate_n_s[foot_index].detach().clone(),
          vertical_speed_m_s=foot_vertical_speed_m_s[foot_index].detach().clone(),
          corner_downward_speed_m_s=corner_downward_speed_m_s[foot_index].detach().clone(),
          substep_index=int(substep_index),
          time_offset_s=float(time_offset_s),
          corner_downward_speeds_m_s=corner_downward_speeds_m_s[foot_index].detach().clone(),
        )
      )
      self._seen_regions[foot_index] |= current_regions

    return events
