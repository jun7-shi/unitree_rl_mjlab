"""Realtime foot-grid overlay for the Viser play viewer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import viser

from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.viewer import ViserPlayViewer

from .contact_backends import extract_capsule_contact_point_forces, extract_feet_contact_flags
from .foot_grid import (
  distribute_contact_forces_to_foot_grid,
  foot_grid_point_velocities,
  make_capsule_footprint_sample_offsets,
)
from .regional_contact import world_to_local_foot_points
from .robots import get_foot_proxy_spec, get_robot_spec


FOOT_NAMES = ("left", "right")


@dataclass(frozen=True, slots=True)
class FootGridOverlayConfig:
  """Configuration for the realtime foot-grid Viser overlay."""

  robot_name: str = "g1"
  point_count: int = 300
  width: int = 900
  panel_height: int = 220
  point_radius_px: int = 3
  vz_limit_m_s: float = 1.0
  force_limit_n: float = 0.0


def ensure_foot_grid_capsule_contact_sensor(env_cfg: Any, robot_name: str = "g1") -> None:
  """Attach the capsule contact sensor needed by the foot-grid force overlay."""

  spec = get_robot_spec(robot_name)
  existing_sensors = env_cfg.scene.sensors or ()
  existing_names = {sensor.name for sensor in existing_sensors}
  new_sensors: list[ContactSensorCfg] = []
  if "foot_capsule_ground_contact" not in existing_names:
    new_sensors.append(
      ContactSensorCfg(
        name="foot_capsule_ground_contact",
        primary=ContactMatch(
          mode="geom",
          pattern=spec.foot_collision_geom_names,
          entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
      )
    )
  if "foot_capsule_ground_contact_points" not in existing_names:
    new_sensors.append(
      ContactSensorCfg(
        name="foot_capsule_ground_contact_points",
        primary=ContactMatch(
          mode="geom",
          pattern=spec.foot_collision_geom_names,
          entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force", "pos", "normal", "tangent"),
        reduce="maxforce",
        num_slots=1,
        global_frame=True,
      )
    )
  if not new_sensors:
    return
  env_cfg.scene.sensors = existing_sensors + tuple(new_sensors)


def signed_vz_values_to_rgb(values: np.ndarray, *, limit_m_s: float) -> np.ndarray:
  """Map signed vertical velocities to red-white-blue RGB values."""

  if limit_m_s <= 0.0:
    raise ValueError("limit_m_s must be positive")
  values = np.asarray(values, dtype=np.float32)
  clipped = np.clip(values / limit_m_s, -1.0, 1.0)
  rgb = np.full(values.shape + (3,), 255.0, dtype=np.float32)
  negative = clipped < 0.0
  positive = clipped > 0.0
  red = np.array([185.0, 28.0, 28.0], dtype=np.float32)
  blue = np.array([29.0, 78.0, 216.0], dtype=np.float32)
  white = np.array([255.0, 255.0, 255.0], dtype=np.float32)
  if negative.any():
    t = np.abs(clipped[negative])[..., None]
    rgb[negative] = white * (1.0 - t) + red * t
  if positive.any():
    t = clipped[positive][..., None]
    rgb[positive] = white * (1.0 - t) + blue * t
  return np.clip(np.rint(rgb), 0, 255).astype(np.uint8)


def pressure_values_to_rgb(values: np.ndarray, *, limit_n: float) -> np.ndarray:
  """Map normal-force proxy values to RGB, with exact white at zero."""

  if limit_n <= 0.0:
    raise ValueError("limit_n must be positive")
  values = np.asarray(values, dtype=np.float32)
  t = np.clip(values / limit_n, 0.0, 1.0)
  stops = np.array(
    [
      [255.0, 255.0, 255.0],
      [253.0, 230.0, 138.0],
      [249.0, 115.0, 22.0],
      [153.0, 27.0, 27.0],
    ],
    dtype=np.float32,
  )
  positions = np.array([0.0, 0.25, 0.65, 1.0], dtype=np.float32)
  rgb = np.empty(values.shape + (3,), dtype=np.float32)
  for channel in range(3):
    rgb[..., channel] = np.interp(t, positions, stops[:, channel])
  return np.clip(np.rint(rgb), 0, 255).astype(np.uint8)


class FootGridOverlayRasterizer:
  """Rasterize a two-row foot-grid velocity/force image for a GUI image handle."""

  def __init__(
    self,
    local_xy_m: np.ndarray,
    *,
    width: int = 900,
    panel_height: int = 220,
    point_radius_px: int = 3,
    vz_limit_m_s: float = 1.0,
    force_limit_n: float = 0.0,
  ) -> None:
    if local_xy_m.ndim != 2 or local_xy_m.shape[1] != 2:
      raise ValueError("local_xy_m must have shape [P, 2]")
    if width <= 0 or panel_height <= 0 or point_radius_px <= 0:
      raise ValueError("width, panel_height, and point_radius_px must be positive")
    if force_limit_n < 0.0:
      raise ValueError("force_limit_n must be non-negative")
    self.local_xy_m = local_xy_m.astype(np.float32, copy=True)
    self.width = int(width)
    self.panel_height = int(panel_height)
    self.point_radius_px = int(point_radius_px)
    self.force_limit_n = float(force_limit_n)
    self.height = self.panel_height * 2
    self._panel_width = self.width // 2
    self._base_pixels = self._compute_panel_pixels(self.local_xy_m)
    self.vz_limit_m_s = 1.0
    self.set_vz_limit_m_s(vz_limit_m_s)

  def point_pixel(self, *, row: int, foot_index: int, point_index: int) -> tuple[int, int]:
    """Return the image pixel center for a rendered point."""

    x, y = self._base_pixels[point_index]
    return row * self.panel_height + int(y), foot_index * self._panel_width + int(x)

  def render(
    self,
    *,
    signed_vz: np.ndarray,
    force_n: np.ndarray,
    contact: np.ndarray,
  ) -> np.ndarray:
    """Render one control-step image with signed-vz and force-proxy rows."""

    signed_vz = np.asarray(signed_vz, dtype=np.float32)
    force_n = np.asarray(force_n, dtype=np.float32)
    contact = np.asarray(contact, dtype=bool)
    if signed_vz.shape != force_n.shape:
      raise ValueError("signed_vz and force_n must have matching shape")
    if signed_vz.ndim != 2 or signed_vz.shape[0] != 2:
      raise ValueError("signed_vz and force_n must have shape [2, P]")
    if signed_vz.shape[1] != self.local_xy_m.shape[0]:
      raise ValueError("point count does not match local_xy_m")
    if contact.shape[0] != 2:
      raise ValueError("contact must have shape [2]")

    image = np.full((self.height, self.width, 3), 255, dtype=np.uint8)
    image[:, self._panel_width - 1:self._panel_width + 1] = 210
    image[self.panel_height - 1:self.panel_height + 1, :] = 210

    for foot_idx in range(2):
      contact_color = np.array([220, 38, 38], dtype=np.uint8) if contact[foot_idx] else np.array([107, 114, 128], dtype=np.uint8)
      x0 = foot_idx * self._panel_width
      x1 = x0 + self._panel_width
      image[0:8, x0:x1] = contact_color
      image[self.panel_height:self.panel_height + 8, x0:x1] = contact_color
      self._draw_points(
        image,
        row=0,
        foot_index=foot_idx,
        colors=signed_vz_values_to_rgb(
          signed_vz[foot_idx],
          limit_m_s=self.vz_limit_m_s,
        ),
      )
      self._draw_points(
        image,
        row=1,
        foot_index=foot_idx,
        colors=pressure_values_to_rgb(
          force_n[foot_idx],
          limit_n=self._pressure_limit(force_n),
        ),
      )
    return image

  def _compute_panel_pixels(self, local_xy_m: np.ndarray) -> np.ndarray:
    padding = max(14, self.point_radius_px + 8)
    return _local_xy_to_human_view_pixels(
      local_xy_m,
      panel_width=self._panel_width,
      panel_height=self.panel_height,
      padding=padding,
    )

  def _pressure_limit(self, force_n: np.ndarray) -> float:
    if self.force_limit_n > 0.0:
      return self.force_limit_n
    max_force = float(np.nanmax(force_n)) if force_n.size else 0.0
    return max(max_force, 1.0e-6)

  def _draw_points(
    self,
    image: np.ndarray,
    *,
    row: int,
    foot_index: int,
    colors: np.ndarray,
  ) -> None:
    x_offset = foot_index * self._panel_width
    y_offset = row * self.panel_height
    radius = self.point_radius_px
    for point_idx, (base_x, base_y) in enumerate(self._base_pixels):
      cx = int(base_x) + x_offset
      cy = int(base_y) + y_offset
      x0 = max(cx - radius, 0)
      x1 = min(cx + radius + 1, self.width)
      y0 = max(cy - radius, 0)
      y1 = min(cy + radius + 1, self.height)
      yy, xx = np.ogrid[y0:y1, x0:x1]
      disk = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
      image[y0:y1, x0:x1][disk] = colors[point_idx]

  def set_vz_limit_m_s(self, limit_m_s: float) -> None:
    """Set the signed-vz display range used by subsequent renders."""

    if limit_m_s <= 0.0:
      raise ValueError("limit_m_s must be positive")
    self.vz_limit_m_s = float(limit_m_s)


class ViserFootGridOverlay:
  """GUI image overlay for realtime foot-grid velocity and force proxy."""

  def __init__(self, env: Any, config: FootGridOverlayConfig) -> None:
    self.env = env
    self.config = config
    self.spec = get_robot_spec(config.robot_name)
    self.proxy = get_foot_proxy_spec(config.robot_name)
    self._body_ids: list[int] | None = None
    self._local_offsets: torch.Tensor | None = None
    self._local_xy_np: np.ndarray | None = None
    self._rasterizer: FootGridOverlayRasterizer | None = None
    self._image_handle: viser.GuiImageHandle | None = None
    self._status_handle: viser.GuiMarkdownHandle | None = None
    self._vz_limit_handle: Any | None = None
    self._last_signed_vz: np.ndarray | None = None
    self._last_force_n: np.ndarray | None = None
    self._last_contact: np.ndarray | None = None
    self._last_step_count = 0
    self._step = 0

  def setup(self, server: viser.ViserServer) -> None:
    """Create the Viser GUI tab and image handle."""

    self._ensure_geometry()
    assert self._local_xy_np is not None
    self._rasterizer = FootGridOverlayRasterizer(
      self._local_xy_np,
      width=self.config.width,
      panel_height=self.config.panel_height,
      point_radius_px=self.config.point_radius_px,
      vz_limit_m_s=self.config.vz_limit_m_s,
      force_limit_n=self.config.force_limit_n,
    )
    tab_group = server.gui.add_tab_group()
    with tab_group.add_tab("Foot Grid"):
      self._status_handle = server.gui.add_markdown("")
      initial_vz_limit = float(self.config.vz_limit_m_s)
      self._vz_limit_handle = server.gui.add_slider(
        "Vz range +/- m/s",
        min=0.05,
        max=max(2.0, initial_vz_limit),
        step=0.05,
        initial_value=initial_vz_limit,
        marks=((0.1, "0.1"), (0.5, "0.5"), (1.0, "1.0"), (2.0, "2.0")),
        hint="Signed Vz color range.",
      )

      @self._vz_limit_handle.on_update
      def _(_) -> None:
        self._set_vz_limit_from_gui()

      self._image_handle = server.gui.add_image(
        image=np.full(
          (self._rasterizer.height, self._rasterizer.width, 3),
          255,
          dtype=np.uint8,
        ),
        label="signed Vz / normal force proxy",
        format="jpeg",
        jpeg_quality=90,
      )

  def update(self, env_idx: int, step_count: int) -> None:
    """Update the GUI image from the current environment state."""

    if self._image_handle is None or self._rasterizer is None:
      return
    self._ensure_geometry()
    assert self._local_offsets is not None
    robot = self.env.scene["robot"]
    body_pos = robot.data.body_link_pos_w[:, self._body_ids]
    body_quat = robot.data.body_link_quat_w[:, self._body_ids]
    body_lin_vel = robot.data.body_link_lin_vel_w[:, self._body_ids]
    body_ang_vel = robot.data.body_link_ang_vel_w[:, self._body_ids]
    local_offsets = self._local_offsets.to(dtype=body_pos.dtype, device=body_pos.device)
    grid_velocity = foot_grid_point_velocities(
      body_pos_w=body_pos,
      body_quat_w=body_quat,
      body_lin_vel_w=body_lin_vel,
      body_ang_vel_w=body_ang_vel,
      local_offsets_m=local_offsets,
    )
    contact_names, contact_force, contact_mask, contact_pos = extract_capsule_contact_point_forces(
      self.env,
      self.config.robot_name,
    )
    if contact_names != self.spec.foot_collision_geom_names:
      raise RuntimeError("Capsule contact-point ordering mismatch")
    foot_contact = extract_feet_contact_flags(self.env, self.config.robot_name)
    num_feet = len(self.proxy.foot_names)
    env_idx = int(env_idx)
    signed_vz = grid_velocity[env_idx, :, :, 2].detach().cpu().numpy()
    contact_force_by_foot = contact_force.reshape(
      contact_force.shape[0],
      num_feet,
      -1,
    )
    contact_mask_by_foot = contact_mask.reshape(
      contact_mask.shape[0],
      num_feet,
      -1,
    )
    contact_pos_local = world_to_local_foot_points(
      body_pos_w=body_pos,
      body_quat_w=body_quat,
      points_w=contact_pos.reshape(
        contact_pos.shape[0],
        num_feet,
        -1,
        3,
      ),
    )
    force_n = distribute_contact_forces_to_foot_grid(
      contact_force_n=contact_force_by_foot,
      contact_pos_local_xy_m=contact_pos_local[..., :2],
      contact_mask=contact_mask_by_foot,
      local_xy_m=local_offsets[:, :2],
    )[env_idx].detach().cpu().numpy()
    contact = foot_contact[env_idx].detach().cpu().numpy().astype(bool)
    self._last_signed_vz = signed_vz.astype(np.float32, copy=False)
    self._last_force_n = force_n.astype(np.float32, copy=False)
    self._last_contact = contact
    self._last_step_count = step_count
    self._render_cached_frame()
    self._step = step_count

  def _set_vz_limit_from_gui(self) -> None:
    if self._vz_limit_handle is None or self._rasterizer is None:
      return
    self._rasterizer.set_vz_limit_m_s(float(self._vz_limit_handle.value))
    self._render_cached_frame()

  def _render_cached_frame(self) -> None:
    if (
      self._image_handle is None
      or self._rasterizer is None
      or self._last_signed_vz is None
      or self._last_force_n is None
      or self._last_contact is None
    ):
      return
    self._image_handle.image = self._rasterizer.render(
      signed_vz=self._last_signed_vz,
      force_n=self._last_force_n,
      contact=self._last_contact,
    )
    if self._status_handle is not None:
      self._status_handle.content = (
        f"Step `{self._last_step_count}` | "
        f"left `{'CONTACT' if self._last_contact[0] else 'AIR'}` | "
        f"right `{'CONTACT' if self._last_contact[1] else 'AIR'}` | "
        f"vz range `+/-{self._rasterizer.vz_limit_m_s:.2f} m/s` | "
        f"max force `{float(self._last_force_n.max()):.1f} N`"
      )

  def _ensure_geometry(self) -> None:
    if self._body_ids is not None:
      return
    if self.config.point_count <= 0:
      raise ValueError("foot-grid overlay point count must be positive")
    if not self.proxy.foot_body_names:
      raise ValueError(f"robot '{self.config.robot_name}' does not define foot bodies")
    robot = self.env.scene["robot"]
    body_ids, _ = robot.find_bodies(self.proxy.foot_body_names, preserve_order=True)
    self._body_ids = body_ids
    local_offsets = make_capsule_footprint_sample_offsets(
      capsule_fromto_xy_m=self.spec.foot_collision_fromto_xy_m[:7],
      capsule_radius_m=self.spec.foot_collision_radius_m[:7],
      z_m=-0.025,
      target_count=self.config.point_count,
    )
    self._local_offsets = local_offsets
    self._local_xy_np = local_offsets[:, :2].cpu().numpy()


class FootGridViserPlayViewer(ViserPlayViewer):
  """Viser play viewer that updates a foot-grid GUI image once per env step."""

  def __init__(
    self,
    env: Any,
    policy: Any,
    *,
    foot_grid_config: FootGridOverlayConfig,
  ) -> None:
    super().__init__(env, policy)
    self._foot_grid_overlay = ViserFootGridOverlay(env.unwrapped, foot_grid_config)

  def setup(self) -> None:
    super().setup()
    self._foot_grid_overlay.setup(self._server)

  def _execute_step(self) -> bool:
    ok = super()._execute_step()
    if ok:
      self._foot_grid_overlay.update(self._scene.env_idx, self._step_count)
    return ok

  def reset_environment(self) -> None:
    super().reset_environment()
    self._foot_grid_overlay.update(self._scene.env_idx, self._step_count)


def _local_xy_to_human_view_pixels(
  local_xy_m: np.ndarray,
  *,
  panel_width: int,
  panel_height: int,
  padding: int,
) -> np.ndarray:
  local_xy_m = np.asarray(local_xy_m, dtype=np.float32)
  local_x = local_xy_m[:, 0]
  local_y = local_xy_m[:, 1]
  usable_width = max(panel_width - 2 * padding - 1, 1)
  usable_height = max(panel_height - 2 * padding - 1, 1)
  x_min = float(local_x.min())
  x_max = float(local_x.max())
  y_min = float(local_y.min())
  y_max = float(local_y.max())
  x_span = x_max - x_min
  y_span = y_max - y_min
  eps = 1.0e-9
  width_scale = usable_width / y_span if y_span > eps else np.inf
  height_scale = usable_height / x_span if x_span > eps else np.inf
  pixels_per_m = min(width_scale, height_scale)
  if not np.isfinite(pixels_per_m):
    px = np.full_like(local_y, padding + usable_width * 0.5, dtype=np.float32)
    py = np.full_like(local_x, padding + usable_height * 0.5, dtype=np.float32)
    return np.stack((np.rint(px), np.rint(py)), axis=1).astype(np.int32)

  if y_span > eps:
    drawn_width = y_span * pixels_per_m
    left = padding + (usable_width - drawn_width) * 0.5
    px = left + (local_y - y_min) * pixels_per_m
  else:
    px = np.full_like(local_y, padding + usable_width * 0.5, dtype=np.float32)

  if x_span > eps:
    drawn_height = x_span * pixels_per_m
    top = padding + (usable_height - drawn_height) * 0.5
    py = top + (x_max - local_x) * pixels_per_m
  else:
    py = np.full_like(local_x, padding + usable_height * 0.5, dtype=np.float32)

  return np.stack((np.rint(px), np.rint(py)), axis=1).astype(np.int32)
