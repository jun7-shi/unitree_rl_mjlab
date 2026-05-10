from dataclasses import dataclass

import numpy as np

from src.evaluation.silent_walking.viser_foot_grid_overlay import (
  FootGridOverlayRasterizer,
  ensure_foot_grid_capsule_contact_sensor,
  pressure_values_to_rgb,
)


def test_pressure_values_to_rgb_maps_zero_to_white():
  colors = pressure_values_to_rgb(np.array([0.0, 50.0, 100.0], dtype=np.float32), limit_n=100.0)

  assert colors.dtype == np.uint8
  assert colors.shape == (3, 3)
  assert colors[0].tolist() == [255, 255, 255]
  assert colors[-1].tolist() == [153, 27, 27]


def test_foot_grid_overlay_rasterizer_draws_control_step_image():
  local_xy = np.stack(
    (
      np.linspace(-0.05, 0.13, 300, dtype=np.float32),
      np.zeros(300, dtype=np.float32),
    ),
    axis=1,
  )
  rasterizer = FootGridOverlayRasterizer(local_xy, width=640, panel_height=160)
  signed_vz = np.zeros((2, 300), dtype=np.float32)
  signed_vz[0, 10] = -0.5
  signed_vz[1, 20] = 0.5
  force = np.zeros((2, 300), dtype=np.float32)
  force[0, 10] = 20.0
  contact = np.array([True, False])

  image = rasterizer.render(signed_vz=signed_vz, force_n=force, contact=contact)

  assert image.dtype == np.uint8
  assert image.shape == (320, 640, 3)
  assert image.min() < 255
  zero_force_pixel = rasterizer.point_pixel(row=1, foot_index=1, point_index=20)
  y, x = zero_force_pixel
  assert image[y, x].tolist() == [255, 255, 255]


def test_ensure_foot_grid_capsule_contact_sensor_appends_once():
  @dataclass
  class FakeSensor:
    name: str

  @dataclass
  class FakeScene:
    sensors: tuple[object, ...] = ()

  @dataclass
  class FakeCfg:
    scene: FakeScene

  cfg = FakeCfg(scene=FakeScene(sensors=(FakeSensor(name="feet_ground_contact"),)))

  ensure_foot_grid_capsule_contact_sensor(cfg, "g1")
  ensure_foot_grid_capsule_contact_sensor(cfg, "g1")

  names = [sensor.name for sensor in cfg.scene.sensors]
  assert names.count("foot_capsule_ground_contact") == 1
