from dataclasses import dataclass

import numpy as np

from src.evaluation.silent_walking.viser_foot_grid_overlay import (
  FootGridOverlayRasterizer,
  FootGridOverlayConfig,
  FootGridViserPlayViewer,
  SilentFootGridOverlayEnvWrapper,
  ensure_foot_grid_capsule_contact_sensor,
  prepare_silent_foot_grid_overlay_env_cfg,
  pressure_values_to_rgb,
  wrap_env_for_silent_foot_grid_overlay,
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


def test_foot_grid_overlay_rasterizer_uses_human_view_xy_orientation():
  local_xy = np.array(
    [
      [-0.05, -0.02],
      [0.13, -0.02],
      [-0.05, 0.02],
    ],
    dtype=np.float32,
  )
  rasterizer = FootGridOverlayRasterizer(local_xy, width=240, panel_height=180)

  rear_y, rear_x = rasterizer.point_pixel(row=0, foot_index=0, point_index=0)
  toe_y, toe_x = rasterizer.point_pixel(row=0, foot_index=0, point_index=1)
  lateral_y, lateral_x = rasterizer.point_pixel(row=0, foot_index=0, point_index=2)

  assert abs(rear_x - toe_x) < abs(rear_y - toe_y)
  assert abs(rear_x - lateral_x) > abs(rear_y - lateral_y)
  assert toe_y < rear_y


def test_foot_grid_overlay_rasterizer_preserves_local_xy_aspect_ratio():
  local_xy = np.array(
    [
      [-0.05, -0.03],
      [0.13, -0.03],
      [-0.05, 0.03],
    ],
    dtype=np.float32,
  )
  rasterizer = FootGridOverlayRasterizer(local_xy, width=900, panel_height=220)

  rear_y, rear_x = rasterizer.point_pixel(row=0, foot_index=0, point_index=0)
  toe_y, toe_x = rasterizer.point_pixel(row=0, foot_index=0, point_index=1)
  lateral_y, lateral_x = rasterizer.point_pixel(row=0, foot_index=0, point_index=2)

  pixel_fore_aft = abs(toe_y - rear_y)
  pixel_lateral = abs(lateral_x - rear_x)
  actual_ratio = pixel_lateral / pixel_fore_aft
  expected_ratio = 0.06 / 0.18

  assert abs(actual_ratio - expected_ratio) < 0.05
  assert abs(toe_x - rear_x) <= 1
  assert abs(lateral_y - rear_y) <= 1


def test_foot_grid_overlay_pressure_uses_adaptive_scale_for_small_forces():
  local_xy = np.array([[-0.05, -0.02], [0.13, 0.02]], dtype=np.float32)
  rasterizer = FootGridOverlayRasterizer(local_xy, width=240, panel_height=180)
  signed_vz = np.zeros((2, 2), dtype=np.float32)
  force = np.zeros((2, 2), dtype=np.float32)
  force[0, 0] = 0.2
  contact = np.array([True, False])

  image = rasterizer.render(signed_vz=signed_vz, force_n=force, contact=contact)

  y, x = rasterizer.point_pixel(row=1, foot_index=0, point_index=0)
  assert image[y, x].tolist() == [153, 27, 27]


def test_foot_grid_overlay_rasterizer_updates_signed_vz_limit_live():
  local_xy = np.array([[-0.05, -0.02], [0.13, 0.02]], dtype=np.float32)
  rasterizer = FootGridOverlayRasterizer(
    local_xy,
    width=240,
    panel_height=180,
    vz_limit_m_s=1.0,
  )
  signed_vz = np.zeros((2, 2), dtype=np.float32)
  signed_vz[0, 0] = 0.5
  force = np.zeros((2, 2), dtype=np.float32)
  contact = np.array([False, False])

  image_wide = rasterizer.render(signed_vz=signed_vz, force_n=force, contact=contact)
  rasterizer.set_vz_limit_m_s(0.5)
  image_narrow = rasterizer.render(signed_vz=signed_vz, force_n=force, contact=contact)

  y, x = rasterizer.point_pixel(row=0, foot_index=0, point_index=0)
  assert image_wide[y, x].tolist() != image_narrow[y, x].tolist()
  assert image_narrow[y, x].tolist() == [29, 78, 216]


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

  ensure_foot_grid_capsule_contact_sensor(cfg, "g1", force_slots=4)
  ensure_foot_grid_capsule_contact_sensor(cfg, "g1", force_slots=4)

  names = [sensor.name for sensor in cfg.scene.sensors]
  assert names.count("foot_capsule_ground_contact") == 1
  assert names.count("foot_capsule_ground_contact_points") == 1
  point_sensor = next(
    sensor for sensor in cfg.scene.sensors if sensor.name == "foot_capsule_ground_contact_points"
  )
  assert point_sensor.reduce == "maxforce"
  assert point_sensor.global_frame is True
  assert point_sensor.num_slots == 4
  assert {"found", "force", "pos", "normal", "tangent"}.issubset(point_sensor.fields)


def test_prepare_silent_foot_grid_overlay_env_cfg_returns_cfg_and_appends_sensors():
  @dataclass
  class FakeScene:
    sensors: tuple[object, ...] = ()

  @dataclass
  class FakeCfg:
    scene: FakeScene

  cfg = FakeCfg(scene=FakeScene())

  returned = prepare_silent_foot_grid_overlay_env_cfg(cfg, "g1", force_slots=2)

  assert returned is cfg
  names = [sensor.name for sensor in cfg.scene.sensors]
  assert names == ["foot_capsule_ground_contact", "foot_capsule_ground_contact_points"]
  assert cfg.scene.sensors[-1].num_slots == 2


def test_silent_foot_grid_overlay_env_wrapper_delegates_env_and_overlay_calls():
  class FakeOverlay:
    def __init__(self, env, config):
      self.env = env
      self.config = config
      self.setup_calls = []
      self.update_calls = []

    def setup(self, server):
      self.setup_calls.append(server)

    def update(self, env_idx, step_count, *, substep_index=None, substep_count=None):
      self.update_calls.append((env_idx, step_count, substep_index, substep_count))

  class FakeEnv:
    def __init__(self):
      self.unwrapped = object()
      self.step_calls = []
      self.cfg = object()

    def step(self, action):
      self.step_calls.append(action)
      return "step-result"

    def get_observations(self):
      return "obs"

  env = FakeEnv()
  config = FootGridOverlayConfig(point_count=4)

  wrapper = wrap_env_for_silent_foot_grid_overlay(
    env,
    config,
    overlay_factory=FakeOverlay,
  )

  assert isinstance(wrapper, SilentFootGridOverlayEnvWrapper)
  assert wrapper.unwrapped is env.unwrapped
  assert wrapper.cfg is env.cfg
  assert wrapper.get_observations() == "obs"
  assert wrapper.step("action") == "step-result"
  assert env.step_calls == ["action"]

  wrapper.setup_silent_foot_grid_overlay("server")
  wrapper.update_silent_foot_grid_overlay(
    1,
    2,
    substep_index=3,
    substep_count=4,
  )

  assert wrapper.silent_foot_grid_overlay.setup_calls == ["server"]
  assert wrapper.silent_foot_grid_overlay.update_calls == [(1, 2, 3, 4)]


def test_foot_grid_sim_update_rate_steps_by_physics_dt():
  class FakeUnwrapped:
    physics_dt = 0.005
    step_dt = 0.02

  class FakeEnv:
    unwrapped = FakeUnwrapped()

  viewer = object.__new__(FootGridViserPlayViewer)
  viewer.env = FakeEnv()
  viewer.frame_time = 1.0
  viewer._foot_grid_update_rate = "sim"
  viewer._sim_budget = 0.0
  viewer._time_multiplier = 1.0
  viewer._was_capped = False
  viewer.sync_viewer_to_env = lambda: None
  steps = []

  def execute_sim_substep():
    steps.append(len(steps))
    return True

  viewer._execute_sim_substep = execute_sim_substep

  viewer._step_physics(0.019)

  assert len(steps) == 3
  assert 0.003 < viewer._sim_budget < 0.005
