import torch
import csv
from matplotlib import colors

from src.evaluation.silent_walking.foot_grid import (
  distribute_capsule_forces_to_foot_grid,
  distribute_contact_forces_to_foot_grid,
  foot_grid_down_speed_frames,
  foot_grid_point_velocities,
  make_capsule_footprint_sample_offsets,
  make_rectangular_foot_grid_offsets,
  summarize_post_contact_foot_grid,
  write_foot_grid_raw_csv,
)
from src.evaluation.silent_walking.plotting import (
  _pressure_video_colormap,
  _signed_vz_video_colormap,
  save_foot_grid_velocity_video,
)
from src.evaluation.silent_walking.types import EpisodeMetricTrace


def _minimal_trace(**overrides) -> EpisodeMetricTrace:
  base = dict(
    foot_z_force_n=torch.zeros(4, 2),
    foot_contact_flag=torch.zeros(4, 2),
    foot_vertical_velocity_m_s=torch.zeros(4, 2),
    capsule_names=(),
    capsule_layout_xy_m=torch.zeros(0, 2),
    capsule_outline_fromto_xy_m=torch.zeros(0, 2, 2),
    capsule_radius_m=torch.zeros(0),
    capsule_z_force_n=torch.zeros(4, 0),
    capsule_contact_flag=torch.zeros(4, 0),
    capsule_vertical_velocity_m_s=torch.zeros(4, 0),
    peak_force_bw=torch.zeros(1),
    loading_rate_bw_s=torch.zeros(1),
    touchdown_peak_force_bw=torch.zeros(1),
    touchdown_loading_rate_bw_s=torch.zeros(1),
    touchdown_vertical_speed_m_s=torch.zeros(1),
    left_touchdown_peak_force_bw=torch.zeros(1),
    right_touchdown_peak_force_bw=torch.zeros(1),
    left_touchdown_loading_rate_bw_s=torch.zeros(1),
    right_touchdown_loading_rate_bw_s=torch.zeros(1),
    left_touchdown_vertical_speed_m_s=torch.zeros(1),
    right_touchdown_vertical_speed_m_s=torch.zeros(1),
    touchdown_peak_asymmetry_bw=torch.zeros(1),
    capsule_touchdown_peak_force_bw=torch.zeros(0),
    capsule_touchdown_loading_rate_bw_s=torch.zeros(0),
    capsule_touchdown_vertical_speed_m_s=torch.zeros(0),
    capsule_touchdown_count=torch.zeros(0),
    action_rate_l2=torch.zeros(4),
    command_velocity=torch.zeros(4, 3),
    actual_linear_velocity=torch.zeros(4, 2),
    actual_yaw_rate=torch.zeros(4),
    linear_velocity_error=torch.zeros(4),
    yaw_rate_error=torch.zeros(4),
    contact_quietness_score=torch.zeros(1),
  )
  base.update(overrides)
  return EpisodeMetricTrace(**base)


def test_make_rectangular_foot_grid_offsets_creates_uniform_30_point_grid():
  offsets = make_rectangular_foot_grid_offsets(
    x_range_m=(-0.05, 0.13),
    y_range_m=(-0.03, 0.03),
    z_m=-0.025,
    shape=(5, 6),
  )

  assert offsets.shape == (30, 3)
  assert torch.allclose(offsets[0], torch.tensor([-0.05, -0.03, -0.025]))
  assert torch.allclose(offsets[-1], torch.tensor([0.13, 0.03, -0.025]))
  assert torch.allclose(offsets[:6, 0], torch.full((6,), -0.05))
  assert torch.allclose(offsets[:6, 1], torch.linspace(-0.03, 0.03, 6))


def test_make_capsule_footprint_sample_offsets_follows_collision_sole_shape():
  fromto_xy = (
    ((0.10, -0.026), (0.05, -0.027)),
    ((-0.044, -0.018), (0.123, -0.018)),
    ((-0.052, -0.010), (0.130, -0.010)),
    ((-0.054, 0.000), (0.132, 0.000)),
    ((-0.052, 0.010), (0.130, 0.010)),
    ((-0.044, 0.018), (0.123, 0.018)),
    ((0.10, 0.026), (0.05, 0.026)),
  )

  offsets = make_capsule_footprint_sample_offsets(
    capsule_fromto_xy_m=fromto_xy,
    z_m=-0.025,
    target_count=30,
  )

  assert offsets.shape == (30, 3)
  assert torch.allclose(offsets[:, 2], torch.full((30,), -0.025))
  assert float(offsets[:, 0].min().item()) < -0.05
  assert float(offsets[:, 0].max().item()) > 0.125
  assert float((offsets[:, 0].max() - offsets[:, 0].min()).item()) > 0.17
  assert float((offsets[:, 1].max() - offsets[:, 1].min()).item()) > 0.045
  assert bool(
    (
      _min_distance_to_segments(offsets[:, :2], fromto_xy)
      <= torch.tensor(0.010 + 1.0e-5)
    ).all().item()
  )


def test_make_capsule_footprint_sample_offsets_uniformly_fills_sole_area():
  fromto_xy = (
    ((0.10, -0.026), (0.05, -0.027)),
    ((-0.044, -0.018), (0.123, -0.018)),
    ((-0.052, -0.010), (0.130, -0.010)),
    ((-0.054, 0.000), (0.132, 0.000)),
    ((-0.052, 0.010), (0.130, 0.010)),
    ((-0.044, 0.018), (0.123, 0.018)),
    ((0.10, 0.026), (0.05, 0.026)),
  )

  offsets = make_capsule_footprint_sample_offsets(
    capsule_fromto_xy_m=fromto_xy,
    z_m=-0.025,
    target_count=80,
  )

  assert offsets.shape == (80, 3)
  unique_y = torch.unique(torch.round(offsets[:, 1] * 10000.0) / 10000.0)
  assert unique_y.numel() > len(fromto_xy)
  assert bool(
    (
      _min_distance_to_segments(offsets[:, :2], fromto_xy)
      <= torch.tensor(0.010 + 1.0e-5)
    ).all().item()
  )


def test_make_capsule_footprint_sample_offsets_avoids_lattice_rows():
  fromto_xy = (
    ((0.10, -0.026), (0.05, -0.027)),
    ((-0.044, -0.018), (0.123, -0.018)),
    ((-0.052, -0.010), (0.130, -0.010)),
    ((-0.054, 0.000), (0.132, 0.000)),
    ((-0.052, 0.010), (0.130, 0.010)),
    ((-0.044, 0.018), (0.123, 0.018)),
    ((0.10, 0.026), (0.05, 0.026)),
  )

  offsets = make_capsule_footprint_sample_offsets(
    capsule_fromto_xy_m=fromto_xy,
    z_m=-0.025,
    target_count=300,
  )

  local_xy = offsets[:, :2]
  rounded_y = torch.unique(torch.round(local_xy[:, 1] * 10000.0) / 10000.0)
  distance = torch.cdist(local_xy, local_xy)
  distance.fill_diagonal_(float("inf"))
  nearest = distance.min(dim=1).values

  assert rounded_y.numel() > 100
  assert float(nearest.min().item()) > 0.0015


def test_foot_grid_point_velocities_include_body_angular_velocity():
  offsets = torch.tensor([[[0.10, 0.0, 0.0]]])
  body_pos_w = torch.zeros(1, 1, 3)
  body_quat_w = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]])
  body_lin_vel_w = torch.tensor([[[0.0, 0.0, -1.0]]])
  body_ang_vel_w = torch.tensor([[[0.0, 10.0, 0.0]]])

  velocity = foot_grid_point_velocities(
    body_pos_w=body_pos_w,
    body_quat_w=body_quat_w,
    body_lin_vel_w=body_lin_vel_w,
    body_ang_vel_w=body_ang_vel_w,
    local_offsets_m=offsets,
  )

  assert velocity.shape == (1, 1, 1, 3)
  assert torch.allclose(velocity[0, 0, 0], torch.tensor([0.0, 0.0, -2.0]))


def test_distribute_capsule_forces_to_foot_grid_conserves_force_near_segments():
  capsule_force = torch.tensor([[10.0, 20.0, 30.0, 40.0]])
  capsule_fromto_xy = (
    ((0.0, -0.02), (0.1, -0.02)),
    ((0.0, 0.02), (0.1, 0.02)),
    ((0.0, -0.02), (0.1, -0.02)),
    ((0.0, 0.02), (0.1, 0.02)),
  )
  capsule_radius = (0.01, 0.01, 0.01, 0.01)
  local_xy = torch.tensor([[0.05, -0.02], [0.05, 0.02]])

  point_force = distribute_capsule_forces_to_foot_grid(
    capsule_z_force_n=capsule_force,
    capsule_fromto_xy_m=capsule_fromto_xy,
    capsule_radius_m=capsule_radius,
    local_xy_m=local_xy,
  )

  assert point_force.shape == (1, 2, 2)
  assert torch.allclose(point_force[0, 0], torch.tensor([10.0, 20.0]))
  assert torch.allclose(point_force[0, 1], torch.tensor([30.0, 40.0]))
  assert torch.allclose(point_force.sum(dim=2), capsule_force.reshape(1, 2, 2).sum(dim=2))


def test_distribute_contact_forces_to_foot_grid_only_lights_contact_points():
  contact_force = torch.tensor([[[12.0]]])
  contact_pos_local_xy = torch.tensor([[[[0.0, 0.0]]]])
  contact_mask = torch.tensor([[[True]]])
  local_xy = torch.tensor([[0.0, 0.0], [0.04, 0.0], [0.0, 0.04]])

  point_force = distribute_contact_forces_to_foot_grid(
    contact_force_n=contact_force,
    contact_pos_local_xy_m=contact_pos_local_xy,
    contact_mask=contact_mask,
    local_xy_m=local_xy,
    contact_radius_m=0.02,
  )

  assert point_force.shape == (1, 1, 3)
  assert torch.allclose(point_force[0, 0], torch.tensor([12.0, 0.0, 0.0]))

  no_contact_force = distribute_contact_forces_to_foot_grid(
    contact_force_n=contact_force,
    contact_pos_local_xy_m=contact_pos_local_xy,
    contact_mask=torch.tensor([[[False]]]),
    local_xy_m=local_xy,
    contact_radius_m=0.02,
  )
  assert torch.equal(no_contact_force, torch.zeros_like(no_contact_force))


def test_summarize_post_contact_foot_grid_uses_touchdown_edges_and_post_steps():
  foot_contact = torch.tensor(
    [
      [False, False],
      [True, False],
      [True, True],
      [False, True],
    ]
  )
  grid_velocity = torch.zeros(4, 2, 2, 3)
  grid_velocity[1, 0, 0, 2] = -0.4
  grid_velocity[1, 0, 1, 2] = -0.2
  grid_velocity[2, 0, 0, 2] = -0.1
  grid_velocity[2, 0, 1, 2] = -0.6
  grid_velocity[2, 1, 0, 2] = -0.3
  grid_velocity[2, 1, 1, 2] = -0.5
  grid_velocity[3, 1, 0, 2] = -0.7
  grid_velocity[3, 1, 1, 2] = -0.9
  trace = _minimal_trace(
    foot_contact_flag=foot_contact.float(),
    foot_grid_names=("p00", "p01"),
    foot_grid_shape=(1, 2),
    foot_grid_local_xy_m=torch.tensor([[0.0, -0.03], [0.0, 0.03]]),
    foot_grid_velocity_m_s=grid_velocity,
  )

  rows = summarize_post_contact_foot_grid(trace, post_steps=(0, 1))

  left_p00 = next(row for row in rows if row.foot == "left" and row.point_name == "p00")
  right_p01 = next(row for row in rows if row.foot == "right" and row.point_name == "p01")
  assert left_p00.event_count == 1
  assert left_p00.down_speed_mean_by_step_m_s[0] == 0.4
  assert left_p00.down_speed_mean_by_step_m_s[1] == 0.1
  assert right_p01.event_count == 1
  assert right_p01.down_speed_mean_by_step_m_s[0] == 0.5
  assert right_p01.down_speed_mean_by_step_m_s[1] == 0.9


def test_foot_grid_down_speed_frames_returns_time_varying_heatmap():
  grid_velocity = torch.zeros(3, 2, 2, 3)
  grid_velocity[0, 0, 0, 2] = -0.1
  grid_velocity[1, 0, 1, 2] = -0.4
  grid_velocity[2, 1, 0, 2] = -0.7
  trace = _minimal_trace(
    foot_grid_names=("p00", "p01"),
    foot_grid_shape=(1, 2),
    foot_grid_local_xy_m=torch.tensor([[0.0, -0.03], [0.0, 0.03]]),
    foot_grid_velocity_m_s=grid_velocity,
  )

  frames = foot_grid_down_speed_frames(trace)

  assert frames.shape == (3, 2, 1, 2)
  assert torch.allclose(frames[0, 0, 0, 0], torch.tensor(0.1))
  assert torch.allclose(frames[1, 0, 0, 1], torch.tensor(0.4))
  assert torch.allclose(frames[2, 1, 0, 0], torch.tensor(0.7))


def test_save_foot_grid_velocity_video_writes_animated_gif(tmp_path):
  from PIL import Image

  grid_velocity = torch.zeros(3, 2, 2, 3)
  grid_velocity[0, 0, 0, 2] = -0.1
  grid_velocity[1, 0, 1, 2] = -0.4
  grid_velocity[2, 1, 0, 2] = -0.7
  trace = _minimal_trace(
    foot_grid_names=("p00", "p01"),
    foot_grid_shape=(1, 2),
    foot_grid_local_xy_m=torch.tensor([[0.0, -0.03], [0.0, 0.03]]),
    foot_grid_velocity_m_s=grid_velocity,
  )

  output = save_foot_grid_velocity_video(
    trace,
    tmp_path / "foot_grid.gif",
    title="synthetic",
    dt=0.02,
    fps=30,
  )

  assert output.exists()
  assert output.stat().st_size > 0
  with Image.open(output) as image:
    assert 25 <= image.info["duration"] <= 35


def test_save_foot_grid_velocity_video_accepts_signed_vz_metric(tmp_path):
  grid_velocity = torch.zeros(3, 2, 2, 3)
  grid_velocity[0, 0, 0, 2] = -0.1
  grid_velocity[1, 0, 1, 2] = 0.4
  trace = _minimal_trace(
    foot_grid_names=("p00", "p01"),
    foot_grid_shape=(1, 2),
    foot_grid_local_xy_m=torch.tensor([[0.0, -0.03], [0.0, 0.03]]),
    foot_grid_velocity_m_s=grid_velocity,
  )

  output = save_foot_grid_velocity_video(
    trace,
    tmp_path / "foot_grid_signed.gif",
    title="synthetic",
    dt=0.02,
    fps=30,
    metric="signed_vz",
  )

  assert output.exists()
  assert output.stat().st_size > 0


def test_save_foot_grid_velocity_video_can_add_pressure_row(tmp_path):
  grid_velocity = torch.zeros(3, 2, 2, 3)
  grid_velocity[0, 0, 0, 2] = -0.1
  grid_velocity[1, 0, 1, 2] = 0.4
  grid_force = torch.zeros(3, 2, 2)
  grid_force[1, 0, 0] = 12.0
  grid_force[1, 1, 1] = 18.0
  trace = _minimal_trace(
    foot_grid_names=("p00", "p01"),
    foot_grid_shape=(1, 2),
    foot_grid_local_xy_m=torch.tensor([[0.0, -0.03], [0.0, 0.03]]),
    foot_grid_velocity_m_s=grid_velocity,
    foot_grid_force_n=grid_force,
  )

  output = save_foot_grid_velocity_video(
    trace,
    tmp_path / "foot_grid_pressure.gif",
    title="synthetic",
    dt=0.02,
    fps=30,
    metric="signed_vz",
    include_pressure=True,
  )

  assert output.exists()
  assert output.stat().st_size > 0


def test_signed_vz_video_colormap_centers_zero_on_white():
  cmap = _signed_vz_video_colormap()

  assert colors.to_hex(cmap(0.0)) == "#b91c1c"
  assert colors.to_hex(cmap(0.5)) == "#ffffff"
  assert colors.to_hex(cmap(1.0)) == "#1d4ed8"


def test_pressure_video_colormap_maps_zero_force_to_white():
  cmap = _pressure_video_colormap()

  assert colors.to_hex(cmap(0.0)) == "#ffffff"
  assert colors.to_hex(cmap(1.0)) == "#991b1b"


def test_write_foot_grid_raw_csv_exports_per_frame_full_velocity(tmp_path):
  grid_velocity = torch.zeros(2, 2, 2, 3)
  grid_velocity[0, 0, 0] = torch.tensor([0.1, 0.2, -0.3])
  grid_velocity[1, 1, 1] = torch.tensor([0.4, 0.0, 0.5])
  trace = _minimal_trace(
    foot_contact_flag=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
    foot_grid_names=("p00", "p01"),
    foot_grid_shape=(1, 2),
    foot_grid_local_xy_m=torch.tensor([[0.0, -0.03], [0.0, 0.03]]),
    foot_grid_velocity_m_s=grid_velocity,
  )

  output = write_foot_grid_raw_csv(trace, tmp_path / "raw.csv", dt=0.02)

  with output.open(newline="") as f:
    rows = list(csv.DictReader(f))
  assert len(rows) == 8
  first = rows[0]
  assert first["step"] == "0"
  assert first["time_s"] == "0.0"
  assert first["foot"] == "left"
  assert first["point_name"] == "p00"
  assert float(first["vx_m_s"]) == 0.1
  assert float(first["vy_m_s"]) == 0.2
  assert float(first["vz_m_s"]) == -0.3
  assert float(first["downward_speed_m_s"]) == 0.3
  assert first["foot_contact"] == "1"


def _min_distance_to_segments(
  points_xy: torch.Tensor,
  fromto_xy: tuple[tuple[tuple[float, float], tuple[float, float]], ...],
) -> torch.Tensor:
  distances = []
  for start_xy, end_xy in fromto_xy:
    start = torch.tensor(start_xy, dtype=points_xy.dtype)
    end = torch.tensor(end_xy, dtype=points_xy.dtype)
    segment = end - start
    length_sq = torch.dot(segment, segment).clamp_min(torch.finfo(points_xy.dtype).eps)
    t = torch.sum((points_xy - start) * segment, dim=1) / length_sq
    t = torch.clamp(t, min=0.0, max=1.0)
    projection = start + t[:, None] * segment
    distances.append(torch.linalg.norm(points_xy - projection, dim=1))
  return torch.stack(distances, dim=1).min(dim=1).values
