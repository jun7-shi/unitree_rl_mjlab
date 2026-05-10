import unittest

import torch

from src.evaluation.silent_walking.contact_backends import (
  contact_points_to_foot_region_flags,
)
from src.evaluation.silent_walking.robots import get_foot_proxy_spec
from src.evaluation.silent_walking.telemetry import (
  SilentTelemetryCollector,
  detect_touchdown,
  positive_loading_rate,
  virtual_heel_toe_points,
)


class TelemetryHelperTests(unittest.TestCase):
  def test_positive_loading_rate_uses_largest_positive_force_delta(self):
    force = torch.tensor([[0.0, 10.0], [8.0, 12.0], [5.0, 30.0]])

    rate = positive_loading_rate(force, dt=0.02)

    self.assertTrue(
      torch.allclose(
        rate,
        torch.tensor([[0.0, 0.0], [400.0, 100.0], [0.0, 900.0]]),
      )
    )

  def test_detect_touchdown_uses_contact_rising_edge(self):
    previous = torch.tensor([False, True])
    current = torch.tensor([True, True])

    touchdown = detect_touchdown(previous, current)

    self.assertTrue(torch.equal(touchdown, torch.tensor([True, False])))

  def test_g1_foot_proxy_defines_sites_bodies_capsules_and_offsets(self):
    spec = get_foot_proxy_spec("g1")

    self.assertEqual(spec.foot_names, ("left", "right"))
    self.assertEqual(spec.foot_site_names, ("left_foot", "right_foot"))
    self.assertEqual(
      spec.foot_body_names,
      ("left_ankle_roll_link", "right_ankle_roll_link"),
    )
    self.assertEqual(spec.heel_local_offset_m, (-0.055, 0.0, -0.025))
    self.assertEqual(spec.toe_local_offset_m, (0.13, 0.0, -0.025))
    self.assertEqual(
      spec.foot_corner_names,
      ("rear_left", "rear_right", "front_left", "front_right"),
    )
    self.assertEqual(
      spec.foot_corner_local_offsets_m,
      (
        (-0.05, 0.025, -0.03),
        (-0.05, -0.025, -0.03),
        (0.12, 0.03, -0.03),
        (0.12, -0.03, -0.03),
      ),
    )
    self.assertEqual(spec.heel_region_max_x_m, -0.02)
    self.assertEqual(spec.toe_region_min_x_m, 0.09)
    self.assertEqual(len(spec.foot_collision_geom_names), 14)
    self.assertEqual(spec.foot_grid_shape, (0, 0))
    self.assertEqual(len(spec.foot_grid_names), 30)
    self.assertEqual(len(spec.foot_grid_local_offsets_m), 30)

  def test_virtual_heel_toe_points_apply_offsets_with_identity_quaternion(self):
    body_pos = torch.zeros(1, 2, 3)
    body_quat = torch.tensor(
      [[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]]
    )

    heel, toe = virtual_heel_toe_points(
      body_pos_w=body_pos,
      body_quat_w=body_quat,
      heel_local_offset_m=(-0.055, 0.0, -0.025),
      toe_local_offset_m=(0.13, 0.0, -0.025),
    )

    self.assertTrue(torch.allclose(heel[0, 0], torch.tensor([-0.055, 0.0, -0.025])))
    self.assertTrue(torch.allclose(toe[0, 1], torch.tensor([0.13, 0.0, -0.025])))

  def test_contact_points_to_region_flags_groups_capsules_by_foot_order(self):
    body_pos = torch.zeros(1, 2, 3)
    body_quat = torch.tensor(
      [[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]]
    )
    contact_pos = torch.zeros(1, 14, 3)
    contact_mask = torch.zeros(1, 14, dtype=torch.bool)
    contact_pos[0, 0] = torch.tensor([-0.05, 0.0, -0.03])
    contact_mask[0, 0] = True
    contact_pos[0, 7] = torch.tensor([0.12, 0.0, -0.03])
    contact_mask[0, 7] = True

    regions = contact_points_to_foot_region_flags(
      robot_name="g1",
      contact_pos_w=contact_pos,
      contact_mask=contact_mask,
      foot_body_pos_w=body_pos,
      foot_body_quat_w=body_quat,
    )

    self.assertTrue(torch.equal(regions[0, 0], torch.tensor([True, False, False])))
    self.assertTrue(torch.equal(regions[0, 1], torch.tensor([False, False, True])))


class SilentTelemetryCollectorTests(unittest.TestCase):
  def test_collector_records_touchdown_peak_loading_and_downward_speed(self):
    collector = SilentTelemetryCollector(
      robot_name="g1",
      dt=0.02,
      body_weight_newton=400.0,
    )

    collector.record_sample(
      action=torch.zeros(1, 2),
      command_velocity=torch.tensor([[0.5, 0.0, 0.0]]),
      actual_linear_velocity=torch.tensor([[0.4, 0.0]]),
      actual_yaw_rate=torch.tensor([0.0]),
      foot_force_n=torch.tensor([[0.0, 0.0]]),
      foot_contact=torch.tensor([[False, False]]),
      foot_site_vz_m_s=torch.tensor([[0.0, 0.0]]),
      capsule_names=("left_foot1_collision", "right_foot1_collision"),
      capsule_force_n=torch.tensor([[0.0, 0.0]]),
      capsule_contact=torch.tensor([[False, False]]),
      capsule_vz_m_s=torch.tensor([[0.0, 0.0]]),
      heel_pos_w=torch.zeros(1, 2, 3),
      toe_pos_w=torch.zeros(1, 2, 3),
      foot_roll_angle_rad=torch.zeros(1, 2),
    )
    collector.record_sample(
      action=torch.ones(1, 2),
      command_velocity=torch.tensor([[0.5, 0.0, 0.0]]),
      actual_linear_velocity=torch.tensor([[0.45, 0.0]]),
      actual_yaw_rate=torch.tensor([0.1]),
      foot_force_n=torch.tensor([[120.0, 0.0]]),
      foot_contact=torch.tensor([[True, False]]),
      foot_site_vz_m_s=torch.tensor([[-0.4, 0.0]]),
      capsule_names=("left_foot1_collision", "right_foot1_collision"),
      capsule_force_n=torch.tensor([[120.0, 0.0]]),
      capsule_contact=torch.tensor([[True, False]]),
      capsule_vz_m_s=torch.tensor([[-0.3, 0.0]]),
      heel_pos_w=torch.tensor([[[0.0, 0.0, 0.01], [0.0, 0.0, 0.02]]]),
      toe_pos_w=torch.tensor([[[0.1, 0.0, 0.03], [0.1, 0.0, 0.02]]]),
      foot_roll_angle_rad=torch.tensor([[0.1, 0.0]]),
    )

    trace, summary = collector.to_trace_and_summary()

    self.assertTrue(torch.allclose(trace.touchdown_vertical_speed_m_s, torch.tensor([0.4])))
    self.assertTrue(torch.allclose(trace.touchdown_peak_force_bw, torch.tensor([0.3])))
    self.assertAlmostEqual(trace.foot_loading_rate_n_s[1, 0].item(), 6000.0)
    self.assertLessEqual(summary.contact_quietness, 1.0)

  def test_collector_records_foot_grid_velocity_and_force_proxy(self):
    collector = SilentTelemetryCollector(
      robot_name="g1",
      dt=0.02,
      body_weight_newton=400.0,
    )
    grid_velocity = torch.zeros(1, 2, 2, 3)
    grid_velocity[0, 0, 0, 2] = -0.3
    grid_force = torch.tensor([[[12.0, 0.0], [0.0, 8.0]]])

    collector.record_sample(
      action=torch.zeros(1, 2),
      command_velocity=torch.tensor([[0.5, 0.0, 0.0]]),
      actual_linear_velocity=torch.tensor([[0.4, 0.0]]),
      actual_yaw_rate=torch.tensor([0.0]),
      foot_force_n=torch.tensor([[12.0, 8.0]]),
      foot_contact=torch.tensor([[True, True]]),
      foot_site_vz_m_s=torch.tensor([[-0.2, -0.1]]),
      capsule_names=("left_foot1_collision", "right_foot1_collision"),
      capsule_force_n=torch.tensor([[12.0, 8.0]]),
      capsule_contact=torch.tensor([[True, True]]),
      capsule_vz_m_s=torch.tensor([[-0.2, -0.1]]),
      heel_pos_w=torch.zeros(1, 2, 3),
      toe_pos_w=torch.zeros(1, 2, 3),
      foot_roll_angle_rad=torch.zeros(1, 2),
      foot_grid_names=("p00", "p01"),
      foot_grid_local_xy_m=torch.tensor([[0.0, -0.03], [0.0, 0.03]]),
      foot_grid_velocity_m_s=grid_velocity,
      foot_grid_force_n=grid_force,
    )

    trace, _ = collector.to_trace_and_summary()

    self.assertEqual(trace.foot_grid_names, ("p00", "p01"))
    self.assertTrue(
      torch.equal(
        trace.foot_grid_local_xy_m,
        torch.tensor([[0.0, -0.03], [0.0, 0.03]]),
      )
    )
    self.assertTrue(torch.equal(trace.foot_grid_velocity_m_s, grid_velocity))
    self.assertTrue(torch.equal(trace.foot_grid_force_n, grid_force))

  def test_collector_records_corner_velocity_and_secondary_region_event(self):
    collector = SilentTelemetryCollector(
      robot_name="g1",
      dt=0.02,
      body_weight_newton=400.0,
    )

    common = {
      "action": torch.zeros(1, 2),
      "command_velocity": torch.tensor([[0.5, 0.0, 0.0]]),
      "actual_linear_velocity": torch.tensor([[0.5, 0.0]]),
      "actual_yaw_rate": torch.tensor([0.0]),
      "capsule_names": ("left_foot1_collision", "right_foot1_collision"),
      "capsule_force_n": torch.tensor([[0.0, 0.0]]),
      "capsule_contact": torch.tensor([[False, False]]),
      "capsule_vz_m_s": torch.tensor([[0.0, 0.0]]),
      "heel_pos_w": torch.zeros(1, 2, 3),
      "toe_pos_w": torch.zeros(1, 2, 3),
      "foot_roll_angle_rad": torch.zeros(1, 2),
    }
    collector.record_sample(
      **common,
      foot_force_n=torch.tensor([[0.0, 0.0]]),
      foot_contact=torch.tensor([[False, False]]),
      foot_site_vz_m_s=torch.tensor([[0.0, 0.0]]),
      foot_corner_pos_w=torch.zeros(1, 2, 4, 3),
      foot_region_contact=torch.zeros(1, 2, 3, dtype=torch.bool),
    )
    collector.record_sample(
      **common,
      foot_force_n=torch.tensor([[100.0, 0.0]]),
      foot_contact=torch.tensor([[True, False]]),
      foot_site_vz_m_s=torch.tensor([[-0.3, 0.0]]),
      foot_corner_pos_w=torch.tensor(
        [[
          [
            [0.0, 0.0, -0.010],
            [0.0, 0.0, -0.012],
            [0.0, 0.0, 0.020],
            [0.0, 0.0, 0.020],
          ],
          torch.zeros(4, 3).tolist(),
        ]]
      ),
      foot_region_contact=torch.tensor([[[True, False, False], [False, False, False]]]),
    )
    collector.record_sample(
      **common,
      foot_force_n=torch.tensor([[150.0, 0.0]]),
      foot_contact=torch.tensor([[True, False]]),
      foot_site_vz_m_s=torch.tensor([[0.0, 0.0]]),
      foot_corner_pos_w=torch.tensor(
        [[
          [
            [0.0, 0.0, -0.006],
            [0.0, 0.0, -0.008],
            [0.0, 0.0, 0.010],
            [0.0, 0.0, 0.008],
          ],
          torch.zeros(4, 3).tolist(),
        ]]
      ),
      foot_region_contact=torch.tensor([[[True, False, True], [False, False, False]]]),
    )

    trace, _ = collector.to_trace_and_summary()

    self.assertEqual(trace.foot_corner_vertical_velocity_m_s.shape, (3, 2, 4))
    self.assertAlmostEqual(trace.foot_corner_vertical_velocity_m_s[2, 0, 2].item(), -0.5)
    self.assertEqual(trace.region_event_type, ("touchdown", "secondary"))
    self.assertEqual(trace.region_event_regions, ("heel", "toe"))
    self.assertTrue(torch.allclose(trace.region_event_peak_force_bw, torch.tensor([0.25, 0.375])))
    self.assertTrue(torch.equal(trace.region_event_stance_id, torch.tensor([0, 0])))
    self.assertTrue(torch.equal(trace.region_event_step, torch.tensor([1, 2])))
    self.assertTrue(torch.equal(trace.region_event_foot_index, torch.tensor([0, 0])))
    self.assertTrue(
      torch.allclose(trace.region_event_corner_downward_speed_m_s, torch.tensor([0.6, 0.6]))
    )
    self.assertTrue(
      torch.allclose(
        trace.region_event_corner_downward_speeds_m_s,
        torch.tensor(
          [
            [0.5, 0.6, 0.0, 0.0],
            [0.0, 0.0, 0.5, 0.6],
          ]
        ),
      )
    )

  def test_region_events_use_substep_contact_and_velocity_when_available(self):
    collector = SilentTelemetryCollector(
      robot_name="g1",
      dt=0.02,
      body_weight_newton=400.0,
    )

    common = {
      "action": torch.zeros(1, 2),
      "command_velocity": torch.tensor([[0.5, 0.0, 0.0]]),
      "actual_linear_velocity": torch.tensor([[0.5, 0.0]]),
      "actual_yaw_rate": torch.tensor([0.0]),
      "capsule_names": ("left_foot1_collision", "right_foot1_collision"),
      "capsule_force_n": torch.tensor([[0.0, 0.0]]),
      "capsule_contact": torch.tensor([[False, False]]),
      "capsule_vz_m_s": torch.tensor([[0.0, 0.0]]),
      "heel_pos_w": torch.zeros(1, 2, 3),
      "toe_pos_w": torch.zeros(1, 2, 3),
      "foot_roll_angle_rad": torch.zeros(1, 2),
    }
    collector.record_sample(
      **common,
      foot_force_n=torch.tensor([[0.0, 0.0]]),
      foot_contact=torch.tensor([[False, False]]),
      foot_site_vz_m_s=torch.tensor([[0.0, 0.0]]),
      foot_region_contact=torch.zeros(1, 2, 3, dtype=torch.bool),
    )
    collector.record_sample(
      **common,
      foot_force_n=torch.tensor([[160.0, 0.0]]),
      foot_contact=torch.tensor([[True, False]]),
      foot_site_vz_m_s=torch.tensor([[0.0, 0.0]]),
      foot_region_contact=torch.tensor([[[True, False, True], [False, False, False]]]),
      substep_dt=0.005,
      substep_foot_force_n=torch.tensor(
        [
          [100.0, 0.0],
          [160.0, 0.0],
        ]
      ),
      substep_foot_site_vz_m_s=torch.tensor(
        [
          [-0.8, 0.0],
          [-0.1, 0.0],
        ]
      ),
      substep_foot_region_contact=torch.tensor(
        [
          [[True, False, False], [False, False, False]],
          [[True, False, True], [False, False, False]],
        ]
      ),
      substep_corner_downward_speeds_m_s=torch.tensor(
        [
          [[0.7, 0.6, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
          [[0.0, 0.0, 0.5, 0.6], [0.0, 0.0, 0.0, 0.0]],
        ]
      ),
    )

    trace, _ = collector.to_trace_and_summary()

    self.assertEqual(trace.region_event_type, ("touchdown", "secondary"))
    self.assertEqual(trace.region_event_regions, ("heel", "toe"))
    self.assertTrue(torch.equal(trace.region_event_step, torch.tensor([1, 1])))
    self.assertTrue(torch.equal(trace.region_event_substep_index, torch.tensor([0, 1])))
    self.assertTrue(
      torch.allclose(trace.region_event_time_offset_s, torch.tensor([0.005, 0.010]))
    )
    self.assertTrue(
      torch.allclose(trace.region_event_vertical_speed_m_s, torch.tensor([0.8, 0.1]))
    )
    self.assertTrue(
      torch.allclose(trace.region_event_corner_downward_speed_m_s, torch.tensor([0.7, 0.6]))
    )


if __name__ == "__main__":
  unittest.main()
