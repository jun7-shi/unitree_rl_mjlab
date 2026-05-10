import unittest

import torch

from src.evaluation.silent_walking.post_event_dynamics import (
  summarize_four_corner_post_event_dynamics,
)
from src.evaluation.silent_walking.types import EpisodeMetricTrace


class PostEventDynamicsTests(unittest.TestCase):
  def test_summarizes_corner_speed_height_and_region_contact_after_event(self):
    trace = _minimal_trace(
      foot_region_contact_flag=torch.tensor(
        [
          [[1.0, 0.0, 0.0]],
          [[1.0, 0.0, 0.0]],
          [[1.0, 0.0, 1.0]],
          [[0.0, 0.0, 1.0]],
        ]
      ),
      foot_corner_height_m=torch.tensor(
        [
          [[0.030, 0.030, 0.040, 0.040]],
          [[0.010, 0.020, 0.030, 0.020]],
          [[0.020, 0.010, 0.010, 0.000]],
          [[0.030, 0.020, 0.020, 0.010]],
        ]
      ),
      foot_corner_vertical_velocity_m_s=torch.tensor(
        [
          [[-0.2, -0.3, -0.1, 0.0]],
          [[-0.1, -0.2, 0.05, -0.4]],
          [[0.1, 0.0, -0.2, -0.1]],
          [[0.0, -0.05, 0.1, -0.2]],
        ]
      ),
    )

    rows = summarize_four_corner_post_event_dynamics(trace)

    rear_left = _row(rows, "touchdown", "rear_left")
    self.assertEqual(rear_left.event_count, 1)
    self.assertAlmostEqual(rear_left.event_down_speed_mean_m_s, 0.2)
    self.assertAlmostEqual(rear_left.post_1_step_down_speed_mean_m_s, 0.1)
    self.assertAlmostEqual(rear_left.post_3_step_down_speed_mean_m_s, 0.0)
    self.assertAlmostEqual(rear_left.post_3_step_min_height_mean_m, 0.01)
    self.assertAlmostEqual(rear_left.post_3_step_region_contact_ratio_mean, 2.0 / 3.0)

    front_right = _row(rows, "touchdown", "front_right")
    self.assertAlmostEqual(front_right.event_down_speed_mean_m_s, 0.0)
    self.assertAlmostEqual(front_right.post_1_step_down_speed_mean_m_s, 0.4)
    self.assertAlmostEqual(front_right.post_3_step_down_speed_mean_m_s, 0.2)
    self.assertAlmostEqual(front_right.post_3_step_min_height_mean_m, 0.0)
    self.assertAlmostEqual(front_right.post_3_step_region_contact_ratio_mean, 2.0 / 3.0)


def _row(rows, event_type: str, corner_name: str):
  for row in rows:
    if row.event_type == event_type and row.corner_name == corner_name:
      return row
  raise AssertionError(f"Missing row for {event_type} {corner_name}")


def _minimal_trace(
  *,
  foot_region_contact_flag: torch.Tensor,
  foot_corner_height_m: torch.Tensor,
  foot_corner_vertical_velocity_m_s: torch.Tensor,
) -> EpisodeMetricTrace:
  return EpisodeMetricTrace(
    foot_z_force_n=torch.zeros(4, 1),
    foot_contact_flag=torch.zeros(4, 1),
    foot_vertical_velocity_m_s=torch.zeros(4, 1),
    capsule_names=(),
    capsule_layout_xy_m=torch.zeros(0, 2),
    capsule_outline_fromto_xy_m=torch.zeros(0, 2, 2),
    capsule_radius_m=torch.zeros(0),
    capsule_z_force_n=torch.zeros(4, 0),
    capsule_contact_flag=torch.zeros(4, 0),
    capsule_vertical_velocity_m_s=torch.zeros(4, 0),
    peak_force_bw=torch.zeros(1),
    loading_rate_bw_s=torch.zeros(1),
    touchdown_peak_force_bw=torch.zeros(0),
    touchdown_loading_rate_bw_s=torch.zeros(0),
    touchdown_vertical_speed_m_s=torch.zeros(0),
    left_touchdown_peak_force_bw=torch.zeros(0),
    right_touchdown_peak_force_bw=torch.zeros(0),
    left_touchdown_loading_rate_bw_s=torch.zeros(0),
    right_touchdown_loading_rate_bw_s=torch.zeros(0),
    left_touchdown_vertical_speed_m_s=torch.zeros(0),
    right_touchdown_vertical_speed_m_s=torch.zeros(0),
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
    foot_region_names=("heel", "midfoot", "toe"),
    foot_region_contact_flag=foot_region_contact_flag,
    foot_corner_names=("rear_left", "rear_right", "front_left", "front_right"),
    foot_corner_height_m=foot_corner_height_m,
    foot_corner_vertical_velocity_m_s=foot_corner_vertical_velocity_m_s,
    region_event_type=("touchdown",),
    region_event_stance_id=torch.tensor([0]),
    region_event_step=torch.tensor([0]),
    region_event_foot_index=torch.tensor([0]),
    region_event_foot=("left",),
    region_event_regions=("heel",),
    region_event_peak_force_bw=torch.tensor([0.1]),
    region_event_loading_rate_bw_s=torch.tensor([0.2]),
    region_event_vertical_speed_m_s=torch.tensor([0.3]),
    region_event_corner_downward_speed_m_s=torch.tensor([0.3]),
    region_event_corner_downward_speeds_m_s=torch.tensor([[0.2, 0.3, 0.1, 0.0]]),
  )


if __name__ == "__main__":
  unittest.main()
