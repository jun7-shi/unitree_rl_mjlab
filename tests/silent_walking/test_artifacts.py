from pathlib import Path
import json
import tempfile
import unittest

import torch

from src.evaluation.silent_walking.artifacts import write_eval_artifacts
from src.evaluation.silent_walking.runner import SilentEvalResult
from src.evaluation.silent_walking.types import EpisodeMetricSummary, EpisodeMetricTrace


class ArtifactWriterTests(unittest.TestCase):
  def test_write_eval_artifacts_creates_report_summary_trace_and_touchdowns(self):
    trace = EpisodeMetricTrace(
      foot_z_force_n=torch.tensor([[10.0, 0.0], [0.0, 12.0]]),
      foot_contact_flag=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
      foot_vertical_velocity_m_s=torch.tensor([[-0.2, 0.0], [0.0, -0.3]]),
      capsule_names=("left_foot1_collision", "right_foot1_collision"),
      capsule_layout_xy_m=torch.zeros(2, 2),
      capsule_outline_fromto_xy_m=torch.zeros(0, 2, 2),
      capsule_radius_m=torch.zeros(0),
      capsule_z_force_n=torch.tensor([[10.0, 0.0], [0.0, 12.0]]),
      capsule_contact_flag=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
      capsule_vertical_velocity_m_s=torch.tensor([[-0.2, 0.0], [0.0, -0.3]]),
      peak_force_bw=torch.tensor([0.2]),
      loading_rate_bw_s=torch.tensor([1.0]),
      touchdown_peak_force_bw=torch.tensor([0.1, 0.2]),
      touchdown_loading_rate_bw_s=torch.tensor([0.5, 1.0]),
      touchdown_vertical_speed_m_s=torch.tensor([0.2, 0.3]),
      left_touchdown_peak_force_bw=torch.tensor([0.1]),
      right_touchdown_peak_force_bw=torch.tensor([0.2]),
      left_touchdown_loading_rate_bw_s=torch.tensor([0.5]),
      right_touchdown_loading_rate_bw_s=torch.tensor([1.0]),
      left_touchdown_vertical_speed_m_s=torch.tensor([0.2]),
      right_touchdown_vertical_speed_m_s=torch.tensor([0.3]),
      touchdown_peak_asymmetry_bw=torch.tensor([0.1]),
      capsule_touchdown_peak_force_bw=torch.tensor([0.1, 0.2]),
      capsule_touchdown_loading_rate_bw_s=torch.tensor([0.5, 1.0]),
      capsule_touchdown_vertical_speed_m_s=torch.tensor([0.2, 0.3]),
      capsule_touchdown_count=torch.tensor([1.0, 1.0]),
      action_rate_l2=torch.tensor([0.0, 0.1]),
      command_velocity=torch.tensor([[0.2, 0.0, 0.0], [0.2, 0.0, 0.0]]),
      actual_linear_velocity=torch.tensor([[0.1, 0.0], [0.2, 0.0]]),
      actual_yaw_rate=torch.tensor([0.0, 0.0]),
      linear_velocity_error=torch.tensor([0.1, 0.0]),
      yaw_rate_error=torch.tensor([0.0, 0.0]),
      contact_quietness_score=torch.tensor([0.5]),
      foot_loading_rate_bw_s=torch.tensor([[0.0, 0.0], [0.0, 0.7]]),
      heel_vertical_velocity_m_s=torch.tensor([[0.0, 0.0], [0.0, -0.2]]),
      toe_vertical_velocity_m_s=torch.tensor([[0.0, 0.0], [0.0, -0.1]]),
      heel_height_m=torch.tensor([[0.01, 0.02], [0.03, 0.01]]),
      toe_height_m=torch.tensor([[0.02, 0.02], [0.04, 0.02]]),
      foot_roll_angle_rad=torch.tensor([[0.1, 0.0], [0.2, 0.1]]),
      touchdown_heel_vz_m_s=torch.tensor([-0.2, -0.1]),
      touchdown_toe_vz_m_s=torch.tensor([-0.1, -0.05]),
      touchdown_roll_angle_rad=torch.tensor([0.2, 0.1]),
      touchdown_first_contact_region=("heel", "flat"),
      foot_region_names=("heel", "midfoot", "toe"),
      foot_region_contact_flag=torch.tensor(
        [
          [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
          [[1.0, 0.0, 1.0], [0.0, 0.0, 0.0]],
        ]
      ),
      foot_corner_names=("rear_left", "rear_right", "front_left", "front_right"),
      foot_corner_height_m=torch.zeros(2, 2, 4),
      foot_corner_vertical_velocity_m_s=torch.tensor(
        [
          [[0.0, 0.0, -0.1, -0.1], [0.0, 0.0, 0.0, 0.0]],
          [[0.1, 0.1, -0.2, -0.3], [0.0, 0.0, 0.0, 0.0]],
        ]
      ),
      region_event_type=("touchdown", "secondary"),
      region_event_stance_id=torch.tensor([0, 0]),
      region_event_step=torch.tensor([0, 1]),
      region_event_foot_index=torch.tensor([0, 0]),
      region_event_foot=("left", "left"),
      region_event_regions=("heel", "toe"),
      region_event_peak_force_bw=torch.tensor([0.1, 0.2]),
      region_event_loading_rate_bw_s=torch.tensor([0.5, 0.8]),
      region_event_vertical_speed_m_s=torch.tensor([0.2, 0.1]),
      region_event_corner_downward_speed_m_s=torch.tensor([0.3, 0.4]),
      region_event_corner_downward_speeds_m_s=torch.tensor(
        [
          [0.2, 0.3, 0.1, 0.0],
          [0.0, 0.0, 0.4, 0.35],
        ]
      ),
    )
    result = SilentEvalResult(
      robot_name="g1",
      task_id="Unitree-G1-Tracking",
      steps=2,
      step_dt=0.02,
      trace=trace,
      summary=EpisodeMetricSummary(0.5, 0.6, 0.7, 0.6),
    )

    with tempfile.TemporaryDirectory() as tmpdir:
      paths = write_eval_artifacts(result, policy_label="policy.pt", output_dir=tmpdir)

      self.assertEqual(
        set(paths.keys()),
        {"summary", "report", "trace", "touchdowns", "regional_events", "stance_patterns"},
      )
      for path in paths.values():
        self.assertTrue(Path(path).exists())
      self.assertIn("foot0_loading_rate_bw_s", Path(paths["trace"]).read_text())
      self.assertIn("foot0_region_heel_contact", Path(paths["trace"]).read_text())
      self.assertIn("foot0_rear_left_vz_m_s", Path(paths["trace"]).read_text())
      self.assertIn("first_contact_region", Path(paths["touchdowns"]).read_text())
      regional_events = Path(paths["regional_events"]).read_text()
      self.assertIn("stance_id", regional_events)
      self.assertIn("step", regional_events)
      self.assertIn("substep_index", regional_events)
      self.assertIn("time_offset_s", regional_events)
      self.assertIn("rear_left_down_speed_m_s", regional_events)
      self.assertIn("front_right_down_speed_m_s", regional_events)
      self.assertIn("secondary", regional_events)
      stance_patterns = Path(paths["stance_patterns"]).read_text()
      self.assertIn("heel_to_toe_rollover", stance_patterns)
      self.assertIn("0,left,0,1,heel,heel+toe,2,1,0,1,1,0", stance_patterns)
      report = Path(paths["report"]).read_text()
      self.assertIn("TD Site Vz", report)
      self.assertIn("Region Event Speeds", report)
      self.assertIn("Regional Event Distribution", report)
      self.assertIn(
        "| `touchdown` | `1` | `1` | `0` | `0` | `0` | `0` | `0` |",
        report,
      )
      self.assertIn(
        "| `secondary` | `1` | `0` | `0` | `1` | `0` | `0` | `0` |",
        report,
      )
      self.assertIn("Four-Corner Post-Event Dynamics", report)
      self.assertIn("Stance-Level Contact Patterns", report)
      self.assertIn("Corner Max Down P95", report)
      summary = json.loads(Path(paths["summary"]).read_text())
      self.assertEqual(summary["raw_metrics"]["stance_count"], 1)
      self.assertEqual(summary["raw_metrics"]["rollover_stance_count"], 1)
      self.assertEqual(summary["raw_metrics"]["heel_to_toe_rollover_stance_count"], 1)
      self.assertEqual(summary["raw_metrics"]["multi_region_first_touchdown_stance_count"], 0)
      self.assertIn(
        "touchdown_front_right_post_1_step_down_speed_mean_m_s",
        summary["raw_metrics"],
      )
      self.assertIn(
        "touchdown_front_right_post_3_step_region_contact_ratio_mean",
        summary["raw_metrics"],
      )
      self.assertEqual(summary["raw_metrics"]["touchdown_region_event_count"], 1)
      self.assertEqual(summary["raw_metrics"]["secondary_region_event_count"], 1)
      self.assertAlmostEqual(
        summary["raw_metrics"]["touchdown_corner_downward_speed_p95_m_s"],
        0.3,
      )
      self.assertAlmostEqual(
        summary["raw_metrics"]["secondary_corner_downward_speed_p95_m_s"],
        0.4,
      )


if __name__ == "__main__":
  unittest.main()
