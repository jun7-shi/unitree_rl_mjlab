from src.evaluation.silent_walking.reporting import render_markdown_report
from src.evaluation.silent_walking.types import EpisodeMetricSummary, EpisodeMetricTrace
import torch


def test_render_markdown_report_contains_robot_policy_and_scores():
  summary = EpisodeMetricSummary(
    contact_quietness=0.75,
    body_smoothness=0.80,
    task_compliance=0.90,
    total_score=0.81,
  )

  report = render_markdown_report("g1", "policy.onnx", summary)

  assert "# Silent Walking Report" in report
  assert "- Robot: `g1`" in report
  assert "policy.onnx" in report
  assert "- Contact Quietness: `0.75`" in report
  assert "- Body Smoothness: `0.80`" in report
  assert "- Task Compliance: `0.90`" in report
  assert "0.81" in report


def test_render_markdown_report_escapes_backticks_in_labels():
  summary = EpisodeMetricSummary(
    contact_quietness=0.10,
    body_smoothness=0.20,
    task_compliance=0.30,
    total_score=0.40,
  )

  report = render_markdown_report("g`1", "policy`.onnx", summary)

  assert "``g`1``" in report
  assert "``policy`.onnx``" in report


def test_render_markdown_report_includes_raw_metrics_when_trace_is_present():
  summary = EpisodeMetricSummary(
    contact_quietness=0.10,
    body_smoothness=0.20,
    task_compliance=0.30,
    total_score=0.40,
  )
  trace = EpisodeMetricTrace(
    foot_z_force_n=torch.tensor([[10.0, 12.0], [0.0, 8.0]]),
    foot_contact_flag=torch.tensor([[1.0, 1.0], [0.0, 1.0]]),
    foot_vertical_velocity_m_s=torch.tensor([[-0.2, -0.3], [0.0, -0.1]]),
    capsule_names=("left_foot1_collision", "right_foot1_collision"),
    capsule_layout_xy_m=torch.tensor([[-0.01, 0.02], [0.01, -0.02]]),
    capsule_z_force_n=torch.tensor([[5.0, 7.0], [0.0, 6.0]]),
    capsule_contact_flag=torch.tensor([[1.0, 1.0], [0.0, 1.0]]),
    capsule_vertical_velocity_m_s=torch.tensor([[-0.1, -0.2], [0.0, -0.1]]),
    peak_force_bw=torch.tensor([1.5]),
    loading_rate_bw_s=torch.tensor([2.5]),
    touchdown_peak_force_bw=torch.tensor([1.2, 1.8]),
    touchdown_loading_rate_bw_s=torch.tensor([2.0, 3.0]),
    touchdown_vertical_speed_m_s=torch.tensor([0.2, 0.3]),
    left_touchdown_peak_force_bw=torch.tensor([1.2]),
    right_touchdown_peak_force_bw=torch.tensor([1.8]),
    left_touchdown_loading_rate_bw_s=torch.tensor([2.0]),
    right_touchdown_loading_rate_bw_s=torch.tensor([3.0]),
    left_touchdown_vertical_speed_m_s=torch.tensor([0.2]),
    right_touchdown_vertical_speed_m_s=torch.tensor([0.3]),
    touchdown_peak_asymmetry_bw=torch.tensor([0.6]),
    capsule_touchdown_peak_force_bw=torch.tensor([1.1, 1.7]),
    capsule_touchdown_loading_rate_bw_s=torch.tensor([1.9, 2.9]),
    capsule_touchdown_vertical_speed_m_s=torch.tensor([0.1, 0.2]),
    capsule_touchdown_count=torch.tensor([1.0, 2.0]),
    action_rate_l2=torch.tensor([0.5, 1.5]),
    command_velocity=torch.tensor([[0.3, 0.0, 0.1], [0.2, 0.0, 0.1]]),
    actual_linear_velocity=torch.tensor([[0.2, 0.0], [0.1, 0.0]]),
    actual_yaw_rate=torch.tensor([0.08, 0.09]),
    linear_velocity_error=torch.tensor([0.2]),
    yaw_rate_error=torch.tensor([0.1]),
    contact_quietness_score=torch.tensor([0.1]),
  )

  report = render_markdown_report("g1", "policy.onnx", summary, trace=trace)

  assert "## Raw Metrics" in report
  assert "Peak Force / BW" in report
  assert "Loading Rate / BW/s" in report
  assert "Touchdown Peak Force / BW" in report
  assert "Touchdown Loading Rate / BW/s" in report
  assert "Touchdown Vertical Speed / m/s" in report
  assert "Left Touchdown Peak / BW" in report
  assert "Right Touchdown Peak / BW" in report
  assert "Touchdown Peak Asymmetry / BW" in report
  assert "Per-Capsule Touchdown Metrics" in report
  assert "left_foot1_collision" in report
  assert "Action Rate L2" in report
  assert "Linear Velocity Error" in report
  assert "Yaw Rate Error" in report
