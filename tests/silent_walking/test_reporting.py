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
    peak_force_bw=torch.tensor([1.5]),
    loading_rate_bw_s=torch.tensor([2.5]),
    action_rate_l2=torch.tensor([0.5, 1.5]),
    linear_velocity_error=torch.tensor([0.2]),
    yaw_rate_error=torch.tensor([0.1]),
    contact_quietness_score=torch.tensor([0.1]),
  )

  report = render_markdown_report("g1", "policy.onnx", summary, trace=trace)

  assert "## Raw Metrics" in report
  assert "Peak Force / BW" in report
  assert "Loading Rate / BW/s" in report
  assert "Action Rate L2" in report
  assert "Linear Velocity Error" in report
  assert "Yaw Rate Error" in report
