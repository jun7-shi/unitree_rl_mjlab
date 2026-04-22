from src.evaluation.silent_walking.reporting import render_markdown_report
from src.evaluation.silent_walking.types import EpisodeMetricSummary


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

  assert "`g\\`1`" in report
  assert "`policy\\`.onnx`" in report
