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
  assert "policy.onnx" in report
  assert "0.81" in report
