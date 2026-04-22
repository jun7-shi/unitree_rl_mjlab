"""Markdown reporting helpers for silent walking evaluation."""

from __future__ import annotations

from .types import EpisodeMetricSummary


def _escape_inline_code(value: str) -> str:
  """Escape backticks before embedding user strings in markdown code spans."""

  return value.replace("`", "\\`")


def render_markdown_report(
  robot_name: str, policy_label: str, summary: EpisodeMetricSummary
) -> str:
  """Render a compact markdown report for one evaluated policy."""

  return f"""# Silent Walking Report

- Robot: `{_escape_inline_code(robot_name)}`
- Policy: `{_escape_inline_code(policy_label)}`
- Contact Quietness: `{summary.contact_quietness:.2f}`
- Body Smoothness: `{summary.body_smoothness:.2f}`
- Task Compliance: `{summary.task_compliance:.2f}`
- Total Score: `{summary.total_score:.2f}`
"""
