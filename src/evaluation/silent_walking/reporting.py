"""Markdown reporting helpers for silent walking evaluation."""

from __future__ import annotations

from .types import EpisodeMetricSummary


def _escape_inline_code(value: str) -> str:
  """Wrap values in a safe Markdown code span fence."""

  max_run = 0
  current_run = 0
  for char in value:
    if char == "`":
      current_run += 1
      max_run = max(max_run, current_run)
    else:
      current_run = 0

  fence = "`" * (max_run + 1)
  return f"{fence}{value}{fence}"


def render_markdown_report(
  robot_name: str, policy_label: str, summary: EpisodeMetricSummary
) -> str:
  """Render a compact markdown report for one evaluated policy."""

  return f"""# Silent Walking Report

- Robot: {_escape_inline_code(robot_name)}
- Policy: {_escape_inline_code(policy_label)}
- Contact Quietness: `{summary.contact_quietness:.2f}`
- Body Smoothness: `{summary.body_smoothness:.2f}`
- Task Compliance: `{summary.task_compliance:.2f}`
- Total Score: `{summary.total_score:.2f}`
"""
