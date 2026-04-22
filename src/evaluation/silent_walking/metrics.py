"""Pure metric helpers for silent walking evaluation."""

from __future__ import annotations

import torch


def normalize_force_by_body_weight(
  force_newton: torch.Tensor | float, body_weight_newton: float
) -> torch.Tensor:
  """Normalize a force value by body weight."""

  force_tensor = torch.as_tensor(force_newton, dtype=torch.float32)
  return force_tensor / body_weight_newton


def compute_loading_rate(
  force_series: torch.Tensor | list[float], dt: float
) -> torch.Tensor:
  """Compute a simple loading-rate proxy over a force window."""

  force_tensor = torch.as_tensor(force_series, dtype=torch.float32)
  return (force_tensor.max() - force_tensor.min()) / dt


def compute_contact_quietness_score(
  peak_force_bw: torch.Tensor | float,
  loading_rate_bw_s: torch.Tensor | float,
) -> torch.Tensor:
  """Combine normalized peak force and loading rate into a bounded score."""

  peak_force_tensor = torch.as_tensor(peak_force_bw, dtype=torch.float32)
  loading_rate_tensor = torch.as_tensor(loading_rate_bw_s, dtype=torch.float32)
  penalty = 0.5 * peak_force_tensor + 0.05 * loading_rate_tensor
  return torch.clamp(1.0 - penalty, min=0.0, max=1.0)
