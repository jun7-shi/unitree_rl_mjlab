"""Pure metric helpers for silent walking evaluation."""

from __future__ import annotations

from typing import Sequence

import torch


def _to_tensor(value: torch.Tensor | float | Sequence[float]) -> torch.Tensor:
  """Convert scalars and sequences to tensors without dropping tensor metadata."""

  if isinstance(value, torch.Tensor):
    return value
  return torch.as_tensor(value)


def normalize_force_by_body_weight(
  force_newton: torch.Tensor | float, body_weight_newton: float
) -> torch.Tensor:
  """Normalize a force value by body weight."""

  if body_weight_newton <= 0:
    raise ValueError("body_weight_newton must be > 0")

  force_tensor = _to_tensor(force_newton)
  return force_tensor / body_weight_newton


def compute_loading_rate(
  force_series: torch.Tensor | Sequence[float], dt: float
) -> torch.Tensor:
  """Compute a simple loading-rate proxy over a force window."""

  if dt <= 0:
    raise ValueError("dt must be > 0")

  force_tensor = _to_tensor(force_series)
  if force_tensor.numel() == 0:
    raise ValueError("force_series must not be empty")

  return (force_tensor.max() - force_tensor.min()) / dt


def compute_contact_quietness_score(
  peak_force_bw: torch.Tensor | float,
  loading_rate_bw_s: torch.Tensor | float,
) -> torch.Tensor:
  """Combine normalized peak force and loading rate into a bounded score."""

  peak_force_tensor = _to_tensor(peak_force_bw)
  loading_rate_tensor = _to_tensor(loading_rate_bw_s)
  penalty = 0.5 * peak_force_tensor + 0.05 * loading_rate_tensor
  return torch.clamp(1.0 - penalty, min=0.0, max=1.0)
