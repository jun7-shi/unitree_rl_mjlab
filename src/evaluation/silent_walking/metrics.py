"""Pure metric helpers for silent walking evaluation."""

from __future__ import annotations

from typing import Sequence

import torch


def _to_tensor(value: torch.Tensor | float | Sequence[float]) -> torch.Tensor:
  """Convert scalars and sequences to tensors without dropping tensor metadata."""

  if isinstance(value, torch.Tensor):
    return value
  return torch.as_tensor(value)


def _validate_finite(name: str, tensor: torch.Tensor) -> None:
  """Reject NaN and Inf inputs before metric computation."""

  if not torch.isfinite(tensor).all():
    raise ValueError(f"{name} must contain only finite values")


def _validate_positive_scalar(name: str, value: float) -> None:
  """Reject non-finite or non-positive scalar denominators."""

  value_tensor = torch.as_tensor(value)
  _validate_finite(name, value_tensor)
  if value <= 0:
    raise ValueError(f"{name} must be > 0")


def _validate_non_negative(name: str, tensor: torch.Tensor) -> None:
  """Reject physically invalid negative-valued metric inputs."""

  if (tensor < 0).any():
    raise ValueError(f"{name} must contain only non-negative values")


def normalize_force_by_body_weight(
  force_newton: torch.Tensor | float, body_weight_newton: float
) -> torch.Tensor:
  """Normalize a force value by body weight."""

  _validate_positive_scalar("body_weight_newton", body_weight_newton)

  force_tensor = _to_tensor(force_newton)
  _validate_finite("force_newton", force_tensor)
  return force_tensor / body_weight_newton


def compute_loading_rate(
  force_series: torch.Tensor | Sequence[float], dt: float
) -> torch.Tensor:
  """Compute a simple loading-rate proxy over a force window."""

  _validate_positive_scalar("dt", dt)

  force_tensor = _to_tensor(force_series)
  if force_tensor.numel() == 0:
    raise ValueError("force_series must not be empty")
  _validate_finite("force_series", force_tensor)

  return (force_tensor.max() - force_tensor.min()) / dt


def compute_contact_quietness_score(
  peak_force_bw: torch.Tensor | float,
  loading_rate_bw_s: torch.Tensor | float,
) -> torch.Tensor:
  """Combine normalized peak force and loading rate into a bounded score."""

  peak_force_tensor = _to_tensor(peak_force_bw)
  loading_rate_tensor = _to_tensor(loading_rate_bw_s)
  _validate_finite("peak_force_bw", peak_force_tensor)
  _validate_finite("loading_rate_bw_s", loading_rate_tensor)
  _validate_non_negative("peak_force_bw", peak_force_tensor)
  _validate_non_negative("loading_rate_bw_s", loading_rate_tensor)
  penalty = 0.5 * peak_force_tensor + 0.05 * loading_rate_tensor
  return torch.clamp(1.0 - penalty, min=0.0, max=1.0)
