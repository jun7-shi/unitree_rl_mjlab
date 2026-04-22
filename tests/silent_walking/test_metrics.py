import pytest
import torch

from src.evaluation.silent_walking.metrics import (
  compute_contact_quietness_score,
  compute_loading_rate,
  normalize_force_by_body_weight,
)


def test_normalize_force_by_body_weight_returns_one_for_body_weight():
  force = torch.tensor([343.35])
  normalized = normalize_force_by_body_weight(force, body_weight_newton=343.35)

  assert torch.allclose(normalized, torch.tensor([1.0]))


def test_compute_loading_rate_uses_peak_force_over_dt():
  force = torch.tensor([0.0, 10.0, 30.0, 40.0])
  rate = compute_loading_rate(force, dt=0.01)

  assert rate.item() == 4000.0


def test_compute_contact_quietness_score_is_bounded():
  score = compute_contact_quietness_score(
    peak_force_bw=torch.tensor([1.2]),
    loading_rate_bw_s=torch.tensor([8.0]),
  )

  assert 0.0 <= score.item() <= 1.0


def test_metric_helpers_reject_non_finite_and_invalid_denominators():
  with pytest.raises(ValueError):
    normalize_force_by_body_weight(torch.tensor([1.0]), body_weight_newton=float("nan"))

  with pytest.raises(ValueError):
    compute_loading_rate(torch.tensor([1.0, 2.0]), dt=0.0)

  with pytest.raises(ValueError):
    compute_contact_quietness_score(
      peak_force_bw=torch.tensor([float("inf")]),
      loading_rate_bw_s=torch.tensor([8.0]),
    )

  with pytest.raises(ValueError):
    compute_contact_quietness_score(
      peak_force_bw=torch.tensor([-0.1]),
      loading_rate_bw_s=torch.tensor([8.0]),
    )
