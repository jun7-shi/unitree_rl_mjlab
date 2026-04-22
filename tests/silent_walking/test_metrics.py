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
