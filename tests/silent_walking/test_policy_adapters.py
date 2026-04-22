import torch

from src.evaluation.silent_walking.policy_adapters import (
  ZeroPolicyAdapter,
  adapt_policy_path,
)


def test_zero_policy_adapter_matches_requested_action_dim():
  adapter = ZeroPolicyAdapter(action_dim=12, device="cpu")

  out = adapter.act(torch.zeros(1, 4))

  assert tuple(out.shape) == (1, 12)


def test_adapt_policy_path_rejects_unknown_suffix():
  try:
    adapt_policy_path("policy.invalid", device="cpu")
  except ValueError as exc:
    assert "Unsupported policy format" in str(exc)
  else:
    raise AssertionError("Expected ValueError")
