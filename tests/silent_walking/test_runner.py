import torch

from src.evaluation.silent_walking.contact_backends import supports_contact_backend
from src.evaluation.silent_walking.runner import _extract_actor_obs, _validate_runtime_support


def test_supports_contact_backend_only_for_g1():
  assert supports_contact_backend("g1") is True
  assert supports_contact_backend("bumi") is False


def test_extract_actor_obs_handles_tuple_wrapped_actor_dict():
  actor_obs = torch.ones(1, 4)

  assert torch.equal(_extract_actor_obs(({"actor": actor_obs}, {})), actor_obs)


def test_validate_runtime_support_fails_fast_for_missing_bumi_assets():
  try:
    _validate_runtime_support("bumi")
  except FileNotFoundError as exc:
    assert "bumi" in str(exc).lower()
  else:
    raise AssertionError("Expected missing bumi assets to fail runtime validation")
