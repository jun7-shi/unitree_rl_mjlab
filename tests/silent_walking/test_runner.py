from dataclasses import dataclass

import torch

from src.evaluation.silent_walking.contact_backends import supports_contact_backend
from src.evaluation.silent_walking.runner import (
  _CallablePolicyAdapter,
  _compute_touchdown_metrics,
  _extract_actor_obs,
  _resolve_policy,
  _validate_runtime_support,
)


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


def test_callable_policy_adapter_wraps_plain_callable():
  adapter = _CallablePolicyAdapter(lambda obs: obs + 2.0)

  out = adapter.act(torch.zeros(1, 3))

  assert torch.allclose(out, torch.full((1, 3), 2.0))


def test_resolve_policy_loads_checkpoint_through_runtime_loader(monkeypatch):
  monkeypatch.setattr(
    "src.evaluation.silent_walking.runner._load_checkpoint_policy",
    lambda task_id, env, checkpoint_path, device: (lambda obs: obs + 1.0),
  )

  adapter = _resolve_policy("Unitree-G1-Flat", object(), "model_100.pt", "cpu")
  out = adapter.act(torch.zeros(1, 2))

  assert torch.allclose(out, torch.ones(1, 2))


def test_load_checkpoint_policy_uses_runner(monkeypatch):
  @dataclass
  class FakeAgentCfg:
    clip_actions: bool = True

  class FakePolicy:
    def __call__(self, obs):
      return obs + 2.0

  class FakeRunner:
    init_args = None
    load_args = None
    policy_device = None

    def __init__(self, env, agent_cfg, device):
      FakeRunner.init_args = (env, agent_cfg, device)

    def load(self, path, load_cfg, strict, map_location):
      FakeRunner.load_args = (path, load_cfg, strict, map_location)

    def get_inference_policy(self, device):
      FakeRunner.policy_device = device
      return FakePolicy()

  class FakeWrappedEnv:
    pass

  fake_env = object()

  monkeypatch.setattr(
    "src.evaluation.silent_walking.runner.load_rl_cfg",
    lambda task_id: FakeAgentCfg(),
  )
  monkeypatch.setattr(
    "src.evaluation.silent_walking.runner.load_runner_cls",
    lambda task_id: FakeRunner,
  )
  monkeypatch.setattr(
    "src.evaluation.silent_walking.runner.RslRlVecEnvWrapper",
    lambda env, clip_actions: FakeWrappedEnv(),
  )

  from src.evaluation.silent_walking.runner import _load_checkpoint_policy

  policy = _load_checkpoint_policy(
    "g1_velocity",
    env=fake_env,
    checkpoint_path="model_000100.pt",
    device="cpu",
  )

  assert FakeRunner.init_args == (FakeRunner.init_args[0], {"clip_actions": True}, "cpu")
  assert isinstance(FakeRunner.init_args[0], FakeWrappedEnv)
  assert FakeRunner.load_args == (
    "model_000100.pt",
    {"actor": True},
    True,
    "cpu",
  )
  assert FakeRunner.policy_device == "cpu"
  assert torch.allclose(policy(torch.zeros(1, 4)), torch.full((1, 4), 2.0))


def test_compute_touchdown_metrics_extracts_event_windows():
  forces = torch.tensor(
    [
      [0.0, 0.0],
      [10.0, 0.0],
      [20.0, 5.0],
      [15.0, 8.0],
      [0.0, 0.0],
      [0.0, 12.0],
      [0.0, 18.0],
    ]
  )
  contacts = torch.tensor(
    [
      [False, False],
      [True, False],
      [True, True],
      [True, True],
      [False, False],
      [False, True],
      [False, True],
    ]
  )

  peaks, rates = _compute_touchdown_metrics(
    forces, contacts, dt=0.02, loading_window_steps=3
  )

  assert len(peaks) == 2
  assert len(rates) == 2
  assert torch.allclose(peaks[0], torch.tensor([20.0]))
  assert torch.allclose(peaks[1], torch.tensor([8.0, 18.0]))
  assert torch.allclose(rates[0], torch.tensor([500.0]))
  assert torch.allclose(rates[1], torch.tensor([400.0, 300.0]))
