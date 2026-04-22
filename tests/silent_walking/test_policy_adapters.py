import torch

from src.evaluation.silent_walking.policy_adapters import (
  OnnxPolicyAdapter,
  TorchscriptPolicyAdapter,
  ZeroPolicyAdapter,
  adapt_policy_path,
  is_checkpoint_path,
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


def test_is_checkpoint_path_matches_pt_only():
  assert is_checkpoint_path("model_100.pt") is True
  assert is_checkpoint_path("policy.onnx") is False


def test_adapt_policy_path_rejects_checkpoint_pt_files():
  try:
    adapt_policy_path("model_100.pt", device="cpu")
  except ValueError as exc:
    assert "runtime runner" in str(exc)
  else:
    raise AssertionError("Expected ValueError")


def test_torchscript_policy_adapter_uses_loaded_module(monkeypatch):
  class FakePolicy:
    def eval(self):
      return self

    def __call__(self, obs):
      return obs + 1.0

  monkeypatch.setattr(torch.jit, "load", lambda path, map_location=None: FakePolicy())

  adapter = TorchscriptPolicyAdapter("policy.pt", device="cpu")
  out = adapter.act(torch.zeros(1, 4))

  assert torch.allclose(out, torch.ones(1, 4))


def test_onnx_policy_adapter_uses_device_specific_providers(monkeypatch):
  class FakeSession:
    def __init__(self, path, providers):
      self.path = path
      self.providers = providers

    def get_providers(self):
      return ["CPUExecutionProvider"]

    def get_inputs(self):
      class FakeInput:
        name = "obs"

      return [FakeInput()]

    def run(self, _, inputs):
      return [inputs["obs"]]

  class FakeOrt:
    InferenceSession = FakeSession

  monkeypatch.setattr(
    "src.evaluation.silent_walking.policy_adapters.ort",
    FakeOrt(),
  )

  try:
    OnnxPolicyAdapter("policy.onnx", device="cuda:0")
  except RuntimeError as exc:
    assert "CUDAExecutionProvider is not active" in str(exc)
  else:
    raise AssertionError("Expected RuntimeError")


def test_onnx_policy_adapter_runs_inference_on_cpu(monkeypatch):
  class FakeSession:
    def __init__(self, path, providers):
      self.path = path
      self.providers = providers

    def get_providers(self):
      return ["CPUExecutionProvider"]

    def get_inputs(self):
      class FakeInput:
        name = "obs"

      return [FakeInput()]

    def run(self, _, inputs):
      return [inputs["obs"]]

  class FakeOrt:
    InferenceSession = FakeSession

  monkeypatch.setattr(
    "src.evaluation.silent_walking.policy_adapters.ort",
    FakeOrt(),
  )

  adapter = OnnxPolicyAdapter("policy.onnx", device="cpu")
  out = adapter.act(torch.ones(1, 4))

  assert adapter._session.providers == ["CPUExecutionProvider"]
  assert torch.allclose(out, torch.ones(1, 4))
