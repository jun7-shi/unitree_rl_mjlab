"""Policy adapters for silent walking evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import torch

try:
  import onnxruntime as ort
except ImportError:  # pragma: no cover - handled when ONNX adapter is used.
  ort = None


def _onnx_providers_for_device(device: str) -> list[str]:
  """Return a provider preference list matching the requested device."""

  normalized = device.lower()
  if normalized.startswith("cuda"):
    return ["CUDAExecutionProvider", "CPUExecutionProvider"]
  return ["CPUExecutionProvider"]


class PolicyAdapter(Protocol):
  """Minimal inference interface required by the evaluator."""

  def reset(self) -> None:
    """Reset internal adapter state before a new rollout."""

  def act(self, obs: torch.Tensor) -> torch.Tensor:
    """Compute an action tensor from an observation tensor."""


class ZeroPolicyAdapter:
  """Adapter that always returns zero actions."""

  def __init__(self, action_dim: int, device: str):
    self._action_dim = action_dim
    self._device = device

  def reset(self) -> None:
    return None

  def act(self, obs: torch.Tensor) -> torch.Tensor:
    return torch.zeros(obs.shape[0], self._action_dim, device=self._device)


class TorchscriptPolicyAdapter:
  """Adapter for TorchScript policy files."""

  def __init__(self, path: str, device: str):
    self._device = device
    self._policy = torch.jit.load(path, map_location=device)
    self._policy.eval()

  def reset(self) -> None:
    return None

  def act(self, obs: torch.Tensor) -> torch.Tensor:
    with torch.inference_mode():
      return self._policy(obs.to(self._device))


class OnnxPolicyAdapter:
  """Adapter for ONNX policy files."""

  def __init__(self, path: str, device: str):
    if ort is None:
      raise ImportError("onnxruntime is required to load ONNX policies")

    self._device = torch.device(device)
    self._session = ort.InferenceSession(
      path,
      providers=_onnx_providers_for_device(device),
    )
    self._input_name = self._session.get_inputs()[0].name

  def reset(self) -> None:
    return None

  def act(self, obs: torch.Tensor) -> torch.Tensor:
    obs_tensor = obs.detach().to("cpu")
    outputs = self._session.run(None, {self._input_name: obs_tensor.numpy()})
    return torch.as_tensor(outputs[0], device=self._device)


def adapt_policy_path(path: str, device: str) -> PolicyAdapter:
  """Create the appropriate policy adapter from a file path."""

  suffix = Path(path).suffix.lower()
  if suffix == ".onnx":
    return OnnxPolicyAdapter(path, device=device)
  if suffix in {".pt", ".jit"}:
    return TorchscriptPolicyAdapter(path, device=device)
  raise ValueError(f"Unsupported policy format: {suffix}")
