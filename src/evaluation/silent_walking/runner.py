"""Runtime runner for silent walking evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

import torch

_RUNTIME_CACHE_ROOT = Path(
  os.environ.get("SILENT_WALKING_CACHE_ROOT", "/tmp/unitree_rl_mjlab/silent_walking")
)
_MPL_CACHE_DIR = _RUNTIME_CACHE_ROOT / "matplotlib"
_WARP_CACHE_DIR = _RUNTIME_CACHE_ROOT / "warp"

for cache_dir in (_MPL_CACHE_DIR, _WARP_CACHE_DIR):
  cache_dir.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE_DIR))
os.environ.setdefault("WARP_CACHE_PATH", str(_WARP_CACHE_DIR))

import mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.torch import configure_torch_backends
import src.tasks  # noqa: F401

from .contact_backends import extract_feet_net_forces, supports_contact_backend
from .metrics import (
  compute_contact_quietness_score,
  compute_loading_rate,
  normalize_force_by_body_weight,
)
from .policy_adapters import PolicyAdapter
from .robots import get_robot_spec
from .types import EpisodeMetricSummary, EpisodeMetricTrace


@dataclass(frozen=True, slots=True)
class SilentEvalResult:
  """Evaluation result bundle for one policy rollout."""

  robot_name: str
  task_id: str
  steps: int
  trace: EpisodeMetricTrace
  summary: EpisodeMetricSummary


def _validate_runtime_support(robot_name: str) -> None:
  """Fail fast when the selected robot is not ready for runtime evaluation."""

  spec = get_robot_spec(robot_name)
  if spec.asset_status != "ready":
    raise FileNotFoundError(
      f"Robot '{spec.name}' assets are not available at {spec.asset_path}"
    )
  if not supports_contact_backend(robot_name):
    raise NotImplementedError(
      f"Silent walking runtime evaluation is not wired yet for robot '{spec.name}'"
    )


def _extract_actor_obs(obs: Any) -> torch.Tensor:
  """Normalize env observation outputs to a single actor observation tensor."""

  if isinstance(obs, tuple):
    if not obs:
      raise TypeError("Received empty observation tuple.")
    return _extract_actor_obs(obs[0])
  if isinstance(obs, dict):
    actor_obs = obs.get("policy") or obs.get("actor")
    if isinstance(actor_obs, torch.Tensor):
      return actor_obs
  if isinstance(obs, torch.Tensor):
    return obs
  raise TypeError(f"Unsupported observation type: {type(obs)!r}")


def _step_env(env: ManagerBasedRlEnv, action: torch.Tensor) -> torch.Tensor:
  """Step the environment and return the next actor observation tensor."""

  step_result = env.step(action)
  if not isinstance(step_result, tuple):
    raise TypeError(f"Unsupported env.step return type: {type(step_result)!r}")
  if len(step_result) == 4:
    obs, _, _, _ = step_result
  elif len(step_result) == 5:
    obs, _, _, _, _ = step_result
  else:
    raise TypeError(f"Unsupported env.step tuple length: {len(step_result)}")
  return _extract_actor_obs(obs)


def run_silent_eval(
  robot_name: str,
  policy_adapter: PolicyAdapter,
  steps: int,
  device: str,
) -> SilentEvalResult:
  """Run a minimal silent-walking evaluation rollout."""

  configure_torch_backends()

  spec = get_robot_spec(robot_name)
  _validate_runtime_support(robot_name)
  env_cfg = load_env_cfg(spec.task_id, play=True)
  env_cfg.scene.num_envs = 1
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)

  try:
    obs = _extract_actor_obs(env.reset())
    policy_adapter.reset()

    peak_force_samples: list[torch.Tensor] = []

    for _ in range(steps):
      action = policy_adapter.act(obs)
      obs = _step_env(env, action)
      foot_force = extract_feet_net_forces(env, robot_name=robot_name)
      peak_force_samples.append(foot_force[..., 2].abs().max(dim=1).values.squeeze(0))

    peak_force_series = torch.stack(peak_force_samples)
    peak_force_bw = normalize_force_by_body_weight(
      peak_force_series.max(),
      body_weight_newton=spec.mass_normalization,
    )
    loading_rate_bw_s = compute_loading_rate(
      normalize_force_by_body_weight(
        peak_force_series.flatten(),
        body_weight_newton=spec.mass_normalization,
      ),
      dt=env.step_dt,
    )
    contact_quietness = compute_contact_quietness_score(
      peak_force_bw=peak_force_bw,
      loading_rate_bw_s=loading_rate_bw_s,
    )
    trace = EpisodeMetricTrace(
      peak_force_bw=peak_force_bw.reshape(1),
      loading_rate_bw_s=loading_rate_bw_s.reshape(1),
      contact_quietness_score=contact_quietness.reshape(1),
    )
    summary = EpisodeMetricSummary(
      contact_quietness=float(contact_quietness.item()),
      body_smoothness=0.0,
      task_compliance=0.0,
      total_score=float(contact_quietness.item()),
    )
    return SilentEvalResult(
      robot_name=robot_name,
      task_id=spec.task_id,
      steps=steps,
      trace=trace,
      summary=summary,
    )
  finally:
    env.close()
