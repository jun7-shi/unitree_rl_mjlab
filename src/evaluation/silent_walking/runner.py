"""Runtime runner for silent walking evaluation."""

from __future__ import annotations

from dataclasses import asdict
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
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg
from mjlab.tasks.registry import load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
import src.tasks  # noqa: F401

from .contact_backends import (
  extract_capsule_contact_forces,
  extract_capsule_vertical_velocities,
  extract_feet_contact_flags,
  extract_feet_net_forces,
  extract_foot_vertical_velocities,
  supports_contact_backend,
)
from .metrics import (
  combine_total_score,
  compute_body_smoothness_score,
  compute_contact_quietness_score,
  compute_loading_rate,
  compute_task_compliance_score,
  normalize_force_by_body_weight,
)
from .policy_adapters import PolicyAdapter, adapt_policy_path, is_checkpoint_path
from .robots import get_robot_spec
from .types import EpisodeMetricSummary, EpisodeMetricTrace


@dataclass(frozen=True, slots=True)
class SilentEvalResult:
  """Evaluation result bundle for one policy rollout."""

  robot_name: str
  task_id: str
  steps: int
  step_dt: float
  trace: EpisodeMetricTrace
  summary: EpisodeMetricSummary


class _CallablePolicyAdapter:
  """Wrap a plain callable policy with the evaluator adapter interface."""

  def __init__(self, policy_callable, use_raw_obs: bool = False):
    self._policy_callable = policy_callable
    self.use_raw_obs = use_raw_obs

  def reset(self) -> None:
    return None

  def act(self, obs) -> torch.Tensor:
    with torch.inference_mode():
      return self._policy_callable(obs)


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


def _unwrap_obs(obs: Any) -> Any:
  """Strip wrapper tuples from env observations while preserving inner structure."""

  if isinstance(obs, tuple):
    if not obs:
      raise TypeError("Received empty observation tuple.")
    return _unwrap_obs(obs[0])
  return obs


def _extract_actor_obs(obs: Any) -> torch.Tensor:
  """Normalize env observation outputs to a single actor observation tensor."""

  obs = _unwrap_obs(obs)
  if isinstance(obs, dict):
    actor_obs = obs.get("policy") or obs.get("actor")
    if isinstance(actor_obs, torch.Tensor):
      return actor_obs
  if isinstance(obs, torch.Tensor):
    return obs
  raise TypeError(f"Unsupported observation type: {type(obs)!r}")


def _step_env(env: ManagerBasedRlEnv, action: torch.Tensor) -> tuple[Any, torch.Tensor]:
  """Step the environment and return the raw and actor observations."""

  step_result = env.step(action)
  if not isinstance(step_result, tuple):
    raise TypeError(f"Unsupported env.step return type: {type(step_result)!r}")
  if len(step_result) == 4:
    obs, _, _, _ = step_result
  elif len(step_result) == 5:
    obs, _, _, _, _ = step_result
  else:
    raise TypeError(f"Unsupported env.step tuple length: {len(step_result)}")
  return obs, _extract_actor_obs(obs)


def _compute_touchdown_metrics(
  normal_force_series: torch.Tensor,
  contact_flag_series: torch.Tensor,
  dt: float,
  vertical_velocity_series: torch.Tensor | None = None,
  loading_window_steps: int = 5,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
  """Compute per-touchdown peak force, loading rate, and vertical speed."""

  touchdown_peaks: list[torch.Tensor] = []
  touchdown_rates: list[torch.Tensor] = []
  touchdown_vertical_speeds: list[torch.Tensor] = []
  num_steps, num_feet = normal_force_series.shape

  for foot_idx in range(num_feet):
    foot_touchdown_peaks: list[torch.Tensor] = []
    foot_touchdown_rates: list[torch.Tensor] = []
    foot_touchdown_vertical_speeds: list[torch.Tensor] = []
    foot_forces = normal_force_series[:, foot_idx]
    foot_contacts = contact_flag_series[:, foot_idx]
    foot_vertical_velocity = (
      vertical_velocity_series[:, foot_idx]
      if vertical_velocity_series is not None
      else None
    )
    previous_contact = False

    for step_idx in range(num_steps):
      in_contact = bool(foot_contacts[step_idx].item())
      if in_contact and not previous_contact:
        end_idx = min(step_idx + loading_window_steps, num_steps)
        window = foot_forces[step_idx:end_idx]
        foot_touchdown_peaks.append(window.max())
        foot_touchdown_rates.append(compute_loading_rate(window, dt=dt))
        if foot_vertical_velocity is not None:
          foot_touchdown_vertical_speeds.append(
            torch.clamp(-foot_vertical_velocity[step_idx], min=0.0)
          )
      previous_contact = in_contact

    if foot_touchdown_peaks:
      touchdown_peaks.append(torch.stack(foot_touchdown_peaks))
      touchdown_rates.append(torch.stack(foot_touchdown_rates))
      if foot_touchdown_vertical_speeds:
        touchdown_vertical_speeds.append(torch.stack(foot_touchdown_vertical_speeds))
      else:
        touchdown_vertical_speeds.append(torch.zeros(1, device=normal_force_series.device))
    else:
      zero = torch.zeros(1, device=normal_force_series.device)
      touchdown_peaks.append(zero)
      touchdown_rates.append(zero)
      touchdown_vertical_speeds.append(zero)

  return touchdown_peaks, touchdown_rates, touchdown_vertical_speeds


def _summarize_channel_events(channel_events: list[torch.Tensor]) -> torch.Tensor:
  """Reduce variable-length event lists into one mean value per channel."""

  return torch.stack([events.mean() for events in channel_events])


def _load_checkpoint_policy(
  task_id: str,
  env: ManagerBasedRlEnv,
  checkpoint_path: str,
  device: str,
):
  """Load an inference policy from an RSL-RL checkpoint."""

  agent_cfg = load_rl_cfg(task_id)
  wrapped_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
  runner = runner_cls(wrapped_env, asdict(agent_cfg), device=device)
  runner.load(
    checkpoint_path,
    load_cfg={"actor": True},
    strict=True,
    map_location=device,
  )
  return runner.get_inference_policy(device=device)


def _resolve_policy(
  task_id: str,
  env: ManagerBasedRlEnv,
  policy_source: str | PolicyAdapter,
  device: str,
):
  """Resolve a runtime policy source to a callable rollout policy."""

  if isinstance(policy_source, str):
    if is_checkpoint_path(policy_source):
      return _CallablePolicyAdapter(
        _load_checkpoint_policy(task_id, env, policy_source, device),
        use_raw_obs=True,
      )
    return adapt_policy_path(policy_source, device=device)
  return policy_source


def run_silent_eval(
  robot_name: str,
  policy_adapter: str | PolicyAdapter,
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
    policy = _resolve_policy(spec.task_id, env, policy_adapter, device)
    raw_obs = env.reset()
    obs = _extract_actor_obs(raw_obs)
    policy.reset()

    peak_force_samples: list[torch.Tensor] = []
    contact_flag_samples: list[torch.Tensor] = []
    foot_vertical_velocity_samples: list[torch.Tensor] = []
    capsule_force_samples: list[torch.Tensor] = []
    capsule_contact_flag_samples: list[torch.Tensor] = []
    capsule_vertical_velocity_samples: list[torch.Tensor] = []
    action_rate_samples: list[torch.Tensor] = []
    command_velocity_samples: list[torch.Tensor] = []
    actual_linear_velocity_samples: list[torch.Tensor] = []
    actual_yaw_rate_samples: list[torch.Tensor] = []
    linear_velocity_error_samples: list[torch.Tensor] = []
    yaw_rate_error_samples: list[torch.Tensor] = []
    previous_action: torch.Tensor | None = None

    for _ in range(steps):
      policy_obs = _unwrap_obs(raw_obs) if getattr(policy, "use_raw_obs", False) else obs
      action = policy.act(policy_obs)
      if previous_action is None:
        action_rate_samples.append(torch.zeros(1, device=action.device))
      else:
        action_rate_samples.append(
          torch.linalg.norm(action - previous_action, dim=1)
        )
      previous_action = action.detach().clone()
      raw_obs, obs = _step_env(env, action)
      foot_force = extract_feet_net_forces(env, robot_name=robot_name)
      foot_contact = extract_feet_contact_flags(env, robot_name=robot_name)
      foot_vertical_velocity = extract_foot_vertical_velocities(env, robot_name=robot_name)
      capsule_names, capsule_force, capsule_contact = extract_capsule_contact_forces(
        env,
        robot_name=robot_name,
      )
      capsule_velocity_names, capsule_vertical_velocity = extract_capsule_vertical_velocities(
        env,
        robot_name=robot_name,
      )
      if capsule_names != capsule_velocity_names:
        raise RuntimeError("Capsule force and velocity ordering mismatch")
      peak_force_samples.append(foot_force[..., 2].abs().squeeze(0))
      contact_flag_samples.append(foot_contact.squeeze(0))
      foot_vertical_velocity_samples.append(foot_vertical_velocity.squeeze(0))
      capsule_force_samples.append(capsule_force.squeeze(0))
      capsule_contact_flag_samples.append(capsule_contact.squeeze(0))
      capsule_vertical_velocity_samples.append(capsule_vertical_velocity.squeeze(0))

      twist_command = env.command_manager.get_command("twist")
      if twist_command is None:
        raise RuntimeError("Velocity evaluation requires a 'twist' command term")
      robot = env.scene["robot"]
      command_velocity_samples.append(twist_command.squeeze(0).detach().clone())
      actual_linear_velocity_samples.append(
        robot.data.root_link_lin_vel_b[:, :2].squeeze(0).detach().clone()
      )
      actual_yaw_rate_samples.append(
        robot.data.root_link_ang_vel_b[:, 2].squeeze(0).detach().clone()
      )
      linear_velocity_error_samples.append(
        torch.linalg.norm(
          twist_command[:, :2] - robot.data.root_link_lin_vel_b[:, :2],
          dim=1,
        )
      )
      yaw_rate_error_samples.append(
        torch.abs(twist_command[:, 2] - robot.data.root_link_ang_vel_b[:, 2])
      )

    peak_force_series = torch.stack(peak_force_samples)
    contact_flag_series = torch.stack(contact_flag_samples)
    foot_vertical_velocity_series = torch.stack(foot_vertical_velocity_samples)
    capsule_force_series = torch.stack(capsule_force_samples)
    capsule_contact_flag_series = torch.stack(capsule_contact_flag_samples)
    capsule_vertical_velocity_series = torch.stack(capsule_vertical_velocity_samples)
    action_rate_series = torch.stack(action_rate_samples)
    command_velocity_series = torch.stack(command_velocity_samples)
    actual_linear_velocity_series = torch.stack(actual_linear_velocity_samples)
    actual_yaw_rate_series = torch.stack(actual_yaw_rate_samples)
    linear_velocity_error_series = torch.stack(linear_velocity_error_samples)
    yaw_rate_error_series = torch.stack(yaw_rate_error_samples)
    (
      touchdown_peak_force_by_foot,
      touchdown_loading_rate_by_foot,
      touchdown_vertical_speed_by_foot,
    ) = _compute_touchdown_metrics(
      peak_force_series,
      contact_flag_series,
      dt=env.step_dt,
      vertical_velocity_series=foot_vertical_velocity_series,
    )
    (
      touchdown_peak_force_by_capsule,
      touchdown_loading_rate_by_capsule,
      touchdown_vertical_speed_by_capsule,
    ) = _compute_touchdown_metrics(
      capsule_force_series,
      capsule_contact_flag_series,
      dt=env.step_dt,
      vertical_velocity_series=capsule_vertical_velocity_series,
    )
    left_touchdown_peak_force = touchdown_peak_force_by_foot[0]
    right_touchdown_peak_force = touchdown_peak_force_by_foot[1]
    left_touchdown_loading_rate = touchdown_loading_rate_by_foot[0]
    right_touchdown_loading_rate = touchdown_loading_rate_by_foot[1]
    left_touchdown_vertical_speed = touchdown_vertical_speed_by_foot[0]
    right_touchdown_vertical_speed = touchdown_vertical_speed_by_foot[1]
    touchdown_peak_force = torch.cat(touchdown_peak_force_by_foot)
    touchdown_loading_rate = torch.cat(touchdown_loading_rate_by_foot)
    touchdown_vertical_speed = torch.cat(touchdown_vertical_speed_by_foot)
    peak_force_bw = normalize_force_by_body_weight(
      touchdown_peak_force.mean(),
      body_weight_newton=spec.mass_normalization,
    )
    touchdown_peak_force_bw = normalize_force_by_body_weight(
      touchdown_peak_force,
      body_weight_newton=spec.mass_normalization,
    )
    touchdown_loading_rate_bw_s = normalize_force_by_body_weight(
      touchdown_loading_rate,
      body_weight_newton=spec.mass_normalization,
    )
    left_touchdown_peak_force_bw = normalize_force_by_body_weight(
      left_touchdown_peak_force,
      body_weight_newton=spec.mass_normalization,
    )
    right_touchdown_peak_force_bw = normalize_force_by_body_weight(
      right_touchdown_peak_force,
      body_weight_newton=spec.mass_normalization,
    )
    left_touchdown_loading_rate_bw_s = normalize_force_by_body_weight(
      left_touchdown_loading_rate,
      body_weight_newton=spec.mass_normalization,
    )
    right_touchdown_loading_rate_bw_s = normalize_force_by_body_weight(
      right_touchdown_loading_rate,
      body_weight_newton=spec.mass_normalization,
    )
    capsule_touchdown_peak_force_bw = normalize_force_by_body_weight(
      _summarize_channel_events(touchdown_peak_force_by_capsule),
      body_weight_newton=spec.mass_normalization,
    )
    capsule_touchdown_loading_rate_bw_s = normalize_force_by_body_weight(
      _summarize_channel_events(touchdown_loading_rate_by_capsule),
      body_weight_newton=spec.mass_normalization,
    )
    capsule_touchdown_vertical_speed_m_s = _summarize_channel_events(
      touchdown_vertical_speed_by_capsule
    )
    capsule_touchdown_count = torch.tensor(
      [events.numel() for events in touchdown_peak_force_by_capsule],
      dtype=torch.float32,
    )
    touchdown_peak_asymmetry_bw = torch.abs(
      left_touchdown_peak_force_bw.mean() - right_touchdown_peak_force_bw.mean()
    ).reshape(1)
    loading_rate_bw_s = touchdown_loading_rate_bw_s.mean()
    contact_quietness = compute_contact_quietness_score(
      peak_force_bw=peak_force_bw,
      loading_rate_bw_s=loading_rate_bw_s,
    )
    body_smoothness = compute_body_smoothness_score(action_rate_series.mean())
    task_compliance = compute_task_compliance_score(
      linear_velocity_error_series.mean(),
      yaw_rate_error_series.mean(),
    )
    total_score = combine_total_score(
      contact_quietness,
      body_smoothness,
      task_compliance,
    )
    trace = EpisodeMetricTrace(
      foot_z_force_n=peak_force_series,
      foot_contact_flag=contact_flag_series.float(),
      foot_vertical_velocity_m_s=foot_vertical_velocity_series,
      capsule_names=capsule_names,
      capsule_z_force_n=capsule_force_series,
      capsule_contact_flag=capsule_contact_flag_series.float(),
      capsule_vertical_velocity_m_s=capsule_vertical_velocity_series,
      peak_force_bw=peak_force_bw.reshape(1),
      loading_rate_bw_s=loading_rate_bw_s.reshape(1),
      touchdown_peak_force_bw=touchdown_peak_force_bw.flatten(),
      touchdown_loading_rate_bw_s=touchdown_loading_rate_bw_s.flatten(),
      touchdown_vertical_speed_m_s=touchdown_vertical_speed.flatten(),
      left_touchdown_peak_force_bw=left_touchdown_peak_force_bw.flatten(),
      right_touchdown_peak_force_bw=right_touchdown_peak_force_bw.flatten(),
      left_touchdown_loading_rate_bw_s=left_touchdown_loading_rate_bw_s.flatten(),
      right_touchdown_loading_rate_bw_s=right_touchdown_loading_rate_bw_s.flatten(),
      left_touchdown_vertical_speed_m_s=left_touchdown_vertical_speed.flatten(),
      right_touchdown_vertical_speed_m_s=right_touchdown_vertical_speed.flatten(),
      touchdown_peak_asymmetry_bw=touchdown_peak_asymmetry_bw,
      capsule_touchdown_peak_force_bw=capsule_touchdown_peak_force_bw.flatten(),
      capsule_touchdown_loading_rate_bw_s=capsule_touchdown_loading_rate_bw_s.flatten(),
      capsule_touchdown_vertical_speed_m_s=capsule_touchdown_vertical_speed_m_s.flatten(),
      capsule_touchdown_count=capsule_touchdown_count.flatten(),
      action_rate_l2=action_rate_series.flatten(),
      command_velocity=command_velocity_series,
      actual_linear_velocity=actual_linear_velocity_series,
      actual_yaw_rate=actual_yaw_rate_series.flatten(),
      linear_velocity_error=linear_velocity_error_series.flatten(),
      yaw_rate_error=yaw_rate_error_series.flatten(),
      contact_quietness_score=contact_quietness.reshape(1),
    )
    summary = EpisodeMetricSummary(
      contact_quietness=float(contact_quietness.item()),
      body_smoothness=float(body_smoothness.item()),
      task_compliance=float(task_compliance.item()),
      total_score=float(total_score.item()),
    )
    return SilentEvalResult(
      robot_name=robot_name,
      task_id=spec.task_id,
      steps=steps,
      step_dt=env.step_dt,
      trace=trace,
      summary=summary,
    )
  finally:
    env.close()
