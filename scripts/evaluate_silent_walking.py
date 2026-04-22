"""Run silent walking evaluation in MuJoCo."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
  sys.path.insert(0, str(PROJECT_ROOT))


def build_parser() -> argparse.ArgumentParser:
  """Create the CLI parser for silent walking evaluation."""

  parser = argparse.ArgumentParser(description="Evaluate silent walking policies.")
  parser.add_argument("--robot", choices=["g1", "bumi"], required=True)
  parser.add_argument("--policy", required=True)
  parser.add_argument("--steps", type=int, default=500)
  parser.add_argument("--device", default=None)
  parser.add_argument("--plot-file", default=None)
  return parser


def main() -> None:
  parser = build_parser()
  args = parser.parse_args()

  try:
    import torch

    from src.evaluation.silent_walking.policy_adapters import (
      ZeroPolicyAdapter,
      adapt_policy_path,
      is_checkpoint_path,
    )
    from src.evaluation.silent_walking.robots import get_robot_spec
    from src.evaluation.silent_walking.runner import run_silent_eval
    from src.evaluation.silent_walking.plotting import (
      save_capsule_distribution_plot,
      save_capsule_force_timeseries_plot,
      save_trace_plot,
    )
    from src.evaluation.silent_walking.reporting import render_markdown_report
  except ModuleNotFoundError as exc:
    raise SystemExit(
      "Missing runtime dependency for silent walking evaluation: "
      f"{exc.name}. Activate the project environment before running this script."
    ) from exc

  device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  robot_spec = get_robot_spec(args.robot)

  if args.policy == "zero":
    policy_adapter = ZeroPolicyAdapter(action_dim=robot_spec.action_dim, device=device)
  else:
    policy_path = Path(args.policy)
    if not policy_path.exists():
      raise FileNotFoundError(f"Policy file not found: {policy_path}")
    if is_checkpoint_path(str(policy_path)):
      policy_adapter = str(policy_path)
    else:
      policy_adapter = adapt_policy_path(str(policy_path), device=device)

  result = run_silent_eval(
    robot_name=args.robot,
    policy_adapter=policy_adapter,
    steps=args.steps,
    device=device,
  )
  print(
    render_markdown_report(
      result.robot_name,
      args.policy,
      result.summary,
      trace=result.trace,
    )
  )
  if args.plot_file is not None:
    trace_path = save_trace_plot(
      result.trace,
      args.plot_file,
      title=f"{result.robot_name}: {Path(args.policy).name}",
      dt=result.step_dt,
    )
    footmap_path = save_capsule_distribution_plot(
      result.trace,
      Path(args.plot_file).with_name(f"{Path(args.plot_file).stem}_foot_distribution.png"),
      title=f"{result.robot_name}: {Path(args.policy).name} footprint",
    )
    capsule_force_path = save_capsule_force_timeseries_plot(
      result.trace,
      Path(args.plot_file).with_name(f"{Path(args.plot_file).stem}_capsule_force_timeseries.png"),
      title=f"{result.robot_name}: {Path(args.policy).name} capsule force traces",
      dt=result.step_dt,
    )
    print(f"\nPlot saved to: {trace_path}")
    print(f"Foot distribution saved to: {footmap_path}")
    print(f"Capsule force time series saved to: {capsule_force_path}")


if __name__ == "__main__":
  main()
