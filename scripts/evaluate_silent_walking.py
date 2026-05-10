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
  parser.add_argument("--task-id", default=None)
  parser.add_argument("--motion-file", default=None)
  parser.add_argument(
    "--disable-foot-phase-observation",
    action="store_true",
    help="Remove motion_foot_phase observations for legacy 160-dim tracking checkpoints.",
  )
  parser.add_argument(
    "--no-terminations",
    action="store_true",
    help="Disable task terminations during evaluator rollouts.",
  )
  parser.add_argument(
    "--fixed-command",
    nargs=3,
    type=float,
    metavar=("LIN_X", "LIN_Y", "YAW"),
    default=None,
    help="Force a constant velocity command for velocity tasks.",
  )
  parser.add_argument("--plot-file", default=None)
  parser.add_argument("--output-dir", default=None)
  parser.add_argument("--foot-grid-output-dir", default=None)
  parser.add_argument("--foot-grid-post-step", type=int, default=0)
  parser.add_argument("--foot-grid-video-file", default=None)
  parser.add_argument("--foot-grid-video-stride", type=int, default=1)
  parser.add_argument("--foot-grid-video-fps", type=int, default=None)
  parser.add_argument(
    "--foot-grid-video-metric",
    choices=["downward_speed", "signed_vz", "speed"],
    default="downward_speed",
  )
  parser.add_argument("--foot-grid-video-pressure-row", action="store_true")
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
    from src.evaluation.silent_walking.artifacts import write_eval_artifacts
    from src.evaluation.silent_walking.foot_grid import (
      summarize_post_contact_foot_grid,
      write_foot_grid_heatmap_csv,
      write_foot_grid_raw_csv,
    )
    from src.evaluation.silent_walking.robots import get_robot_spec
    from src.evaluation.silent_walking.runner import run_silent_eval
    from src.evaluation.silent_walking.plotting import (
      save_capsule_distribution_plot,
      save_capsule_force_timeseries_plot,
      save_foot_grid_post_contact_heatmap,
      save_foot_grid_velocity_video,
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
    task_id=args.task_id,
    motion_file=args.motion_file,
    disable_foot_phase_observation=args.disable_foot_phase_observation,
    no_terminations=args.no_terminations,
    fixed_command=tuple(args.fixed_command) if args.fixed_command is not None else None,
  )
  print(
    render_markdown_report(
      result.robot_name,
      args.policy,
      result.summary,
      trace=result.trace,
    )
  )
  if args.output_dir is not None:
    paths = write_eval_artifacts(
      result,
      policy_label=args.policy,
      output_dir=args.output_dir,
    )
    print("\nArtifacts saved to:")
    for name, path in paths.items():
      print(f"- {name}: {path}")
  if args.foot_grid_output_dir is not None:
    output_dir = Path(args.foot_grid_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = summarize_post_contact_foot_grid(
      result.trace,
      post_steps=(0, 1, 3),
    )
    csv_path = write_foot_grid_heatmap_csv(
      output_dir / "foot_grid_post_contact_heatmap.csv",
      rows,
      post_steps=(0, 1, 3),
    )
    heatmap_path = save_foot_grid_post_contact_heatmap(
      result.trace,
      output_dir / "foot_grid_post_contact_heatmap.png",
      title=f"{result.robot_name}: {Path(args.policy).name} foot-grid post-contact heatmap",
      post_step=args.foot_grid_post_step,
    )
    raw_csv_path = write_foot_grid_raw_csv(
      result.trace,
      output_dir / "foot_grid_raw_velocity.csv",
      dt=result.step_dt,
    )
    print(f"Foot grid heatmap CSV saved to: {csv_path}")
    print(f"Foot grid heatmap PNG saved to: {heatmap_path}")
    print(f"Foot grid raw velocity CSV saved to: {raw_csv_path}")
  if args.foot_grid_video_file is not None:
    video_path = save_foot_grid_velocity_video(
      result.trace,
      args.foot_grid_video_file,
      title=f"{result.robot_name}: {Path(args.policy).name} foot-grid velocity",
      dt=result.step_dt,
      stride=args.foot_grid_video_stride,
      fps=args.foot_grid_video_fps,
      metric=args.foot_grid_video_metric,
      include_pressure=args.foot_grid_video_pressure_row,
    )
    print(f"Foot grid velocity video saved to: {video_path}")
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
    foot_grid_heatmap_path = save_foot_grid_post_contact_heatmap(
      result.trace,
      Path(args.plot_file).with_name(f"{Path(args.plot_file).stem}_foot_grid_heatmap.png"),
      title=f"{result.robot_name}: {Path(args.policy).name} foot-grid post-contact heatmap",
      post_step=args.foot_grid_post_step,
    )
    print(f"\nPlot saved to: {trace_path}")
    print(f"Foot distribution saved to: {footmap_path}")
    print(f"Capsule force time series saved to: {capsule_force_path}")
    print(f"Foot grid heatmap saved to: {foot_grid_heatmap_path}")


if __name__ == "__main__":
  main()
