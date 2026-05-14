from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import mujoco

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from src.motion.bvh_full_body import load_soma_bvh_full_body_targets
from src.motion.sew_mimic import BRANCH_V2_ALGORITHM, SEW_ALGORITHM_CONFIGS
from src.motion.sew_full_body import G1FullBodySEWRetargeter, FullBodyRetargetResult, retarget_full_body_targets


@dataclass(frozen=True)
class PreviewConfig:
  bvh_path: Path
  start_frame: int = 0
  max_frames: int | None = None
  fps: float = 120.0
  no_viewer: bool = False
  loop: bool = True
  apply_orientation_offsets: bool = True
  align_upper_arm_axes_to_g1: bool = True
  apply_lower_body_offsets: bool = True
  remove_initial_heading: bool = True
  localize_to_body_frame: bool = True
  algorithm_version: str = BRANCH_V2_ALGORITHM.name


@dataclass(frozen=True)
class PreviewSummary:
  frame_count: int
  success_count: int
  max_errors: dict[str, float]


def run_preview(config: PreviewConfig) -> PreviewSummary:
  if config.start_frame < 0:
    raise ValueError(f"start_frame must be non-negative, got {config.start_frame}")
  if config.max_frames is not None and config.max_frames <= 0:
    raise ValueError(f"max_frames must be positive when set, got {config.max_frames}")
  if config.fps <= 0.0:
    raise ValueError(f"fps must be positive, got {config.fps}")

  frame_stop = None if config.max_frames is None else config.start_frame + config.max_frames
  frame_slice = slice(config.start_frame, frame_stop)
  targets = load_soma_bvh_full_body_targets(
    config.bvh_path,
    frame_slice=frame_slice,
    apply_orientation_offsets=config.apply_orientation_offsets,
    align_upper_arm_axes_to_g1=config.align_upper_arm_axes_to_g1,
    apply_lower_body_offsets=config.apply_lower_body_offsets,
    remove_initial_heading=config.remove_initial_heading,
    localize_to_body_frame=config.localize_to_body_frame and config.apply_lower_body_offsets,
  )
  retargeter = G1FullBodySEWRetargeter(algorithm_version=config.algorithm_version)
  results = retarget_full_body_targets(targets, retargeter=retargeter)
  summary = _summarize_results(results)

  if config.no_viewer:
    return summary

  _play_results_in_mujoco(retargeter, results, fps=config.fps, loop=config.loop)
  return summary


def _summarize_results(results: Sequence[FullBodyRetargetResult]) -> PreviewSummary:
  if not results:
    return PreviewSummary(frame_count=0, success_count=0, max_errors={})
  keys = sorted(results[0].errors)
  max_errors = {
    key: max(float(result.errors[key]) for result in results)
    for key in keys
  }
  return PreviewSummary(
    frame_count=len(results),
    success_count=sum(1 for result in results if result.success),
    max_errors=max_errors,
  )


def _play_results_in_mujoco(
  retargeter: G1FullBodySEWRetargeter,
  results: Sequence[FullBodyRetargetResult],
  *,
  fps: float,
  loop: bool,
) -> None:
  import mujoco.viewer

  data = mujoco.MjData(retargeter.model)
  frame_dt = 1.0 / fps
  with mujoco.viewer.launch_passive(retargeter.model, data) as viewer:
    while viewer.is_running():
      for result in results:
        if not viewer.is_running():
          break
        frame_start = time.time()
        data.qpos[:] = result.full_qpos
        mujoco.mj_forward(retargeter.model, data)
        viewer.sync()
        elapsed = time.time() - frame_start
        if elapsed < frame_dt:
          time.sleep(frame_dt - elapsed)
      if not loop:
        break


def _print_summary(summary: PreviewSummary) -> None:
  print(f"frames: {summary.frame_count}")
  print(f"successful frames: {summary.success_count}/{summary.frame_count}")
  for key, value in summary.max_errors.items():
    print(f"max {key}: {value:.6f}")


def build_arg_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description="Preview full-body SEW-Mimic BVH retargeting on Unitree G1.")
  parser.add_argument("--bvh", required=True, type=Path, help="Input SOMA/bones-seed BVH file.")
  parser.add_argument("--start-frame", type=int, default=0, help="First BVH frame to retarget.")
  parser.add_argument("--max-frames", type=int, default=None, help="Retarget only the first N frames.")
  parser.add_argument("--fps", type=float, default=120.0, help="Playback frame rate.")
  parser.add_argument(
    "--algorithm-version",
    choices=tuple(SEW_ALGORITHM_CONFIGS),
    default=BRANCH_V2_ALGORITHM.name,
    help="SEW candidate-selection version to use.",
  )
  parser.add_argument("--no-viewer", action="store_true", help="Retarget and print diagnostics without opening MuJoCo.")
  parser.add_argument("--once", action="store_true", help="Play the sequence once instead of looping until closed.")
  parser.add_argument(
    "--raw-orientations",
    action="store_true",
    help="Use raw BVH Chest/Hand orientations instead of soma-retargeter SOMA-to-G1 orientation offsets.",
  )
  parser.add_argument(
    "--raw-upper-arm-axes",
    action="store_true",
    help="Use raw BVH upper-arm segment directions instead of flipping them to match G1 shoulder axes.",
  )
  parser.add_argument(
    "--raw-lower-body-offsets",
    action="store_true",
    help="Use raw BVH leg keypoints instead of soma-retargeter SOMA-to-G1 lower-body scaler offsets.",
  )
  parser.add_argument(
    "--keep-global-heading",
    action="store_true",
    help="Keep the BVH global heading instead of removing the first frame's heading for fixed-base preview.",
  )
  parser.add_argument(
    "--world-frame-targets",
    action="store_true",
    help="Retarget BVH targets in world frame instead of localizing each frame to the body frame.",
  )
  return parser


def main(argv: Sequence[str] | None = None) -> int:
  args = build_arg_parser().parse_args(argv)
  summary = run_preview(
    PreviewConfig(
      bvh_path=args.bvh,
      start_frame=args.start_frame,
      max_frames=args.max_frames,
      fps=args.fps,
      no_viewer=args.no_viewer,
      loop=not args.once,
      apply_orientation_offsets=not args.raw_orientations,
      align_upper_arm_axes_to_g1=not args.raw_upper_arm_axes,
      apply_lower_body_offsets=not args.raw_lower_body_offsets,
      remove_initial_heading=not args.keep_global_heading,
      localize_to_body_frame=not args.world_frame_targets,
      algorithm_version=args.algorithm_version,
    )
  )
  _print_summary(summary)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
