from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from src.motion.bvh_full_body import load_soma_bvh_full_body_targets
from src.motion.seed_bones import G1_29DOF_JOINT_COLUMNS
from src.motion.sew_full_body import G1FullBodySEWRetargeter, retarget_full_body_targets

G1_FULL_BODY_JOINT_COLUMNS = tuple(G1_29DOF_JOINT_COLUMNS)
G1_LOWER_BODY_JOINT_COLUMNS = tuple(G1_29DOF_JOINT_COLUMNS[:12])


@dataclass(frozen=True)
class CompareConfig:
  bvh_path: Path
  seed_csv_path: Path
  start_frame: int = 0
  max_frames: int | None = None
  apply_orientation_offsets: bool = True
  align_upper_arm_axes_to_g1: bool = True
  apply_lower_body_offsets: bool = True
  remove_initial_heading: bool = True


@dataclass(frozen=True)
class CompareSummary:
  frame_count: int
  success_count: int
  max_errors: dict[str, float]
  mean_abs_joint_diff_deg: float
  max_abs_joint_diff_deg: float
  lower_mean_abs_joint_diff_deg: float
  lower_max_abs_joint_diff_deg: float
  per_joint_max_abs_diff_deg: dict[str, float]


def load_seed_full_body_joint_angles(path: str | Path) -> np.ndarray:
  input_path = Path(path)
  with input_path.open(newline="", encoding="utf-8") as handle:
    reader = csv.DictReader(handle)
    if reader.fieldnames is None:
      raise ValueError(f"Seed G1 CSV has no header: {input_path}")
    missing = [column for column in G1_FULL_BODY_JOINT_COLUMNS if column not in reader.fieldnames]
    if missing:
      raise ValueError(f"Seed G1 CSV is missing full-body columns: {', '.join(missing)}")
    rows = [
      [math.radians(float(row[column])) for column in G1_FULL_BODY_JOINT_COLUMNS]
      for row in reader
    ]
  return np.asarray(rows, dtype=float)


def compare_retarget_with_seed_csv(config: CompareConfig) -> CompareSummary:
  if config.start_frame < 0:
    raise ValueError(f"start_frame must be non-negative, got {config.start_frame}")
  if config.max_frames is not None and config.max_frames <= 0:
    raise ValueError(f"max_frames must be positive when set, got {config.max_frames}")

  frame_stop = None if config.max_frames is None else config.start_frame + config.max_frames
  frame_slice = slice(config.start_frame, frame_stop)
  seed_q = load_seed_full_body_joint_angles(config.seed_csv_path)[frame_slice]
  targets = load_soma_bvh_full_body_targets(
    config.bvh_path,
    frame_slice=frame_slice,
    apply_orientation_offsets=config.apply_orientation_offsets,
    align_upper_arm_axes_to_g1=config.align_upper_arm_axes_to_g1,
    apply_lower_body_offsets=config.apply_lower_body_offsets,
    remove_initial_heading=config.remove_initial_heading,
  )
  if len(targets) != len(seed_q):
    raise ValueError(
      f"Frame count mismatch after slicing: BVH has {len(targets)} targets, "
      f"seed CSV has {len(seed_q)} rows"
    )

  retargeter = G1FullBodySEWRetargeter()
  results = retarget_full_body_targets(targets, retargeter=retargeter)
  if not results:
    return CompareSummary(
      frame_count=0,
      success_count=0,
      max_errors={},
      mean_abs_joint_diff_deg=0.0,
      max_abs_joint_diff_deg=0.0,
      lower_mean_abs_joint_diff_deg=0.0,
      lower_max_abs_joint_diff_deg=0.0,
      per_joint_max_abs_diff_deg={},
    )

  retarget_q = np.asarray([result.joint_angles for result in results], dtype=float)
  abs_diff_deg = np.abs(np.rad2deg(retarget_q - seed_q))
  lower_abs_diff_deg = abs_diff_deg[:, : len(G1_LOWER_BODY_JOINT_COLUMNS)]
  max_errors = {
    key: max(float(result.errors[key]) for result in results)
    for key in sorted(results[0].errors)
  }
  per_joint_max = {
    column.removesuffix("_dof"): float(value)
    for column, value in zip(G1_FULL_BODY_JOINT_COLUMNS, np.max(abs_diff_deg, axis=0))
  }
  return CompareSummary(
    frame_count=len(results),
    success_count=sum(1 for result in results if result.success),
    max_errors=max_errors,
    mean_abs_joint_diff_deg=float(np.mean(abs_diff_deg)),
    max_abs_joint_diff_deg=float(np.max(abs_diff_deg)),
    lower_mean_abs_joint_diff_deg=float(np.mean(lower_abs_diff_deg)),
    lower_max_abs_joint_diff_deg=float(np.max(lower_abs_diff_deg)),
    per_joint_max_abs_diff_deg=per_joint_max,
  )


def _print_summary(summary: CompareSummary, *, top_joints: int) -> None:
  print(f"frames: {summary.frame_count}")
  print(f"successful frames: {summary.success_count}/{summary.frame_count}")
  for key, value in summary.max_errors.items():
    print(f"max target {key}: {value:.6f}")
  print(f"mean abs joint diff deg: {summary.mean_abs_joint_diff_deg:.6f}")
  print(f"max abs joint diff deg: {summary.max_abs_joint_diff_deg:.6f}")
  print(f"lower mean abs joint diff deg: {summary.lower_mean_abs_joint_diff_deg:.6f}")
  print(f"lower max abs joint diff deg: {summary.lower_max_abs_joint_diff_deg:.6f}")
  ranked = sorted(
    summary.per_joint_max_abs_diff_deg.items(),
    key=lambda item: item[1],
    reverse=True,
  )
  for name, value in ranked[:top_joints]:
    print(f"max diff {name}: {value:.6f}")


def build_arg_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
    description="Compare full-body SEW-Mimic BVH retargeting against a bones-seed G1 CSV."
  )
  parser.add_argument("--bvh", required=True, type=Path, help="Input SOMA/bones-seed BVH file.")
  parser.add_argument("--seed-csv", required=True, type=Path, help="bones-seed G1 CSV for the same motion.")
  parser.add_argument("--start-frame", type=int, default=0, help="First frame to compare.")
  parser.add_argument("--max-frames", type=int, default=None, help="Compare only N frames.")
  parser.add_argument("--top-joints", type=int, default=12, help="Number of largest per-joint differences to print.")
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
    help="Keep the BVH global heading instead of removing the first frame's heading before comparison.",
  )
  return parser


def main(argv: Sequence[str] | None = None) -> int:
  args = build_arg_parser().parse_args(argv)
  summary = compare_retarget_with_seed_csv(
    CompareConfig(
      bvh_path=args.bvh,
      seed_csv_path=args.seed_csv,
      start_frame=args.start_frame,
      max_frames=args.max_frames,
      apply_orientation_offsets=not args.raw_orientations,
      align_upper_arm_axes_to_g1=not args.raw_upper_arm_axes,
      apply_lower_body_offsets=not args.raw_lower_body_offsets,
      remove_initial_heading=not args.keep_global_heading,
    )
  )
  _print_summary(summary, top_joints=args.top_joints)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
