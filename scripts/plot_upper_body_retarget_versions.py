from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from scripts.compare_sew_full_body_with_seed_csv import load_seed_full_body_joint_angles
from src.motion.bvh_full_body import load_soma_bvh_full_body_targets
from src.motion.seed_bones import G1_29DOF_JOINT_COLUMNS
from src.motion.sew_full_body import G1FullBodySEWRetargeter, retarget_full_body_targets
from src.motion.sew_mimic import BRANCH_V2_ALGORITHM, SEW_ALGORITHM_CONFIGS


UPPER_START = 12
UPPER_COLUMNS = tuple(G1_29DOF_JOINT_COLUMNS[UPPER_START:])
UPPER_NAMES = tuple(column.removesuffix("_joint_dof").removesuffix("_dof") for column in UPPER_COLUMNS)
VERSION_COLORS = {
  "paper_v1": "#d62728",
  "branch_v2": "#2ca02c",
}


@dataclass(frozen=True)
class PlotConfig:
  bvh_path: Path
  seed_csv_path: Path
  output_dir: Path
  start_frame: int = 0
  max_frames: int | None = None
  versions: tuple[str, ...] = ("paper_v1", BRANCH_V2_ALGORITHM.name)
  apply_orientation_offsets: bool = True
  align_upper_arm_axes_to_g1: bool = True
  apply_lower_body_offsets: bool = True
  remove_initial_heading: bool = True
  localize_to_body_frame: bool = True


def plot_upper_body_retarget_versions(config: PlotConfig) -> dict[str, Path]:
  if config.start_frame < 0:
    raise ValueError(f"start_frame must be non-negative, got {config.start_frame}")
  if config.max_frames is not None and config.max_frames <= 0:
    raise ValueError(f"max_frames must be positive when set, got {config.max_frames}")
  for version in config.versions:
    if version not in SEW_ALGORITHM_CONFIGS:
      options = ", ".join(sorted(SEW_ALGORITHM_CONFIGS))
      raise ValueError(f"Unknown SEW algorithm version '{version}'. Expected one of: {options}")

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
    localize_to_body_frame=config.localize_to_body_frame and config.apply_lower_body_offsets,
  )
  if len(targets) != len(seed_q):
    raise ValueError(
      f"Frame count mismatch after slicing: BVH has {len(targets)} targets, "
      f"seed CSV has {len(seed_q)} rows"
    )

  config.output_dir.mkdir(parents=True, exist_ok=True)
  frames = np.arange(config.start_frame, config.start_frame + len(targets))
  seed_upper = np.rad2deg(seed_q[:, UPPER_START:])
  version_upper = {
    version: _retarget_upper_degrees(targets, version)
    for version in config.versions
  }
  limits = np.rad2deg(G1FullBodySEWRetargeter(algorithm_version=config.versions[0]).upper.joint_limits)

  stem = f"{Path(config.bvh_path).stem}_upper_{config.start_frame}_{frames[-1] if len(frames) else config.start_frame}"
  plot_path = config.output_dir / f"{stem}.png"
  stats_path = config.output_dir / f"{stem}_stats.csv"
  arrays_path = config.output_dir / f"{stem}_arrays.npz"

  _plot_upper(plot_path, frames, seed_upper, version_upper, limits)
  _write_stats(stats_path, seed_upper, version_upper, limits, start_frame=config.start_frame)
  np.savez(
    arrays_path,
    frames=frames,
    seed_upper_deg=seed_upper,
    upper_names=np.asarray(UPPER_NAMES),
    joint_limits_deg=limits,
    **{f"{version}_upper_deg": values for version, values in version_upper.items()},
  )
  return {
    "plot": plot_path,
    "stats": stats_path,
    "arrays": arrays_path,
  }


def _retarget_upper_degrees(targets, version: str) -> np.ndarray:
  retargeter = G1FullBodySEWRetargeter(algorithm_version=version)
  results = retarget_full_body_targets(targets, retargeter=retargeter)
  q = np.asarray([result.joint_angles for result in results], dtype=float)
  return np.rad2deg(q[:, UPPER_START:])


def _plot_upper(
  path: Path,
  frames: np.ndarray,
  seed_upper: np.ndarray,
  version_upper: dict[str, np.ndarray],
  limits: np.ndarray,
) -> None:
  fig, axes = plt.subplots(len(UPPER_NAMES), 1, figsize=(16, 2.2 * len(UPPER_NAMES)), sharex=True)
  for joint_index, ax in enumerate(axes):
    ax.plot(frames, seed_upper[:, joint_index], color="#1f77b4", linewidth=1.2, label="bones-seed G1.csv")
    for version, values in version_upper.items():
      ax.plot(
        frames,
        values[:, joint_index],
        color=VERSION_COLORS.get(version),
        linewidth=1.0,
        label=version,
      )
    ax.axhline(limits[joint_index, 0], color="0.65", linestyle="--", linewidth=0.7)
    ax.axhline(limits[joint_index, 1], color="0.65", linestyle="--", linewidth=0.7)
    ax.set_ylabel(_short_name(UPPER_NAMES[joint_index]), rotation=0, ha="right", va="center")
    ax.grid(True, linewidth=0.3, alpha=0.5)
    if joint_index == 0:
      ax.legend(loc="upper right")
  axes[-1].set_xlabel("frame")
  fig.suptitle("Upper-body joints: SWE versions vs bones-seed G1.csv")
  fig.tight_layout(rect=(0, 0, 1, 0.995))
  fig.savefig(path, dpi=150)
  plt.close(fig)


def _write_stats(
  path: Path,
  seed_upper: np.ndarray,
  version_upper: dict[str, np.ndarray],
  limits: np.ndarray,
  *,
  start_frame: int,
) -> None:
  fieldnames = [
    "version",
    "joint",
    "limit_low_deg",
    "limit_high_deg",
    "mean_diff_deg",
    "mean_abs_diff_deg",
    "median_abs_diff_deg",
    "p95_abs_diff_deg",
    "max_abs_diff_deg",
    "max_abs_diff_frame",
    "version_at_max_deg",
    "seed_at_max_deg",
  ]
  with path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    for version, values in version_upper.items():
      diff = values - seed_upper
      abs_diff = np.abs(diff)
      for joint_index, joint in enumerate(UPPER_NAMES):
        max_index = int(np.argmax(abs_diff[:, joint_index]))
        writer.writerow(
          {
            "version": version,
            "joint": joint,
            "limit_low_deg": float(limits[joint_index, 0]),
            "limit_high_deg": float(limits[joint_index, 1]),
            "mean_diff_deg": float(np.mean(diff[:, joint_index])),
            "mean_abs_diff_deg": float(np.mean(abs_diff[:, joint_index])),
            "median_abs_diff_deg": float(np.median(abs_diff[:, joint_index])),
            "p95_abs_diff_deg": float(np.percentile(abs_diff[:, joint_index], 95)),
            "max_abs_diff_deg": float(abs_diff[max_index, joint_index]),
            "max_abs_diff_frame": start_frame + max_index,
            "version_at_max_deg": float(values[max_index, joint_index]),
            "seed_at_max_deg": float(seed_upper[max_index, joint_index]),
          }
        )


def _short_name(name: str) -> str:
  return name.replace("left_", "L_").replace("right_", "R_").replace("waist_", "W_")


def build_arg_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
    description="Plot upper-body SWE retargeting versions against a bones-seed G1 CSV."
  )
  parser.add_argument("--bvh", required=True, type=Path, help="Input SOMA/bones-seed BVH file.")
  parser.add_argument("--seed-csv", required=True, type=Path, help="bones-seed G1 CSV for the same motion.")
  parser.add_argument("--output-dir", required=True, type=Path, help="Directory for PNG, CSV, and NPZ outputs.")
  parser.add_argument("--start-frame", type=int, default=0, help="First frame to plot.")
  parser.add_argument("--max-frames", type=int, default=None, help="Plot only N frames.")
  parser.add_argument(
    "--versions",
    nargs="+",
    choices=sorted(SEW_ALGORITHM_CONFIGS),
    default=["paper_v1", BRANCH_V2_ALGORITHM.name],
    help="SEW algorithm versions to plot.",
  )
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
    help="Keep the BVH global heading instead of removing the first frame's heading before plotting.",
  )
  parser.add_argument(
    "--world-frame-targets",
    action="store_true",
    help="Retarget BVH targets in world frame instead of localizing each frame to the body frame.",
  )
  return parser


def main(argv: Sequence[str] | None = None) -> int:
  args = build_arg_parser().parse_args(argv)
  outputs = plot_upper_body_retarget_versions(
    PlotConfig(
      bvh_path=args.bvh,
      seed_csv_path=args.seed_csv,
      output_dir=args.output_dir,
      start_frame=args.start_frame,
      max_frames=args.max_frames,
      versions=tuple(args.versions),
      apply_orientation_offsets=not args.raw_orientations,
      align_upper_arm_axes_to_g1=not args.raw_upper_arm_axes,
      apply_lower_body_offsets=not args.raw_lower_body_offsets,
      remove_initial_heading=not args.keep_global_heading,
      localize_to_body_frame=not args.world_frame_targets,
    )
  )
  for label, path in outputs.items():
    print(f"{label}: {path}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
