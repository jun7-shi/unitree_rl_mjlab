from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from scripts.compare_sew_with_seed_csv import load_seed_upper_body_joint_angles
from src.motion.bvh_upper_body import load_soma_bvh_upper_body_human_keypoint_targets
from src.motion.sew_mimic import normalize, solve_two_axis_rotation
from src.motion.sew_upper_body import G1UpperBodySEWRetargeter

DATA_ROOT = Path("/data/jun7.shi/datasets/bones-seed")
BVH_DIR = DATA_ROOT / "soma_uniform/bvh/231006"
CSV_DIR = DATA_ROOT / "g1/csv/231006"
OUT_DIR = REPO_ROOT / "analysis/a464_multimotion_l_shoulder_pitch_paper_pure"
LEFT_SHOULDER_PITCH_UPPER_INDEX = 3

MOTIONS = (
  "dance_breezy_001__A464",
  "dance_click_002__A464",
  "dance_crossbounce_006__A464",
  "dance_dance_moves_002__A464",
  "dance_deep_dab_001__A464",
  "dance_dip_001__A464",
  "dance_disco_fever_001__A464",
  "dance_distraction_dance_001__A464",
  "dance_drippin_flavor_005__A464",
  "dance_feel_the_flow_002__A464",
  "dance_fishin_002__A464",
  "dance_floss_002__A464",
  "dance_forget_me_not_001__A464",
  "dance_freedom_wheels_001__A464",
  "dance_fresh_001__A464",
  "dance_fright_funk_001__A464",
)


def unwrap_to(value_deg: float, reference_deg: float) -> float:
  return value_deg + 360.0 * round((reference_deg - value_deg) / 360.0)


def compute_motion_rows(retargeter: G1UpperBodySEWRetargeter, motion: str) -> list[tuple]:
  bvh_path = BVH_DIR / f"{motion}.bvh"
  csv_path = CSV_DIR / f"{motion}.csv"
  targets = load_soma_bvh_upper_body_human_keypoint_targets(
    bvh_path,
    apply_orientation_offsets=True,
    remove_initial_heading=True,
  )
  seed_q = load_seed_upper_body_joint_angles(csv_path)
  frame_count = min(len(targets), len(seed_q))

  q_prev = np.zeros(17)
  rows: list[tuple] = []
  for frame_index in range(frame_count):
    target = targets[frame_index]
    q = retargeter._solve_orientation_group(
      q_prev.copy(),
      slice(0, 3),
      retargeter._torso_body_id,
      target.chest_orientation,
    )
    upper_arm = normalize(target.left_arm.elbow - target.left_arm.shoulder)
    indexes = (3, 4)
    base = q.copy()
    base[list(indexes)] = 0.0
    retargeter._set_upper_body_joint_angles(base)
    first_axis = normalize(retargeter.data.xaxis[retargeter.controlled_joint_ids[indexes[0]]])
    second_axis = normalize(retargeter.data.xaxis[retargeter.controlled_joint_ids[indexes[1]]])
    initial_axis = normalize(retargeter.data.xaxis[retargeter._axis_joint_ids["left"]["upper"]])
    candidates = solve_two_axis_rotation(initial_axis, upper_arm, first_axis, second_axis)
    if not candidates:
      raise RuntimeError(f"{motion} frame {frame_index}: no SEW candidates returned")
    pitches_rad = [pair[0] for pair in candidates]
    if len(pitches_rad) == 1:
      pitches_rad = pitches_rad * 2

    seed_deg = float(np.degrees(seed_q[frame_index, LEFT_SHOULDER_PITCH_UPPER_INDEX]))
    principal0 = float(np.degrees(pitches_rad[0]))
    principal1 = float(np.degrees(pitches_rad[1]))
    unwrapped0 = unwrap_to(principal0, seed_deg)
    unwrapped1 = unwrap_to(principal1, seed_deg)
    diff0 = abs(unwrapped0 - seed_deg)
    diff1 = abs(unwrapped1 - seed_deg)
    nearest_id = 0 if diff0 <= diff1 else 1
    nearest_diff = min(diff0, diff1)
    rows.append(
      (
        frame_index,
        seed_deg,
        principal0,
        principal1,
        unwrapped0,
        unwrapped1,
        nearest_diff,
        nearest_id,
      )
    )
    q_prev = q
  return rows


def _write_values_csv(out_dir: Path, all_rows: list[tuple]) -> Path:
  values_path = out_dir / "a464_multimotion_l_shoulder_pitch_direct_candidates_all_frames_values.csv"
  with values_path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.writer(handle)
    writer.writerow(
      [
        "motion",
        "frame",
        "seed_deg",
        "principal_cand0_deg",
        "principal_cand1_deg",
        "unwrapped_cand0_deg",
        "unwrapped_cand1_deg",
        "nearest_candidate_abs_diff_deg",
        "nearest_candidate_id",
      ]
    )
    writer.writerows(all_rows)
  return values_path


def _write_summary_csv(out_dir: Path, summaries: list[tuple]) -> Path:
  summary_path = out_dir / "a464_multimotion_l_shoulder_pitch_direct_candidates_all_frames_summary.csv"
  with summary_path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.writer(handle)
    writer.writerow(
      [
        "motion",
        "frames",
        "nearest_candidate_mean_abs_diff_deg",
        "nearest_candidate_p95_abs_diff_deg",
        "nearest_candidate_max_abs_diff_deg",
        "nearest_candidate_max_abs_diff_frame",
        "candidate_choice_switches",
        "cand0_min_deg",
        "cand0_max_deg",
        "cand1_min_deg",
        "cand1_max_deg",
      ]
    )
    writer.writerows(summaries)
  return summary_path


def _plot_grid(out_dir: Path, motion_rows: dict[str, list[tuple]]) -> tuple[Path, Path]:
  motion_count = len(motion_rows)
  ncols = 4
  nrows = (motion_count + ncols - 1) // ncols
  fig, axes = plt.subplots(nrows, ncols, figsize=(22, 4.2 * nrows), sharex=False)
  axes = np.asarray(axes).flatten()
  for axis, motion in zip(axes, motion_rows):
    rows = motion_rows[motion]
    frames = np.array([row[0] for row in rows])
    seed = np.array([row[1] for row in rows])
    unwrapped0 = np.array([row[4] for row in rows])
    unwrapped1 = np.array([row[5] for row in rows])
    axis.plot(frames, seed, label="seed", color="black", linewidth=1.6)
    axis.plot(frames, unwrapped0, label="unwrapped cand0", color="tab:blue", linewidth=0.9, alpha=0.85)
    axis.plot(frames, unwrapped1, label="unwrapped cand1", color="tab:orange", linewidth=0.9, alpha=0.85)
    axis.set_title(f"{motion} ({len(rows)} frames)", fontsize=9)
    axis.set_xlabel("frame")
    axis.set_ylabel("left shoulder pitch (deg)")
    axis.grid(True, alpha=0.3)
    axis.legend(fontsize=7, loc="best")
  for axis in axes[motion_count:]:
    axis.set_visible(False)
  fig.suptitle("A464 left shoulder pitch — seed vs SEW closed-form candidates (all frames, paper-pure BVH input)", fontsize=14)
  fig.tight_layout(rect=[0, 0, 1, 0.97])
  png_path = out_dir / "a464_multimotion_l_shoulder_pitch_direct_candidates_all_frames.png"
  pdf_path = out_dir / "a464_multimotion_l_shoulder_pitch_direct_candidates_all_frames.pdf"
  fig.savefig(png_path, dpi=150)
  fig.savefig(pdf_path)
  plt.close(fig)
  return png_path, pdf_path


def main() -> int:
  OUT_DIR.mkdir(parents=True, exist_ok=True)
  retargeter = G1UpperBodySEWRetargeter()

  all_rows: list[tuple] = []
  summaries: list[tuple] = []
  motion_rows: dict[str, list[tuple]] = {}
  for motion in MOTIONS:
    print(f"processing {motion}...")
    rows = compute_motion_rows(retargeter, motion)
    motion_rows[motion] = rows
    for row in rows:
      all_rows.append((motion, *row))

    diffs = np.array([row[6] for row in rows])
    nearest_ids = np.array([row[7] for row in rows])
    unwrapped0 = np.array([row[4] for row in rows])
    unwrapped1 = np.array([row[5] for row in rows])
    switches = int(np.sum(np.abs(np.diff(nearest_ids)) > 0))
    summaries.append(
      (
        motion,
        len(rows),
        float(np.mean(diffs)),
        float(np.percentile(diffs, 95)),
        float(np.max(diffs)),
        int(np.argmax(diffs)),
        switches,
        float(np.min(unwrapped0)),
        float(np.max(unwrapped0)),
        float(np.min(unwrapped1)),
        float(np.max(unwrapped1)),
      )
    )

  values_path = _write_values_csv(OUT_DIR, all_rows)
  summary_path = _write_summary_csv(OUT_DIR, summaries)
  png_path, pdf_path = _plot_grid(OUT_DIR, motion_rows)
  print(f"wrote values: {values_path}")
  print(f"wrote summary: {summary_path}")
  print(f"wrote plot: {png_path}")
  print(f"wrote plot: {pdf_path}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
