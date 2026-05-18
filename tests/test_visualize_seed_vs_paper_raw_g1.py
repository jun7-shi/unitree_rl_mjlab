import csv
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

from src.motion.seed_bones import G1_29DOF_JOINT_COLUMNS
from tests.test_bvh_full_body import (
  _write_tiny_full_body_bvh,
  _write_tiny_humanoid_full_body_bvh,
  _write_turning_humanoid_full_body_bvh,
)


def test_seed_vs_paper_raw_script_help_lists_visualization_options():
  result = subprocess.run(
    [
      sys.executable,
      "scripts/visualize_seed_vs_paper_raw_g1.py",
      "--help",
    ],
    cwd=Path(__file__).resolve().parents[1],
    check=False,
    capture_output=True,
    text=True,
  )

  assert result.returncode == 0, result.stderr
  assert "seed G1 CSV against paper_v1 raw full-body retargeting" in result.stdout
  assert "--paper-raw-offset-to-seed" in result.stdout
  assert "--global-root" in result.stdout


def test_seed_vs_paper_raw_arg_parser_defaults_to_full_motion_and_inferred_csv():
  from scripts.visualize_seed_vs_paper_raw_g1 import build_arg_parser

  args = build_arg_parser().parse_args(["--bvh", "soma_uniform/bvh/231006/full.bvh"])

  assert args.seed_csv is None
  assert args.start_frame == 0
  assert args.end_frame is None
  assert args.display_mode == "body-centric"


def test_seed_vs_paper_raw_arg_parser_keeps_global_root_as_world_mode_alias():
  from scripts.visualize_seed_vs_paper_raw_g1 import display_mode_from_args, build_arg_parser

  args = build_arg_parser().parse_args(
    ["--bvh", "soma_uniform/bvh/231006/full.bvh", "--global-root"]
  )

  assert display_mode_from_args(args) == "world"


def test_paper_raw_motion_row_preserves_seed_root_and_replaces_all_joints():
  from scripts.visualize_seed_vs_paper_raw_g1 import (
    FULL_BODY_DOF,
    paper_raw_motion_row_from_seed,
  )

  seed_row = [float(index) for index in range(7 + 29)]
  paper_full_q = np.arange(FULL_BODY_DOF, dtype=float) + 100.0

  paper_row = paper_raw_motion_row_from_seed(seed_row, paper_full_q)

  assert paper_row[:7] == seed_row[:7]
  assert paper_row[7:] == list(paper_full_q)


def test_unwrap_to_moves_angle_to_reference_neighborhood():
  from scripts.visualize_seed_vs_paper_raw_g1 import unwrap_to

  assert unwrap_to(-343.0, -50.0) == 17.0
  assert unwrap_to(197.0, 0.0) == -163.0


def test_comparison_offsets_arrange_columns_on_side_axis():
  from scripts.visualize_seed_vs_paper_raw_g1 import comparison_offsets

  offsets = comparison_offsets(spacing=1.25)

  np.testing.assert_allclose(offsets["bvh"], [0.0, -1.25, 0.0])
  np.testing.assert_allclose(offsets["seed"], [0.0, 0.0, 0.0])
  np.testing.assert_allclose(offsets["paper"], [0.0, 1.25, 0.0])


def test_comparison_frames_include_raw_bvh_full_body_skeleton(tmp_path):
  from scripts.visualize_seed_vs_paper_raw_g1 import VisualizerConfig, build_comparison_frames

  bvh_path = tmp_path / "full.bvh"
  _write_tiny_humanoid_full_body_bvh(bvh_path)
  csv_path = tmp_path / "seed.csv"
  with csv_path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.writer(handle)
    writer.writerow(
      [
        "Frame",
        "root_translateX",
        "root_translateY",
        "root_translateZ",
        "root_rotateX",
        "root_rotateY",
        "root_rotateZ",
        *G1_29DOF_JOINT_COLUMNS,
      ]
    )
    writer.writerow([0, 0, 0, 100, 0, 0, 0, *([0] * len(G1_29DOF_JOINT_COLUMNS))])

  frame = build_comparison_frames(
    VisualizerConfig(
      bvh_path=bvh_path,
      seed_csv_path=csv_path,
      start_frame=0,
      end_frame=0,
      apply_orientation_offsets=False,
      align_upper_arm_axes_to_g1=False,
      remove_initial_heading=False,
    )
  )[0]

  assert frame.bvh_full_target.lower.left_leg.hip.shape == (3,)
  assert frame.bvh_full_target.upper.left_arm.shoulder.shape == (3,)


def _write_seed_csv(path: Path, frame_count: int) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.writer(handle)
    writer.writerow(
      [
        "Frame",
        "root_translateX",
        "root_translateY",
        "root_translateZ",
        "root_rotateX",
        "root_rotateY",
        "root_rotateZ",
        *G1_29DOF_JOINT_COLUMNS,
      ]
    )
    for frame in range(frame_count):
      writer.writerow([frame, 0, 0, 100, 0, 0, 0, *([0] * len(G1_29DOF_JOINT_COLUMNS))])


def test_default_comparison_frame_range_uses_full_available_motion(tmp_path):
  from scripts.visualize_seed_vs_paper_raw_g1 import VisualizerConfig, build_comparison_frames

  bvh_path = tmp_path / "full.bvh"
  _write_turning_humanoid_full_body_bvh(bvh_path)
  csv_path = tmp_path / "seed.csv"
  _write_seed_csv(csv_path, frame_count=2)

  frames = build_comparison_frames(
    VisualizerConfig(
      bvh_path=bvh_path,
      seed_csv_path=csv_path,
      apply_orientation_offsets=False,
      align_upper_arm_axes_to_g1=False,
      remove_initial_heading=False,
    )
  )

  assert [frame.frame_index for frame in frames] == [0, 1]


def test_comparison_frames_infer_seed_csv_path_from_bones_seed_bvh_path(tmp_path):
  from scripts.visualize_seed_vs_paper_raw_g1 import VisualizerConfig, build_comparison_frames

  dataset_root = tmp_path / "bones-seed"
  bvh_path = dataset_root / "soma_uniform" / "bvh" / "231006" / "full.bvh"
  bvh_path.parent.mkdir(parents=True)
  _write_tiny_humanoid_full_body_bvh(bvh_path)
  csv_path = dataset_root / "g1" / "csv" / "231006" / "full.csv"
  _write_seed_csv(csv_path, frame_count=1)

  frames = build_comparison_frames(
    VisualizerConfig(
      bvh_path=bvh_path,
      seed_csv_path=None,
      start_frame=0,
      end_frame=0,
      apply_orientation_offsets=False,
      align_upper_arm_axes_to_g1=False,
      remove_initial_heading=False,
    )
  )

  assert frames[0].csv_frame == 0


def test_next_frame_index_wraps_playback_range():
  from scripts.visualize_seed_vs_paper_raw_g1 import next_frame_index

  frames = [250, 251, 252]

  assert next_frame_index(250, frames) == 251
  assert next_frame_index(251, frames) == 252
  assert next_frame_index(252, frames) == 250
  assert next_frame_index(999, frames) == 250


def test_articulated_g1_pose_snapshot_updates_geom_transforms_not_meshes():
  from scripts.visualize_seed_vs_paper_raw_g1 import (
    BODY_CENTRIC_DISPLAY_MODE,
    LEFT_SHOULDER_PITCH_UPPER_INDEX,
    UPPER_BODY_START,
    _articulated_geom_pose_snapshot,
  )
  from scripts.visualize_seed_bvh_axes import _seed_g1_model_context

  model, data, joint_qpos_addresses, visual_geom_ids = _seed_g1_model_context()
  seed_row = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, *([0.0] * len(G1_29DOF_JOINT_COLUMNS))]
  moved_row = list(seed_row)
  moved_row[7 + UPPER_BODY_START + LEFT_SHOULDER_PITCH_UPPER_INDEX] = 0.75

  neutral = _articulated_geom_pose_snapshot(
    model,
    data,
    joint_qpos_addresses,
    visual_geom_ids,
    seed_row,
    offset=np.zeros(3),
    display_mode=BODY_CENTRIC_DISPLAY_MODE,
  )
  moved = _articulated_geom_pose_snapshot(
    model,
    data,
    joint_qpos_addresses,
    visual_geom_ids,
    moved_row,
    offset=np.zeros(3),
    display_mode=BODY_CENTRIC_DISPLAY_MODE,
  )

  assert neutral.geom_ids == moved.geom_ids
  assert neutral.positions.shape == moved.positions.shape == (len(visual_geom_ids), 3)
  assert neutral.wxyzs.shape == moved.wxyzs.shape == (len(visual_geom_ids), 4)
  assert any(
    not np.allclose(lhs, rhs)
    for lhs, rhs in zip(neutral.positions, moved.positions)
  )


def test_articulated_g1_world_display_preserves_root_translation():
  from scripts.visualize_seed_vs_paper_raw_g1 import (
    BODY_CENTRIC_DISPLAY_MODE,
    WORLD_DISPLAY_MODE,
    _articulated_geom_pose_snapshot,
  )
  from scripts.visualize_seed_bvh_axes import _seed_g1_model_context

  model, data, joint_qpos_addresses, visual_geom_ids = _seed_g1_model_context()
  body_row = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, *([0.0] * len(G1_29DOF_JOINT_COLUMNS))]
  world_row = list(body_row)
  world_row[0] = 1.25
  geom_ids = visual_geom_ids[:3]

  body_origin_pose = _articulated_geom_pose_snapshot(
    model,
    data,
    joint_qpos_addresses,
    geom_ids,
    body_row,
    offset=np.zeros(3),
    display_mode=BODY_CENTRIC_DISPLAY_MODE,
  )
  body_shifted_pose = _articulated_geom_pose_snapshot(
    model,
    data,
    joint_qpos_addresses,
    geom_ids,
    world_row,
    offset=np.zeros(3),
    display_mode=BODY_CENTRIC_DISPLAY_MODE,
  )
  world_origin_pose = _articulated_geom_pose_snapshot(
    model,
    data,
    joint_qpos_addresses,
    geom_ids,
    body_row,
    offset=np.zeros(3),
    display_mode=WORLD_DISPLAY_MODE,
  )
  world_shifted_pose = _articulated_geom_pose_snapshot(
    model,
    data,
    joint_qpos_addresses,
    geom_ids,
    world_row,
    offset=np.zeros(3),
    display_mode=WORLD_DISPLAY_MODE,
  )

  np.testing.assert_allclose(body_shifted_pose.positions, body_origin_pose.positions, atol=1e-8)
  np.testing.assert_allclose(
    world_shifted_pose.positions - world_origin_pose.positions,
    np.tile([1.25, 0.0, 0.0], (len(geom_ids), 1)),
    atol=1e-8,
  )


def test_bvh_display_snapshot_updates_points_with_stable_shapes(tmp_path):
  from scripts.visualize_seed_vs_paper_raw_g1 import BODY_CENTRIC_DISPLAY_MODE, _bvh_display_snapshot
  from src.motion.bvh_full_body import load_soma_bvh_full_body_targets

  bvh_path = tmp_path / "full.bvh"
  _write_tiny_full_body_bvh(bvh_path)
  targets = load_soma_bvh_full_body_targets(
    bvh_path,
    apply_orientation_offsets=False,
    align_upper_arm_axes_to_g1=False,
    remove_initial_heading=False,
  )

  first = _bvh_display_snapshot(targets[0], np.zeros(3), display_mode=BODY_CENTRIC_DISPLAY_MODE)
  second = _bvh_display_snapshot(
    targets[0],
    np.array([0.0, 0.5, 0.0]),
    display_mode=BODY_CENTRIC_DISPLAY_MODE,
  )

  assert first.skeleton_segments.shape == second.skeleton_segments.shape
  assert first.axis_segments.shape == second.axis_segments.shape
  assert first.keypoints.shape == second.keypoints.shape
  assert not np.allclose(first.skeleton_segments, second.skeleton_segments)


def test_bvh_world_display_uses_fixed_initial_alignment(tmp_path):
  from scripts.visualize_seed_vs_paper_raw_g1 import (
    WORLD_DISPLAY_MODE,
    bvh_world_alignment_from_frame,
    _bvh_display_snapshot,
  )
  from src.motion.bvh_full_body import load_soma_bvh_full_body_targets

  bvh_path = tmp_path / "full.bvh"
  _write_tiny_full_body_bvh(bvh_path)
  target = load_soma_bvh_full_body_targets(
    bvh_path,
    apply_orientation_offsets=False,
    align_upper_arm_axes_to_g1=False,
    remove_initial_heading=False,
  )[0]
  seed_row = [1.25, -0.5, 0.8, 0.0, 0.0, 0.0, 1.0, *([0.0] * len(G1_29DOF_JOINT_COLUMNS))]
  alignment = bvh_world_alignment_from_frame(target, seed_row)

  snapshot = _bvh_display_snapshot(
    target,
    np.zeros(3),
    display_mode=WORLD_DISPLAY_MODE,
    world_alignment=alignment,
  )

  left_hip = snapshot.keypoints[7]
  right_hip = snapshot.keypoints[10]
  np.testing.assert_allclose(0.5 * (left_hip + right_hip), seed_row[:3], atol=1e-8)

  bvh_delta = np.array([0.4, 0.0, 0.0])
  shifted_target = _translate_full_body_target(target, bvh_delta)
  shifted = _bvh_display_snapshot(
    shifted_target,
    np.zeros(3),
    display_mode=WORLD_DISPLAY_MODE,
    world_alignment=alignment,
  )
  np.testing.assert_allclose(
    shifted.skeleton_segments - snapshot.skeleton_segments,
    np.full_like(snapshot.skeleton_segments, bvh_delta),
    atol=1e-8,
  )

  yaw_90_row = [1.25, -0.5, 0.8, 0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5), *([0.0] * len(G1_29DOF_JOINT_COLUMNS))]
  yaw_90_alignment = bvh_world_alignment_from_frame(target, yaw_90_row)
  rotated = _bvh_display_snapshot(
    target,
    np.zeros(3),
    display_mode=WORLD_DISPLAY_MODE,
    world_alignment=yaw_90_alignment,
  )
  np.testing.assert_allclose(rotated.skeleton_segments, snapshot.skeleton_segments, atol=1e-8)


def _translate_full_body_target(target, delta: np.ndarray):
  def arm(value):
    return replace(
      value,
      shoulder=value.shoulder + delta,
      elbow=value.elbow + delta,
      wrist=value.wrist + delta,
    )

  def leg(value):
    return replace(
      value,
      hip=value.hip + delta,
      knee=value.knee + delta,
      ankle=value.ankle + delta,
    )

  return replace(
    target,
    lower=replace(
      target.lower,
      left_leg=leg(target.lower.left_leg),
      right_leg=leg(target.lower.right_leg),
    ),
    upper=replace(
      target.upper,
      chest_position=target.upper.chest_position + delta,
      left_arm=arm(target.upper.left_arm),
      right_arm=arm(target.upper.right_arm),
    ),
  )
