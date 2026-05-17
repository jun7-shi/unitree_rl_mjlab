import csv
import math
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np

from src.motion.seed_bones import G1_29DOF_JOINT_COLUMNS
from tests.test_bvh_full_body import _write_tiny_full_body_bvh


def test_visualizer_script_help_lists_axis_overlay_options():
  result = subprocess.run(
    [
      sys.executable,
      "scripts/visualize_seed_bvh_axes.py",
      "--help",
    ],
    cwd=Path(__file__).resolve().parents[1],
    check=False,
    capture_output=True,
    text=True,
  )

  assert result.returncode == 0, result.stderr
  assert "--seed-csv" in result.stdout
  assert "--start-frame" in result.stdout
  assert "--axis-length" in result.stdout
  assert "--hide-seed-g1-mesh" in result.stdout
  assert "--global-seed-g1-mesh" in result.stdout
  assert "--layout" in result.stdout
  assert "seed/BVH arm axes" in result.stdout


def test_set_qpos_from_seed_motion_row_uses_mujoco_freejoint_quat_order():
  from scripts.visualize_seed_bvh_axes import _set_qpos_from_seed_motion_row

  model = mujoco.MjModel.from_xml_path(
    str(Path(__file__).resolve().parents[1] / "src/assets/robots/unitree_g1/xmls/g1.xml")
  )
  data = mujoco.MjData(model)
  joint_qpos_addresses = [
    int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, column.removesuffix("_dof"))])
    for column in G1_29DOF_JOINT_COLUMNS
  ]
  row = [1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 0.9, *([0.0] * len(G1_29DOF_JOINT_COLUMNS))]

  _set_qpos_from_seed_motion_row(data, joint_qpos_addresses, row)

  np.testing.assert_allclose(data.qpos[:3], [1.0, 2.0, 3.0])
  np.testing.assert_allclose(data.qpos[3:7], [0.9, 0.1, 0.2, 0.3])


def test_load_seed_rows_preserves_csv_frame_numbers(tmp_path):
  from scripts.visualize_seed_bvh_axes import _load_seed_motion_rows

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
    writer.writerow([260, 0, 0, 100, 0, 0, 0, *([0] * len(G1_29DOF_JOINT_COLUMNS))])
    writer.writerow([275, 0, 0, 100, 0, 0, 0, *([0] * len(G1_29DOF_JOINT_COLUMNS))])

  frame_numbers, rows = _load_seed_motion_rows(csv_path)

  assert frame_numbers == [260, 275]
  assert len(rows) == 2
  assert math.isclose(rows[0][2], 1.0)


def test_angle_degrees_reports_unsigned_axis_angle():
  from scripts.visualize_seed_bvh_axes import _axis_angle_degrees

  assert math.isclose(_axis_angle_degrees(np.array([1, 0, 0]), np.array([1, 0, 0])), 0.0)
  assert math.isclose(_axis_angle_degrees(np.array([1, 0, 0]), np.array([0, 1, 0])), 90.0)


def test_seed_g1_mesh_snapshot_can_be_centered_on_seed_upper_body():
  from scripts.visualize_seed_bvh_axes import (
    _seed_g1_physical_target,
    _seed_g1_model_context,
    _set_qpos_from_seed_motion_row,
    _seed_g1_mesh_snapshot,
    _target_origin,
  )

  row = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, *([0.0] * len(G1_29DOF_JOINT_COLUMNS))]
  model, data, joint_qpos_addresses, visual_geom_ids = _seed_g1_model_context()
  snapshot = _seed_g1_mesh_snapshot(
    model,
    data,
    joint_qpos_addresses,
    visual_geom_ids,
    row,
    center=np.zeros(3),
  )

  assert snapshot.vertices.ndim == 2
  assert snapshot.vertices.shape[1] == 3
  assert snapshot.faces.ndim == 2
  assert snapshot.faces.shape[1] == 3
  assert len(snapshot.vertices) > 0

  _set_qpos_from_seed_motion_row(data, joint_qpos_addresses, row)
  mujoco.mj_forward(model, data)
  physical = _seed_g1_physical_target(model, data)
  np.testing.assert_allclose(_target_origin(physical), physical.left_arm.shoulder * 0.5 + physical.right_arm.shoulder * 0.5)


def test_body_frame_seed_g1_mesh_ignores_seed_root_pose():
  from scripts.visualize_seed_bvh_axes import (
    _seed_g1_model_context,
    _seed_g1_mesh_snapshot,
  )

  zero_root = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, *([0.0] * len(G1_29DOF_JOINT_COLUMNS))]
  rotated_root = [
    10.0,
    -3.0,
    2.0,
    0.0,
    0.0,
    math.sqrt(0.5),
    math.sqrt(0.5),
    *([0.0] * len(G1_29DOF_JOINT_COLUMNS)),
  ]
  model, data, joint_qpos_addresses, visual_geom_ids = _seed_g1_model_context()

  snapshot_a = _seed_g1_mesh_snapshot(
    model,
    data,
    joint_qpos_addresses,
    visual_geom_ids,
    zero_root,
    use_global_root=False,
  )
  snapshot_b = _seed_g1_mesh_snapshot(
    model,
    data,
    joint_qpos_addresses,
    visual_geom_ids,
    rotated_root,
    use_global_root=False,
  )

  np.testing.assert_allclose(snapshot_a.vertices, snapshot_b.vertices)


def test_debug_frames_keep_raw_bvh_skeleton_separate_from_sew_axis_target(tmp_path):
  from scripts.visualize_seed_bvh_axes import VisualizerConfig, _build_debug_frames

  bvh_path = tmp_path / "full.bvh"
  _write_tiny_full_body_bvh(bvh_path)
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

  frames = _build_debug_frames(
    VisualizerConfig(
      bvh_path=bvh_path,
      seed_csv_path=csv_path,
      start_frame=0,
      end_frame=0,
      show_seed_g1_mesh=False,
      apply_lower_body_offsets=False,
      localize_to_body_frame=False,
      align_upper_arm_axes_to_g1=True,
    )
  )

  assert len(frames) == 1
  frame = frames[0]
  assert frame.raw_bvh is not frame.bvh
  assert not np.allclose(frame.raw_bvh.left_arm.shoulder, frame.bvh.left_arm.shoulder)


def test_story_markdown_explains_real_skeletons_and_axis_only_targets(tmp_path):
  from scripts.visualize_seed_bvh_axes import (
    VisualizerConfig,
    _build_debug_frames,
    _format_debug_markdown,
  )

  bvh_path = tmp_path / "full.bvh"
  _write_tiny_full_body_bvh(bvh_path)
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

  frame = _build_debug_frames(
    VisualizerConfig(
      bvh_path=bvh_path,
      seed_csv_path=csv_path,
      start_frame=0,
      end_frame=0,
      show_seed_g1_mesh=False,
      apply_lower_body_offsets=False,
      localize_to_body_frame=False,
    )
  )[0]

  markdown = _format_debug_markdown(frame, layout="story", step="2 BVH SEW target axes")

  assert "Read left to right" in markdown
  assert "real skeleton/keypoints" in markdown
  assert "axis-only" in markdown
  assert "not elbow/wrist positions" in markdown
  assert "Current step" in markdown
