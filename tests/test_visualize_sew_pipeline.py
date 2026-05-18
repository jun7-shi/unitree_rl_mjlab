import csv
import subprocess
import sys
from pathlib import Path

from src.motion.seed_bones import G1_29DOF_JOINT_COLUMNS
from tests.test_bvh_full_body import _write_tiny_full_body_bvh


def test_pipeline_script_help_lists_step_controls():
  result = subprocess.run(
    [
      sys.executable,
      "scripts/visualize_sew_pipeline.py",
      "--help",
    ],
    cwd=Path(__file__).resolve().parents[1],
    check=False,
    capture_output=True,
    text=True,
  )

  assert result.returncode == 0, result.stderr
  assert "full SEW-Mimic pipeline" in result.stdout
  assert "--pipeline-step" in result.stdout
  assert "--body-part" in result.stdout
  assert "--show-all-candidates" in result.stdout


def test_pipeline_steps_cover_raw_preprocess_solver_and_seed_comparison():
  from scripts.visualize_sew_pipeline import PIPELINE_STEPS

  assert len(PIPELINE_STEPS) == 24
  assert [step.number for step in PIPELINE_STEPS] == list(range(1, 25))
  titles = " ".join(step.title for step in PIPELINE_STEPS)
  assert "Raw BVH hierarchy" in titles
  assert "G1 axis proxy target synthesis" in titles
  assert "Subproblem 2 raw candidates" in titles
  assert "Final G1 output + seed comparison" in titles
  assert any(step.axis_only for step in PIPELINE_STEPS)
  assert any(step.real_skeleton for step in PIPELINE_STEPS)


def test_pipeline_trace_keeps_source_converted_target_and_seed_layers_separate(tmp_path):
  from scripts.visualize_sew_pipeline import PipelineVisualizerConfig, build_pipeline_traces

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

  traces = build_pipeline_traces(
    PipelineVisualizerConfig(
      bvh_path=bvh_path,
      seed_csv_path=csv_path,
      start_frame=0,
      end_frame=0,
      show_seed_g1_mesh=False,
      apply_lower_body_offsets=False,
      localize_to_body_frame=False,
    )
  )

  assert len(traces) == 1
  trace = traces[0]
  assert trace.source_skeleton.coordinate_system == "bvh_source"
  assert trace.converted_skeleton.coordinate_system == "mjlab"
  assert trace.debug_frame.raw_bvh is not trace.debug_frame.bvh
  assert trace.seed_proxy is trace.debug_frame.seed


def test_pipeline_trace_stages_lower_offsets_before_upper_axis_flip(tmp_path):
  from scripts.visualize_sew_pipeline import PipelineVisualizerConfig, build_pipeline_traces

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

  trace = build_pipeline_traces(
    PipelineVisualizerConfig(
      bvh_path=bvh_path,
      seed_csv_path=csv_path,
      start_frame=0,
      end_frame=0,
      show_seed_g1_mesh=False,
      apply_lower_body_offsets=False,
      localize_to_body_frame=False,
    )
  )[0]

  assert trace.lower_processed_target is not trace.processed_full_target
  assert trace.lower_processed_target.upper.left_arm.shoulder is not trace.raw_full_target.upper.left_arm.shoulder
  assert trace.lower_processed_target.upper.left_arm.shoulder.tolist() == trace.raw_full_target.upper.left_arm.shoulder.tolist()
  assert trace.processed_full_target.upper.left_arm.shoulder.tolist() != trace.raw_full_target.upper.left_arm.shoulder.tolist()


def test_pipeline_trace_keeps_seed_motion_row_without_precomputing_mesh(tmp_path):
  from scripts.visualize_sew_pipeline import PipelineVisualizerConfig, build_pipeline_traces

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

  trace = build_pipeline_traces(
    PipelineVisualizerConfig(
      bvh_path=bvh_path,
      seed_csv_path=csv_path,
      start_frame=0,
      end_frame=0,
      show_seed_g1_mesh=True,
      apply_lower_body_offsets=False,
      localize_to_body_frame=False,
    )
  )[0]

  assert trace.seed_motion_row[2] == 1.0
  assert trace.debug_frame.seed_g1_mesh is None


def test_pipeline_trace_records_solver_full_qpos_for_final_retargeted_mesh(tmp_path):
  from scripts.visualize_sew_pipeline import PipelineVisualizerConfig, build_pipeline_traces

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

  trace = build_pipeline_traces(
    PipelineVisualizerConfig(
      bvh_path=bvh_path,
      seed_csv_path=csv_path,
      start_frame=0,
      end_frame=0,
      show_seed_g1_mesh=False,
      apply_lower_body_offsets=False,
      localize_to_body_frame=False,
    )
  )[0]

  assert trace.solver_full_qpos.ndim == 1
  assert trace.solver_full_qpos.shape[0] > len(G1_29DOF_JOINT_COLUMNS)
  assert trace.solver_full_qpos[3] == 1.0


def test_pipeline_step_markdown_explains_current_layer_and_proxy_status():
  from scripts.visualize_sew_pipeline import PIPELINE_STEPS, format_pipeline_step_markdown

  upper_flip = PIPELINE_STEPS[10]
  markdown = format_pipeline_step_markdown(
    upper_flip,
    frame_index=270,
    csv_frame=270,
    body_part="left arm",
    diagnostic_lines=["candidate 0: pitch 12.0 roll -5.0"],
  )

  assert "**Step 11/24:** G1 axis proxy target synthesis" in markdown
  assert "left arm" in markdown
  assert "axis-only" in markdown
  assert "not physical elbow/wrist positions" in markdown
  assert "candidate 0" in markdown
