from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from scripts.visualize_seed_bvh_axes import (
  ArmColumnFrame,
  DebugFrame,
  MeshSnapshot,
  VisualizerConfig,
  _axis_angle_degrees,
  _axis_segments,
  _body_segments,
  _build_debug_frames,
  _keypoint_cloud,
  _load_seed_motion_rows,
  _seed_g1_model_context,
  _seed_g1_mesh_snapshot,
  _seed_g1_physical_target,
  _seed_g1_shoulder_center,
  _set_qpos_from_seed_motion_row,
)
from src.motion.bvh_full_body import load_soma_bvh_full_body_targets
from src.motion.bvh_sew import BVHMotion, BVHFramePose, load_bvh, soma_mujoco_to_mjlab_rotation
from src.motion.sew_full_body import FullBodyTarget, G1FullBodySEWRetargeter
from src.motion.sew_mimic import PAPER_V1_ALGORITHM, normalize
from src.motion.sew_upper_body import UpperBodyTarget


@dataclass(frozen=True)
class PipelineStep:
  number: int
  title: str
  description: str
  code_path: str
  layer: str
  real_skeleton: bool = False
  axis_only: bool = False


PIPELINE_STEPS = (
  PipelineStep(
    1,
    "Raw BVH hierarchy",
    "BVH joints, parents, offsets, and local axes before motion channels are evaluated.",
    "src/motion/bvh_sew.py:load_bvh",
    "input",
    real_skeleton=True,
  ),
  PipelineStep(
    2,
    "BVH frame pose",
    "One motion row expanded through the BVH hierarchy into world-space source-coordinate keypoints.",
    "src/motion/bvh_sew.py:BVHMotion.frame_pose",
    "input",
    real_skeleton=True,
  ),
  PipelineStep(
    3,
    "Unit scale + SOMA/MuJoCo rotation",
    "Centimeter BVH positions are scaled to meters and rotated with the +90 degree X mapping used by soma-retargeter.",
    "src/motion/bvh_sew.py:soma_mujoco_to_mjlab_rotation",
    "preprocess",
    real_skeleton=True,
  ),
  PipelineStep(
    4,
    "Converted real skeleton",
    "The converted mjlab-coordinate keypoints before SEW target conventions are applied.",
    "src/motion/bvh_full_body.py:load_soma_bvh_full_body_targets",
    "preprocess",
    real_skeleton=True,
  ),
  PipelineStep(
    5,
    "Body frame construction",
    "The hip/chest body frame used to make targets comparable frame-to-frame.",
    "src/motion/bvh_full_body.py:_body_frame_from_pose",
    "preprocess",
    real_skeleton=True,
  ),
  PipelineStep(
    6,
    "Heading removal / body localization",
    "The full-body target after optional heading removal and body-frame localization.",
    "src/motion/bvh_full_body.py:_localize_targets_to_body_frame",
    "preprocess",
    real_skeleton=True,
  ),
  PipelineStep(
    7,
    "soma_retargeter orientation offsets",
    "Chest, hand, and foot orientation offsets copied from soma_retargeter are applied to orientation targets.",
    "src/motion/bvh_upper_body.py:soma_to_g1_upper_body_orientation_offsets",
    "preprocess",
  ),
  PipelineStep(
    8,
    "soma_retargeter lower-body offsets",
    "Lower-body effectors can use soma_retargeter scale and position/orientation offsets.",
    "src/motion/bvh_full_body.py:soma_to_g1_lower_body_effector_config",
    "preprocess",
    real_skeleton=True,
  ),
  PipelineStep(
    9,
    "Extract FullBodyTarget",
    "Final full-body target object split into upper-body and lower-body targets.",
    "src/motion/sew_full_body.py:FullBodyTarget",
    "target",
    real_skeleton=True,
  ),
  PipelineStep(
    10,
    "Real segment vectors",
    "Normalized arm and leg segment vectors computed from target keypoints.",
    "src/motion/sew_upper_body.py:_solve_arm",
    "target",
    real_skeleton=True,
  ),
  PipelineStep(
    11,
    "G1 axis proxy target synthesis",
    "The human upper-arm target is converted into the G1 shoulder-axis proxy convention. After this step the arm drawing is axis-only and not physical elbow/wrist positions.",
    "src/motion/bvh_upper_body.py:synthesize_g1_axis_proxy_arm_target",
    "target",
    axis_only=True,
  ),
  PipelineStep(
    12,
    "Final SEW target axes",
    "The exact upper/lower arm axes sent into the closed-form solver.",
    "src/motion/sew_upper_body.py:G1UpperBodySEWRetargeter._solve_arm",
    "target",
    axis_only=True,
  ),
  PipelineStep(
    13,
    "Seed G1 reference physical anchors",
    "The bones-seed CSV pose rendered on the real G1 model; this is the reference output, not the solver result.",
    "src/motion/sew_upper_body.py:G1UpperBodySEWRetargeter._keypoint_joint_ids",
    "robot",
    real_skeleton=True,
  ),
  PipelineStep(
    14,
    "G1 SEW proxy axes",
    "The G1 axis proxy built from joint axes, separate from physical anchors.",
    "src/motion/sew_upper_body.py:_target_arm_from_current_configuration",
    "robot",
    axis_only=True,
  ),
  PipelineStep(
    15,
    "Previous-frame q / warm start",
    "The solver starts from the previous frame's selected joint branch.",
    "src/motion/sew_full_body.py:retarget_full_body_targets",
    "solver",
  ),
  PipelineStep(
    16,
    "Waist/chest solve",
    "Waist yaw/roll/pitch are solved to match the target chest orientation.",
    "src/motion/sew_upper_body.py:_solve_orientation_group",
    "solver",
  ),
  PipelineStep(
    17,
    "Selected limb 2-axis setup",
    "For one 2-DOF group, collect initial axis, first joint axis, second joint axis, and target axis.",
    "src/motion/sew_upper_body.py:_solve_axis_group",
    "solver",
    axis_only=True,
  ),
  PipelineStep(
    18,
    "Subproblem 4 intermediate",
    "Subproblem 4 builds one or two angle roots from the plane/circle relation used by Subproblem 2.",
    "src/motion/sew_mimic.py:subproblem4",
    "solver",
    axis_only=True,
  ),
  PipelineStep(
    19,
    "Subproblem 2 raw candidates",
    "Raw two-axis pitch/roll candidates before equivalent-angle expansion or branch selection.",
    "src/motion/sew_mimic.py:solve_two_axis_rotation",
    "solver",
    axis_only=True,
  ),
  PipelineStep(
    20,
    "Equivalent angle expansion",
    "Each raw angle can represent multiple equivalent branches separated by 360 degrees.",
    "src/motion/sew_mimic.py:bounded_equivalent_angle_candidates",
    "solver",
    axis_only=True,
  ),
  PipelineStep(
    21,
    "Candidate scoring / selection",
    "Candidates are scored by axis error, joint distance, branch distance, and clipping state.",
    "src/motion/sew_mimic.py:select_limit_projected_sew_candidate",
    "solver",
    axis_only=True,
  ),
  PipelineStep(
    22,
    "Apply selected 2-axis group",
    "The selected 2-axis candidate is applied and compared directly with the target axis.",
    "src/motion/sew_upper_body.py:_select_axis_group_candidate",
    "solver",
    axis_only=True,
  ),
  PipelineStep(
    23,
    "Next segment + wrist/foot orientation",
    "The next segment group and end-effector orientation solve complete each limb.",
    "src/motion/sew_upper_body.py:_solve_wrist_group",
    "solver",
    axis_only=True,
  ),
  PipelineStep(
    24,
    "Final G1 output + seed comparison",
    "Final solver-retargeted G1 mesh is shown next to the bones-seed G1 reference, seed proxy axes, and BVH target axes.",
    "scripts/visualize_sew_pipeline.py:render_final_comparison",
    "comparison",
    real_skeleton=True,
    axis_only=True,
  ),
)


@dataclass(frozen=True)
class PipelineVisualizerConfig(VisualizerConfig):
  initial_pipeline_step: str = "01 Raw BVH hierarchy"
  body_part: str = "left arm"
  show_all_candidates: bool = False


@dataclass(frozen=True)
class SkeletonSnapshot:
  positions: dict[str, np.ndarray]
  parents: dict[str, str | None]
  rotations: dict[str, np.ndarray]
  coordinate_system: str


@dataclass(frozen=True)
class PipelineTrace:
  frame_index: int
  csv_frame: int
  source_skeleton: SkeletonSnapshot
  converted_skeleton: SkeletonSnapshot
  raw_full_target: FullBodyTarget
  lower_processed_target: FullBodyTarget
  processed_full_target: FullBodyTarget
  debug_frame: DebugFrame
  seed_proxy: UpperBodyTarget
  seed_motion_row: list[float]
  solver_full_qpos: np.ndarray


def _target_kwargs(config: PipelineVisualizerConfig) -> dict[str, bool]:
  return {
    "apply_orientation_offsets": config.apply_orientation_offsets,
    "apply_lower_body_offsets": config.apply_lower_body_offsets,
    "remove_initial_heading": config.remove_initial_heading,
    "localize_to_body_frame": config.localize_to_body_frame and config.apply_lower_body_offsets,
  }


def _skeleton_snapshot(
  motion: BVHMotion,
  pose: BVHFramePose,
  *,
  coordinate_system: str,
) -> SkeletonSnapshot:
  return SkeletonSnapshot(
    positions={name: position.copy() for name, position in pose.positions.items()},
    parents={name: joint.parent for name, joint in motion.joints.items()},
    rotations={name: rotation.copy() for name, rotation in pose.rotations.items()},
    coordinate_system=coordinate_system,
  )


def build_pipeline_traces(config: PipelineVisualizerConfig) -> list[PipelineTrace]:
  motion = load_bvh(config.bvh_path, scale=0.01)
  _csv_frames, seed_rows = _load_seed_motion_rows(config.seed_csv_path)
  debug_frames = _build_debug_frames(replace(config, show_seed_g1_mesh=False))
  common_target_kwargs = {
    "apply_orientation_offsets": config.apply_orientation_offsets,
    "remove_initial_heading": config.remove_initial_heading,
    "localize_to_body_frame": config.localize_to_body_frame,
  }
  raw_full_targets = load_soma_bvh_full_body_targets(
    config.bvh_path,
    apply_lower_body_offsets=False,
    align_upper_arm_axes_to_g1=False,
    **common_target_kwargs,
  )
  lower_processed_targets = load_soma_bvh_full_body_targets(
    config.bvh_path,
    apply_lower_body_offsets=config.apply_lower_body_offsets,
    align_upper_arm_axes_to_g1=False,
    **common_target_kwargs,
  )
  processed_full_targets = load_soma_bvh_full_body_targets(
    config.bvh_path,
    apply_lower_body_offsets=config.apply_lower_body_offsets,
    align_upper_arm_axes_to_g1=config.align_upper_arm_axes_to_g1,
    **common_target_kwargs,
  )
  retargeter = G1FullBodySEWRetargeter(algorithm_version=PAPER_V1_ALGORITHM)
  q_previous = np.zeros(29, dtype=float)
  solver_full_qpos_by_frame: dict[int, np.ndarray] = {}
  for frame_index, target in enumerate(processed_full_targets[: config.end_frame + 1]):
    result = retargeter.retarget(q_previous, target)
    solver_full_qpos_by_frame[frame_index] = result.full_qpos.copy()
    q_previous = (
      result.solver_joint_angles
      if result.solver_joint_angles is not None
      else result.joint_angles
    )
  traces: list[PipelineTrace] = []
  for debug_frame in debug_frames:
    frame_index = debug_frame.frame_index
    source_pose = motion.frame_pose(frame_index)
    converted_pose = motion.frame_pose(
      frame_index,
      coordinate_rotation=soma_mujoco_to_mjlab_rotation(),
    )
    traces.append(
      PipelineTrace(
        frame_index=frame_index,
        csv_frame=debug_frame.csv_frame,
        source_skeleton=_skeleton_snapshot(
          motion,
          source_pose,
          coordinate_system="bvh_source",
        ),
        converted_skeleton=_skeleton_snapshot(
          motion,
          converted_pose,
          coordinate_system="mjlab",
        ),
        raw_full_target=raw_full_targets[frame_index],
        lower_processed_target=lower_processed_targets[frame_index],
        processed_full_target=processed_full_targets[frame_index],
        debug_frame=debug_frame,
        seed_proxy=debug_frame.seed,
        seed_motion_row=seed_rows[frame_index],
        solver_full_qpos=solver_full_qpos_by_frame[frame_index],
      )
    )
  return traces


def _step_label(step: PipelineStep) -> str:
  return f"{step.number:02d} {step.title}"


def _step_by_label(label: str) -> PipelineStep:
  for step in PIPELINE_STEPS:
    if label == _step_label(step):
      return step
  return PIPELINE_STEPS[0]


def _candidate_lines(frame: DebugFrame) -> list[str]:
  debug = frame.left_candidate_debug
  lines = [
    f"before left pitch/roll solve: {debug.q_before_pitch_roll_deg[0]:.2f}, {debug.q_before_pitch_roll_deg[1]:.2f} deg",
    f"target dot pitch axis: {debug.target_dot_pitch_axis:.4f}",
    f"target dot initial upper axis: {debug.target_dot_initial_axis:.4f}",
  ]
  for index, (pitch, roll) in enumerate(debug.candidates_pitch_roll_deg):
    lines.append(f"candidate {index}: pitch {pitch:.2f} deg, roll {roll:.2f} deg")
  return lines


def _diagnostic_lines(trace: PipelineTrace, step: PipelineStep) -> list[str]:
  frame = trace.debug_frame
  if step.number in {17, 18, 19, 20, 21}:
    return _candidate_lines(frame)
  if step.number in {11, 12, 22, 24}:
    return [
      f"seed vs BVH left upper axis: {frame.seed_bvh_left_upper_axis_error_deg:.2f} deg",
      f"seed vs BVH left lower axis: {frame.seed_bvh_left_lower_axis_error_deg:.2f} deg",
      f"BVH elbow angle - seed elbow angle: {frame.seed_bvh_left_elbow_angle_delta_deg:.2f} deg",
    ]
  if step.number == 16:
    return [f"solver diagnostics compare target chest orientation against G1 torso orientation"]
  return []


def format_pipeline_step_markdown(
  step: PipelineStep,
  *,
  frame_index: int,
  csv_frame: int,
  body_part: str,
  diagnostic_lines: Sequence[str] = (),
) -> str:
  flags: list[str] = []
  if step.real_skeleton:
    flags.append("real skeleton/keypoints")
  if step.axis_only:
    flags.append("axis-only")
  flag_text = ", ".join(flags) if flags else "orientation / solver state"
  proxy_note = (
    "<br>**Proxy warning:** axis-only proxy endpoints are not physical elbow/wrist positions."
    if step.axis_only
    else ""
  )
  diagnostics = ""
  if diagnostic_lines:
    diagnostics = "<br><br>**Diagnostics**<br>" + "<br>".join(diagnostic_lines)
  return (
    f"**Step {step.number}/24:** {step.title}<br>"
    f"**Frame:** {frame_index}  **CSV Frame:** {csv_frame}<br>"
    f"**Body part:** {body_part}<br>"
    f"**Layer:** {step.layer} ({flag_text})<br>"
    f"**Code:** `{step.code_path}`<br><br>"
    f"{step.description}"
    f"{proxy_note}"
    f"{diagnostics}"
  )


def _snapshot_origin(snapshot: SkeletonSnapshot) -> np.ndarray:
  for pair in (("LeftArm", "RightArm"), ("LeftLeg", "RightLeg"), ("Hips", "Chest")):
    if pair[0] in snapshot.positions and pair[1] in snapshot.positions:
      return 0.5 * (snapshot.positions[pair[0]] + snapshot.positions[pair[1]])
  return np.mean(np.asarray(list(snapshot.positions.values()), dtype=float), axis=0)


def _snapshot_segments(snapshot: SkeletonSnapshot, offset: np.ndarray) -> np.ndarray:
  origin = _snapshot_origin(snapshot)
  segments = []
  for name, parent in snapshot.parents.items():
    if parent is None or name not in snapshot.positions or parent not in snapshot.positions:
      continue
    start = snapshot.positions[parent] - origin + offset
    end = snapshot.positions[name] - origin + offset
    segments.append([start, end])
  return np.asarray(segments, dtype=float)


def _snapshot_points(snapshot: SkeletonSnapshot, offset: np.ndarray) -> np.ndarray:
  origin = _snapshot_origin(snapshot)
  return np.asarray(
    [position - origin + offset for position in snapshot.positions.values()],
    dtype=float,
  )


def _full_target_origin(target: FullBodyTarget) -> np.ndarray:
  left = target.lower.left_leg.hip
  right = target.lower.right_leg.hip
  return 0.5 * (left + right)


def _full_body_segments(target: FullBodyTarget, offset: np.ndarray) -> np.ndarray:
  origin = _full_target_origin(target)

  def p(value: np.ndarray) -> np.ndarray:
    return np.asarray(value, dtype=float) - origin + offset

  upper = target.upper
  lower = target.lower
  segments = [
    (upper.left_arm.shoulder, upper.right_arm.shoulder),
    (upper.chest_position, upper.left_arm.shoulder),
    (upper.chest_position, upper.right_arm.shoulder),
    (upper.left_arm.shoulder, upper.left_arm.elbow),
    (upper.left_arm.elbow, upper.left_arm.wrist),
    (upper.right_arm.shoulder, upper.right_arm.elbow),
    (upper.right_arm.elbow, upper.right_arm.wrist),
    (lower.left_leg.hip, lower.right_leg.hip),
    (lower.left_leg.hip, lower.left_leg.knee),
    (lower.left_leg.knee, lower.left_leg.ankle),
    (lower.right_leg.hip, lower.right_leg.knee),
    (lower.right_leg.knee, lower.right_leg.ankle),
    (upper.chest_position, 0.5 * (lower.left_leg.hip + lower.right_leg.hip)),
  ]
  return np.asarray([[p(start), p(end)] for start, end in segments], dtype=float)


def _full_body_points(target: FullBodyTarget, offset: np.ndarray) -> np.ndarray:
  origin = _full_target_origin(target)
  upper = target.upper
  lower = target.lower
  points = [
    upper.chest_position,
    upper.left_arm.shoulder,
    upper.left_arm.elbow,
    upper.left_arm.wrist,
    upper.right_arm.shoulder,
    upper.right_arm.elbow,
    upper.right_arm.wrist,
    lower.left_leg.hip,
    lower.left_leg.knee,
    lower.left_leg.ankle,
    lower.right_leg.hip,
    lower.right_leg.knee,
    lower.right_leg.ankle,
  ]
  return np.asarray([point - origin + offset for point in points], dtype=float)


def _body_frame_axes(snapshot: SkeletonSnapshot, offset: np.ndarray, length: float) -> np.ndarray:
  positions = snapshot.positions
  left_hip = positions.get("LeftLeg")
  right_hip = positions.get("RightLeg")
  chest = positions.get("Chest")
  if left_hip is None or right_hip is None or chest is None:
    return np.empty((0, 2, 3), dtype=float)
  origin_world = 0.5 * (left_hip + right_hip)
  left_axis = normalize(left_hip - right_hip)
  up_candidate = chest - origin_world
  up_axis = normalize(up_candidate - left_axis * float(np.dot(up_candidate, left_axis)))
  forward_axis = normalize(np.cross(left_axis, up_axis))
  axes = (forward_axis, left_axis, up_axis)
  origin = _snapshot_origin(snapshot)
  start = origin_world - origin + offset
  return np.asarray([[start, start + axis * length] for axis in axes], dtype=float)


def _arm_axis_pair(target: UpperBodyTarget, offset: np.ndarray, axis_length: float) -> np.ndarray:
  return _axis_segments(target, offset, axis_length)


def _add_label(server, handles: list[object], name: str, text: str, offset: np.ndarray) -> None:
  handles.append(
    server.scene.add_label(
      f"/{name}/label",
      text=text,
      position=offset + np.array([0.0, 0.0, 0.75]),
      font_size_mode="scene",
      font_scene_height=0.055,
      anchor="center-center",
    )
  )


def _mesh_snapshot_from_qpos(
  model: mujoco.MjModel,
  data: mujoco.MjData,
  visual_geom_ids: Sequence[int],
  qpos: np.ndarray,
) -> MeshSnapshot:
  from mjlab.viewer.viser.conversions import merge_geoms_global

  data.qpos[:] = np.asarray(qpos, dtype=float)
  mujoco.mj_forward(model, data)
  mesh = merge_geoms_global(model, data, list(visual_geom_ids))
  return MeshSnapshot(
    vertices=np.asarray(mesh.vertices, dtype=float) - _seed_g1_shoulder_center(model, data),
    faces=np.asarray(mesh.faces, dtype=np.int32),
  )


def _run_viser(traces: Sequence[PipelineTrace], config: PipelineVisualizerConfig) -> None:
  import viser

  server = viser.ViserServer(port=config.port)
  server.scene.add_grid("/grid", width=4.0, height=2.5, cell_size=0.1)
  trace_by_frame = {trace.frame_index: trace for trace in traces}
  frame_indices = list(trace_by_frame)
  handles: list[object] = []
  step_options = tuple(_step_label(step) for step in PIPELINE_STEPS)
  seed_mesh_context = _seed_g1_model_context() if config.show_seed_g1_mesh else None
  seed_mesh_cache: dict[int, tuple[MeshSnapshot, UpperBodyTarget]] = {}
  solver_mesh_context = _seed_g1_model_context()
  solver_mesh_cache: dict[int, MeshSnapshot] = {}

  with server.gui.add_folder("Pipeline"):
    frame_slider = server.gui.add_slider(
      "Frame",
      min=min(frame_indices),
      max=max(frame_indices),
      step=1,
      initial_value=min(frame_indices),
    )
    step_dropdown = server.gui.add_dropdown(
      "Pipeline step",
      options=step_options,
      initial_value=config.initial_pipeline_step if config.initial_pipeline_step in step_options else step_options[0],
    )
    body_part_dropdown = server.gui.add_dropdown(
      "Body part",
      options=("left arm", "right arm", "left leg", "right leg", "full body"),
      initial_value=config.body_part,
    )
    show_all_candidates = server.gui.add_checkbox(
      "Show all candidates",
      initial_value=config.show_all_candidates,
    )
    play_checkbox = server.gui.add_checkbox("Play", initial_value=False)
    fps_slider = server.gui.add_slider("FPS", min=1.0, max=60.0, step=1.0, initial_value=config.fps)
    info_markdown = server.gui.add_markdown("")

  def clear_scene() -> None:
    while handles:
      handles.pop().remove()

  def add_snapshot(name: str, snapshot: SkeletonSnapshot, offset: np.ndarray, color: tuple[int, int, int]) -> None:
    segments = _snapshot_segments(snapshot, offset)
    if len(segments):
      handles.append(
        server.scene.add_line_segments(
          f"/{name}/segments",
          points=segments,
          colors=np.asarray(color, dtype=np.uint8),
          line_width=3.0,
        )
      )
    points = _snapshot_points(snapshot, offset)
    handles.append(
      server.scene.add_point_cloud(
        f"/{name}/points",
        points=points,
        colors=np.asarray(color, dtype=np.uint8),
        point_size=0.025,
        point_shape="circle",
      )
    )
    _add_label(server, handles, name, name, offset)

  def add_full_target(name: str, target: FullBodyTarget, offset: np.ndarray, color: tuple[int, int, int]) -> None:
    handles.append(
      server.scene.add_line_segments(
        f"/{name}/segments",
        points=_full_body_segments(target, offset),
        colors=np.asarray(color, dtype=np.uint8),
        line_width=4.0,
      )
    )
    handles.append(
      server.scene.add_point_cloud(
        f"/{name}/points",
        points=_full_body_points(target, offset),
        colors=np.asarray(color, dtype=np.uint8),
        point_size=0.032,
        point_shape="circle",
      )
    )
    _add_label(server, handles, name, name, offset)

  def add_arm_column(column: ArmColumnFrame) -> None:
    if column.show_body:
      handles.append(
        server.scene.add_line_segments(
          f"/{column.name}/body",
          points=_body_segments(column.target, column.offset),
          colors=np.asarray(column.color, dtype=np.uint8),
          line_width=4.0,
        )
      )
    if column.show_axes:
      handles.append(
        server.scene.add_line_segments(
          f"/{column.name}/axes",
          points=_arm_axis_pair(column.target, column.offset, config.axis_length),
          colors=np.asarray(
            [
              [[240, 70, 70], [240, 70, 70]],
              [[120, 90, 255], [120, 90, 255]],
              [[240, 70, 70], [240, 70, 70]],
              [[120, 90, 255], [120, 90, 255]],
            ],
            dtype=np.uint8,
          ),
          line_width=7.0,
        )
      )
    if column.show_keypoints:
      handles.append(
        server.scene.add_point_cloud(
          f"/{column.name}/points",
          points=_keypoint_cloud(column.target, column.offset),
          colors=np.asarray(column.color, dtype=np.uint8),
          point_size=0.032,
          point_shape="circle",
        )
      )
    _add_label(server, handles, column.name, column.name, column.offset)

  def seed_mesh_for_trace(trace: PipelineTrace) -> tuple[MeshSnapshot, UpperBodyTarget] | None:
    if seed_mesh_context is None:
      return None
    if trace.frame_index in seed_mesh_cache:
      return seed_mesh_cache[trace.frame_index]
    model, data, joint_qpos_addresses, visual_geom_ids = seed_mesh_context
    mesh = _seed_g1_mesh_snapshot(
      model,
      data,
      joint_qpos_addresses,
      visual_geom_ids,
      trace.seed_motion_row,
      use_global_root=config.seed_g1_mesh_use_global_root,
    )
    seed_row = list(trace.seed_motion_row)
    if not config.seed_g1_mesh_use_global_root:
      seed_row = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, *seed_row[7:]]
    _set_qpos_from_seed_motion_row(data, joint_qpos_addresses, seed_row)
    mujoco.mj_forward(model, data)
    physical = _seed_g1_physical_target(model, data)
    seed_mesh_cache[trace.frame_index] = (mesh, physical)
    return mesh, physical

  def add_seed_mesh(trace: PipelineTrace, offset: np.ndarray) -> None:
    seed_mesh = seed_mesh_for_trace(trace)
    if seed_mesh is None:
      _add_label(server, handles, "seed_mesh_hidden", "seed G1 mesh hidden", offset)
      return
    mesh, physical = seed_mesh
    handles.append(
      server.scene.add_mesh_simple(
        "/seed_g1_model/mesh",
        vertices=mesh.vertices,
        faces=mesh.faces,
        color=(80, 190, 115),
        opacity=0.55,
        flat_shading=False,
        side="double",
        position=offset,
      )
    )
    add_arm_column(
      ArmColumnFrame(
        "seed physical anchors",
        physical,
        (245, 245, 245),
        offset,
        show_axes=False,
      )
    )
    _add_label(server, handles, "seed_g1_model", "seed G1 model", offset)

  def solver_mesh_for_trace(trace: PipelineTrace) -> MeshSnapshot:
    if trace.frame_index not in solver_mesh_cache:
      model, data, _joint_qpos_addresses, visual_geom_ids = solver_mesh_context
      solver_mesh_cache[trace.frame_index] = _mesh_snapshot_from_qpos(
        model,
        data,
        visual_geom_ids,
        trace.solver_full_qpos,
      )
    return solver_mesh_cache[trace.frame_index]

  def add_solver_mesh(trace: PipelineTrace, offset: np.ndarray) -> None:
    mesh = solver_mesh_for_trace(trace)
    handles.append(
      server.scene.add_mesh_simple(
        "/solver_g1_model/mesh",
        vertices=mesh.vertices,
        faces=mesh.faces,
        color=(80, 145, 245),
        opacity=0.62,
        flat_shading=False,
        side="double",
        position=offset,
      )
    )
    _add_label(server, handles, "solver_g1_model", "solver retargeted G1", offset)

  def add_body_frame(name: str, snapshot: SkeletonSnapshot, offset: np.ndarray) -> None:
    axes = _body_frame_axes(snapshot, offset, config.axis_length * 1.6)
    if not len(axes):
      return
    handles.append(
      server.scene.add_line_segments(
        f"/{name}/body_frame",
        points=axes,
        colors=np.asarray(
          [
            [[240, 70, 70], [240, 70, 70]],
            [[70, 200, 110], [70, 200, 110]],
            [[70, 140, 255], [70, 140, 255]],
          ],
          dtype=np.uint8,
        ),
        line_width=8.0,
      )
    )

  def add_candidate_labels(trace: PipelineTrace, offset: np.ndarray) -> None:
    if not show_all_candidates.value:
      return
    for index, (pitch, roll) in enumerate(trace.debug_frame.left_candidate_debug.candidates_pitch_roll_deg):
      handles.append(
        server.scene.add_label(
          f"/candidate_{index}/label",
          text=f"cand {index}: pitch {pitch:.1f}, roll {roll:.1f}",
          position=offset + np.array([0.0, 0.12 * index, 0.45]),
          font_size_mode="scene",
          font_scene_height=0.04,
          anchor="center-center",
        )
      )

  def render(trace: PipelineTrace, step: PipelineStep) -> None:
    spacing = config.column_spacing
    frame = trace.debug_frame
    if step.number == 1:
      add_snapshot("1 raw BVH hierarchy", trace.source_skeleton, np.zeros(3), (242, 143, 52))
    elif step.number == 2:
      add_snapshot("2 BVH frame pose", trace.source_skeleton, np.zeros(3), (242, 143, 52))
    elif step.number == 3:
      add_snapshot("source BVH", trace.source_skeleton, np.array([-0.65 * spacing, 0.0, 0.0]), (242, 143, 52))
      add_snapshot("mjlab converted", trace.converted_skeleton, np.array([0.65 * spacing, 0.0, 0.0]), (95, 170, 245))
    elif step.number == 4:
      add_snapshot("4 converted real skeleton", trace.converted_skeleton, np.zeros(3), (95, 170, 245))
    elif step.number == 5:
      add_snapshot("5 body-frame source", trace.converted_skeleton, np.zeros(3), (95, 170, 245))
      add_body_frame("5 body-frame source", trace.converted_skeleton, np.zeros(3))
    elif step.number == 6:
      add_full_target("6 localized full-body target", trace.raw_full_target, np.zeros(3), (245, 185, 80))
    elif step.number == 7:
      add_arm_column(
        ArmColumnFrame(
          "7 orientation-offset target",
          frame.raw_bvh,
          (245, 185, 80),
          np.zeros(3),
          show_axes=True,
        )
      )
    elif step.number == 8:
      add_full_target("lower raw/keypoint target", trace.raw_full_target, np.array([-0.55 * spacing, 0.0, 0.0]), (245, 185, 80))
      add_full_target("lower processed target (upper not flipped)", trace.lower_processed_target, np.array([0.55 * spacing, 0.0, 0.0]), (80, 200, 120))
    elif step.number == 9:
      add_full_target("9 FullBodyTarget before upper flip", trace.lower_processed_target, np.zeros(3), (80, 200, 120))
    elif step.number == 10:
      add_full_target("10 real segment vectors", trace.lower_processed_target, np.zeros(3), (80, 200, 120))
      add_arm_column(
        ArmColumnFrame(
          "upper-body segment axes",
          frame.raw_bvh,
          (245, 185, 80),
          np.zeros(3),
          show_body=False,
          show_keypoints=False,
        )
      )
    elif step.number == 11:
      add_arm_column(
        ArmColumnFrame("before upper-arm flip", frame.raw_bvh, (245, 185, 80), np.array([-0.55 * spacing, 0.0, 0.0]), show_axes=True)
      )
      add_arm_column(
        ArmColumnFrame("after G1 convention flip", frame.bvh, (80, 200, 120), np.array([0.55 * spacing, 0.0, 0.0]), show_body=False, show_keypoints=False)
      )
    elif step.number == 12:
      add_arm_column(
        ArmColumnFrame("12 final BVH SEW target axes", frame.bvh, (80, 200, 120), np.zeros(3), show_body=False, show_keypoints=False)
      )
    elif step.number == 13:
      add_seed_mesh(trace, np.zeros(3))
    elif step.number == 14:
      add_arm_column(
        ArmColumnFrame("14 seed G1 SEW proxy axes", frame.seed, (58, 166, 85), np.zeros(3), show_body=False, show_keypoints=False)
      )
    elif step.number == 15:
      add_arm_column(
        ArmColumnFrame("15 previous-branch context", frame.solver, (80, 145, 245), np.zeros(3), show_body=False, show_keypoints=False)
      )
      add_candidate_labels(trace, np.zeros(3))
    elif step.number == 16:
      add_arm_column(
        ArmColumnFrame("target chest + arms", frame.bvh, (80, 200, 120), np.array([-0.55 * spacing, 0.0, 0.0]), show_axes=True)
      )
      add_arm_column(
        ArmColumnFrame("solver torso/arms", frame.solver, (80, 145, 245), np.array([0.55 * spacing, 0.0, 0.0]), show_axes=True)
      )
    elif step.number in {17, 18, 19, 20, 21}:
      add_arm_column(
        ArmColumnFrame(f"{step.number} BVH target axis", frame.bvh, (80, 200, 120), np.array([-0.4 * spacing, 0.0, 0.0]), show_body=False, show_keypoints=False)
      )
      add_arm_column(
        ArmColumnFrame(f"{step.number} solver candidate context", frame.solver, (80, 145, 245), np.array([0.4 * spacing, 0.0, 0.0]), show_body=False, show_keypoints=False)
      )
      add_candidate_labels(trace, np.array([0.0, 0.0, 0.0]))
    elif step.number == 22:
      add_arm_column(
        ArmColumnFrame("22 selected solver axes", frame.solver, (80, 145, 245), np.array([-0.45 * spacing, 0.0, 0.0]), show_body=False, show_keypoints=False)
      )
      add_arm_column(
        ArmColumnFrame("22 target axes", frame.bvh, (80, 200, 120), np.array([0.45 * spacing, 0.0, 0.0]), show_body=False, show_keypoints=False)
      )
    elif step.number == 23:
      add_arm_column(
        ArmColumnFrame("23 completed limb solve", frame.solver, (80, 145, 245), np.zeros(3), show_axes=True)
      )
    elif step.number == 24:
      add_arm_column(
        ArmColumnFrame("BVH target axes", frame.bvh, (245, 185, 80), np.array([-1.8 * spacing, 0.0, 0.0]), show_body=False, show_keypoints=False)
      )
      add_seed_mesh(trace, np.array([-0.55 * spacing, 0.0, 0.0]))
      add_solver_mesh(trace, np.array([0.55 * spacing, 0.0, 0.0]))
      add_arm_column(
        ArmColumnFrame("seed proxy axes", frame.seed, (58, 166, 85), np.array([1.55 * spacing, 0.0, 0.0]), show_body=False, show_keypoints=False)
      )
      add_arm_column(
        ArmColumnFrame("solver proxy axes", frame.solver, (80, 145, 245), np.array([2.25 * spacing, 0.0, 0.0]), show_body=False, show_keypoints=False)
      )

  def update_frame(frame_index: int) -> None:
    trace = trace_by_frame[int(frame_index)]
    step = _step_by_label(str(step_dropdown.value))
    clear_scene()
    render(trace, step)
    info_markdown.content = format_pipeline_step_markdown(
      step,
      frame_index=trace.frame_index,
      csv_frame=trace.csv_frame,
      body_part=str(body_part_dropdown.value),
      diagnostic_lines=_diagnostic_lines(trace, step),
    )

  @frame_slider.on_update
  def _(event) -> None:
    update_frame(int(event.target.value))

  @step_dropdown.on_update
  def _(_) -> None:
    update_frame(int(frame_slider.value))

  @body_part_dropdown.on_update
  def _(_) -> None:
    update_frame(int(frame_slider.value))

  @show_all_candidates.on_update
  def _(_) -> None:
    update_frame(int(frame_slider.value))

  update_frame(min(frame_indices))
  print(f"Viser full SEW-Mimic pipeline viewer running on http://localhost:{config.port}")
  while True:
    if play_checkbox.value:
      next_frame = int(frame_slider.value) + 1
      if next_frame > max(frame_indices):
        next_frame = min(frame_indices)
      frame_slider.value = next_frame
      update_frame(next_frame)
      time.sleep(1.0 / max(float(fps_slider.value), 1e-6))
    else:
      time.sleep(0.1)


def build_arg_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
    description="Visualize the full SEW-Mimic pipeline from raw BVH skeleton to G1 seed comparison."
  )
  parser.add_argument("--bvh", required=True, type=Path, help="Input SOMA/bones-seed BVH file.")
  parser.add_argument("--seed-csv", required=True, type=Path, help="Input bones-seed G1 CSV file.")
  parser.add_argument("--start-frame", type=int, default=260, help="First frame to show.")
  parser.add_argument("--end-frame", type=int, default=285, help="Last frame to show, inclusive.")
  parser.add_argument("--fps", type=float, default=10.0, help="Autoplay FPS for the browser viewer.")
  parser.add_argument("--port", type=int, default=8089, help="Viser web server port.")
  parser.add_argument("--axis-length", type=float, default=0.18, help="Displayed axis length in meters.")
  parser.add_argument("--column-spacing", type=float, default=0.9, help="Spacing between compared drawings.")
  parser.add_argument(
    "--pipeline-step",
    choices=tuple(_step_label(step) for step in PIPELINE_STEPS),
    default=_step_label(PIPELINE_STEPS[0]),
    help="Initial pipeline step to display.",
  )
  parser.add_argument(
    "--body-part",
    choices=("left arm", "right arm", "left leg", "right leg", "full body"),
    default="left arm",
    help="Body part label for the explanation panel.",
  )
  parser.add_argument(
    "--show-all-candidates",
    action="store_true",
    help="Show candidate labels for the two-axis solve steps.",
  )
  parser.add_argument("--hide-seed-g1-mesh", action="store_true", help="Hide seed G1 MuJoCo mesh.")
  parser.add_argument("--global-seed-g1-mesh", action="store_true", help="Render seed G1 mesh with CSV root pose.")
  parser.add_argument("--raw-orientations", action="store_true", help="Disable soma-retargeter orientation offsets.")
  parser.add_argument("--raw-upper-arm-axes", action="store_true", help="Disable G1 upper-arm axis convention flip.")
  parser.add_argument("--raw-lower-body-offsets", action="store_true", help="Disable soma-retargeter lower-body offsets.")
  parser.add_argument("--keep-global-heading", action="store_true", help="Keep the BVH global heading.")
  parser.add_argument("--world-frame-targets", action="store_true", help="Do not localize targets to body frame.")
  return parser


def main(argv: Sequence[str] | None = None) -> int:
  args = build_arg_parser().parse_args(argv)
  config = PipelineVisualizerConfig(
    bvh_path=args.bvh,
    seed_csv_path=args.seed_csv,
    start_frame=args.start_frame,
    end_frame=args.end_frame,
    fps=args.fps,
    port=args.port,
    axis_length=args.axis_length,
    column_spacing=args.column_spacing,
    initial_pipeline_step=args.pipeline_step,
    body_part=args.body_part,
    show_all_candidates=args.show_all_candidates,
    show_seed_g1_mesh=not args.hide_seed_g1_mesh,
    seed_g1_mesh_use_global_root=args.global_seed_g1_mesh,
    apply_orientation_offsets=not args.raw_orientations,
    align_upper_arm_axes_to_g1=not args.raw_upper_arm_axes,
    apply_lower_body_offsets=not args.raw_lower_body_offsets,
    remove_initial_heading=not args.keep_global_heading,
    localize_to_body_frame=not args.world_frame_targets,
  )
  traces = build_pipeline_traces(config)
  _run_viser(traces, config)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
