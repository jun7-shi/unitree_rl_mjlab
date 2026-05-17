from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from src.motion.bvh_full_body import load_soma_bvh_full_body_targets
from src.motion.seed_bones import (
  G1_29DOF_JOINT_COLUMNS,
  REQUIRED_COLUMNS,
  convert_seed_bones_row_to_motion_row,
)
from src.motion.sew_mimic import (
  PAPER_V1_ALGORITHM,
  ArmKeypointTarget,
  normalize,
  solve_two_axis_rotation,
)
from src.motion.sew_upper_body import G1UpperBodySEWRetargeter, UpperBodyTarget


UPPER_BODY_START = 12
UPPER_BODY_DOF = 17
LEFT_UPPER_GROUP = np.array([3, 4], dtype=int)
STORY_STEP_OPTIONS = (
  "all steps",
  "1 raw BVH skeleton",
  "2 BVH SEW target axes",
  "3 seed G1 real robot",
  "4 seed SEW proxy axes",
  "5 paper_v1 solver axes",
)


@dataclass(frozen=True)
class VisualizerConfig:
  bvh_path: Path
  seed_csv_path: Path
  start_frame: int = 260
  end_frame: int = 285
  fps: float = 10.0
  port: int = 8088
  axis_length: float = 0.18
  column_spacing: float = 0.8
  initial_layout: str = "story"
  show_seed_g1_mesh: bool = True
  seed_g1_mesh_use_global_root: bool = False
  apply_orientation_offsets: bool = True
  align_upper_arm_axes_to_g1: bool = True
  apply_lower_body_offsets: bool = True
  remove_initial_heading: bool = True
  localize_to_body_frame: bool = True


@dataclass(frozen=True)
class ArmColumnFrame:
  name: str
  target: UpperBodyTarget
  color: tuple[int, int, int]
  offset: np.ndarray
  show_body: bool = True
  show_axes: bool = True
  show_keypoints: bool = True


@dataclass(frozen=True)
class LeftUpperCandidateDebug:
  q_before_pitch_roll_deg: tuple[float, float]
  candidates_pitch_roll_deg: tuple[tuple[float, float], ...]
  target_dot_pitch_axis: float
  target_dot_initial_axis: float


@dataclass(frozen=True)
class MeshSnapshot:
  vertices: np.ndarray
  faces: np.ndarray


@dataclass(frozen=True)
class DebugFrame:
  frame_index: int
  csv_frame: int
  raw_bvh: UpperBodyTarget
  bvh: UpperBodyTarget
  seed_physical: UpperBodyTarget | None
  seed: UpperBodyTarget
  seed_g1_mesh: MeshSnapshot | None
  solver: UpperBodyTarget
  seed_left_joints_deg: tuple[float, float, float, float]
  solver_left_joints_deg: tuple[float, float, float, float]
  left_candidate_debug: LeftUpperCandidateDebug
  seed_bvh_left_upper_axis_error_deg: float
  seed_bvh_left_lower_axis_error_deg: float
  seed_bvh_left_elbow_angle_delta_deg: float


def _load_seed_motion_rows(path: str | Path) -> tuple[list[int], list[list[float]]]:
  input_path = Path(path)
  frame_numbers: list[int] = []
  rows: list[list[float]] = []
  with input_path.open(newline="", encoding="utf-8") as handle:
    reader = csv.DictReader(handle)
    if reader.fieldnames is None:
      raise ValueError(f"Seed-bones CSV has no header: {input_path}")
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in reader.fieldnames]
    if missing_columns:
      raise ValueError(
        f"Seed-bones CSV is missing required columns: {', '.join(missing_columns)}"
      )
    for csv_row in reader:
      frame_numbers.append(int(float(csv_row["Frame"])))
      rows.append(convert_seed_bones_row_to_motion_row(csv_row))
  return frame_numbers, rows


def _set_qpos_from_seed_motion_row(
  data: mujoco.MjData,
  joint_qpos_addresses: Sequence[int],
  row: Sequence[float],
) -> None:
  data.qpos[:] = 0.0
  data.qpos[0:3] = row[0:3]
  # Seed motion rows store root quaternions as xyzw; MuJoCo freejoint qpos uses wxyz.
  data.qpos[3:7] = [row[6], row[3], row[4], row[5]]
  for qpos_address, joint_position in zip(joint_qpos_addresses, row[7:]):
    data.qpos[int(qpos_address)] = float(joint_position)


def _seed_g1_model_context() -> tuple[mujoco.MjModel, mujoco.MjData, list[int], list[int]]:
  model = mujoco.MjModel.from_xml_path(
    str(REPO_ROOT / "src/assets/robots/unitree_g1/xmls/g1.xml")
  )
  data = mujoco.MjData(model)
  joint_qpos_addresses: list[int] = []
  for column in G1_29DOF_JOINT_COLUMNS:
    joint_name = column.removesuffix("_dof")
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
      raise ValueError(f"G1 model is missing joint '{joint_name}'")
    joint_qpos_addresses.append(int(model.jnt_qposadr[joint_id]))
  visual_geom_ids = [
    geom_id
    for geom_id in range(model.ngeom)
    if int(model.geom_group[geom_id]) < 3
  ]
  return model, data, joint_qpos_addresses, visual_geom_ids


def _joint_anchor(model: mujoco.MjModel, data: mujoco.MjData, joint_name: str) -> np.ndarray:
  joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
  if joint_id < 0:
    raise ValueError(f"G1 model is missing joint '{joint_name}'")
  return data.xanchor[joint_id].copy()


def _body_position(model: mujoco.MjModel, data: mujoco.MjData, body_name: str) -> np.ndarray:
  body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
  if body_id < 0:
    raise ValueError(f"G1 model is missing body '{body_name}'")
  return data.xpos[body_id].copy()


def _body_orientation(model: mujoco.MjModel, data: mujoco.MjData, body_name: str) -> np.ndarray:
  body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
  if body_id < 0:
    raise ValueError(f"G1 model is missing body '{body_name}'")
  return data.xmat[body_id].reshape(3, 3).copy()


def _site_orientation(model: mujoco.MjModel, data: mujoco.MjData, site_name: str) -> np.ndarray:
  site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
  if site_id < 0:
    raise ValueError(f"G1 model is missing site '{site_name}'")
  return data.site_xmat[site_id].reshape(3, 3).copy()


def _seed_g1_shoulder_center(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
  return 0.5 * (
    _joint_anchor(model, data, "left_shoulder_pitch_joint")
    + _joint_anchor(model, data, "right_shoulder_pitch_joint")
  )


def _seed_g1_physical_target(model: mujoco.MjModel, data: mujoco.MjData) -> UpperBodyTarget:
  return UpperBodyTarget(
    chest_position=_body_position(model, data, "torso_link"),
    chest_orientation=_body_orientation(model, data, "torso_link"),
    left_arm=ArmKeypointTarget(
      shoulder=_joint_anchor(model, data, "left_shoulder_pitch_joint"),
      elbow=_joint_anchor(model, data, "left_elbow_joint"),
      wrist=_joint_anchor(model, data, "left_wrist_pitch_joint"),
      hand_orientation=_site_orientation(model, data, "left_palm"),
    ),
    right_arm=ArmKeypointTarget(
      shoulder=_joint_anchor(model, data, "right_shoulder_pitch_joint"),
      elbow=_joint_anchor(model, data, "right_elbow_joint"),
      wrist=_joint_anchor(model, data, "right_wrist_pitch_joint"),
      hand_orientation=_site_orientation(model, data, "right_palm"),
    ),
  )


def _seed_g1_mesh_snapshot(
  model: mujoco.MjModel,
  data: mujoco.MjData,
  joint_qpos_addresses: Sequence[int],
  visual_geom_ids: Sequence[int],
  row: Sequence[float],
  *,
  center: np.ndarray | None = None,
  use_global_root: bool = False,
) -> MeshSnapshot:
  from mjlab.viewer.viser.conversions import merge_geoms_global

  seed_row = list(row)
  if not use_global_root:
    seed_row = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, *seed_row[7:]]
  _set_qpos_from_seed_motion_row(data, joint_qpos_addresses, seed_row)
  mujoco.mj_forward(model, data)
  mesh = merge_geoms_global(model, data, list(visual_geom_ids))
  mesh_center = _seed_g1_shoulder_center(model, data) if center is None else np.asarray(center, dtype=float)
  return MeshSnapshot(
    vertices=np.asarray(mesh.vertices, dtype=float) - mesh_center,
    faces=np.asarray(mesh.faces, dtype=np.int32),
  )


def _axis_angle_degrees(lhs: np.ndarray, rhs: np.ndarray) -> float:
  cosine = float(np.clip(np.dot(normalize(lhs), normalize(rhs)), -1.0, 1.0))
  return math.degrees(math.acos(cosine))


def _upper_q_from_seed_row(row: Sequence[float]) -> np.ndarray:
  start = 7 + UPPER_BODY_START
  stop = start + UPPER_BODY_DOF
  return np.asarray(row[start:stop], dtype=float)


def _left_seed_joints_deg(row: Sequence[float]) -> tuple[float, float, float, float]:
  upper_q = _upper_q_from_seed_row(row)
  return tuple(float(v) for v in np.rad2deg(upper_q[3:7]))


def _arm_axes(target: ArmKeypointTarget) -> tuple[np.ndarray, np.ndarray]:
  return normalize(target.elbow - target.shoulder), normalize(target.wrist - target.elbow)


def _elbow_angle_degrees(target: ArmKeypointTarget) -> float:
  upper, lower = _arm_axes(target)
  return _axis_angle_degrees(upper, lower)


def _left_upper_candidate_debug(
  retargeter: G1UpperBodySEWRetargeter,
  q_previous: np.ndarray,
  target: UpperBodyTarget,
) -> LeftUpperCandidateDebug:
  q = retargeter._clip_joint_angles(np.asarray(q_previous, dtype=float))
  q = retargeter._solve_orientation_group(
    q,
    slice(0, 3),
    retargeter._torso_body_id,
    target.chest_orientation,
  )
  base = q.copy()
  base[LEFT_UPPER_GROUP] = 0.0
  retargeter._set_upper_body_joint_angles(base)
  first_axis = normalize(retargeter.data.xaxis[retargeter.controlled_joint_ids[LEFT_UPPER_GROUP[0]]])
  second_axis = normalize(retargeter.data.xaxis[retargeter.controlled_joint_ids[LEFT_UPPER_GROUP[1]]])
  joint_id = retargeter._axis_joint_ids["left"]["upper"]
  initial_axis = normalize(retargeter.data.xaxis[joint_id])
  target_axis = normalize(target.left_arm.elbow - target.left_arm.shoulder)
  candidates = solve_two_axis_rotation(initial_axis, target_axis, first_axis, second_axis)
  return LeftUpperCandidateDebug(
    q_before_pitch_roll_deg=tuple(float(v) for v in np.rad2deg(q[LEFT_UPPER_GROUP])),
    candidates_pitch_roll_deg=tuple(
      (float(math.degrees(first)), float(math.degrees(second)))
      for first, second in candidates
    ),
    target_dot_pitch_axis=float(np.dot(target_axis, first_axis)),
    target_dot_initial_axis=float(np.dot(target_axis, initial_axis)),
  )


def _build_debug_frames(config: VisualizerConfig) -> list[DebugFrame]:
  if config.start_frame < 0:
    raise ValueError(f"start_frame must be non-negative, got {config.start_frame}")
  if config.end_frame < config.start_frame:
    raise ValueError(
      f"end_frame must be greater than or equal to start_frame, got {config.end_frame}"
    )

  csv_frames, seed_rows = _load_seed_motion_rows(config.seed_csv_path)
  target_kwargs = dict(
    apply_orientation_offsets=config.apply_orientation_offsets,
    apply_lower_body_offsets=config.apply_lower_body_offsets,
    remove_initial_heading=config.remove_initial_heading,
    localize_to_body_frame=config.localize_to_body_frame and config.apply_lower_body_offsets,
  )
  targets = load_soma_bvh_full_body_targets(
    config.bvh_path,
    align_upper_arm_axes_to_g1=config.align_upper_arm_axes_to_g1,
    **target_kwargs,
  )
  raw_targets = load_soma_bvh_full_body_targets(
    config.bvh_path,
    align_upper_arm_axes_to_g1=False,
    **target_kwargs,
  )
  frame_count = min(len(seed_rows), len(targets))
  if config.end_frame >= frame_count:
    raise ValueError(
      f"end_frame {config.end_frame} is out of range for {frame_count} aligned frames"
    )

  seed_probe = G1UpperBodySEWRetargeter(algorithm_version=PAPER_V1_ALGORITHM)
  solver = G1UpperBodySEWRetargeter(algorithm_version=PAPER_V1_ALGORITHM)
  solver_probe = G1UpperBodySEWRetargeter(algorithm_version=PAPER_V1_ALGORITHM)
  seed_mesh_context = _seed_g1_model_context() if config.show_seed_g1_mesh else None
  q_previous = np.zeros(UPPER_BODY_DOF)
  debug_frames: list[DebugFrame] = []

  for frame_index in range(config.end_frame + 1):
    bvh_upper = targets[frame_index].upper
    raw_bvh_upper = raw_targets[frame_index].upper
    candidate_debug = _left_upper_candidate_debug(solver, q_previous, bvh_upper)
    solver_result = solver.retarget(q_previous, bvh_upper)
    solver_q = solver_result.solver_joint_angles if solver_result.solver_joint_angles is not None else solver_result.joint_angles

    if frame_index >= config.start_frame:
      seed_upper_q = _upper_q_from_seed_row(seed_rows[frame_index])
      seed_upper = seed_probe.target_from_configuration(seed_upper_q)
      solver_upper = solver_probe.target_from_configuration(solver_q)
      seed_g1_mesh = (
        _seed_g1_mesh_snapshot(
          *seed_mesh_context,
          seed_rows[frame_index],
          use_global_root=config.seed_g1_mesh_use_global_root,
        )
        if seed_mesh_context is not None
        else None
      )
      seed_physical = None
      if seed_mesh_context is not None:
        mesh_model, mesh_data, mesh_joint_qpos_addresses, _mesh_visual_geom_ids = seed_mesh_context
        seed_row = list(seed_rows[frame_index])
        if not config.seed_g1_mesh_use_global_root:
          seed_row = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, *seed_row[7:]]
        _set_qpos_from_seed_motion_row(mesh_data, mesh_joint_qpos_addresses, seed_row)
        mujoco.mj_forward(mesh_model, mesh_data)
        seed_physical = _seed_g1_physical_target(mesh_model, mesh_data)
      seed_upper_axis, seed_lower_axis = _arm_axes(seed_upper.left_arm)
      bvh_upper_axis, bvh_lower_axis = _arm_axes(bvh_upper.left_arm)
      debug_frames.append(
        DebugFrame(
          frame_index=frame_index,
          csv_frame=csv_frames[frame_index],
          raw_bvh=raw_bvh_upper,
          bvh=bvh_upper,
          seed_physical=seed_physical,
          seed=seed_upper,
          seed_g1_mesh=seed_g1_mesh,
          solver=solver_upper,
          seed_left_joints_deg=_left_seed_joints_deg(seed_rows[frame_index]),
          solver_left_joints_deg=tuple(float(v) for v in np.rad2deg(solver_q[3:7])),
          left_candidate_debug=candidate_debug,
          seed_bvh_left_upper_axis_error_deg=_axis_angle_degrees(seed_upper_axis, bvh_upper_axis),
          seed_bvh_left_lower_axis_error_deg=_axis_angle_degrees(seed_lower_axis, bvh_lower_axis),
          seed_bvh_left_elbow_angle_delta_deg=(
            _elbow_angle_degrees(bvh_upper.left_arm) - _elbow_angle_degrees(seed_upper.left_arm)
          ),
        )
      )
    q_previous = solver_q
  return debug_frames


def _centered_point(point: np.ndarray, origin: np.ndarray, offset: np.ndarray) -> np.ndarray:
  return np.asarray(point, dtype=float) - origin + offset


def _target_origin(target: UpperBodyTarget) -> np.ndarray:
  return 0.5 * (target.left_arm.shoulder + target.right_arm.shoulder)


def _body_segments(target: UpperBodyTarget, offset: np.ndarray) -> np.ndarray:
  origin = _target_origin(target)

  def p(value: np.ndarray) -> np.ndarray:
    return _centered_point(value, origin, offset)

  segments = [
    (target.left_arm.shoulder, target.right_arm.shoulder),
    (target.chest_position, target.left_arm.shoulder),
    (target.chest_position, target.right_arm.shoulder),
    (target.left_arm.shoulder, target.left_arm.elbow),
    (target.left_arm.elbow, target.left_arm.wrist),
    (target.right_arm.shoulder, target.right_arm.elbow),
    (target.right_arm.elbow, target.right_arm.wrist),
  ]
  return np.asarray([[p(start), p(end)] for start, end in segments], dtype=float)


def _axis_segments(target: UpperBodyTarget, offset: np.ndarray, axis_length: float) -> np.ndarray:
  origin = _target_origin(target)

  def p(value: np.ndarray) -> np.ndarray:
    return _centered_point(value, origin, offset)

  left_upper, left_lower = _arm_axes(target.left_arm)
  right_upper, right_lower = _arm_axes(target.right_arm)
  starts_and_axes = [
    (target.left_arm.shoulder, left_upper),
    (target.left_arm.elbow, left_lower),
    (target.right_arm.shoulder, right_upper),
    (target.right_arm.elbow, right_lower),
  ]
  return np.asarray(
    [[p(start), p(start + axis * axis_length)] for start, axis in starts_and_axes],
    dtype=float,
  )


def _keypoint_cloud(target: UpperBodyTarget, offset: np.ndarray) -> np.ndarray:
  origin = _target_origin(target)
  points = [
    target.chest_position,
    target.left_arm.shoulder,
    target.left_arm.elbow,
    target.left_arm.wrist,
    target.right_arm.shoulder,
    target.right_arm.elbow,
    target.right_arm.wrist,
  ]
  return np.asarray([_centered_point(point, origin, offset) for point in points], dtype=float)


def _candidate_lines(debug: LeftUpperCandidateDebug) -> str:
  if not debug.candidates_pitch_roll_deg:
    return "no candidates"
  return "<br>".join(
    f"cand {index}: pitch {pitch:7.2f}, roll {roll:7.2f}"
    for index, (pitch, roll) in enumerate(debug.candidates_pitch_roll_deg)
  )


def _story_step_description(step: str) -> str:
  descriptions = {
    "1 raw BVH skeleton": (
      "Shows the real skeleton/keypoints imported from BVH after the same coordinate conversion "
      "used by the retargeting pipeline."
    ),
    "2 BVH SEW target axes": (
      "Shows the axis-only solver target after preprocessing; the upper-arm axis may be flipped "
      "to match the G1 SEW convention. These axes are not elbow/wrist positions."
    ),
    "3 seed G1 real robot": (
      "Shows the real G1 mesh and physical shoulder/elbow/wrist anchors driven directly by the "
      "bones-seed CSV joint angles."
    ),
    "4 seed SEW proxy axes": (
      "Shows the axis-only proxy extracted from the seed G1 pose. It is built from G1 joint axes, "
      "so the displayed elbow/wrist points are not elbow/wrist positions."
    ),
    "5 paper_v1 solver axes": (
      "Shows the axis-only proxy produced by the current closed-form paper_v1 solver from the "
      "BVH SEW target axes."
    ),
  }
  return descriptions.get(step, "Shows every stage in the left-to-right pipeline.")


def _format_debug_markdown(
  frame: DebugFrame,
  *,
  layout: str = "story",
  step: str = "all steps",
) -> str:
  seed_pitch, seed_roll, seed_yaw, seed_elbow = frame.seed_left_joints_deg
  solver_pitch, solver_roll, solver_yaw, solver_elbow = frame.solver_left_joints_deg
  before_pitch, before_roll = frame.left_candidate_debug.q_before_pitch_roll_deg
  story_prefix = ""
  if layout == "story":
    story_prefix = (
      "**Read left to right.**<br>"
      "1. **Raw BVH skeleton**: real skeleton/keypoints from the BVH after coordinate conversion.<br>"
      "2. **BVH SEW target axes**: axis-only target after preprocessing; upper arm may be flipped to match G1 SEW convention.<br>"
      "3. **Seed G1 real robot**: real G1 mesh plus real shoulder/elbow/wrist anchors from bones-seed CSV.<br>"
      "4. **Seed SEW proxy axes**: axis-only proxy extracted from the seed G1 pose; these are not elbow/wrist positions.<br>"
      "5. **Paper solver axes**: axis-only proxy produced by the current closed-form retargeter.<br>"
      "Colors: red = upper-arm/proxy upper axis, purple = lower-arm/proxy lower axis.<br><br>"
    )
    if step != "all steps":
      story_prefix += f"**Current step:** {step}<br>{_story_step_description(step)}<br><br>"
  return (
    story_prefix +
    f"**Frame:** {frame.frame_index}  **CSV Frame:** {frame.csv_frame}<br>"
    f"**Seed vs BVH left upper axis:** {frame.seed_bvh_left_upper_axis_error_deg:.2f} deg<br>"
    f"**Seed vs BVH left lower axis:** {frame.seed_bvh_left_lower_axis_error_deg:.2f} deg<br>"
    f"**BVH elbow angle - seed elbow angle:** {frame.seed_bvh_left_elbow_angle_delta_deg:.2f} deg<br><br>"
    f"**seed left joints:** pitch {seed_pitch:.2f}, roll {seed_roll:.2f}, "
    f"yaw {seed_yaw:.2f}, elbow {seed_elbow:.2f}<br>"
    f"**solver left joints:** pitch {solver_pitch:.2f}, roll {solver_roll:.2f}, "
    f"yaw {solver_yaw:.2f}, elbow {solver_elbow:.2f}<br>"
    f"**before left pitch/roll solve:** pitch {before_pitch:.2f}, roll {before_roll:.2f}<br>"
    f"**target dot pitch axis:** {frame.left_candidate_debug.target_dot_pitch_axis:.4f}<br>"
    f"**target dot initial upper axis:** {frame.left_candidate_debug.target_dot_initial_axis:.4f}<br><br>"
    f"**seed G1 mesh:** {'shown' if frame.seed_g1_mesh is not None else 'hidden'}<br>"
    f"**seed physical links:** {'shown' if frame.seed_physical is not None else 'hidden'}<br>"
    f"**Note:** skeleton columns are real keypoints; SEW columns are axis-only proxy targets.<br><br>"
    f"**left pitch/roll candidates**<br>{_candidate_lines(frame.left_candidate_debug)}"
  )


def _run_viser(debug_frames: Sequence[DebugFrame], config: VisualizerConfig) -> None:
  import viser

  server = viser.ViserServer(port=config.port)
  server.scene.add_grid("/grid", width=3.5, height=2.0, cell_size=0.1)

  frame_by_index = {frame.frame_index: frame for frame in debug_frames}
  frame_indices = list(frame_by_index)
  handles: list[object] = []

  with server.gui.add_folder("Playback"):
    frame_slider = server.gui.add_slider(
      "Frame",
      min=min(frame_indices),
      max=max(frame_indices),
      step=1,
      initial_value=min(frame_indices),
    )
    layout_dropdown = server.gui.add_dropdown(
      "Layout",
      options=("story", "overview"),
      initial_value=config.initial_layout,
    )
    step_dropdown = server.gui.add_dropdown(
      "Story step",
      options=STORY_STEP_OPTIONS,
      initial_value="all steps",
    )
    play_checkbox = server.gui.add_checkbox("Play", initial_value=False)
    fps_slider = server.gui.add_slider("FPS", min=1.0, max=60.0, step=1.0, initial_value=config.fps)
    info_markdown = server.gui.add_markdown("")

  def clear_scene() -> None:
    while handles:
      handle = handles.pop()
      handle.remove()

  def add_column(column: ArmColumnFrame) -> None:
    if column.show_body:
      body_segments = _body_segments(column.target, column.offset)
      handles.append(
        server.scene.add_line_segments(
          f"/{column.name}/body",
          points=body_segments,
          colors=np.asarray(column.color, dtype=np.uint8),
          line_width=4.0,
        )
      )
    if column.show_axes:
      axis_segments = _axis_segments(column.target, column.offset, config.axis_length)
      handles.append(
        server.scene.add_line_segments(
          f"/{column.name}/axes",
          points=axis_segments,
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
      keypoints = _keypoint_cloud(column.target, column.offset)
      handles.append(
        server.scene.add_point_cloud(
          f"/{column.name}/keypoints",
          points=keypoints,
          colors=np.asarray(column.color, dtype=np.uint8),
          point_size=0.035,
          point_shape="circle",
        )
      )
    handles.append(
      server.scene.add_label(
        f"/{column.name}/label",
        text=column.name,
        position=column.offset + np.array([0.0, 0.0, 0.55]),
        font_size_mode="scene",
        font_scene_height=0.05,
        anchor="center-center",
      )
    )

  def add_seed_mesh_column(frame: DebugFrame, offset: np.ndarray) -> None:
    if frame.seed_g1_mesh is None:
      return
    handles.append(
      server.scene.add_mesh_simple(
        "/seed_g1_model/mesh",
        vertices=frame.seed_g1_mesh.vertices,
        faces=frame.seed_g1_mesh.faces,
        color=(80, 190, 115),
        opacity=0.55,
        flat_shading=False,
        side="double",
        position=offset,
      )
    )
    if frame.seed_physical is not None:
      add_column(
        ArmColumnFrame(
          "seed G1 physical links",
          frame.seed_physical,
          (245, 245, 245),
          offset,
          show_axes=False,
        )
      )
    handles.append(
      server.scene.add_label(
        "/seed_g1_model/label",
        text="seed G1 model",
        position=offset + np.array([0.0, 0.0, 0.75]),
        font_size_mode="scene",
        font_scene_height=0.05,
        anchor="center-center",
      )
    )

  def update_frame(frame_index: int) -> None:
    frame = frame_by_index[int(frame_index)]
    clear_scene()
    spacing = config.column_spacing
    layout = str(layout_dropdown.value)
    step = str(step_dropdown.value)
    if layout == "story":
      if step == "1 raw BVH skeleton":
        columns = [
          ArmColumnFrame(
            "1 raw BVH skeleton (real keypoints)",
            frame.raw_bvh,
            (242, 143, 52),
            np.zeros(3),
            show_axes=False,
          )
        ]
      elif step == "2 BVH SEW target axes":
        columns = [
          ArmColumnFrame(
            "2 BVH SEW target axes (axis-only)",
            frame.bvh,
            (242, 143, 52),
            np.zeros(3),
            show_body=False,
            show_keypoints=False,
          )
        ]
      elif step == "3 seed G1 real robot":
        columns = []
        if frame.seed_g1_mesh is not None:
          add_seed_mesh_column(frame, np.zeros(3))
        else:
          handles.append(
            server.scene.add_label(
              "/seed_g1_model/hidden_label",
              text="3 seed G1 real robot hidden",
              position=np.array([0.0, 0.0, 0.55]),
              font_size_mode="scene",
              font_scene_height=0.05,
              anchor="center-center",
            )
          )
      elif step == "4 seed SEW proxy axes":
        columns = [
          ArmColumnFrame(
            "4 seed SEW proxy axes (axis-only)",
            frame.seed,
            (58, 166, 85),
            np.zeros(3),
            show_body=False,
            show_keypoints=False,
          )
        ]
      elif step == "5 paper_v1 solver axes":
        columns = [
          ArmColumnFrame(
            "5 paper_v1 solver axes (axis-only)",
            frame.solver,
            (80, 145, 245),
            np.zeros(3),
            show_body=False,
            show_keypoints=False,
          )
        ]
      elif frame.seed_g1_mesh is None:
        columns = [
          ArmColumnFrame("1 raw BVH skeleton (real keypoints)", frame.raw_bvh, (242, 143, 52), np.array([-1.5 * spacing, 0.0, 0.0]), show_axes=False),
          ArmColumnFrame("2 BVH SEW target axes (axis-only)", frame.bvh, (242, 143, 52), np.array([-0.5 * spacing, 0.0, 0.0]), show_body=False, show_keypoints=False),
          ArmColumnFrame("4 seed SEW proxy axes (axis-only)", frame.seed, (58, 166, 85), np.array([0.5 * spacing, 0.0, 0.0]), show_body=False, show_keypoints=False),
          ArmColumnFrame("5 paper_v1 solver axes (axis-only)", frame.solver, (80, 145, 245), np.array([1.5 * spacing, 0.0, 0.0]), show_body=False, show_keypoints=False),
        ]
      else:
        columns = [
          ArmColumnFrame("1 raw BVH skeleton (real keypoints)", frame.raw_bvh, (242, 143, 52), np.array([-2.0 * spacing, 0.0, 0.0]), show_axes=False),
          ArmColumnFrame("2 BVH SEW target axes (axis-only)", frame.bvh, (242, 143, 52), np.array([-1.0 * spacing, 0.0, 0.0]), show_body=False, show_keypoints=False),
          ArmColumnFrame("4 seed SEW proxy axes (axis-only)", frame.seed, (58, 166, 85), np.array([1.0 * spacing, 0.0, 0.0]), show_body=False, show_keypoints=False),
          ArmColumnFrame("5 paper_v1 solver axes (axis-only)", frame.solver, (80, 145, 245), np.array([2.0 * spacing, 0.0, 0.0]), show_body=False, show_keypoints=False),
        ]
        add_seed_mesh_column(frame, np.array([0.0, 0.0, 0.0]))
    elif frame.seed_g1_mesh is None:
      columns = [
        ArmColumnFrame("raw BVH skeleton", frame.raw_bvh, (242, 143, 52), np.array([-1.5 * spacing, 0.0, 0.0]), show_axes=False),
        ArmColumnFrame("BVH SEW target axes", frame.bvh, (242, 143, 52), np.array([-0.5 * spacing, 0.0, 0.0]), show_body=False, show_keypoints=False),
        ArmColumnFrame("seed SEW proxy axes", frame.seed, (58, 166, 85), np.array([0.5 * spacing, 0.0, 0.0]), show_body=False, show_keypoints=False),
        ArmColumnFrame("paper_v1 solver axes", frame.solver, (80, 145, 245), np.array([1.5 * spacing, 0.0, 0.0]), show_body=False, show_keypoints=False),
      ]
    else:
      columns = [
        ArmColumnFrame("raw BVH skeleton", frame.raw_bvh, (242, 143, 52), np.array([-2.0 * spacing, 0.0, 0.0]), show_axes=False),
        ArmColumnFrame("BVH SEW target axes", frame.bvh, (242, 143, 52), np.array([-1.0 * spacing, 0.0, 0.0]), show_body=False, show_keypoints=False),
        ArmColumnFrame("seed SEW proxy axes", frame.seed, (58, 166, 85), np.array([1.0 * spacing, 0.0, 0.0]), show_body=False, show_keypoints=False),
        ArmColumnFrame("paper_v1 solver axes", frame.solver, (80, 145, 245), np.array([2.0 * spacing, 0.0, 0.0]), show_body=False, show_keypoints=False),
      ]
      add_seed_mesh_column(frame, np.array([0.0, 0.0, 0.0]))
    for column in columns:
      add_column(column)
    info_markdown.content = _format_debug_markdown(frame, layout=layout, step=step)

  @frame_slider.on_update
  def _(event) -> None:
    update_frame(int(event.target.value))

  @layout_dropdown.on_update
  def _(_) -> None:
    update_frame(int(frame_slider.value))

  @step_dropdown.on_update
  def _(_) -> None:
    update_frame(int(frame_slider.value))

  update_frame(min(frame_indices))
  print(f"Viser seed/BVH axis viewer running on http://localhost:{config.port}")
  frame_dt = 1.0 / max(config.fps, 1e-6)
  while True:
    if play_checkbox.value:
      next_frame = int(frame_slider.value) + 1
      if next_frame > max(frame_indices):
        next_frame = min(frame_indices)
      frame_slider.value = next_frame
      update_frame(next_frame)
      frame_dt = 1.0 / max(float(fps_slider.value), 1e-6)
      time.sleep(frame_dt)
    else:
      time.sleep(0.1)


def build_arg_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
    description="Visualize seed/BVH arm axes for bones-seed SEW-Mimic diagnostics."
  )
  parser.add_argument("--bvh", required=True, type=Path, help="Input SOMA/bones-seed BVH file.")
  parser.add_argument("--seed-csv", required=True, type=Path, help="Input bones-seed G1 CSV file.")
  parser.add_argument("--start-frame", type=int, default=260, help="First frame to show.")
  parser.add_argument("--end-frame", type=int, default=285, help="Last frame to show, inclusive.")
  parser.add_argument("--fps", type=float, default=10.0, help="Autoplay FPS for the browser viewer.")
  parser.add_argument("--port", type=int, default=8088, help="Viser web server port.")
  parser.add_argument("--axis-length", type=float, default=0.18, help="Displayed upper/lower arm axis length in meters.")
  parser.add_argument("--column-spacing", type=float, default=0.8, help="Spacing between BVH, seed, and solver columns.")
  parser.add_argument(
    "--layout",
    choices=("story", "overview"),
    default="story",
    help="Initial viewer layout. Use story for a left-to-right algorithm explanation.",
  )
  parser.add_argument(
    "--hide-seed-g1-mesh",
    action="store_true",
    help="Hide the direct G1 model mesh driven by the seed CSV.",
  )
  parser.add_argument(
    "--global-seed-g1-mesh",
    action="store_true",
    help=(
      "Render the seed G1 model with the CSV root translation/orientation. "
      "By default only 29DOF joint angles are used so the mesh matches the body-frame axis overlays."
    ),
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
    help="Keep the BVH global heading instead of removing the first frame's heading.",
  )
  parser.add_argument(
    "--world-frame-targets",
    action="store_true",
    help="Show BVH targets in world frame instead of localizing each frame to the body frame.",
  )
  return parser


def main(argv: Sequence[str] | None = None) -> int:
  args = build_arg_parser().parse_args(argv)
  config = VisualizerConfig(
    bvh_path=args.bvh,
    seed_csv_path=args.seed_csv,
    start_frame=args.start_frame,
    end_frame=args.end_frame,
    fps=args.fps,
    port=args.port,
    axis_length=args.axis_length,
    column_spacing=args.column_spacing,
    initial_layout=args.layout,
    show_seed_g1_mesh=not args.hide_seed_g1_mesh,
    seed_g1_mesh_use_global_root=args.global_seed_g1_mesh,
    apply_orientation_offsets=not args.raw_orientations,
    align_upper_arm_axes_to_g1=not args.raw_upper_arm_axes,
    apply_lower_body_offsets=not args.raw_lower_body_offsets,
    remove_initial_heading=not args.keep_global_heading,
    localize_to_body_frame=not args.world_frame_targets,
  )
  debug_frames = _build_debug_frames(config)
  _run_viser(debug_frames, config)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
