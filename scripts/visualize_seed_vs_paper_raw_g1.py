from __future__ import annotations

import argparse
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

from scripts.visualize_seed_bvh_axes import (
  MeshSnapshot,
  _load_seed_motion_rows,
  _seed_g1_model_context,
  _seed_g1_shoulder_center,
  _set_qpos_from_seed_motion_row,
)
from src.motion.bvh_full_body import load_soma_bvh_full_body_targets
from src.motion.bvh_upper_body import load_soma_bvh_upper_body_targets
from src.motion.sew_full_body import FullBodyTarget
from src.motion.sew_mimic import PAPER_V1_ALGORITHM, normalize
from src.motion.sew_upper_body import G1UpperBodySEWRetargeter


UPPER_BODY_START = 12
UPPER_BODY_DOF = 17
LEFT_SHOULDER_PITCH_UPPER_INDEX = 3
BODY_CENTRIC_DISPLAY_MODE = "body-centric"
WORLD_DISPLAY_MODE = "world"
DISPLAY_MODES = (BODY_CENTRIC_DISPLAY_MODE, WORLD_DISPLAY_MODE)


@dataclass(frozen=True)
class VisualizerConfig:
  bvh_path: Path
  seed_csv_path: Path | None = None
  start_frame: int = 0
  end_frame: int | None = None
  port: int = 8090
  fps: float = 10.0
  display_mode: str = BODY_CENTRIC_DISPLAY_MODE
  paper_raw_offset_to_seed: bool = False
  apply_orientation_offsets: bool = True
  align_upper_arm_axes_to_g1: bool = True
  remove_initial_heading: bool = True

  def __post_init__(self) -> None:
    if self.display_mode not in DISPLAY_MODES:
      raise ValueError(f"display_mode must be one of {DISPLAY_MODES}, got {self.display_mode!r}")


@dataclass(frozen=True)
class ComparisonFrame:
  frame_index: int
  csv_frame: int
  bvh_full_target: FullBodyTarget
  seed_motion_row: list[float]
  paper_raw_motion_row: list[float]
  paper_raw_upper_q: np.ndarray
  paper_raw_left_shoulder_pitch_deg: float
  displayed_left_shoulder_pitch_deg: float
  seed_left_shoulder_pitch_deg: float
  display_offset_k_360: int


@dataclass(frozen=True)
class ArticulatedGeomPoseSnapshot:
  geom_ids: tuple[int, ...]
  positions: np.ndarray
  wxyzs: np.ndarray


@dataclass(frozen=True)
class ArticulatedRobotHandles:
  geom_ids: tuple[int, ...]
  mesh_handles: tuple[object, ...]


@dataclass(frozen=True)
class BvhDisplaySnapshot:
  skeleton_segments: np.ndarray
  axis_segments: np.ndarray
  keypoints: np.ndarray


@dataclass(frozen=True)
class BvhDisplayHandles:
  skeleton_handle: object
  axes_handle: object
  keypoints_handle: object


def unwrap_to(value_deg: float, reference_deg: float) -> float:
  return value_deg + 360.0 * round((reference_deg - value_deg) / 360.0)


def paper_raw_motion_row_from_seed(
  seed_motion_row: Sequence[float],
  paper_raw_upper_q: Sequence[float],
) -> list[float]:
  row = list(seed_motion_row)
  upper_q = list(np.asarray(paper_raw_upper_q, dtype=float))
  if len(upper_q) != UPPER_BODY_DOF:
    raise ValueError(f"paper_raw_upper_q must have {UPPER_BODY_DOF} values, got {len(upper_q)}")
  row[7 + UPPER_BODY_START :] = upper_q
  return row


def infer_seed_csv_path_from_bvh(bvh_path: Path) -> Path:
  parts = Path(bvh_path).parts
  for index in range(len(parts) - 1):
    if parts[index] == "soma_uniform" and parts[index + 1] == "bvh":
      return Path(*parts[:index], "g1", "csv", *parts[index + 2 :]).with_suffix(".csv")
  raise ValueError(
    "Cannot infer seed CSV path from BVH path. Expected a bones-seed path "
    "containing 'soma_uniform/bvh'; pass --seed-csv explicitly."
  )


def _resolve_seed_csv_path(config: VisualizerConfig) -> Path:
  if config.seed_csv_path is not None:
    return config.seed_csv_path
  return infer_seed_csv_path_from_bvh(config.bvh_path)


def _paper_raw_upper_body_q(config: VisualizerConfig, frame_count: int | None = None) -> list[np.ndarray]:
  targets = load_soma_bvh_upper_body_targets(
    config.bvh_path,
    apply_orientation_offsets=config.apply_orientation_offsets,
    align_upper_arm_axes_to_g1=config.align_upper_arm_axes_to_g1,
    remove_initial_heading=config.remove_initial_heading,
  )
  retargeter = G1UpperBodySEWRetargeter(algorithm_version=PAPER_V1_ALGORITHM)
  q_previous = np.zeros(UPPER_BODY_DOF, dtype=float)
  rows: list[np.ndarray] = []
  selected_targets = targets if frame_count is None else targets[:frame_count]
  for target in selected_targets:
    result = retargeter.retarget(q_previous, target)
    raw_q = (
      result.solver_joint_angles
      if result.solver_joint_angles is not None
      else result.joint_angles
    )
    rows.append(np.asarray(raw_q, dtype=float).copy())
    q_previous = raw_q
  return rows


def build_comparison_frames(config: VisualizerConfig) -> list[ComparisonFrame]:
  if config.start_frame < 0:
    raise ValueError(f"start_frame must be non-negative, got {config.start_frame}")
  if config.end_frame is not None and config.end_frame < config.start_frame:
    raise ValueError(f"end_frame must be >= start_frame, got {config.end_frame}")

  seed_csv_path = _resolve_seed_csv_path(config)
  csv_frames, seed_rows = _load_seed_motion_rows(seed_csv_path)
  if not seed_rows:
    raise ValueError(f"Seed CSV has no motion rows: {seed_csv_path}")
  requested_frame_count = None if config.end_frame is None else config.end_frame + 1
  paper_upper_rows = _paper_raw_upper_body_q(config, requested_frame_count)
  bvh_full_targets = load_soma_bvh_full_body_targets(
    config.bvh_path,
    apply_orientation_offsets=config.apply_orientation_offsets,
    align_upper_arm_axes_to_g1=False,
    apply_lower_body_offsets=False,
    remove_initial_heading=config.remove_initial_heading,
    localize_to_body_frame=False,
  )
  available_frame_count = min(len(seed_rows), len(paper_upper_rows), len(bvh_full_targets))
  if available_frame_count <= 0:
    raise ValueError("No overlapping frames are available between seed CSV and BVH")

  end_frame = config.end_frame if config.end_frame is not None else available_frame_count - 1
  if end_frame >= len(seed_rows):
    raise ValueError(f"end_frame {end_frame} is out of range for {len(seed_rows)} seed rows")
  if end_frame >= len(paper_upper_rows):
    raise ValueError(f"BVH has only {len(paper_upper_rows)} upper-body targets")
  if end_frame >= len(bvh_full_targets):
    raise ValueError(f"BVH has only {len(bvh_full_targets)} display targets")
  if config.start_frame > end_frame:
    raise ValueError(f"start_frame {config.start_frame} is out of range for end_frame {end_frame}")

  frames: list[ComparisonFrame] = []
  for frame_index in range(config.start_frame, end_frame + 1):
    seed_row = seed_rows[frame_index]
    paper_upper_q = paper_upper_rows[frame_index]
    seed_pitch_deg = float(np.degrees(seed_row[7 + UPPER_BODY_START + LEFT_SHOULDER_PITCH_UPPER_INDEX]))
    raw_pitch_deg = float(np.degrees(paper_upper_q[LEFT_SHOULDER_PITCH_UPPER_INDEX]))
    displayed_pitch_deg = (
      unwrap_to(raw_pitch_deg, seed_pitch_deg)
      if config.paper_raw_offset_to_seed
      else raw_pitch_deg
    )
    display_offset_k = int(round((displayed_pitch_deg - raw_pitch_deg) / 360.0))
    display_upper_q = paper_upper_q.copy()
    display_upper_q[LEFT_SHOULDER_PITCH_UPPER_INDEX] = np.radians(displayed_pitch_deg)
    frames.append(
      ComparisonFrame(
        frame_index=frame_index,
        csv_frame=csv_frames[frame_index],
        bvh_full_target=bvh_full_targets[frame_index],
        seed_motion_row=seed_row,
        paper_raw_motion_row=paper_raw_motion_row_from_seed(seed_row, display_upper_q),
        paper_raw_upper_q=paper_upper_q,
        paper_raw_left_shoulder_pitch_deg=raw_pitch_deg,
        displayed_left_shoulder_pitch_deg=displayed_pitch_deg,
        seed_left_shoulder_pitch_deg=seed_pitch_deg,
        display_offset_k_360=display_offset_k,
      )
    )
  return frames


def _body_frame_row(row: Sequence[float]) -> list[float]:
  return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, *list(row)[7:]]


def _wxyz_from_xmat(xmat: Sequence[float]) -> np.ndarray:
  quat = np.empty(4, dtype=float)
  mujoco.mju_mat2Quat(quat, np.asarray(xmat, dtype=float).reshape(9))
  return quat


def _rotation_matrix_from_seed_motion_row(row: Sequence[float]) -> np.ndarray:
  values = list(row)
  quat_wxyz = np.asarray([values[6], values[3], values[4], values[5]], dtype=float)
  matrix = np.empty(9, dtype=float)
  mujoco.mju_quat2Mat(matrix, quat_wxyz)
  return matrix.reshape(3, 3)


def _articulated_geom_pose_snapshot(
  model: mujoco.MjModel,
  data: mujoco.MjData,
  joint_qpos_addresses: Sequence[int],
  geom_ids: Sequence[int],
  row: Sequence[float],
  *,
  offset: np.ndarray,
  display_mode: str,
) -> ArticulatedGeomPoseSnapshot:
  if display_mode not in DISPLAY_MODES:
    raise ValueError(f"display_mode must be one of {DISPLAY_MODES}, got {display_mode!r}")
  body_centric = display_mode == BODY_CENTRIC_DISPLAY_MODE
  motion_row = _body_frame_row(row) if body_centric else list(row)
  _set_qpos_from_seed_motion_row(data, joint_qpos_addresses, motion_row)
  mujoco.mj_forward(model, data)
  center = _seed_g1_shoulder_center(model, data) if body_centric else np.zeros(3)
  geom_id_tuple = tuple(int(geom_id) for geom_id in geom_ids)
  return ArticulatedGeomPoseSnapshot(
    geom_ids=geom_id_tuple,
    positions=np.asarray(
      [
        np.asarray(data.geom_xpos[geom_id], dtype=float) - center + offset
        for geom_id in geom_id_tuple
      ],
      dtype=float,
    ),
    wxyzs=np.asarray(
      [_wxyz_from_xmat(data.geom_xmat[geom_id]) for geom_id in geom_id_tuple],
      dtype=float,
    ),
  )


def _geom_mesh_snapshot(model: mujoco.MjModel, geom_id: int) -> MeshSnapshot:
  from mjlab.viewer.viser.conversions import create_primitive_mesh, mujoco_mesh_to_trimesh

  if int(model.geom_type[geom_id]) == int(mujoco.mjtGeom.mjGEOM_MESH):
    mesh = mujoco_mesh_to_trimesh(model, geom_id, verbose=False)
  else:
    mesh = create_primitive_mesh(model, geom_id)
  return MeshSnapshot(
    vertices=np.asarray(mesh.vertices, dtype=float),
    faces=np.asarray(mesh.faces, dtype=np.int32),
  )


def _mesh_snapshot_from_motion_row(
  model: mujoco.MjModel,
  data: mujoco.MjData,
  joint_qpos_addresses: Sequence[int],
  visual_geom_ids: Sequence[int],
  row: Sequence[float],
  *,
  display_mode: str,
) -> MeshSnapshot:
  from mjlab.viewer.viser.conversions import merge_geoms_global

  body_centric = display_mode == BODY_CENTRIC_DISPLAY_MODE
  motion_row = _body_frame_row(row) if body_centric else list(row)
  _set_qpos_from_seed_motion_row(data, joint_qpos_addresses, motion_row)
  mujoco.mj_forward(model, data)
  mesh = merge_geoms_global(model, data, list(visual_geom_ids))
  center = _seed_g1_shoulder_center(model, data) if body_centric else np.zeros(3)
  return MeshSnapshot(
    vertices=np.asarray(mesh.vertices, dtype=float) - center,
    faces=np.asarray(mesh.faces, dtype=np.int32),
  )


def _format_info(frame: ComparisonFrame, *, display_mode: str) -> str:
  return (
    f"**Frame:** {frame.frame_index}  **CSV Frame:** {frame.csv_frame}<br>"
    f"**Root/lower body:** copied from seed CSV for both robots<br>"
    f"**Orange column:** BVH full-body keypoint skeleton after SOMA-to-mjlab conversion; "
    f"no G1 upper-arm axis flip, no lower-body offsets<br>"
    f"**Green column:** seed G1 CSV full pose<br>"
    f"**Blue column:** paper_v1 raw upper-body retarget on seed root/lower-body<br>"
    f"**Display mode:** {display_mode}<br><br>"
    f"**seed left shoulder pitch:** {frame.seed_left_shoulder_pitch_deg:.3f} deg<br>"
    f"**paper raw left shoulder pitch:** {frame.paper_raw_left_shoulder_pitch_deg:.3f} deg<br>"
    f"**displayed paper pitch:** {frame.displayed_left_shoulder_pitch_deg:.3f} deg "
    f"(+ {frame.display_offset_k_360} * 360 deg)<br>"
    f"**displayed-paper minus seed:** "
    f"{frame.displayed_left_shoulder_pitch_deg - frame.seed_left_shoulder_pitch_deg:.3f} deg<br><br>"
    "**Note:** adding +/-360 to a hinge joint does not change the rendered MuJoCo pose; "
    "it only changes the printed angle convention."
  )


def comparison_offsets(spacing: float = 0.95) -> dict[str, np.ndarray]:
  return {
    "bvh": np.array([0.0, -spacing, 0.0]),
    "seed": np.array([0.0, 0.0, 0.0]),
    "paper": np.array([0.0, spacing, 0.0]),
  }


def next_frame_index(current_frame: int, frame_indices: Sequence[int]) -> int:
  ordered = sorted(int(frame_index) for frame_index in frame_indices)
  if not ordered:
    raise ValueError("frame_indices must not be empty")
  for frame_index in ordered:
    if frame_index > int(current_frame):
      return frame_index
  return ordered[0]


def _bvh_full_origin(target: FullBodyTarget) -> np.ndarray:
  return 0.5 * (target.upper.left_arm.shoulder + target.upper.right_arm.shoulder)


def _bvh_display_origin(target: FullBodyTarget, display_mode: str) -> np.ndarray:
  if display_mode == BODY_CENTRIC_DISPLAY_MODE:
    return _bvh_full_origin(target)
  if display_mode == WORLD_DISPLAY_MODE:
    return _bvh_hip_center(target)
  raise ValueError(f"display_mode must be one of {DISPLAY_MODES}, got {display_mode!r}")


def _bvh_hip_center(target: FullBodyTarget) -> np.ndarray:
  return 0.5 * (target.lower.left_leg.hip + target.lower.right_leg.hip)


def _project_bvh_up_axis(candidate: np.ndarray, left_axis: np.ndarray) -> np.ndarray:
  for axis in (candidate, np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0])):
    projected = axis - left_axis * float(np.dot(axis, left_axis))
    norm = float(np.linalg.norm(projected))
    if norm > 1e-12:
      return projected / norm
  raise ValueError("Cannot construct a BVH body frame from collinear hip and chest keypoints")


def _bvh_body_frame(target: FullBodyTarget) -> np.ndarray:
  left_hip = target.lower.left_leg.hip
  right_hip = target.lower.right_leg.hip
  left_axis = normalize(left_hip - right_hip)
  up_axis = _project_bvh_up_axis(target.upper.chest_position - _bvh_hip_center(target), left_axis)
  forward_axis = normalize(np.cross(left_axis, up_axis))
  up_axis = normalize(np.cross(forward_axis, left_axis))
  return np.column_stack([forward_axis, left_axis, up_axis])


def _bvh_display_point_transform(
  target: FullBodyTarget,
  offset: np.ndarray,
  *,
  display_mode: str,
  seed_motion_row: Sequence[float] | None,
):
  if display_mode == BODY_CENTRIC_DISPLAY_MODE:
    origin = _bvh_full_origin(target)

    def transform(point: np.ndarray) -> np.ndarray:
      return np.asarray(point, dtype=float) - origin + offset

    return transform
  if display_mode == WORLD_DISPLAY_MODE:
    if seed_motion_row is None:
      raise ValueError("seed_motion_row is required for BVH world display mode")
    origin = _bvh_hip_center(target)
    body_frame = _bvh_body_frame(target)
    root_position = np.asarray(seed_motion_row[:3], dtype=float)
    root_orientation = _rotation_matrix_from_seed_motion_row(seed_motion_row)

    def transform(point: np.ndarray) -> np.ndarray:
      local = body_frame.T @ (np.asarray(point, dtype=float) - origin)
      return root_position + root_orientation @ local + offset

    return transform
  raise ValueError(f"display_mode must be one of {DISPLAY_MODES}, got {display_mode!r}")


def _full_bvh_segments(
  target: FullBodyTarget,
  offset: np.ndarray,
  *,
  display_mode: str,
  seed_motion_row: Sequence[float] | None,
) -> np.ndarray:
  p = _bvh_display_point_transform(
    target,
    offset,
    display_mode=display_mode,
    seed_motion_row=seed_motion_row,
  )

  upper = target.upper
  lower = target.lower
  hip_center = 0.5 * (lower.left_leg.hip + lower.right_leg.hip)
  segments = [
    (upper.left_arm.shoulder, upper.right_arm.shoulder),
    (upper.chest_position, upper.left_arm.shoulder),
    (upper.chest_position, upper.right_arm.shoulder),
    (upper.left_arm.shoulder, upper.left_arm.elbow),
    (upper.left_arm.elbow, upper.left_arm.wrist),
    (upper.right_arm.shoulder, upper.right_arm.elbow),
    (upper.right_arm.elbow, upper.right_arm.wrist),
    (upper.chest_position, hip_center),
    (lower.left_leg.hip, lower.right_leg.hip),
    (lower.left_leg.hip, lower.left_leg.knee),
    (lower.left_leg.knee, lower.left_leg.ankle),
    (lower.right_leg.hip, lower.right_leg.knee),
    (lower.right_leg.knee, lower.right_leg.ankle),
  ]
  return np.asarray([[p(start), p(end)] for start, end in segments], dtype=float)


def _full_bvh_points(
  target: FullBodyTarget,
  offset: np.ndarray,
  *,
  display_mode: str,
  seed_motion_row: Sequence[float] | None,
) -> np.ndarray:
  transform = _bvh_display_point_transform(
    target,
    offset,
    display_mode=display_mode,
    seed_motion_row=seed_motion_row,
  )
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
  return np.asarray([transform(point) for point in points], dtype=float)


def _full_bvh_axis_segments(
  target: FullBodyTarget,
  offset: np.ndarray,
  *,
  display_mode: str,
  axis_length: float,
  seed_motion_row: Sequence[float] | None,
) -> np.ndarray:
  p = _bvh_display_point_transform(
    target,
    offset,
    display_mode=display_mode,
    seed_motion_row=seed_motion_row,
  )

  upper = target.upper
  left_upper = normalize(upper.left_arm.elbow - upper.left_arm.shoulder)
  left_lower = normalize(upper.left_arm.wrist - upper.left_arm.elbow)
  right_upper = normalize(upper.right_arm.elbow - upper.right_arm.shoulder)
  right_lower = normalize(upper.right_arm.wrist - upper.right_arm.elbow)
  starts_and_axes = [
    (upper.left_arm.shoulder, left_upper),
    (upper.left_arm.elbow, left_lower),
    (upper.right_arm.shoulder, right_upper),
    (upper.right_arm.elbow, right_lower),
  ]
  return np.asarray(
    [[p(start), p(start + axis * axis_length)] for start, axis in starts_and_axes],
    dtype=float,
  )


def _bvh_display_snapshot(
  target: FullBodyTarget,
  offset: np.ndarray,
  *,
  display_mode: str,
  seed_motion_row: Sequence[float] | None = None,
) -> BvhDisplaySnapshot:
  return BvhDisplaySnapshot(
    skeleton_segments=_full_bvh_segments(
      target,
      offset,
      display_mode=display_mode,
      seed_motion_row=seed_motion_row,
    ),
    axis_segments=_full_bvh_axis_segments(
      target,
      offset,
      display_mode=display_mode,
      axis_length=0.18,
      seed_motion_row=seed_motion_row,
    ),
    keypoints=_full_bvh_points(
      target,
      offset,
      display_mode=display_mode,
      seed_motion_row=seed_motion_row,
    ),
  )


def _run_viser(frames: Sequence[ComparisonFrame], config: VisualizerConfig) -> None:
  import viser

  server = viser.ViserServer(port=config.port)
  server.scene.add_grid("/grid", width=3.5, height=2.0, cell_size=0.1)
  frame_by_index = {frame.frame_index: frame for frame in frames}
  frame_indices = list(frame_by_index)
  seed_context = _seed_g1_model_context()
  paper_context = _seed_g1_model_context()
  offsets = comparison_offsets()
  geom_mesh_cache: dict[int, MeshSnapshot] = {}
  last_rendered_frame: int | None = None

  with server.gui.add_folder("Playback"):
    frame_slider = server.gui.add_slider(
      "Frame",
      min=min(frame_indices),
      max=max(frame_indices),
      step=1,
      initial_value=min(frame_indices),
    )
    play_checkbox = server.gui.add_checkbox("Play", initial_value=False)
    fps_slider = server.gui.add_slider("FPS", min=1.0, max=60.0, step=1.0, initial_value=config.fps)
    info_markdown = server.gui.add_markdown("")

  def geom_mesh(geom_id: int) -> MeshSnapshot:
    if geom_id not in geom_mesh_cache:
      geom_mesh_cache[geom_id] = _geom_mesh_snapshot(seed_context[0], geom_id)
    return geom_mesh_cache[geom_id]

  def add_articulated_robot(
    name: str,
    context,
    row: Sequence[float],
    offset: np.ndarray,
    color: tuple[int, int, int],
  ) -> ArticulatedRobotHandles:
    model, data, joint_qpos_addresses, visual_geom_ids = context
    pose = _articulated_geom_pose_snapshot(
      model,
      data,
      joint_qpos_addresses,
      visual_geom_ids,
      row,
      offset=offset,
      display_mode=config.display_mode,
    )
    mesh_handles = []
    for pose_index, geom_id in enumerate(pose.geom_ids):
      mesh = geom_mesh(geom_id)
      mesh_handles.append(
        server.scene.add_mesh_simple(
          f"/{name}/geom_{geom_id}",
          vertices=mesh.vertices,
          faces=mesh.faces,
          color=color,
          opacity=0.65,
          flat_shading=False,
          side="double",
          position=pose.positions[pose_index],
          wxyz=pose.wxyzs[pose_index],
        )
      )
    return ArticulatedRobotHandles(
      geom_ids=pose.geom_ids,
      mesh_handles=tuple(mesh_handles),
    )

  def update_articulated_robot(
    handles: ArticulatedRobotHandles,
    context,
    row: Sequence[float],
    offset: np.ndarray,
  ) -> None:
    model, data, joint_qpos_addresses, _visual_geom_ids = context
    pose = _articulated_geom_pose_snapshot(
      model,
      data,
      joint_qpos_addresses,
      handles.geom_ids,
      row,
      offset=offset,
      display_mode=config.display_mode,
    )
    for handle, position, wxyz in zip(handles.mesh_handles, pose.positions, pose.wxyzs):
      handle.position = position
      handle.wxyz = wxyz

  def add_bvh_target(
    name: str,
    target: FullBodyTarget,
    row: Sequence[float],
    offset: np.ndarray,
  ) -> BvhDisplayHandles:
    snapshot = _bvh_display_snapshot(
      target,
      offset,
      display_mode=config.display_mode,
      seed_motion_row=row,
    )
    skeleton_handle = server.scene.add_line_segments(
        f"/{name}/skeleton",
        points=snapshot.skeleton_segments,
        colors=np.asarray((242, 143, 52), dtype=np.uint8),
        line_width=4.5,
      )
    axes_handle = server.scene.add_line_segments(
        f"/{name}/axes",
        points=snapshot.axis_segments,
        colors=np.asarray(
          [
            [[240, 70, 70], [240, 70, 70]],
            [[120, 90, 255], [120, 90, 255]],
            [[240, 70, 70], [240, 70, 70]],
            [[120, 90, 255], [120, 90, 255]],
          ],
          dtype=np.uint8,
        ),
        line_width=6.0,
      )
    keypoints_handle = server.scene.add_point_cloud(
        f"/{name}/keypoints",
        points=snapshot.keypoints,
        colors=np.asarray((242, 143, 52), dtype=np.uint8),
        point_size=0.035,
        point_shape="circle",
      )
    return BvhDisplayHandles(
      skeleton_handle=skeleton_handle,
      axes_handle=axes_handle,
      keypoints_handle=keypoints_handle,
    )

  def update_bvh_target(
    handles: BvhDisplayHandles,
    target: FullBodyTarget,
    row: Sequence[float],
    offset: np.ndarray,
  ) -> None:
    snapshot = _bvh_display_snapshot(
      target,
      offset,
      display_mode=config.display_mode,
      seed_motion_row=row,
    )
    handles.skeleton_handle.points = snapshot.skeleton_segments
    handles.axes_handle.points = snapshot.axis_segments
    handles.keypoints_handle.points = snapshot.keypoints

  initial_frame = frame_by_index[min(frame_indices)]
  bvh_display = add_bvh_target(
    "BVH full skeleton",
    initial_frame.bvh_full_target,
    initial_frame.seed_motion_row,
    offsets["bvh"],
  )
  seed_robot = add_articulated_robot(
    "seed G1 CSV",
    seed_context,
    initial_frame.seed_motion_row,
    offsets["seed"],
    (80, 190, 115),
  )
  paper_robot = add_articulated_robot(
    "paper raw upper on seed lower",
    paper_context,
    initial_frame.paper_raw_motion_row,
    offsets["paper"],
    (80, 145, 245),
  )

  def update_frame(frame_index: int) -> None:
    nonlocal last_rendered_frame
    frame_index = int(frame_index)
    if last_rendered_frame == frame_index:
      return
    frame = frame_by_index[int(frame_index)]
    update_bvh_target(bvh_display, frame.bvh_full_target, frame.seed_motion_row, offsets["bvh"])
    update_articulated_robot(seed_robot, seed_context, frame.seed_motion_row, offsets["seed"])
    update_articulated_robot(
      paper_robot,
      paper_context,
      frame.paper_raw_motion_row,
      offsets["paper"],
    )
    info_markdown.content = _format_info(frame, display_mode=config.display_mode)
    last_rendered_frame = frame_index

  @frame_slider.on_update
  def _(event) -> None:
    update_frame(int(event.target.value))

  update_frame(min(frame_indices))
  print(f"Viser seed-vs-paper-raw G1 viewer running on http://localhost:{config.port}")
  while True:
    if play_checkbox.value:
      frame_slider.value = next_frame_index(int(frame_slider.value), frame_indices)
      time.sleep(1.0 / max(float(fps_slider.value), 1e-6))
    else:
      time.sleep(0.1)


def build_arg_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
    description=(
      "Visualize seed G1 CSV against paper_v1 raw upper-body retargeting, "
      "with the BVH full-body skeleton beside them."
    )
  )
  parser.add_argument("--bvh", required=True, type=Path, help="Input SOMA/bones-seed BVH file.")
  parser.add_argument(
    "--seed-csv",
    type=Path,
    default=None,
    help="Input bones-seed G1 CSV file. Defaults to the matching g1/csv path inferred from --bvh.",
  )
  parser.add_argument("--start-frame", type=int, default=0, help="First frame to show.")
  parser.add_argument(
    "--end-frame",
    type=int,
    default=None,
    help="Last frame to show, inclusive. Defaults to the full available motion.",
  )
  parser.add_argument("--port", type=int, default=8090, help="Viser web server port.")
  parser.add_argument("--fps", type=float, default=10.0, help="Autoplay FPS.")
  parser.add_argument(
    "--display-mode",
    choices=DISPLAY_MODES,
    default=BODY_CENTRIC_DISPLAY_MODE,
    help="body-centric keeps poses centered for comparison; world preserves BVH/CSV root translation.",
  )
  parser.add_argument(
    "--paper-raw-offset-to-seed",
    action="store_true",
    help="Display paper raw left_shoulder_pitch after adding k*360 to the seed angle neighborhood.",
  )
  parser.add_argument(
    "--global-root",
    action="store_true",
    help="Deprecated alias for --display-mode world.",
  )
  parser.add_argument("--raw-orientations", action="store_true", help="Disable SOMA-to-G1 orientation offsets.")
  parser.add_argument("--raw-upper-arm-axes", action="store_true", help="Disable upper-arm G1 convention flip.")
  parser.add_argument("--keep-global-heading", action="store_true", help="Keep the BVH initial heading.")
  return parser


def display_mode_from_args(args: argparse.Namespace) -> str:
  if getattr(args, "global_root", False):
    return WORLD_DISPLAY_MODE
  return str(args.display_mode)


def main(argv: Sequence[str] | None = None) -> int:
  args = build_arg_parser().parse_args(argv)
  config = VisualizerConfig(
    bvh_path=args.bvh,
    seed_csv_path=args.seed_csv,
    start_frame=args.start_frame,
    end_frame=args.end_frame,
    port=args.port,
    fps=args.fps,
    display_mode=display_mode_from_args(args),
    paper_raw_offset_to_seed=args.paper_raw_offset_to_seed,
    apply_orientation_offsets=not args.raw_orientations,
    align_upper_arm_axes_to_g1=not args.raw_upper_arm_axes,
    remove_initial_heading=not args.keep_global_heading,
  )
  frames = build_comparison_frames(config)
  _run_viser(frames, config)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
