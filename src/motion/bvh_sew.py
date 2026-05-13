from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from src.motion.sew_mimic import ArmKeypointTarget, rotation_about_axis


@dataclass
class BVHJoint:
  name: str
  parent: str | None
  offset: np.ndarray = field(default_factory=lambda: np.zeros(3))
  channels: tuple[str, ...] = ()
  channel_start: int = 0
  children: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class BVHFramePose:
  positions: dict[str, np.ndarray]
  rotations: dict[str, np.ndarray]


@dataclass(frozen=True)
class BVHMotion:
  joints: dict[str, BVHJoint]
  root_name: str
  joint_order: tuple[str, ...]
  frames: np.ndarray
  frame_time: float
  scale: float = 1.0

  def frame_pose(self, frame_index: int, coordinate_rotation: np.ndarray | None = None) -> BVHFramePose:
    if frame_index < 0 or frame_index >= len(self.frames):
      raise IndexError(f"frame_index {frame_index} out of range for {len(self.frames)} frames")
    frame_values = self.frames[frame_index]
    positions: dict[str, np.ndarray] = {}
    rotations: dict[str, np.ndarray] = {}
    self._compute_joint_pose(self.root_name, frame_values, np.zeros(3), np.eye(3), positions, rotations)
    if coordinate_rotation is not None:
      transform = np.asarray(coordinate_rotation, dtype=float).reshape(3, 3)
      positions = {name: transform @ position for name, position in positions.items()}
      rotations = {name: transform @ rotation for name, rotation in rotations.items()}
    return BVHFramePose(positions=positions, rotations=rotations)

  def _compute_joint_pose(
    self,
    joint_name: str,
    frame_values: np.ndarray,
    parent_position: np.ndarray,
    parent_rotation: np.ndarray,
    positions: dict[str, np.ndarray],
    rotations: dict[str, np.ndarray],
  ) -> None:
    joint = self.joints[joint_name]
    translation = np.asarray(joint.offset, dtype=float) * self.scale
    local_rotation = np.eye(3)

    channel_values = frame_values[joint.channel_start : joint.channel_start + len(joint.channels)]
    for channel, value in zip(joint.channels, channel_values):
      if channel == "Xposition":
        translation[0] += float(value) * self.scale
      elif channel == "Yposition":
        translation[1] += float(value) * self.scale
      elif channel == "Zposition":
        translation[2] += float(value) * self.scale
      elif channel == "Xrotation":
        local_rotation = local_rotation @ rotation_about_axis([1.0, 0.0, 0.0], np.deg2rad(value))
      elif channel == "Yrotation":
        local_rotation = local_rotation @ rotation_about_axis([0.0, 1.0, 0.0], np.deg2rad(value))
      elif channel == "Zrotation":
        local_rotation = local_rotation @ rotation_about_axis([0.0, 0.0, 1.0], np.deg2rad(value))
      else:
        raise ValueError(f"Unsupported BVH channel '{channel}' on joint '{joint_name}'")

    world_position = parent_position + parent_rotation @ translation
    world_rotation = parent_rotation @ local_rotation
    positions[joint_name] = world_position
    rotations[joint_name] = world_rotation
    for child_name in joint.children:
      self._compute_joint_pose(
        child_name,
        frame_values,
        world_position,
        world_rotation,
        positions,
        rotations,
      )


def load_bvh(path: str | Path, scale: float = 1.0) -> BVHMotion:
  """Load a BVH hierarchy and motion frames.

  `scale` converts BVH position units to output units. Bones-seed BVH files use
  centimeter-scale offsets, so callers normally pass `scale=0.01` for meters.
  """
  bvh_path = Path(path)
  lines = bvh_path.read_text(encoding="utf-8").splitlines()
  try:
    motion_index = next(index for index, line in enumerate(lines) if line.strip() == "MOTION")
  except StopIteration as exc:
    raise ValueError(f"BVH file has no MOTION section: {bvh_path}") from exc

  joints, root_name, joint_order, channel_count = _parse_hierarchy(lines[:motion_index])
  frames, frame_time = _parse_motion(lines[motion_index + 1 :], channel_count)
  return BVHMotion(
    joints=joints,
    root_name=root_name,
    joint_order=tuple(joint_order),
    frames=frames,
    frame_time=frame_time,
    scale=scale,
  )


def soma_mujoco_to_mjlab_rotation() -> np.ndarray:
  """Return the SOMA/Mujoco-source rotation used by soma-retargeter.

  This matches `SpaceConverter(FacingDirectionType.MUJOCO)`: a +90 degree
  rotation about source X, mapping BVH Y-up coordinates into mjlab Z-up.
  """
  return np.array(
    [
      [1.0, 0.0, 0.0],
      [0.0, 0.0, -1.0],
      [0.0, 1.0, 0.0],
    ],
    dtype=float,
  )


def load_bvh_arm_targets(
  path: str | Path,
  *,
  side: str,
  scale: float = 0.01,
  frame_slice: slice | None = None,
  coordinate_rotation: np.ndarray | None = None,
  joint_names: dict[str, str] | None = None,
) -> list[ArmKeypointTarget]:
  """Extract SEW-Mimic targets from BVH arm keypoints.

  Defaults map bones-seed style names:
  left: LeftArm, LeftForeArm, LeftHand
  right: RightArm, RightForeArm, RightHand
  """
  motion = load_bvh(path, scale=scale)
  names = joint_names or _default_arm_joint_names(side)
  for role in ("shoulder", "elbow", "wrist", "hand"):
    if names[role] not in motion.joints:
      raise ValueError(f"BVH is missing {side} {role} joint '{names[role]}'")

  indices = range(len(motion.frames))[frame_slice or slice(None)]
  targets: list[ArmKeypointTarget] = []
  for frame_index in indices:
    pose = motion.frame_pose(frame_index, coordinate_rotation=coordinate_rotation)
    targets.append(
      ArmKeypointTarget(
        shoulder=pose.positions[names["shoulder"]],
        elbow=pose.positions[names["elbow"]],
        wrist=pose.positions[names["wrist"]],
        hand_orientation=pose.rotations[names["hand"]],
      )
    )
  return targets


def load_soma_bvh_arm_targets(
  path: str | Path,
  *,
  side: str,
  scale: float = 0.01,
  frame_slice: slice | None = None,
  joint_names: dict[str, str] | None = None,
) -> list[ArmKeypointTarget]:
  """Extract SOMA BVH arm targets in mjlab coordinates."""
  return load_bvh_arm_targets(
    path,
    side=side,
    scale=scale,
    frame_slice=frame_slice,
    coordinate_rotation=soma_mujoco_to_mjlab_rotation(),
    joint_names=joint_names,
  )


def _default_arm_joint_names(side: str) -> dict[str, str]:
  if side == "left":
    prefix = "Left"
  elif side == "right":
    prefix = "Right"
  else:
    raise ValueError("side must be 'left' or 'right'")
  return {
    "shoulder": f"{prefix}Arm",
    "elbow": f"{prefix}ForeArm",
    "wrist": f"{prefix}Hand",
    "hand": f"{prefix}Hand",
  }


def _parse_hierarchy(lines: Sequence[str]) -> tuple[dict[str, BVHJoint], str, list[str], int]:
  joints: dict[str, BVHJoint] = {}
  joint_order: list[str] = []
  stack: list[str] = []
  pending_joint: str | None = None
  current_joint: str | None = None
  root_name: str | None = None
  channel_cursor = 0
  pending_end_site = False
  end_site_depth = 0

  for raw_line in lines:
    line = raw_line.strip()
    if not line or line == "HIERARCHY":
      continue
    parts = line.split()
    keyword = parts[0]

    if end_site_depth:
      if keyword == "{":
        end_site_depth += 1
      elif keyword == "}":
        end_site_depth -= 1
      continue

    if pending_end_site:
      if keyword == "{":
        end_site_depth = 1
        pending_end_site = False
      continue

    if keyword in {"ROOT", "JOINT"}:
      if len(parts) != 2:
        raise ValueError(f"Malformed BVH joint line: {line}")
      name = parts[1]
      parent = stack[-1] if stack else None
      if name in joints:
        raise ValueError(f"Duplicate BVH joint name: {name}")
      joints[name] = BVHJoint(name=name, parent=parent)
      joint_order.append(name)
      if parent is not None:
        joints[parent].children.append(name)
      if keyword == "ROOT":
        root_name = name
      pending_joint = name
      current_joint = name
    elif keyword == "End":
      pending_end_site = True
    elif keyword == "{":
      if pending_joint is not None:
        stack.append(pending_joint)
        current_joint = pending_joint
        pending_joint = None
    elif keyword == "}":
      if stack:
        stack.pop()
      current_joint = stack[-1] if stack else None
    elif keyword == "OFFSET":
      if current_joint is None:
        raise ValueError(f"OFFSET outside a joint block: {line}")
      joints[current_joint].offset = np.array([float(value) for value in parts[1:4]], dtype=float)
    elif keyword == "CHANNELS":
      if current_joint is None:
        raise ValueError(f"CHANNELS outside a joint block: {line}")
      count = int(parts[1])
      channels = tuple(parts[2 : 2 + count])
      if len(channels) != count:
        raise ValueError(f"Malformed CHANNELS line: {line}")
      joint = joints[current_joint]
      joint.channels = channels
      joint.channel_start = channel_cursor
      channel_cursor += count

  if root_name is None:
    raise ValueError("BVH hierarchy has no ROOT joint")
  return joints, root_name, joint_order, channel_cursor


def _parse_motion(lines: Sequence[str], channel_count: int) -> tuple[np.ndarray, float]:
  frame_count: int | None = None
  frame_time: float | None = None
  motion_rows: list[list[float]] = []

  for raw_line in lines:
    line = raw_line.strip()
    if not line:
      continue
    if line.startswith("Frames:"):
      frame_count = int(line.split(":", 1)[1].strip())
    elif line.startswith("Frame Time:"):
      frame_time = float(line.split(":", 1)[1].strip())
    else:
      values = [float(value) for value in line.split()]
      if len(values) != channel_count:
        raise ValueError(f"Motion row has {len(values)} values, expected {channel_count}")
      motion_rows.append(values)

  if frame_count is None:
    raise ValueError("BVH motion section has no Frames line")
  if frame_time is None:
    raise ValueError("BVH motion section has no Frame Time line")
  if len(motion_rows) != frame_count:
    raise ValueError(f"BVH has {len(motion_rows)} motion rows, expected {frame_count}")
  return np.asarray(motion_rows, dtype=float), frame_time
