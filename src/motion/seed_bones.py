from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Sequence

G1_29DOF_JOINT_COLUMNS = [
  "left_hip_pitch_joint_dof",
  "left_hip_roll_joint_dof",
  "left_hip_yaw_joint_dof",
  "left_knee_joint_dof",
  "left_ankle_pitch_joint_dof",
  "left_ankle_roll_joint_dof",
  "right_hip_pitch_joint_dof",
  "right_hip_roll_joint_dof",
  "right_hip_yaw_joint_dof",
  "right_knee_joint_dof",
  "right_ankle_pitch_joint_dof",
  "right_ankle_roll_joint_dof",
  "waist_yaw_joint_dof",
  "waist_roll_joint_dof",
  "waist_pitch_joint_dof",
  "left_shoulder_pitch_joint_dof",
  "left_shoulder_roll_joint_dof",
  "left_shoulder_yaw_joint_dof",
  "left_elbow_joint_dof",
  "left_wrist_roll_joint_dof",
  "left_wrist_pitch_joint_dof",
  "left_wrist_yaw_joint_dof",
  "right_shoulder_pitch_joint_dof",
  "right_shoulder_roll_joint_dof",
  "right_shoulder_yaw_joint_dof",
  "right_elbow_joint_dof",
  "right_wrist_roll_joint_dof",
  "right_wrist_pitch_joint_dof",
  "right_wrist_yaw_joint_dof",
]

ROOT_TRANSLATION_COLUMNS = [
  "root_translateX",
  "root_translateY",
  "root_translateZ",
]
ROOT_ROTATION_COLUMNS = [
  "root_rotateX",
  "root_rotateY",
  "root_rotateZ",
]
REQUIRED_COLUMNS = ["Frame", *ROOT_TRANSLATION_COLUMNS, *ROOT_ROTATION_COLUMNS, *G1_29DOF_JOINT_COLUMNS]

# seed-bones G1 CSV root Euler axes need a basis change before being interpreted
# as mjlab/MuJoCo world rotations. Root translation, however, is already aligned
# with the replay world and should only be scaled into meters.
DATASET_TO_MJLAB_ROTATION_FRAME = (
  (0.0, 1.0, 0.0),
  (-1.0, 0.0, 0.0),
  (0.0, 0.0, 1.0),
)


def _quat_multiply(lhs: tuple[float, float, float, float], rhs: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
  lx, ly, lz, lw = lhs
  rx, ry, rz, rw = rhs
  return (
    lw * rx + lx * rw + ly * rz - lz * ry,
    lw * ry - lx * rz + ly * rw + lz * rx,
    lw * rz + lx * ry - ly * rx + lz * rw,
    lw * rw - lx * rx - ly * ry - lz * rz,
  )


def _matmul3(lhs: tuple[tuple[float, float, float], ...], rhs: tuple[tuple[float, float, float], ...]) -> tuple[tuple[float, float, float], ...]:
  return tuple(
    tuple(sum(lhs[row][k] * rhs[k][col] for k in range(3)) for col in range(3))
    for row in range(3)
  )


def _transpose3(matrix: tuple[tuple[float, float, float], ...]) -> tuple[tuple[float, float, float], ...]:
  return tuple(tuple(matrix[col][row] for col in range(3)) for row in range(3))


def _matrix_vector_mul3(
  matrix: tuple[tuple[float, float, float], ...], vector: tuple[float, float, float]
) -> tuple[float, float, float]:
  return tuple(sum(matrix[row][col] * vector[col] for col in range(3)) for row in range(3))


def _rotation_matrix_from_euler_xyz_degrees(
  x_degrees: float, y_degrees: float, z_degrees: float
) -> tuple[tuple[float, float, float], ...]:
  x_radians = math.radians(x_degrees)
  y_radians = math.radians(y_degrees)
  z_radians = math.radians(z_degrees)
  cx = math.cos(x_radians)
  sx = math.sin(x_radians)
  cy = math.cos(y_radians)
  sy = math.sin(y_radians)
  cz = math.cos(z_radians)
  sz = math.sin(z_radians)

  rx = (
    (1.0, 0.0, 0.0),
    (0.0, cx, -sx),
    (0.0, sx, cx),
  )
  ry = (
    (cy, 0.0, sy),
    (0.0, 1.0, 0.0),
    (-sy, 0.0, cy),
  )
  rz = (
    (cz, -sz, 0.0),
    (sz, cz, 0.0),
    (0.0, 0.0, 1.0),
  )
  return _matmul3(_matmul3(rx, ry), rz)


def _quat_xyzw_from_rotation_matrix(
  rotation: tuple[tuple[float, float, float], ...]
) -> tuple[float, float, float, float]:
  m00, m01, m02 = rotation[0]
  m10, m11, m12 = rotation[1]
  m20, m21, m22 = rotation[2]
  trace = m00 + m11 + m22

  if trace > 0.0:
    s = math.sqrt(trace + 1.0) * 2.0
    qw = 0.25 * s
    qx = (m21 - m12) / s
    qy = (m02 - m20) / s
    qz = (m10 - m01) / s
  elif m00 > m11 and m00 > m22:
    s = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
    qw = (m21 - m12) / s
    qx = 0.25 * s
    qy = (m01 + m10) / s
    qz = (m02 + m20) / s
  elif m11 > m22:
    s = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
    qw = (m02 - m20) / s
    qx = (m01 + m10) / s
    qy = 0.25 * s
    qz = (m12 + m21) / s
  else:
    s = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
    qw = (m10 - m01) / s
    qx = (m02 + m20) / s
    qy = (m12 + m21) / s
    qz = 0.25 * s

  quat_xyzw = (qx, qy, qz, qw)
  norm = math.sqrt(sum(value * value for value in quat_xyzw))
  if norm == 0.0:
    raise ValueError("Encountered a zero-length root quaternion after matrix conversion")
  return tuple(value / norm for value in quat_xyzw)


def euler_xyz_degrees_to_quat_xyzw(
  x_degrees: float, y_degrees: float, z_degrees: float
) -> tuple[float, float, float, float]:
  """Convert seed-bones XYZ Euler angles into an mjlab-frame xyzw quaternion."""
  dataset_rotation = _rotation_matrix_from_euler_xyz_degrees(
    x_degrees, y_degrees, z_degrees
  )
  basis_change = DATASET_TO_MJLAB_ROTATION_FRAME
  mjlab_rotation = _matmul3(
    _matmul3(basis_change, dataset_rotation), _transpose3(basis_change)
  )
  return _quat_xyzw_from_rotation_matrix(mjlab_rotation)


def convert_seed_bones_row_to_motion_row(
  row: dict[str, str], translation_scale: float = 0.01
) -> list[float]:
  root_position = [
    float(row[column]) * translation_scale for column in ROOT_TRANSLATION_COLUMNS
  ]
  root_quat_xyzw = list(
    euler_xyz_degrees_to_quat_xyzw(
      float(row["root_rotateX"]),
      float(row["root_rotateY"]),
      float(row["root_rotateZ"]),
    )
  )
  joint_positions = [
    math.radians(float(row[column])) for column in G1_29DOF_JOINT_COLUMNS
  ]
  return [*root_position, *root_quat_xyzw, *joint_positions]


def convert_seed_bones_csv_to_motion_rows(
  input_file: str | Path, translation_scale: float = 0.01
) -> list[list[float]]:
  input_path = Path(input_file)
  with input_path.open(newline="", encoding="utf-8") as handle:
    reader = csv.DictReader(handle)
    if reader.fieldnames is None:
      raise ValueError(f"Seed-bones CSV has no header: {input_path}")
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in reader.fieldnames]
    if missing_columns:
      raise ValueError(
        f"Seed-bones CSV is missing required columns: {', '.join(missing_columns)}"
      )
    return [
      convert_seed_bones_row_to_motion_row(row, translation_scale=translation_scale)
      for row in reader
    ]


def _linear_quantile(values: Sequence[float], quantile: float) -> float:
  if not 0.0 <= quantile <= 1.0:
    raise ValueError(f"quantile must be in [0, 1], got {quantile}")
  if not values:
    raise ValueError("Cannot compute a floor alignment offset from no foot samples")
  sorted_values = sorted(float(value) for value in values)
  if len(sorted_values) == 1:
    return sorted_values[0]
  position = quantile * (len(sorted_values) - 1)
  lower_index = math.floor(position)
  upper_index = math.ceil(position)
  if lower_index == upper_index:
    return sorted_values[lower_index]
  blend = position - lower_index
  return (
    sorted_values[lower_index] * (1.0 - blend)
    + sorted_values[upper_index] * blend
  )


def compute_floor_alignment_offset(
  foot_lower_bounds_m: Sequence[float],
  floor_height_m: float = 0.0,
  clearance_m: float = 0.0,
  quantile: float = 0.0,
) -> float:
  """Return the root-Z offset needed to lift foot lower bounds to the floor."""
  reference_lower_bound = _linear_quantile(foot_lower_bounds_m, quantile)
  return max(0.0, floor_height_m + clearance_m - reference_lower_bound)


def apply_root_z_offset(rows: list[list[float]], offset_m: float) -> list[list[float]]:
  """Return motion rows with only root height shifted by *offset_m*."""
  shifted_rows: list[list[float]] = []
  for row in rows:
    shifted_row = list(row)
    shifted_row[2] += offset_m
    shifted_rows.append(shifted_row)
  return shifted_rows


def write_motion_csv(
  rows: list[list[float]], output_file: str | Path, precision: int = 9
) -> None:
  output_path = Path(output_file)
  with output_path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.writer(handle)
    for row in rows:
      writer.writerow([f"{value:.{precision}f}" for value in row])
