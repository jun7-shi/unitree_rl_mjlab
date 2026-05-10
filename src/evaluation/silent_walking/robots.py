"""Robot registry for silent walking evaluation."""

from dataclasses import dataclass
import xml.etree.ElementTree as ET
from typing import Iterable

from src import SRC_PATH

from .foot_grid import (
  make_capsule_footprint_sample_offsets,
  make_indexed_foot_grid_names,
)
from .types import RobotSpec

_G1_XML = SRC_PATH / "assets" / "robots" / "unitree_g1" / "xmls" / "g1.xml"
_BUMI_XML = SRC_PATH / "assets" / "robots" / "unitree_bumi" / "xmls" / "bumi.xml"


def _mass_normalization_from_xml(xml_path, fallback: float = 1.0) -> float:
  if not xml_path.exists():
    return fallback

  root = ET.parse(xml_path).getroot()
  total_mass = sum(
    float(node.attrib["mass"])
    for node in root.iter("inertial")
    if "mass" in node.attrib
  )
  if total_mass <= 0:
    return fallback
  return total_mass * 9.81


def _foot_collision_geom_names() -> tuple[str, ...]:
  return tuple(
    f"{side}_foot{idx}_collision"
    for side in ("left", "right")
    for idx in range(1, 8)
  )


def _resolve_default_geom_size(root: ET.Element, class_name: str) -> float | None:
  for default in root.iter("default"):
    if default.attrib.get("class") != class_name:
      continue
    geom = default.find("geom")
    if geom is None:
      continue
    size_text = geom.attrib.get("size")
    if not size_text:
      continue
    return float(size_text.split()[0])
  return None


def _parse_fromto_xy(value: str) -> tuple[tuple[float, float], tuple[float, float]]:
  coords = [float(entry) for entry in value.split()]
  if len(coords) != 6:
    raise ValueError(f"Expected 6 fromto coordinates, got {len(coords)}")
  return ((coords[0], coords[1]), (coords[3], coords[4]))


def _foot_collision_geometry_from_xml(
  xml_path,
  geom_names: Iterable[str],
) -> tuple[
  tuple[tuple[tuple[float, float], tuple[float, float]], ...],
  tuple[float, ...],
]:
  if not xml_path.exists():
    return (), ()

  root = ET.parse(xml_path).getroot()
  default_radius = _resolve_default_geom_size(root, "foot_capsule")
  geometry: dict[str, tuple[tuple[tuple[float, float], tuple[float, float]], float]] = {}
  for geom in root.iter("geom"):
    name = geom.attrib.get("name")
    if not name or name not in geom_names:
      continue
    fromto_text = geom.attrib.get("fromto")
    if fromto_text is None:
      continue
    size_text = geom.attrib.get("size")
    radius = float(size_text.split()[0]) if size_text else default_radius
    if radius is None:
      raise ValueError(f"Missing size for foot collision geom '{name}'")
    geometry[name] = (_parse_fromto_xy(fromto_text), radius)

  ordered_names = tuple(geom_names)
  if any(name not in geometry for name in ordered_names):
    missing = [name for name in ordered_names if name not in geometry]
    raise ValueError(f"Missing foot collision geometry in XML: {missing}")

  fromto_xy = tuple(geometry[name][0] for name in ordered_names)
  radii = tuple(geometry[name][1] for name in ordered_names)
  return fromto_xy, radii


_FOOT_COLLISION_GEOM_NAMES = _foot_collision_geom_names()
_G1_FOOT_COLLISION_FROMTO_XY_M, _G1_FOOT_COLLISION_RADIUS_M = _foot_collision_geometry_from_xml(
  _G1_XML,
  _FOOT_COLLISION_GEOM_NAMES,
)
_G1_FOOT_GRID_SHAPE = (0, 0)
_G1_FOOT_GRID_OFFSETS_M = tuple(
  tuple(float(value) for value in row)
  for row in make_capsule_footprint_sample_offsets(
    capsule_fromto_xy_m=_G1_FOOT_COLLISION_FROMTO_XY_M[:7],
    z_m=-0.025,
    target_count=30,
  ).tolist()
)


@dataclass(frozen=True, slots=True)
class FootProxySpec:
  """Robot-specific foot proxy points used by silent walking telemetry."""

  robot_name: str
  foot_names: tuple[str, ...]
  foot_site_names: tuple[str, ...]
  foot_body_names: tuple[str, ...]
  foot_collision_geom_names: tuple[str, ...]
  heel_local_offset_m: tuple[float, float, float]
  toe_local_offset_m: tuple[float, float, float]
  foot_corner_names: tuple[str, ...]
  foot_corner_local_offsets_m: tuple[tuple[float, float, float], ...]
  foot_grid_names: tuple[str, ...]
  foot_grid_shape: tuple[int, int]
  foot_grid_local_offsets_m: tuple[tuple[float, float, float], ...]
  heel_region_max_x_m: float
  toe_region_min_x_m: float
  contact_force_threshold_n: float = 1.0
  touchdown_window_steps: int = 5
  rolling_window_steps: int = 150


ROBOT_REGISTRY: dict[str, RobotSpec] = {
  "g1": RobotSpec(
    name="g1",
    task_id="Unitree-G1-Flat",
    action_dim=29,
    foot_site_names=("left_foot", "right_foot"),
    foot_collision_geom_names=_FOOT_COLLISION_GEOM_NAMES,
    foot_collision_fromto_xy_m=_G1_FOOT_COLLISION_FROMTO_XY_M,
    foot_collision_radius_m=_G1_FOOT_COLLISION_RADIUS_M,
    asset_path=_G1_XML,
    mass_normalization=_mass_normalization_from_xml(_G1_XML),
  ),
  "bumi": RobotSpec(
    name="bumi",
    task_id="Unitree-Bumi-Flat",
    action_dim=12,
    foot_site_names=("left_foot", "right_foot"),
    foot_collision_geom_names=(),
    foot_collision_fromto_xy_m=(),
    foot_collision_radius_m=(),
    asset_path=_BUMI_XML,
    mass_normalization=1.0,
  ),
}

FOOT_PROXY_REGISTRY: dict[str, FootProxySpec] = {
  "g1": FootProxySpec(
    robot_name="g1",
    foot_names=("left", "right"),
    foot_site_names=("left_foot", "right_foot"),
    foot_body_names=("left_ankle_roll_link", "right_ankle_roll_link"),
    foot_collision_geom_names=_FOOT_COLLISION_GEOM_NAMES,
    heel_local_offset_m=(-0.055, 0.0, -0.025),
    toe_local_offset_m=(0.13, 0.0, -0.025),
    # These four analysis points come from scene_g1.xml, not from the default
    # G1 asset XML. They are virtual telemetry points, not collision geoms.
    foot_corner_names=("rear_left", "rear_right", "front_left", "front_right"),
    foot_corner_local_offsets_m=(
      (-0.05, 0.025, -0.03),
      (-0.05, -0.025, -0.03),
      (0.12, 0.03, -0.03),
      (0.12, -0.03, -0.03),
    ),
    foot_grid_names=make_indexed_foot_grid_names(len(_G1_FOOT_GRID_OFFSETS_M)),
    foot_grid_shape=_G1_FOOT_GRID_SHAPE,
    foot_grid_local_offsets_m=_G1_FOOT_GRID_OFFSETS_M,
    heel_region_max_x_m=-0.02,
    toe_region_min_x_m=0.09,
  ),
  "bumi": FootProxySpec(
    robot_name="bumi",
    foot_names=("left", "right"),
    foot_site_names=("left_foot", "right_foot"),
    foot_body_names=("left_ankle_roll_link", "right_ankle_roll_link"),
    foot_collision_geom_names=(),
    heel_local_offset_m=(-0.055, 0.0, -0.025),
    toe_local_offset_m=(0.13, 0.0, -0.025),
    foot_corner_names=("rear_left", "rear_right", "front_left", "front_right"),
    foot_corner_local_offsets_m=(
      (-0.05, 0.025, -0.03),
      (-0.05, -0.025, -0.03),
      (0.12, 0.03, -0.03),
      (0.12, -0.03, -0.03),
    ),
    foot_grid_names=(),
    foot_grid_shape=(0, 0),
    foot_grid_local_offsets_m=(),
    heel_region_max_x_m=-0.02,
    toe_region_min_x_m=0.09,
  ),
}


def list_supported_robots() -> tuple[str, ...]:
  """Return the supported robot names in registry order."""

  return tuple(ROBOT_REGISTRY)


def get_robot_spec(robot_name: str) -> RobotSpec:
  """Return the metadata for a supported robot."""

  return ROBOT_REGISTRY[robot_name]


def get_foot_proxy_spec(robot_name: str) -> FootProxySpec:
  """Return foot proxy metadata for silent walking telemetry."""

  return FOOT_PROXY_REGISTRY[robot_name]
