"""Robot registry for silent walking evaluation."""

import xml.etree.ElementTree as ET

from src import SRC_PATH

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


ROBOT_REGISTRY: dict[str, RobotSpec] = {
  "g1": RobotSpec(
    name="g1",
    task_id="Unitree-G1-Flat",
    action_dim=29,
    foot_site_names=("left_foot", "right_foot"),
    foot_collision_geom_names=_foot_collision_geom_names(),
    asset_path=_G1_XML,
    mass_normalization=_mass_normalization_from_xml(_G1_XML),
  ),
  "bumi": RobotSpec(
    name="bumi",
    task_id="Unitree-Bumi-Flat",
    action_dim=12,
    foot_site_names=("left_foot", "right_foot"),
    foot_collision_geom_names=(),
    asset_path=_BUMI_XML,
    mass_normalization=1.0,
  ),
}


def list_supported_robots() -> tuple[str, ...]:
  """Return the supported robot names in registry order."""

  return tuple(ROBOT_REGISTRY)


def get_robot_spec(robot_name: str) -> RobotSpec:
  """Return the metadata for a supported robot."""

  return ROBOT_REGISTRY[robot_name]
