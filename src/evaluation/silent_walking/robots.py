"""Robot registry for silent walking evaluation."""

from pathlib import Path

from src import SRC_PATH

from .types import RobotSpec

_SUPPORTED_ROBOTS = ("g1", "bumi")

_G1_XML = SRC_PATH / "assets" / "robots" / "unitree_g1" / "xmls" / "g1.xml"
_BUMI_XML = SRC_PATH / "assets" / "robots" / "unitree_bumi" / "xmls" / "bumi.xml"


def _asset_status(xml_path: Path) -> str:
  return "ready" if xml_path.exists() else "missing_assets"


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
    foot_site_names=("left_foot", "right_foot"),
    foot_collision_geom_names=_foot_collision_geom_names(),
    asset_status=_asset_status(_G1_XML),
    mass_normalization=1.0,
  ),
  "bumi": RobotSpec(
    name="bumi",
    task_id="Unitree-Bumi-Flat",
    foot_site_names=("left_foot", "right_foot"),
    foot_collision_geom_names=(),
    asset_status=_asset_status(_BUMI_XML),
    mass_normalization=1.0,
  ),
}


def list_supported_robots() -> tuple[str, ...]:
  """Return the supported robot names in registry order."""

  return _SUPPORTED_ROBOTS


def get_robot_spec(robot_name: str) -> RobotSpec:
  """Return the metadata for a supported robot."""

  return ROBOT_REGISTRY[robot_name]
