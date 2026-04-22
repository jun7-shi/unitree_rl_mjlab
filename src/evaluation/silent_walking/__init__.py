"""Silent walking robot metadata."""

from .robots import ROBOT_REGISTRY, get_robot_spec, list_supported_robots
from .types import RobotSpec

__all__ = [
  "ROBOT_REGISTRY",
  "RobotSpec",
  "get_robot_spec",
  "list_supported_robots",
]
