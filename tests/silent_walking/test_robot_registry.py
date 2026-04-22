from pathlib import Path

from src.evaluation.silent_walking import get_robot_spec, list_supported_robots


def test_list_supported_robots_returns_expected_order():
  assert list_supported_robots() == ("g1", "bumi")


def test_g1_robot_spec_has_expected_metadata():
  spec = get_robot_spec("g1")

  assert spec.task_id == "Unitree-G1-Flat"
  assert spec.foot_site_names == ("left_foot", "right_foot")
  assert "left_foot1_collision" in spec.foot_collision_geom_names


def test_bumi_robot_spec_exists_before_assets():
  spec = get_robot_spec("bumi")

  assert spec.asset_status in {"missing_assets", "ready"}
  assert spec.mass_normalization > 0


def test_asset_status_tracks_current_filesystem_state(monkeypatch):
  spec = get_robot_spec("bumi")
  original_exists = Path.exists
  state = {"exists": False}

  def fake_exists(self: Path) -> bool:
    if self == spec.asset_path:
      return state["exists"]
    return original_exists(self)

  monkeypatch.setattr(Path, "exists", fake_exists)

  assert spec.asset_status == "missing_assets"
  state["exists"] = True
  assert spec.asset_status == "ready"
