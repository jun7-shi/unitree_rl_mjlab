import unittest

import torch

from src.evaluation.silent_walking.regional_contact import RegionalContactEvent
from src.evaluation.silent_walking.telemetry import (
  QuietTelemetrySnapshot,
  TouchdownEvent,
)
from src.evaluation.silent_walking.viewer_overlay import (
  format_quiet_html,
  format_quiet_native_rows,
)


class ViewerOverlayFormattingTests(unittest.TestCase):
  def test_format_quiet_native_rows_includes_live_vz_loading_and_touchdown(self):
    event = TouchdownEvent(
      step=7,
      foot_index=0,
      foot_name="left",
      peak_force_n=torch.tensor(120.0),
      loading_rate_n_s=torch.tensor(800.0),
      vertical_speed_m_s=torch.tensor(0.3),
      heel_vz_m_s=torch.tensor(-0.2),
      toe_vz_m_s=torch.tensor(-0.1),
      roll_angle_rad=torch.tensor(0.15),
      first_contact_region="heel",
    )
    region_event = RegionalContactEvent(
      step=8,
      foot_index=0,
      foot_name="left",
      event_type="secondary",
      regions=("toe",),
      peak_force_n=torch.tensor(150.0),
      loading_rate_n_s=torch.tensor(900.0),
      vertical_speed_m_s=torch.tensor(0.1),
      corner_downward_speed_m_s=torch.tensor(0.4),
    )
    snapshot = QuietTelemetrySnapshot(
      step=8,
      foot_names=("left", "right"),
      foot_contact=torch.tensor([1.0, 0.0]),
      foot_force_n=torch.tensor([100.0, 0.0]),
      foot_loading_rate_n_s=torch.tensor([500.0, 0.0]),
      foot_site_vz_m_s=torch.tensor([-0.3, 0.1]),
      heel_vz_m_s=torch.tensor([-0.2, 0.1]),
      toe_vz_m_s=torch.tensor([-0.1, 0.2]),
      foot_roll_angle_rad=torch.tensor([0.15, -0.05]),
      last_touchdown=event,
      foot_region_names=("heel", "midfoot", "toe"),
      foot_region_contact=torch.tensor([[1.0, 0.0, 1.0], [0.0, 0.0, 0.0]]),
      foot_corner_names=("rear_left", "rear_right", "front_left", "front_right"),
      foot_corner_vz_m_s=torch.tensor([[-0.3, -0.2, 0.1, 0.2], [0.0, 0.0, 0.0, 0.0]]),
      last_region_event=region_event,
    )

    labels, values = format_quiet_native_rows(snapshot, body_weight_newton=400.0)

    self.assertIn("Quiet left", labels)
    self.assertIn("F 0.25BW", values)
    self.assertIn("Vz -0.30", values)
    self.assertIn("R heel+toe", values)
    self.assertIn("C -0.30/-0.20/0.10/0.20", values)
    self.assertIn("Last TD", labels)
    self.assertIn("heel", values)
    self.assertIn("Last Region", labels)
    self.assertIn("secondary", values)

  def test_format_quiet_html_escapes_and_renders_empty_state(self):
    self.assertIn("waiting", format_quiet_html(None, body_weight_newton=400.0))


if __name__ == "__main__":
  unittest.main()
