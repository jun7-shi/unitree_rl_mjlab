import unittest

import numpy as np

from src.analysis.mimic.gait_metrics import (
  clearance_summary,
  compare_contact_sequences,
  contact_events,
  gait_phase_summary,
)
from src.analysis.mimic.reference import derive_foot_contact_from_height
from src.analysis.mimic.reporting import render_markdown_report


class TestMimicGaitMetrics(unittest.TestCase):
  def test_contact_events_detect_touchdown_and_liftoff(self):
    contact = np.array([False, True, True, False, False, True])
    events = contact_events(contact)
    self.assertEqual(events.touchdown_indices, [1, 5])
    self.assertEqual(events.liftoff_indices, [3])

  def test_phase_summary_counts_single_support(self):
    left = np.array([True, True, False, False, True])
    right = np.array([False, True, True, False, False])
    summary = gait_phase_summary(np.stack([left, right], axis=1), fps=50.0)
    self.assertAlmostEqual(summary.single_support_ratio, 3 / 5)
    self.assertAlmostEqual(summary.double_support_ratio, 1 / 5)
    self.assertAlmostEqual(summary.no_support_ratio, 1 / 5)

  def test_compare_contact_sequences_detects_swing_contact_and_extra_touchdown(self):
    ref = np.array(
      [
        [True, False],
        [True, False],
        [False, True],
        [False, True],
        [True, False],
      ]
    )
    policy = np.array(
      [
        [True, True],
        [True, False],
        [False, True],
        [True, True],
        [True, False],
      ]
    )
    comparison = compare_contact_sequences(ref, policy, fps=50.0)
    self.assertGreater(comparison.swing_contact_ratio, 0.0)
    self.assertEqual(comparison.extra_touchdown_count, 1)
    self.assertLess(comparison.single_support_survival, 1.0)

  def test_clearance_summary_uses_reference_swing_windows(self):
    clearance = np.array(
      [
        [0.0, 0.05],
        [0.0, 0.03],
        [0.04, 0.0],
      ]
    )
    ref = np.array(
      [
        [True, False],
        [True, False],
        [False, True],
      ]
    )
    summary = clearance_summary(clearance, ref)
    self.assertAlmostEqual(summary.min_swing_clearance_m, 0.03)
    self.assertAlmostEqual(summary.left_min_swing_clearance_m, 0.04)
    self.assertAlmostEqual(summary.right_min_swing_clearance_m, 0.03)

  def test_derive_foot_contact_from_height_uses_height_and_velocity(self):
    foot_height = np.array([[0.0, 0.05], [0.01, 0.04], [0.05, 0.0]])
    foot_vertical_velocity = np.array([[0.0, 0.2], [0.1, 0.2], [0.3, -0.1]])
    contact = derive_foot_contact_from_height(
      foot_height,
      foot_vertical_velocity,
      height_threshold_m=0.02,
      vertical_speed_threshold_m_s=0.25,
    )
    self.assertEqual(contact.tolist(), [[True, False], [True, False], [False, True]])

  def test_render_markdown_report_contains_core_metrics(self):
    report = render_markdown_report(
      title="Mimic Analysis",
      inputs={"motion_file": "motion.npz"},
      reference_summary={"reference_single_support_ratio": 0.5},
      training_summary=[],
      contact_comparison={"swing_contact_ratio": 0.25},
    )
    self.assertIn("motion.npz", report)
    self.assertIn("reference_single_support_ratio", report)
    self.assertIn("swing_contact_ratio", report)


if __name__ == "__main__":
  unittest.main()
