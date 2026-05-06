import unittest

import numpy as np

from src.analysis.mimic.phase_diagnostics import (
  aggregate_swing_run_rows,
  phase_bin_rows,
  swing_phase_from_contact,
  swing_run_rows,
)


class MimicPhaseDiagnosticsTests(unittest.TestCase):
  def test_swing_phase_from_contact_marks_each_contiguous_swing_run(self):
    reference_contact = np.array(
      [
        [True, True],
        [False, True],
        [False, True],
        [False, False],
        [True, False],
        [True, True],
      ]
    )

    phase, run_id = swing_phase_from_contact(reference_contact)

    np.testing.assert_allclose(phase[:, 0], [0.0, 0.0, 0.5, 1.0, 0.0, 0.0])
    np.testing.assert_allclose(phase[:, 1], [0.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    np.testing.assert_array_equal(run_id[:, 0], [-1, 0, 0, 0, -1, -1])
    np.testing.assert_array_equal(run_id[:, 1], [-1, -1, -1, 0, 0, -1])

  def test_swing_run_rows_summarize_clearance_per_foot_run(self):
    reference_contact = np.array(
      [
        [True, True],
        [False, True],
        [False, True],
        [False, True],
        [True, True],
      ]
    )
    clearance = np.array(
      [
        [0.0, 0.0],
        [0.01, 0.0],
        [0.04, 0.0],
        [0.09, 0.0],
        [0.0, 0.0],
      ]
    )

    rows = swing_run_rows(
      reference_contact=reference_contact,
      clearance_m=clearance,
      fps=50.0,
    )

    self.assertEqual(len(rows), 1)
    self.assertEqual(rows[0]["foot"], "left")
    self.assertEqual(rows[0]["start_frame"], 1)
    self.assertEqual(rows[0]["end_frame"], 3)
    self.assertAlmostEqual(rows[0]["duration_s"], 0.06)
    self.assertAlmostEqual(rows[0]["min_clearance_m"], 0.01)
    self.assertAlmostEqual(rows[0]["max_clearance_m"], 0.09)
    self.assertAlmostEqual(rows[0]["low_clearance_ratio_2cm"], 1.0 / 3.0)

  def test_swing_run_rows_localize_policy_lift_failures_on_true_swings(self):
    reference_contact = np.array(
      [
        [True, True],
        [False, True],
        [False, True],
        [False, True],
        [False, True],
        [False, True],
        [True, True],
        [False, True],
        [True, True],
      ]
    )
    reference_clearance = np.array(
      [
        [0.0, 0.0],
        [0.03, 0.0],
        [0.08, 0.0],
        [0.12, 0.0],
        [0.08, 0.0],
        [0.03, 0.0],
        [0.0, 0.0],
        [0.026, 0.0],
        [0.0, 0.0],
      ]
    )
    policy_contact = np.array(
      [
        [True, True],
        [True, True],
        [True, True],
        [False, True],
        [True, True],
        [True, True],
        [True, True],
        [True, True],
        [True, True],
      ]
    )
    policy_clearance = np.array(
      [
        [0.0, 0.0],
        [0.0, 0.0],
        [0.01, 0.0],
        [0.06, 0.0],
        [0.01, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
      ]
    )
    policy_vertical_velocity = np.array(
      [
        [0.0, 0.0],
        [-0.1, 0.0],
        [-0.2, 0.0],
        [0.1, 0.0],
        [-0.6, 0.0],
        [-0.4, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
      ]
    )

    rows = swing_run_rows(
      reference_contact=reference_contact,
      clearance_m=reference_clearance,
      fps=10.0,
      policy_contact=policy_contact,
      policy_clearance_m=policy_clearance,
      policy_vertical_velocity_m_s=policy_vertical_velocity,
      true_swing_min_duration_s=0.35,
      true_swing_min_peak_clearance_m=0.04,
    )
    summary = aggregate_swing_run_rows(rows)

    self.assertEqual(len(rows), 2)
    true_run = rows[0]
    micro_run = rows[1]
    self.assertFalse(true_run["reference_is_micro_swing"])
    self.assertTrue(micro_run["reference_is_micro_swing"])
    self.assertAlmostEqual(true_run["policy_mid_phase_contact_ratio"], 2.0 / 3.0)
    self.assertAlmostEqual(
      true_run["policy_mid_phase_low_clearance_ratio_2cm"],
      2.0 / 3.0,
    )
    self.assertAlmostEqual(true_run["policy_peak_clearance_ratio_to_reference"], 0.5)
    self.assertEqual(true_run["policy_touchdown_count_during_reference_swing"], 2)
    self.assertTrue(true_run["policy_lift_success_5cm"])
    self.assertAlmostEqual(summary["reference_micro_swing_run_ratio"], 0.5)
    self.assertAlmostEqual(summary["policy_true_swing_lift_success_ratio_5cm"], 1.0)
    self.assertAlmostEqual(summary["policy_true_swing_mid_contact_ratio"], 2.0 / 3.0)

  def test_phase_bin_rows_localize_policy_contacts_and_clearance_deficits(self):
    reference_contact = np.array(
      [
        [True, True],
        [False, True],
        [False, True],
        [False, True],
        [True, True],
      ]
    )
    reference_clearance = np.array(
      [
        [0.0, 0.0],
        [0.01, 0.0],
        [0.06, 0.0],
        [0.04, 0.0],
        [0.0, 0.0],
      ]
    )
    policy_contact = np.array(
      [
        [True, True],
        [True, True],
        [False, True],
        [True, True],
        [True, True],
      ]
    )
    policy_clearance = np.array(
      [
        [0.0, 0.0],
        [0.0, 0.0],
        [0.03, 0.0],
        [0.01, 0.0],
        [0.0, 0.0],
      ]
    )
    policy_vertical_velocity = np.array(
      [
        [0.0, 0.0],
        [-0.1, 0.0],
        [-0.2, 0.0],
        [-0.8, 0.0],
        [0.0, 0.0],
      ]
    )

    rows = phase_bin_rows(
      reference_contact=reference_contact,
      reference_clearance_m=reference_clearance,
      policy_contact=policy_contact,
      policy_clearance_m=policy_clearance,
      policy_vertical_velocity_m_s=policy_vertical_velocity,
      bins=(0.0, 0.5, 1.0),
    )

    left_rows = [row for row in rows if row["foot"] == "left"]
    self.assertEqual(len(left_rows), 2)
    self.assertEqual(left_rows[0]["sample_count"], 1)
    self.assertAlmostEqual(left_rows[0]["policy_swing_contact_ratio"], 1.0)
    self.assertAlmostEqual(left_rows[0]["policy_clearance_deficit_mean_m"], 0.01)
    self.assertEqual(left_rows[1]["sample_count"], 2)
    self.assertAlmostEqual(left_rows[1]["policy_swing_contact_ratio"], 0.5)
    self.assertAlmostEqual(left_rows[1]["policy_clearance_deficit_mean_m"], 0.03)
    self.assertAlmostEqual(
      left_rows[1]["policy_downward_velocity_p95_m_s"],
      0.77,
      places=2,
    )


if __name__ == "__main__":
  unittest.main()
