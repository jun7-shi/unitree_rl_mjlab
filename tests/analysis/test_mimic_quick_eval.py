import unittest

import numpy as np

from src.analysis.mimic.quick_eval import (
  DEFAULT_BASELINE_METRICS,
  classify_quick_eval,
  quick_policy_metrics,
  summarize_scalar_windows,
)


class MimicQuickEvalTests(unittest.TestCase):
  def test_default_baseline_metrics_cover_training_tracking_contact_and_quietness(self):
    categories = {metric["category"] for metric in DEFAULT_BASELINE_METRICS}
    names = {metric["name"] for metric in DEFAULT_BASELINE_METRICS}

    self.assertIn("training", categories)
    self.assertIn("tracking", categories)
    self.assertIn("contact", categories)
    self.assertIn("quietness", categories)
    self.assertIn("action", categories)
    self.assertIn("policy_true_swing_mid_contact_ratio", names)

  def test_quick_policy_metrics_combines_contact_clearance_action_and_quietness(self):
    reference_contact = np.array(
      [
        [True, False],
        [True, False],
        [False, True],
        [False, True],
      ]
    )
    policy_contact = np.array(
      [
        [True, False],
        [True, True],
        [False, True],
        [True, True],
      ]
    )
    policy_clearance = np.array(
      [
        [0.0, 0.03],
        [0.0, 0.01],
        [0.04, 0.0],
        [0.02, 0.0],
      ]
    )
    policy_vertical_velocity = np.array(
      [
        [0.0, -0.1],
        [0.0, -0.5],
        [0.4, 0.0],
        [-0.2, 0.0],
      ]
    )
    action_rate = np.array([0.0, 0.2, 0.4, 0.6])

    metrics = quick_policy_metrics(
      reference_contact=reference_contact,
      policy_contact=policy_contact,
      policy_clearance_m=policy_clearance,
      policy_vertical_velocity_m_s=policy_vertical_velocity,
      action_rate_l2=action_rate,
      fps=50.0,
    )

    self.assertAlmostEqual(metrics["swing_contact_ratio"], 0.5)
    self.assertAlmostEqual(metrics["single_support_survival"], 0.5)
    self.assertEqual(metrics["extra_touchdown_count"], 1)
    self.assertAlmostEqual(metrics["policy_min_swing_clearance_m"], 0.01)
    self.assertAlmostEqual(metrics["policy_p05_swing_clearance_m"], 0.0115)
    self.assertAlmostEqual(metrics["policy_p10_swing_clearance_m"], 0.013)
    self.assertAlmostEqual(metrics["policy_low_swing_clearance_ratio_2cm"], 0.25)
    self.assertAlmostEqual(metrics["policy_left_low_swing_clearance_ratio_2cm"], 0.0)
    self.assertAlmostEqual(metrics["policy_right_low_swing_clearance_ratio_2cm"], 0.5)
    self.assertAlmostEqual(metrics["policy_mean_action_rate_l2"], 0.3)
    self.assertAlmostEqual(metrics["policy_max_abs_swing_foot_vertical_velocity_m_s"], 0.5)

  def test_quick_policy_metrics_can_report_true_swing_run_quality(self):
    reference_contact = np.array(
      [
        [True, True],
        [False, True],
        [False, True],
        [False, True],
        [False, True],
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
      ]
    )
    policy_vertical_velocity = np.zeros_like(policy_clearance)

    metrics = quick_policy_metrics(
      reference_contact=reference_contact,
      reference_clearance_m=reference_clearance,
      policy_contact=policy_contact,
      policy_clearance_m=policy_clearance,
      policy_vertical_velocity_m_s=policy_vertical_velocity,
      action_rate_l2=np.zeros(7),
      fps=10.0,
    )

    self.assertEqual(metrics["reference_true_swing_run_count"], 1)
    self.assertAlmostEqual(metrics["reference_micro_swing_run_ratio"], 0.0)
    self.assertAlmostEqual(metrics["policy_true_swing_lift_success_ratio_5cm"], 1.0)
    self.assertAlmostEqual(metrics["policy_true_swing_mid_contact_ratio"], 2.0 / 3.0)
    self.assertAlmostEqual(
      metrics["policy_true_swing_mid_low_clearance_ratio_2cm"],
      2.0 / 3.0,
    )

  def test_summarize_scalar_windows_uses_named_windows(self):
    rows = [
      {"tag": "Train/mean_reward", "step": 100, "value": 1.0},
      {"tag": "Train/mean_reward", "step": 600, "value": 3.0},
      {"tag": "Metrics/motion/error_body_pos", "step": 600, "value": 0.2},
    ]

    summary = summarize_scalar_windows(
      rows,
      tags=("Train/mean_reward", "Metrics/motion/error_body_pos"),
      windows=((0, 500), (500, 1000)),
    )

    self.assertEqual(summary["Train/mean_reward@0-500_mean"], 1.0)
    self.assertEqual(summary["Train/mean_reward@500-1000_mean"], 3.0)
    self.assertEqual(summary["Metrics/motion/error_body_pos@500-1000_count"], 1)

  def test_classify_quick_eval_promotes_clear_contact_improvement(self):
    decision = classify_quick_eval(
      {
        "swing_contact_ratio": 0.18,
        "single_support_survival": 0.82,
        "extra_touchdown_count": 12,
        "policy_mean_action_rate_l2": 0.4,
      },
      checkpoint_iteration=5000,
    )

    self.assertEqual(decision["decision"], "promote")

  def test_classify_quick_eval_rejects_bad_contact_after_3000_steps(self):
    decision = classify_quick_eval(
      {
        "swing_contact_ratio": 0.5,
        "single_support_survival": 0.4,
        "extra_touchdown_count": 30,
        "policy_mean_action_rate_l2": 0.4,
      },
      checkpoint_iteration=3000,
    )

    self.assertEqual(decision["decision"], "stop")


if __name__ == "__main__":
  unittest.main()
