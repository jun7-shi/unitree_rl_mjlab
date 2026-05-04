import argparse
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.monitor_mimic_training import (
  PauseDecision,
  RunSnapshot,
  latest_checkpoint_iteration,
  parse_latest_iteration,
  parse_latest_metrics,
  should_pause,
)


class MimicTrainingMonitorTests(unittest.TestCase):
  def test_parse_latest_iteration_uses_last_iteration_block(self):
    output = """
    Learning iteration 12/30001
    Mean reward: 1.0
    Learning iteration 42/30001
    Mean reward: 2.5
    """

    self.assertEqual(parse_latest_iteration(output), 42)

  def test_parse_latest_metrics_reads_last_iteration_block(self):
    output = """
    Learning iteration 12/30001
    Mean reward: 1.0
    Metrics/tracking_swing_contact_mean: 0.8
    Learning iteration 42/30001
    Mean reward: 2.5
    Metrics/tracking_swing_contact_mean: 0.2
    """

    metrics = parse_latest_metrics(output)

    self.assertEqual(metrics["Mean reward"], 2.5)
    self.assertEqual(metrics["Metrics/tracking_swing_contact_mean"], 0.2)

  def test_latest_checkpoint_iteration_ignores_non_checkpoint_files(self):
    with TemporaryDirectory() as tmp_dir:
      run_dir = Path(tmp_dir)
      (run_dir / "model_0.pt").touch()
      (run_dir / "model_500.pt").touch()
      (run_dir / "policy.onnx").touch()

      self.assertEqual(latest_checkpoint_iteration(run_dir), 500)

  def test_should_pause_ignores_quality_before_min_iteration(self):
    args = argparse.Namespace(
      stale_seconds=1800,
      quality_min_iteration=5000,
      min_mean_episode_length=60.0,
      max_swing_contact_mean=0.9,
      max_landing_force_mean=700.0,
    )
    snapshot = RunSnapshot(
      run_name="early",
      run_dir=Path("."),
      latest_iteration=300,
      latest_checkpoint_iteration=0,
      seconds_since_event_update=10.0,
      output_tail="Learning iteration 300/30001\nMean episode length: 1.0\n",
      metrics={"Mean episode length": 1.0},
      pids=(123,),
    )

    decision = should_pause(snapshot, args)

    self.assertIsInstance(decision, PauseDecision)
    self.assertFalse(decision.should_pause)

  def test_should_pause_on_hard_failure(self):
    args = argparse.Namespace(
      stale_seconds=1800,
      quality_min_iteration=5000,
      min_mean_episode_length=60.0,
      max_swing_contact_mean=0.9,
      max_landing_force_mean=700.0,
    )
    snapshot = RunSnapshot(
      run_name="failed",
      run_dir=Path("."),
      latest_iteration=1,
      latest_checkpoint_iteration=0,
      seconds_since_event_update=10.0,
      output_tail="Traceback (most recent call last)",
      metrics={},
      pids=(123,),
    )

    decision = should_pause(snapshot, args)

    self.assertTrue(decision.should_pause)


if __name__ == "__main__":
  unittest.main()
