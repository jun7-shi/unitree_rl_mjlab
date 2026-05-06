import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.watch_mimic_quick_experiments import (
  evaluated_iterations,
  parse_checkpoint_schedule,
  pending_checkpoints,
  should_stop_for_quick_decision,
)


class MimicQuickWatchTests(unittest.TestCase):
  def test_parse_checkpoint_schedule_accepts_comma_separated_iterations(self):
    self.assertEqual(parse_checkpoint_schedule("1000,2000,5000"), (1000, 2000, 5000))

  def test_pending_checkpoints_returns_existing_unevaluated_checkpoints(self):
    with TemporaryDirectory() as tmp_dir:
      root = Path(tmp_dir)
      run_dir = root / "run"
      output_dir = root / "out"
      run_dir.mkdir()
      output_dir.mkdir()
      (run_dir / "model_1000.pt").touch()
      (run_dir / "model_2000.pt").touch()
      (run_dir / "model_3000.pt").touch()
      (output_dir / "checkpoint_1000").mkdir()
      (output_dir / "checkpoint_2000").mkdir()
      (output_dir / "checkpoint_2000" / "summary.json").write_text("{}")

      pending = pending_checkpoints(
        run_dir,
        output_dir,
        schedule=(1000, 2000, 3000, 4000),
      )

      self.assertEqual([item.iteration for item in pending], [1000, 3000])

  def test_evaluated_iterations_requires_summary_file(self):
    with TemporaryDirectory() as tmp_dir:
      output_dir = Path(tmp_dir)
      (output_dir / "checkpoint_1000").mkdir()
      (output_dir / "checkpoint_5000").mkdir()
      (output_dir / "checkpoint_5000" / "summary.json").write_text("{}")
      (output_dir / "notes").mkdir()

      self.assertEqual(evaluated_iterations(output_dir), {5000})

  def test_should_stop_for_stop_decision_after_3000(self):
    self.assertTrue(should_stop_for_quick_decision({"decision": "stop"}, 3000))

  def test_should_not_stop_for_wait_or_promote(self):
    self.assertFalse(should_stop_for_quick_decision({"decision": "wait"}, 1000))
    self.assertFalse(should_stop_for_quick_decision({"decision": "promote"}, 5000))


if __name__ == "__main__":
  unittest.main()
