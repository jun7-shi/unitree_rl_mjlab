import json
import tempfile
import unittest
from pathlib import Path

from src.analysis.mimic.training_logs import (
  checkpoint_iteration,
  downsample_scalar_rows,
  list_checkpoints,
  read_wandb_summaries,
)


class TestMimicTrainingLogs(unittest.TestCase):
  def test_checkpoint_iteration_parses_model_name(self):
    self.assertEqual(checkpoint_iteration(Path("model_30000.pt")), 30000)
    self.assertIsNone(checkpoint_iteration(Path("policy.onnx")))

  def test_list_checkpoints_sorts_by_iteration(self):
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      for name in ["model_1000.pt", "model_0.pt", "model_500.pt", "policy.onnx"]:
        (root / name).write_text("", encoding="utf-8")
      checkpoints = list_checkpoints(root)
    self.assertEqual([path.name for path in checkpoints], ["model_0.pt", "model_500.pt", "model_1000.pt"])

  def test_read_wandb_summaries_reads_local_json(self):
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      files = root / "run-20260504_011907-test" / "files"
      files.mkdir(parents=True)
      (files / "wandb-summary.json").write_text(
        json.dumps({"reward": 1.25}), encoding="utf-8"
      )
      summaries = read_wandb_summaries(root)
    self.assertEqual(len(summaries), 1)
    self.assertEqual(summaries[0]["run_dir"], "run-20260504_011907-test")
    self.assertEqual(summaries[0]["summary"]["reward"], 1.25)

  def test_downsample_scalar_rows_keeps_latest_row(self):
    rows = [
      {"tag": "reward", "step": step, "value": float(step), "wall_time": 0.0}
      for step in range(10)
    ]
    sampled = downsample_scalar_rows(rows, max_rows_per_tag=3)
    self.assertLessEqual(len(sampled), 5)
    self.assertEqual(sampled[-1]["step"], 9)


if __name__ == "__main__":
  unittest.main()
