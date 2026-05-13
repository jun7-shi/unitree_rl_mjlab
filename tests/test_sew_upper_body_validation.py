import subprocess
import sys
from pathlib import Path

import numpy as np

from src.motion.sew_upper_body import G1UpperBodySEWRetargeter
from src.motion.sew_upper_body_validation import (
  sample_fk_validation_cases,
  validate_upper_body_retargeter_on_fk_cases,
)


def test_sample_fk_validation_cases_are_deterministic_and_inside_joint_limits():
  retargeter = G1UpperBodySEWRetargeter()

  first = sample_fk_validation_cases(retargeter, count=4, seed=7)
  second = sample_fk_validation_cases(retargeter, count=4, seed=7)

  assert len(first) == 4
  assert [case.name for case in first] == [case.name for case in second]
  for left, right in zip(first, second):
    np.testing.assert_allclose(left.joint_angles, right.joint_angles)
    assert left.joint_angles.shape == (17,)
    assert np.all(left.joint_angles >= retargeter.joint_limits[:, 0])
    assert np.all(left.joint_angles <= retargeter.joint_limits[:, 1])


def test_validate_upper_body_retargeter_round_trips_fk_targets():
  retargeter = G1UpperBodySEWRetargeter()
  cases = sample_fk_validation_cases(retargeter, count=5, seed=0)

  report = validate_upper_body_retargeter_on_fk_cases(
    cases,
    retargeter=retargeter,
    tolerance=1e-3,
  )

  assert report.case_count == 5
  assert report.success_count == 5
  assert report.all_passed
  assert set(report.max_errors) == {
    "left_lower_arm",
    "left_upper_arm",
    "left_wrist",
    "right_lower_arm",
    "right_upper_arm",
    "right_wrist",
    "waist",
  }
  assert all(value <= 1e-3 for value in report.max_errors.values())


def test_validate_sew_upper_body_script_runs_from_repo_root():
  result = subprocess.run(
    [
      sys.executable,
      "scripts/validate_sew_upper_body.py",
      "--samples",
      "3",
      "--seed",
      "0",
      "--tolerance",
      "0.001",
    ],
    cwd=Path(__file__).resolve().parents[1],
    check=False,
    capture_output=True,
    text=True,
  )

  assert result.returncode == 0, result.stderr
  assert "cases: 3" in result.stdout
  assert "successful cases: 3/3" in result.stdout
  assert "max waist:" in result.stdout
