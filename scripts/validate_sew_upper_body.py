from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from src.motion.sew_upper_body import G1UpperBodySEWRetargeter
from src.motion.sew_upper_body_validation import (
  FKValidationReport,
  sample_fk_validation_cases,
  validate_upper_body_retargeter_on_fk_cases,
)


def run_validation(
  *,
  samples: int,
  seed: int,
  tolerance: float,
  xml_path: Path | None = None,
) -> FKValidationReport:
  retargeter = G1UpperBodySEWRetargeter(xml_path=xml_path)
  cases = sample_fk_validation_cases(retargeter, count=samples, seed=seed)
  return validate_upper_body_retargeter_on_fk_cases(
    cases,
    retargeter=retargeter,
    tolerance=tolerance,
  )


def _print_report(report: FKValidationReport) -> None:
  print(f"cases: {report.case_count}")
  print(f"successful cases: {report.success_count}/{report.case_count}")
  for key, value in report.max_errors.items():
    print(f"max {key}: {value:.6f}")
  for case in report.case_results:
    if not case.success:
      worst_key = max(case.errors, key=case.errors.__getitem__)
      print(f"failed {case.name}: {worst_key}={case.errors[worst_key]:.6f}")


def build_arg_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
    description="Validate SEW-Mimic upper-body retargeting with robot-FK round trips."
  )
  parser.add_argument("--samples", type=int, default=16, help="Number of reachable robot poses to validate.")
  parser.add_argument("--seed", type=int, default=0, help="Seed used for additional random FK samples.")
  parser.add_argument("--tolerance", type=float, default=1e-3, help="Maximum allowed geometric residual.")
  parser.add_argument("--xml", type=Path, default=None, help="Optional Unitree G1 MuJoCo XML path.")
  return parser


def main(argv: Sequence[str] | None = None) -> int:
  args = build_arg_parser().parse_args(argv)
  report = run_validation(
    samples=args.samples,
    seed=args.seed,
    tolerance=args.tolerance,
    xml_path=args.xml,
  )
  _print_report(report)
  return 0 if report.all_passed else 1


if __name__ == "__main__":
  raise SystemExit(main())
