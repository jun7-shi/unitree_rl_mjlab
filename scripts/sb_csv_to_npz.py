from __future__ import annotations

import tempfile
from pathlib import Path

import mjlab
import tyro

from csv_to_npz import main as export_motion_npz
from src.motion.g1_floor_alignment import align_g1_motion_rows_to_floor
from src.motion.seed_bones import convert_seed_bones_csv_to_motion_rows, write_motion_csv


def main(
  input_file: str,
  output_name: str,
  input_fps: float = 30.0,
  output_fps: float = 50.0,
  device: str = "cuda:0",
  translation_scale: float = 0.01,
  align_feet_to_floor: bool = False,
  floor_height_m: float = 0.0,
  foot_floor_clearance_m: float = 0.0,
  foot_floor_quantile: float = 0.0,
  alignment_device: str = "cpu",
  render: bool = False,
  line_range: tuple[int, int] | None = None,
):
  """Convert a seed-bones G1 CSV into a tracking NPZ using the existing replay pipeline."""
  rows = convert_seed_bones_csv_to_motion_rows(
    input_file=input_file,
    translation_scale=translation_scale,
  )
  if not rows:
    raise ValueError(f"Seed-bones CSV contains no motion rows: {input_file}")

  # Root-Z lifting is a diagnostic/export convenience, not a retargeting default.
  if align_feet_to_floor:
    rows, alignment_report = align_g1_motion_rows_to_floor(
      rows,
      floor_height_m=floor_height_m,
      clearance_m=foot_floor_clearance_m,
      quantile=foot_floor_quantile,
      device=alignment_device,
    )
    print(
      "[INFO] Foot-floor alignment: "
      f"offset={alignment_report.offset_m:.5f} m, "
      f"min_before={alignment_report.min_lower_bound_m:.5f} m, "
      f"selected_before={alignment_report.selected_lower_bound_m:.5f} m, "
      f"below_floor={alignment_report.frames_below_floor}/"
      f"{alignment_report.frame_count}, "
      f"floor={alignment_report.floor_height_m:.5f} m, "
      f"clearance={alignment_report.clearance_m:.5f} m, "
      f"quantile={alignment_report.quantile:.3f}"
    )

  with tempfile.NamedTemporaryFile(
    mode="w", suffix=".csv", delete=False, encoding="utf-8"
  ) as temp_file:
    temp_motion_path = Path(temp_file.name)

  try:
    write_motion_csv(rows, temp_motion_path)
    export_motion_npz(
      robot="g1",
      input_file=str(temp_motion_path),
      output_name=output_name,
      input_fps=input_fps,
      output_fps=output_fps,
      device=device,
      render=render,
      line_range=line_range,
    )
  finally:
    temp_motion_path.unlink(missing_ok=True)


if __name__ == "__main__":
  tyro.cli(main, config=mjlab.TYRO_FLAGS)
