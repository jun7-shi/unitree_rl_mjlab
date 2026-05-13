import textwrap
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import scripts.preview_sew_upper_body as preview
from scripts.preview_sew_upper_body import PreviewConfig, run_preview


def _write_tiny_upper_body_bvh(path: Path):
  path.write_text(
    textwrap.dedent(
      """\
      HIERARCHY
      ROOT Root
      {
        OFFSET 0.0 0.0 0.0
        CHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation
        JOINT Chest
        {
          OFFSET 0.0 100.0 0.0
          CHANNELS 3 Zrotation Yrotation Xrotation
          JOINT LeftArm
          {
            OFFSET 10.0 0.0 0.0
            CHANNELS 3 Zrotation Yrotation Xrotation
            JOINT LeftForeArm
            {
              OFFSET 20.0 0.0 0.0
              CHANNELS 3 Zrotation Yrotation Xrotation
              JOINT LeftHand
              {
                OFFSET 30.0 0.0 0.0
                CHANNELS 3 Zrotation Yrotation Xrotation
              }
            }
          }
          JOINT RightArm
          {
            OFFSET -10.0 0.0 0.0
            CHANNELS 3 Zrotation Yrotation Xrotation
            JOINT RightForeArm
            {
              OFFSET -20.0 0.0 0.0
              CHANNELS 3 Zrotation Yrotation Xrotation
              JOINT RightHand
              {
                OFFSET -30.0 0.0 0.0
                CHANNELS 3 Zrotation Yrotation Xrotation
              }
            }
          }
        }
      }
      MOTION
      Frames: 1
      Frame Time: 0.008333
      1 2 3 0 0 0 0 0 0 0 0 0 90 0 0 0 0 0 0 0 0 0 0 0 0 0 0
      """
    ),
    encoding="utf-8",
  )


def test_run_preview_headless_retargets_bvh_and_reports_errors(tmp_path):
  bvh_path = tmp_path / "upper.bvh"
  _write_tiny_upper_body_bvh(bvh_path)

  summary = run_preview(
    PreviewConfig(
      bvh_path=bvh_path,
      max_frames=1,
      no_viewer=True,
    )
  )

  assert summary.frame_count == 1
  assert "waist" in summary.max_errors
  assert "left_upper_arm" in summary.max_errors
  assert "right_wrist" in summary.max_errors


def test_preview_script_runs_directly_from_repo_root(tmp_path):
  bvh_path = tmp_path / "upper.bvh"
  _write_tiny_upper_body_bvh(bvh_path)

  result = subprocess.run(
    [
      sys.executable,
      "scripts/preview_sew_upper_body.py",
      "--bvh",
      str(bvh_path),
      "--max-frames",
      "1",
      "--no-viewer",
    ],
    cwd=Path(__file__).resolve().parents[1],
    check=False,
    capture_output=True,
    text=True,
  )

  assert result.returncode == 0, result.stderr
  assert "frames: 1" in result.stdout


def test_run_preview_enables_soma_to_g1_orientation_offsets_by_default(monkeypatch, tmp_path):
  captured = {}

  def fake_load_targets(
    path,
    *,
    frame_slice,
    apply_orientation_offsets,
    align_upper_arm_axes_to_g1,
    remove_initial_heading,
  ):
    captured["apply_orientation_offsets"] = apply_orientation_offsets
    captured["align_upper_arm_axes_to_g1"] = align_upper_arm_axes_to_g1
    captured["remove_initial_heading"] = remove_initial_heading
    return [object()]

  def fake_retarget_targets(targets, *, retargeter):
    return [
      SimpleNamespace(
        success=True,
        errors={"waist": 0.0},
      )
    ]

  monkeypatch.setattr(preview, "load_soma_bvh_upper_body_targets", fake_load_targets)
  monkeypatch.setattr(preview, "retarget_upper_body_targets", fake_retarget_targets)
  monkeypatch.setattr(preview, "G1UpperBodySEWRetargeter", lambda: object())

  run_preview(PreviewConfig(bvh_path=tmp_path / "unused.bvh", no_viewer=True))

  assert captured["apply_orientation_offsets"] is True
  assert captured["align_upper_arm_axes_to_g1"] is True
  assert captured["remove_initial_heading"] is True


def test_run_preview_uses_start_frame_and_max_frames(monkeypatch, tmp_path):
  captured = {}

  def fake_load_targets(
    path,
    *,
    frame_slice,
    apply_orientation_offsets,
    align_upper_arm_axes_to_g1,
    remove_initial_heading,
  ):
    captured["frame_slice"] = frame_slice
    return [object()]

  def fake_retarget_targets(targets, *, retargeter):
    return [
      SimpleNamespace(
        success=True,
        errors={"waist": 0.0},
      )
    ]

  monkeypatch.setattr(preview, "load_soma_bvh_upper_body_targets", fake_load_targets)
  monkeypatch.setattr(preview, "retarget_upper_body_targets", fake_retarget_targets)
  monkeypatch.setattr(preview, "G1UpperBodySEWRetargeter", lambda: object())

  run_preview(
    PreviewConfig(
      bvh_path=tmp_path / "unused.bvh",
      start_frame=450,
      max_frames=200,
      no_viewer=True,
    )
  )

  assert captured["frame_slice"] == slice(450, 650)
