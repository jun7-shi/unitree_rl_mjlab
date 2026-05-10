from subprocess import run


def test_evaluate_silent_walking_cli_help():
  proc = run(
    ["python", "scripts/evaluate_silent_walking.py", "--help"],
    check=False,
    capture_output=True,
    text=True,
  )

  assert proc.returncode == 0
  assert "robot" in proc.stdout
  assert "policy" in proc.stdout
  assert "output-dir" in proc.stdout
  assert "task-id" in proc.stdout
  assert "motion-file" in proc.stdout
  assert "fixed-command" in proc.stdout
  assert "foot-grid-output-dir" in proc.stdout
  assert "foot-grid-video-file" in proc.stdout
