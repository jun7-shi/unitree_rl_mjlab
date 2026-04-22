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
