# Silent Foot-Grid Overlay

This note documents the reusable Viser foot-grid overlay for visualizing
silent-walking sole velocity and contact-force proxies from play scripts.

## What It Shows

The overlay adds a `Foot Grid` tab to Viser with two rows:

- signed vertical velocity (`vz`) at virtual sole sample points
- normal-force proxy distributed from live MuJoCo contact slots

The virtual sole points are visualization probes. They are not collision geoms
and they do not affect simulation.

## Required Sensor Injection

The overlay needs contact sensors to exist before the environment is created.
Call `prepare_silent_foot_grid_overlay_env_cfg()` after loading the env config and
before constructing `ManagerBasedRlEnv`.

```python
from src.evaluation.silent_walking.viser_foot_grid_overlay import (
  prepare_silent_foot_grid_overlay_env_cfg,
)

env_cfg = load_env_cfg(task_id, play=True)
prepare_silent_foot_grid_overlay_env_cfg(
  env_cfg,
  robot_name="g1",
  force_slots=4,
)
env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
```

This adds two mjlab contact sensors through `env_cfg.scene.sensors`:

- `foot_capsule_ground_contact`: per-foot-collision net force and contact flags
- `foot_capsule_ground_contact_points`: strongest contact slots with `pos`,
  `normal`, `tangent`, and force in the global frame

These sensors are not XML collision geoms. They are MuJoCo contact sensors that
mjlab adds while building the environment.

`force_slots` is a per-foot-collision upper bound on retained contact slots. It
does not create more physical contacts. If the live model only produces about a
dozen contact points, increasing `force_slots` to `64` will increase the sensor
capacity but not make the solver generate dozens of new contacts.

## Wrapping An Env

After constructing the env and any RL wrapper, wrap the env once:

```python
from src.evaluation.silent_walking.viser_foot_grid_overlay import (
  FootGridOverlayConfig,
  FootGridViserPlayViewer,
  wrap_env_for_silent_foot_grid_overlay,
)

env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

foot_grid_config = FootGridOverlayConfig(
  robot_name="g1",
  point_count=300,
  update_rate="control",  # or "sim"
  force_slots=4,
)
env = wrap_env_for_silent_foot_grid_overlay(env, foot_grid_config)

FootGridViserPlayViewer(env, policy).run()
```

The wrapper delegates normal env methods and attributes to the wrapped env. It
also carries the foot-grid overlay runtime so play scripts can keep their policy
loading logic unchanged.

## Sim-Step Updates

Use `update_rate="sim"` to update the overlay at physics-step cadence:

```python
foot_grid_config = FootGridOverlayConfig(update_rate="sim")
```

For the default G1 tasks, this means four overlay updates per control step
(`physics_dt=0.005s`, `step_dt=0.02s`). Policy actions are still computed once
per control step and repeated across the physics substeps.

Sim-step updates require `FootGridViserPlayViewer`. A plain `ViserPlayViewer`
only calls `env.step()` at control-step cadence, so it cannot expose the
intermediate physics substeps.

## Existing CLI

The built-in play script already wires the reusable helpers:

```bash
python scripts/play.py Unitree-G1-Flat \
  --checkpoint-file /path/to/model.pt \
  --viewer viser \
  --foot-grid-overlay True \
  --foot-grid-overlay-points 300 \
  --foot-grid-overlay-update-rate sim \
  --foot-grid-overlay-force-slots 4
```

Use the same pattern for tracking tasks, with the task id and motion/checkpoint
arguments appropriate for that policy.
