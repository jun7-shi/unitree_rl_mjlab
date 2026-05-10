from dataclasses import dataclass
import unittest

from scripts.play import PlayConfig, _disable_foot_phase_observation


@dataclass
class _FakeObservationGroup:
  terms: dict[str, object]


@dataclass
class _FakeEnvCfg:
  observations: dict[str, _FakeObservationGroup]


class PlayConfigTests(unittest.TestCase):
  def test_foot_grid_overlay_update_rate_can_request_sim_steps(self):
    cfg = PlayConfig(foot_grid_overlay_update_rate="sim")

    self.assertEqual(cfg.foot_grid_overlay_update_rate, "sim")

  def test_disable_foot_phase_observation_removes_actor_and_critic_terms(self):
    cfg = _FakeEnvCfg(
      observations={
        "actor": _FakeObservationGroup(
          terms={"command": object(), "motion_foot_phase": object()}
        ),
        "critic": _FakeObservationGroup(
          terms={"body_pos": object(), "motion_foot_phase": object()}
        ),
      }
    )

    removed = _disable_foot_phase_observation(cfg)

    self.assertEqual(removed, ("actor.motion_foot_phase", "critic.motion_foot_phase"))
    self.assertEqual(tuple(cfg.observations["actor"].terms.keys()), ("command",))
    self.assertEqual(tuple(cfg.observations["critic"].terms.keys()), ("body_pos",))


if __name__ == "__main__":
  unittest.main()
