import unittest

import torch

from src.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg
from src.tasks.tracking.mdp.observations import foot_phase_observation_from_context
from src.tasks.tracking.mdp.rewards import (
  early_swing_contact_cost_from_masks,
  foot_lift_trajectory_cost_from_heights,
  infer_reference_contacts_from_heights,
  late_swing_downward_velocity_cost_from_velocity,
  landing_force_cost_from_contact_events,
  phase_swing_clearance_cost_from_heights,
  phase_swing_clearance_target,
  single_support_reward_from_masks,
  swing_clearance_margin_cost_from_heights,
  swing_contact_cost_from_masks,
  swing_phase_from_reference_contact,
)


class ContactAwareTrackingRewardTests(unittest.TestCase):
  def test_infer_reference_contacts_from_heights_uses_per_foot_minimum(self):
    reference_heights = torch.tensor(
      [
        [0.02, 0.12],
        [0.03, 0.10],
        [0.11, 0.025],
      ],
      dtype=torch.float32,
    )

    contacts = infer_reference_contacts_from_heights(
      reference_heights,
      clearance_threshold=0.025,
    )

    expected = torch.tensor(
      [
        [True, False],
        [True, False],
        [False, True],
      ]
    )
    torch.testing.assert_close(contacts, expected)

  def test_swing_contact_cost_penalizes_contact_only_during_reference_swing(self):
    reference_contact = torch.tensor([[True, False], [False, True]])
    actual_contact = torch.tensor([[True, True], [False, True]])

    cost = swing_contact_cost_from_masks(reference_contact, actual_contact)

    torch.testing.assert_close(cost, torch.tensor([1.0, 0.0]))

  def test_landing_force_cost_only_counts_first_contact(self):
    first_contact = torch.tensor([[True, False], [False, True]])
    force_magnitude = torch.tensor([[12.0, 100.0], [50.0, 8.0]])

    cost = landing_force_cost_from_contact_events(first_contact, force_magnitude)

    torch.testing.assert_close(cost, torch.tensor([12.0, 8.0]))

  def test_single_support_reward_requires_matching_single_support_foot(self):
    reference_contact = torch.tensor(
      [
        [True, False],
        [False, True],
        [True, True],
        [False, False],
      ]
    )
    actual_contact = torch.tensor(
      [
        [True, False],
        [True, False],
        [True, True],
        [False, False],
      ]
    )

    reward = single_support_reward_from_masks(reference_contact, actual_contact)

    torch.testing.assert_close(reward, torch.tensor([1.0, 0.0, 0.0, 0.0]))

  def test_swing_clearance_margin_cost_only_applies_to_reference_swing_feet(self):
    reference_contact = torch.tensor([[True, False], [False, True]])
    actual_heights = torch.tensor([[0.03, 0.08], [0.07, 0.03]])
    reference_floor_heights = torch.tensor([[0.02, 0.02]])

    cost = swing_clearance_margin_cost_from_heights(
      reference_contact=reference_contact,
      actual_heights=actual_heights,
      reference_floor_heights=reference_floor_heights,
      margin_m=0.10,
    )

    torch.testing.assert_close(cost, torch.tensor([0.04, 0.05]))

  def test_swing_phase_from_reference_contact_marks_contiguous_swing_runs(self):
    reference_contact = torch.tensor(
      [
        [True],
        [False],
        [False],
        [False],
        [True],
      ]
    )

    phase = swing_phase_from_reference_contact(reference_contact)

    torch.testing.assert_close(
      phase[:, 0],
      torch.tensor([0.0, 0.0, 0.5, 1.0, 0.0]),
    )

  def test_phase_swing_clearance_target_is_highest_at_mid_swing(self):
    swing_phase = torch.tensor([[0.0, 0.5, 1.0]])
    reference_clearance = torch.tensor([[0.02, 0.02, 0.02]])

    target = phase_swing_clearance_target(
      swing_phase=swing_phase,
      reference_clearance=reference_clearance,
      base_clearance_m=0.04,
      lift_m=0.08,
      landing_phase_start=0.85,
    )

    self.assertGreater(target[0, 1], target[0, 0])
    self.assertLess(target[0, 2], target[0, 1])
    torch.testing.assert_close(target[0, 2], torch.tensor(0.04))

  def test_phase_swing_clearance_cost_penalizes_deficit_against_phase_target(self):
    reference_contact = torch.tensor([[False, False, True]])
    actual_heights = torch.tensor([[0.04, 0.07, 0.02]])
    reference_floor_heights = torch.tensor([[0.0, 0.0, 0.0]])
    reference_clearance = torch.tensor([[0.02, 0.02, 0.0]])
    swing_phase = torch.tensor([[0.0, 0.5, 0.0]])

    cost = phase_swing_clearance_cost_from_heights(
      reference_contact=reference_contact,
      actual_heights=actual_heights,
      reference_floor_heights=reference_floor_heights,
      reference_clearance=reference_clearance,
      swing_phase=swing_phase,
      base_clearance_m=0.04,
      lift_m=0.08,
      landing_phase_start=0.85,
    )

    torch.testing.assert_close(cost, torch.tensor([0.05]))

  def test_early_swing_contact_cost_allows_contact_near_landing_phase(self):
    reference_contact = torch.tensor([[False, False, True]])
    actual_contact = torch.tensor([[True, True, True]])
    swing_phase = torch.tensor([[0.25, 0.90, 0.0]])

    cost = early_swing_contact_cost_from_masks(
      reference_contact=reference_contact,
      actual_contact=actual_contact,
      swing_phase=swing_phase,
      landing_phase_start=0.85,
    )

    torch.testing.assert_close(cost, torch.tensor([1.0]))

  def test_foot_lift_trajectory_cost_uses_deadband_and_reference_swing_only(self):
    reference_contact = torch.tensor([[False, False, True]])
    actual_heights = torch.tensor([[0.03, 0.09, 0.0]])
    reference_floor_heights = torch.tensor([[0.0, 0.0, 0.0]])
    swing_phase = torch.tensor([[0.0, 0.5, 0.0]])

    cost = foot_lift_trajectory_cost_from_heights(
      reference_contact=reference_contact,
      actual_heights=actual_heights,
      reference_floor_heights=reference_floor_heights,
      swing_phase=swing_phase,
      base_clearance_m=0.04,
      lift_m=0.08,
      phase_power=1.0,
      deadband_m=0.01,
    )

    torch.testing.assert_close(cost, torch.tensor([0.02]))

  def test_late_swing_downward_velocity_cost_only_penalizes_fast_late_descent(self):
    reference_contact = torch.tensor([[False, False, False, True]])
    swing_phase = torch.tensor([[0.50, 0.85, 0.95, 0.0]])
    vertical_velocity = torch.tensor([[-1.0, -0.3, -0.8, -2.0]])

    cost = late_swing_downward_velocity_cost_from_velocity(
      reference_contact=reference_contact,
      vertical_velocity_m_s=vertical_velocity,
      swing_phase=swing_phase,
      velocity_phase_start=0.80,
      max_downward_velocity_m_s=0.40,
    )

    torch.testing.assert_close(cost, torch.tensor([0.40]))

  def test_foot_phase_observation_contains_sin_cos_and_swing_flags(self):
    reference_contact = torch.tensor([[True, False]])
    swing_phase = torch.tensor([[0.0, 0.5]])

    obs = foot_phase_observation_from_context(reference_contact, swing_phase)

    torch.testing.assert_close(
      obs,
      torch.tensor([[0.0, 1.0, 0.0, 1.0, 0.0, 1.0]]),
      atol=1e-6,
      rtol=1e-6,
    )

  def test_g1_tracking_config_has_contact_aware_rewards(self):
    cfg = unitree_g1_flat_tracking_env_cfg()
    sensor_names = {sensor.name for sensor in cfg.scene.sensors or ()}
    actor_terms = cfg.observations["actor"].terms
    critic_terms = cfg.observations["critic"].terms

    self.assertIn("feet_ground_contact", sensor_names)
    self.assertIn("motion_foot_phase", actor_terms)
    self.assertIn("motion_foot_phase", critic_terms)
    self.assertIn("motion_swing_contact", cfg.rewards)
    self.assertIn("motion_soft_landing", cfg.rewards)
    self.assertIn("motion_single_support", cfg.rewards)
    self.assertIn("motion_phase_swing_clearance", cfg.rewards)
    self.assertIn("motion_early_swing_contact", cfg.rewards)
    self.assertIn("motion_foot_lift_trajectory", cfg.rewards)
    self.assertIn("motion_late_swing_velocity", cfg.rewards)


if __name__ == "__main__":
  unittest.main()
