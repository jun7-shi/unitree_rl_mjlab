import unittest

import torch

from src.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg
from src.tasks.tracking.mdp.rewards import (
  infer_reference_contacts_from_heights,
  landing_force_cost_from_contact_events,
  single_support_reward_from_masks,
  swing_clearance_margin_cost_from_heights,
  swing_contact_cost_from_masks,
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

  def test_g1_tracking_config_has_contact_aware_rewards(self):
    cfg = unitree_g1_flat_tracking_env_cfg()
    sensor_names = {sensor.name for sensor in cfg.scene.sensors or ()}

    self.assertIn("feet_ground_contact", sensor_names)
    self.assertIn("motion_swing_contact", cfg.rewards)
    self.assertIn("motion_soft_landing", cfg.rewards)
    self.assertIn("motion_single_support", cfg.rewards)
    self.assertIn("motion_swing_clearance", cfg.rewards)


if __name__ == "__main__":
  unittest.main()
