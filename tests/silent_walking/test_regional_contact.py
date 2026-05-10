import unittest

import torch

from src.evaluation.silent_walking.regional_contact import (
  CORNER_NAMES,
  RegionalContactState,
  classify_contact_regions,
  virtual_foot_corner_points,
)


class RegionalContactHelperTests(unittest.TestCase):
  def test_classify_contact_regions_uses_local_foot_x_not_capsule_id(self):
    points = torch.tensor(
      [[[
        [-0.06, 0.00, -0.03],
        [0.03, 0.00, -0.03],
        [0.13, 0.00, -0.03],
        [0.20, 0.00, -0.03],
      ]]]
    )
    mask = torch.tensor([[[True, True, True, False]]])

    regions = classify_contact_regions(
      contact_pos_local_m=points,
      contact_mask=mask,
      heel_region_max_x_m=-0.02,
      toe_region_min_x_m=0.09,
    )

    self.assertEqual(regions.shape, (1, 1, 3))
    self.assertTrue(torch.equal(regions[0, 0], torch.tensor([True, True, True])))

  def test_virtual_foot_corner_points_use_scene_g1_offsets(self):
    body_pos = torch.zeros(1, 1, 3)
    body_quat = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]])
    offsets = (
      (-0.05, 0.025, -0.03),
      (-0.05, -0.025, -0.03),
      (0.12, 0.03, -0.03),
      (0.12, -0.03, -0.03),
    )

    corners = virtual_foot_corner_points(
      body_pos_w=body_pos,
      body_quat_w=body_quat,
      corner_local_offsets_m=offsets,
    )

    self.assertEqual(CORNER_NAMES, ("rear_left", "rear_right", "front_left", "front_right"))
    self.assertTrue(torch.allclose(corners[0, 0], torch.tensor(offsets)))

  def test_region_state_groups_simultaneous_regions_into_one_touchdown(self):
    state = RegionalContactState(num_feet=1)

    events = state.update(
      step=4,
      region_contact=torch.tensor([[True, False, True]]),
      foot_names=("left",),
      foot_force_n=torch.tensor([120.0]),
      foot_loading_rate_n_s=torch.tensor([5000.0]),
      foot_vertical_speed_m_s=torch.tensor([0.4]),
      corner_downward_speeds_m_s=torch.tensor([[0.5, 0.6, 0.0, 0.0]]),
    )

    self.assertEqual(len(events), 1)
    self.assertEqual(events[0].stance_id, 0)
    self.assertEqual(events[0].event_type, "touchdown")
    self.assertEqual(events[0].regions, ("heel", "toe"))
    self.assertAlmostEqual(events[0].corner_downward_speed_m_s.item(), 0.6)
    self.assertTrue(
      torch.allclose(
        events[0].corner_downward_speeds_m_s,
        torch.tensor([0.5, 0.6, 0.0, 0.0]),
      )
    )

  def test_region_state_emits_secondary_when_new_region_contacts_in_same_stance(self):
    state = RegionalContactState(num_feet=1)
    first_events = state.update(
      step=1,
      region_contact=torch.tensor([[True, False, False]]),
      foot_names=("left",),
      foot_force_n=torch.tensor([80.0]),
      foot_loading_rate_n_s=torch.tensor([2000.0]),
      foot_vertical_speed_m_s=torch.tensor([0.3]),
      corner_downward_speeds_m_s=torch.tensor([[0.3, 0.4, 0.0, 0.0]]),
    )

    events = state.update(
      step=2,
      region_contact=torch.tensor([[True, False, True]]),
      foot_names=("left",),
      foot_force_n=torch.tensor([160.0]),
      foot_loading_rate_n_s=torch.tensor([4000.0]),
      foot_vertical_speed_m_s=torch.tensor([0.1]),
      corner_downward_speeds_m_s=torch.tensor([[0.0, 0.0, 0.2, 0.1]]),
    )

    self.assertEqual(first_events[0].stance_id, 0)
    self.assertEqual(len(events), 1)
    self.assertEqual(events[0].stance_id, 0)
    self.assertEqual(events[0].event_type, "secondary")
    self.assertEqual(events[0].regions, ("toe",))

    state.update(
      step=3,
      region_contact=torch.tensor([[False, False, False]]),
      foot_names=("left",),
      foot_force_n=torch.tensor([0.0]),
      foot_loading_rate_n_s=torch.tensor([0.0]),
      foot_vertical_speed_m_s=torch.tensor([0.0]),
      corner_downward_speeds_m_s=torch.zeros(1, 4),
    )
    next_events = state.update(
      step=4,
      region_contact=torch.tensor([[False, True, True]]),
      foot_names=("left",),
      foot_force_n=torch.tensor([100.0]),
      foot_loading_rate_n_s=torch.tensor([1000.0]),
      foot_vertical_speed_m_s=torch.tensor([0.2]),
      corner_downward_speeds_m_s=torch.tensor([[0.0, 0.0, 0.3, 0.2]]),
    )
    self.assertEqual(next_events[0].stance_id, 1)


if __name__ == "__main__":
  unittest.main()
