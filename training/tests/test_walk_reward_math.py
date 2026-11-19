"""CPU-only checks for geometry and reward shaping; no Newton import."""
import math
import unittest
import torch
from moresense_doge.training.walk_reward_math import (foot_centers_in_base, expanded_footprint,
    footprint_penalty, heading_error, symmetry_penalty, terminal_fall_penalty)


class WalkRewardMathTest(unittest.TestCase):
    def test_frame_transform_and_local_cylinder_offset(self):
        # The robot and all legs are rotated 90 degrees about world Z.
        c = math.sqrt(0.5)
        root = torch.tensor([[10., 20., 3., 0., 0., c, c]])
        links = torch.tensor([[[10., 20., 3., 0., 0., c, c],
                               [10., 21., 3., 0., 0., c, c],
                               [10., 22., 3., 0., 0., c, c],
                               [10., 23., 3., 0., 0., c, c],
                               [10., 24., 3., 0., 0., c, c]]])
        local = torch.tensor([[0., 0.5, 0.]] * 4)
        got = foot_centers_in_base(root, links, [1, 2, 3, 4], local)
        expected = torch.tensor([[[1., 0.5, 0.], [2., 0.5, 0.], [3., 0.5, 0.], [4., 0.5, 0.]]])
        torch.testing.assert_close(got, expected, atol=1e-5, rtol=0)

    def test_soft_penalty_for_stand_and_movement(self):
        nominal = torch.tensor([[-.1, .12, -.14], [-.1, -.12, -.14],
                                [.1, .12, -.14], [.1, -.12, -.14]])
        target = expanded_footprint(nominal)
        torch.testing.assert_close(target[:, :2], torch.tensor([[-.11, .13], [-.11, -.13],
                                                               [.11, .13], [.11, -.13]]))
        feet = target[None].repeat(2, 1, 1)
        feet[:, :, 1] += .020
        result = footprint_penalty(feet, target, torch.tensor([0., .15]))
        self.assertGreater(result[0].item(), 0.)
        self.assertEqual(result[1].item(), 0.)
        feet[0, 0, 0] += .04
        self.assertGreater(symmetry_penalty(feet, target, torch.tensor([0., .15]))[0].item(), 0.)

    def test_heading_and_quaternion_sign(self):
        identity = torch.tensor([0., 0., 0., 1.])
        angles = torch.tensor([0., math.pi/2, -math.pi/2])
        quats = torch.stack((angles*0, angles*0, torch.sin(angles/2), torch.cos(angles/2)), -1)
        torch.testing.assert_close(heading_error(quats, identity), angles)
        torch.testing.assert_close(heading_error(-quats, identity), angles)

    def test_terminal_cost_is_one_off_and_excludes_timeout(self):
        # Values come from _get_dones' terminated output, not time_out.
        terminated = torch.tensor([True, False, False])
        timeout = torch.tensor([False, True, False])
        cost = terminal_fall_penalty(terminated, 2.0)
        torch.testing.assert_close(cost, torch.tensor([2., 0., 0.]))
        self.assertEqual(cost[1].item(), 0.)
        self.assertTrue(timeout[1].item())
        with self.assertRaises(ValueError):
            terminal_fall_penalty(terminated, float('nan'))


if __name__ == '__main__':
    unittest.main()
