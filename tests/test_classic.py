import json
import unittest

from doge_control.classic import ClassicTrot, SWING_HEIGHT_M, sagittal_offset
from doge_control.model import MAX_FORWARD_MPS, ROOT, STAND_POSE, RobotModel


class ClassicTests(unittest.TestCase):
    def test_forward_kinematics_recovers_archived_foot_centers(self):
        pose = json.loads((ROOT / STAND_POSE).read_text(encoding="utf-8"))
        controller = ClassicTrot(RobotModel.load())
        for number, center in enumerate(controller.reference, start=1):
            reported = pose["diagnostics"]["feet"][f"leg_{number}_knee_link"]["center_world_m"]
            for measured, reference, root in zip(reported, center, pose["initial_root_position_m"], strict=True):
                self.assertAlmostEqual(measured, reference + root, places=7)

    def test_diagonal_swing_raises_paired_feet(self):
        controller = ClassicTrot(RobotModel.load())
        controller.targets(None, MAX_FORWARD_MPS, 0)
        targets = controller.targets(None, MAX_FORWARD_MPS, 7)
        heights = [
            leg.foot_center(targets[index * 3:index * 3 + 3])[2] - controller.reference[index][2]
            for index, leg in enumerate(controller.legs)
        ]
        self.assertAlmostEqual(heights[0], SWING_HEIGHT_M, places=4)
        self.assertAlmostEqual(heights[3], heights[0], places=6)
        self.assertAlmostEqual(heights[1], 0.0, places=6)
        self.assertAlmostEqual(heights[2], 0.0, places=6)

    def test_stride_reverses_without_reversing_lift(self):
        forward, lift = sagittal_offset(0.1, MAX_FORWARD_MPS)
        reverse, reverse_lift = sagittal_offset(0.1, -MAX_FORWARD_MPS)
        self.assertEqual(reverse, -forward)
        self.assertEqual(reverse_lift, lift)

    def test_full_cycles_fit_inside_model_limits(self):
        model = RobotModel.load()
        for command in (MAX_FORWARD_MPS, -MAX_FORWARD_MPS):
            controller = ClassicTrot(model)
            for tick in range(80):
                targets = controller.targets(None, command, tick)
                for target, (lower, upper) in zip(targets, model.bounds_rad, strict=True):
                    self.assertLessEqual(lower, target)
                    self.assertLessEqual(target, upper)
