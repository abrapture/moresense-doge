from importlib.util import find_spec
import math
import unittest

from doge_control.model import RobotModel
from doge_control.preview import Observation
from doge_control.rl import RlV1, validate_snapshot


class RlTests(unittest.TestCase):
    def setUp(self):
        self.model = RobotModel.load()
        self.state = Observation(
            tuple(angle + 0.01 for angle in self.model.stand_rad), (0.04,) * 12,
            gyro_rad_s=(0.1, 0.2, 0.3), gravity_body=(0.0, 0.0, -1.0),
        )

    def test_preserved_snapshot_matches_metadata(self):
        validate_snapshot(self.model)

    def test_policy_contract_orders_observations_and_clips_actions(self):
        calls = []

        def predict(observation):
            calls.append(observation)
            return (2.0, -2.0) + (0.0,) * 10

        controller = RlV1(self.model, predict)
        targets = controller.targets(self.state, 0.15, 0)
        controller.targets(self.state, -0.1, 1)
        self.assertEqual(len(calls[0]), 45)
        self.assertEqual(calls[0][:6], self.state.gyro_rad_s + self.state.gravity_body)
        for value in calls[0][6:18]:
            self.assertAlmostEqual(value, 0.01)
        self.assertEqual(calls[0][18:30], self.state.velocities_rad_s)
        self.assertEqual(calls[0][30:42], (0.0,) * 12)
        self.assertEqual(calls[0][42:], (0.0, 0.15, 0.0))
        self.assertEqual(calls[1][30:42], (1.0, -1.0) + (0.0,) * 10)
        self.assertEqual(calls[1][42:], (0.0, -0.1, 0.0))
        self.assertAlmostEqual(targets[0] - self.model.stand_rad[0], 0.25)
        self.assertAlmostEqual(targets[1] - self.model.stand_rad[1], -0.25)
        controller.reset()
        self.assertEqual(controller.previous_action, (0.0,) * 12)

    def test_non_finite_actions_are_rejected(self):
        controller = RlV1(self.model, lambda observation: (math.nan,) * 12)
        with self.assertRaises(ValueError):
            controller.targets(self.state, 0.1, 0)

    @unittest.skipUnless(find_spec("torch"), "PyTorch is required for the archived CPU policy test")
    def test_actual_archived_policy_responds_to_command_on_cpu(self):
        controller = RlV1.load(self.model)
        forward = controller.targets(self.state, 0.15, 0)
        controller.reset()
        reverse = controller.targets(self.state, -0.15, 0)
        self.assertTrue(all(map(math.isfinite, forward)))
        self.assertNotEqual(forward, self.model.stand_rad)
        self.assertNotEqual(forward, reverse)
