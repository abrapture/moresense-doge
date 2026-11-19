import unittest

from doge_control.autonomous import AutonomousSource, RANGE_TIMEOUT_TICKS, STOP_DISTANCE_M


class AutonomousTests(unittest.TestCase):
    def test_obstacle_stops_even_if_path_clears(self):
        source = AutonomousSource([
            {"tick": 0, "distance_m": 0.9},
            {"tick": 1, "distance_m": STOP_DISTANCE_M},
            {"tick": 2, "distance_m": 0.9},
        ])
        self.assertGreater(source(0).forward_mps, 0.0)
        for tick in range(1, 4):
            decision = source(tick)
            self.assertEqual((decision.forward_mps, decision.reason), (0.0, "obstacle"))

    def test_stale_or_missing_range_disallows_motion(self):
        source = AutonomousSource([{"tick": 0, "distance_m": 0.9}])
        self.assertGreater(source(0).forward_mps, 0.0)
        self.assertEqual(source(RANGE_TIMEOUT_TICKS).reason, "range_missing_or_stale")
        self.assertEqual(AutonomousSource([])(0).forward_mps, 0.0)

    def test_clear_route_is_bounded(self):
        source = AutonomousSource([{"tick": tick, "distance_m": 1.0} for tick in range(0, 70, 4)])
        frames = [source(tick) for tick in range(70)]
        self.assertEqual(frames[25].reason, "route_pause")
        self.assertEqual(frames[35].reason, "route_leg_2")
        self.assertTrue(all(frame.reason == "route_complete" for frame in frames[65:]))

    def test_invalid_ranges_are_rejected(self):
        for distance in (-0.1, float("nan"), False, "0.9"):
            with self.subTest(distance=distance), self.assertRaises(ValueError):
                AutonomousSource([{"tick": 0, "distance_m": distance}])
