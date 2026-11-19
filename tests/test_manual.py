import unittest

from doge_control.manual import ManualSource


class ManualTests(unittest.TestCase):
    def test_command_expires_and_explicit_stop_latches(self):
        source = ManualSource([
            {"tick": 1, "forward_mps": 0.1},
            {"tick": 15, "forward_mps": -0.1},
            {"tick": 16, "stop": True},
            {"tick": 17, "forward_mps": 0.15},
        ])
        frames = [source(tick) for tick in range(20)]
        self.assertEqual(frames[0].reason, "operator_wait")
        self.assertEqual(frames[10].forward_mps, 0.1)
        self.assertEqual(frames[11].reason, "operator_timeout")
        self.assertEqual(frames[15].forward_mps, -0.1)
        self.assertTrue(all(frame.reason == "operator_stop" for frame in frames[16:]))

    def test_rejects_malformed_or_out_of_domain_commands(self):
        cases = (
            [{"tick": 0, "forward_mps": 0.16}],
            [{"tick": 0, "forward_mps": float("inf")}],
            [{"tick": True, "forward_mps": 0.1}],
            [{"tick": 0, "stop": False}],
            [{"tick": 0, "stop": True, "forward_mps": 0.1}],
            [{"tick": 3, "stop": True}, {"tick": 3, "forward_mps": 0.1}],
        )
        for rows in cases:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                ManualSource(rows)
