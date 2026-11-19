from contextlib import redirect_stdout
from importlib.util import find_spec
from io import StringIO
import json
import unittest
from unittest.mock import patch

from doge_control.cli import main
from doge_control.model import JOINT_ORDER, RobotModel


def run_preview(*args):
    output = StringIO()
    with redirect_stdout(output):
        main(list(args))
    return [json.loads(line) for line in output.getvalue().splitlines()]


def timeline(frames):
    return [(item["tick"], item["command_vy_mps"], item["reason"]) for item in frames]


class IntegrationTests(unittest.TestCase):
    def test_classic_modes_stop_and_return_to_model_stand(self):
        manual = run_preview("classic-manual", "--ticks", "48")
        autonomous = run_preview("classic-autonomous", "--ticks", "48")
        self.assertEqual(timeline(manual)[18], (18, 0.0, "operator_timeout"))
        self.assertEqual(timeline(manual)[22], (22, -0.1, "operator"))
        self.assertEqual(timeline(manual)[34:], [(tick, 0.0, "operator_stop") for tick in range(34, 48)])
        self.assertEqual(timeline(autonomous)[16:], [(tick, 0.0, "obstacle") for tick in range(16, 48)])
        for frames in (manual, autonomous):
            self.assertEqual(tuple(frames[0]["joint_targets_rad"]), JOINT_ORDER)
            self.assertNotEqual(frames[0]["joint_targets_rad"], frames[1]["joint_targets_rad"])
            self.assertEqual(tuple(frames[-1]["joint_targets_rad"].values()), RobotModel.load().stand_rad)

    def test_manual_can_be_overridden_from_stdin(self):
        with patch("sys.stdin", StringIO('{"tick":1,"forward_mps":0.07}\n{"tick":2,"stop":true}\n')):
            frames = run_preview("classic-manual", "--commands", "-", "--ticks", "4")
        self.assertEqual(timeline(frames), [
            (0, 0.0, "operator_wait"), (1, 0.07, "operator"),
            (2, 0.0, "operator_stop"), (3, 0.0, "operator_stop"),
        ])

    @unittest.skipUnless(find_spec("torch"), "PyTorch is required for the V1 integration test")
    def test_rl_and_classic_use_same_commands_but_different_joint_targets(self):
        for source in ("manual", "autonomous"):
            with self.subTest(source=source):
                classic = run_preview(f"classic-{source}", "--ticks", "48")
                learned = run_preview(f"rl-{source}", "--ticks", "48")
                self.assertEqual(timeline(classic), timeline(learned))
                self.assertNotEqual(classic[0]["joint_targets_rad"], learned[0]["joint_targets_rad"])
                self.assertEqual(classic[-1]["joint_targets_rad"], learned[-1]["joint_targets_rad"])
