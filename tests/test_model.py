import math
from pathlib import Path
import unittest
from xml.etree import ElementTree

from doge_control.model import JOINT_ORDER, MAX_TARGET_RATE_RAD_S, PERIOD_S, ROOT, URDF, RobotModel
from doge_control.preview import Decision, Preview


class ModelTests(unittest.TestCase):
    def test_calibrated_urdf_matches_preview_contract(self):
        model = RobotModel.load()
        robot = ElementTree.parse(ROOT / URDF).getroot()
        self.assertEqual(len(robot.findall("link")), 13)
        joints = {joint.get("name"): joint for joint in robot.findall("joint")}
        self.assertEqual(set(joints), set(JOINT_ORDER))
        for name, (lower, upper) in zip(JOINT_ORDER, model.bounds_rad, strict=True):
            with self.subTest(name=name):
                self.assertEqual(joints[name].get("type"), "revolute")
                bounds = joints[name].find("limit")
                self.assertAlmostEqual(lower, float(bounds.get("lower")))
                self.assertAlmostEqual(upper, float(bounds.get("upper")))
        self.assertGreater(model.stand_rad[8], math.pi)
        self.assertEqual(len(robot.findall(".//collision//box")), 33)
        self.assertEqual(len(robot.findall(".//collision//cylinder")), 4)

    def test_urdf_meshes_resolve_inside_archive(self):
        for urdf in (ROOT / "urdf").glob("*.urdf"):
            robot = ElementTree.parse(urdf)
            for mesh in robot.findall(".//mesh"):
                path = Path(mesh.get("filename"))
                self.assertFalse(path.is_absolute(), path)
                actual = (urdf.parent / path).resolve()
                self.assertTrue(actual.is_relative_to(ROOT / "meshes") and actual.is_file(), actual)

    def test_targets_respect_rate_and_limits(self):
        model = RobotModel.load()
        targets = model.limit((99.0,) * 12, model.stand_rad)
        for old, new, (lower, upper) in zip(model.stand_rad, targets, model.bounds_rad, strict=True):
            self.assertLessEqual(abs(new - old), MAX_TARGET_RATE_RAD_S * PERIOD_S + 1e-12)
            self.assertLessEqual(lower, new)
            self.assertLessEqual(new, upper)
        with self.assertRaises(ValueError):
            model.limit((math.nan,) * 12, model.stand_rad)

    def test_stand_preview_is_stationary(self):
        model = RobotModel.load()
        preview = Preview(model, lambda tick: Decision(0.0, "test"))
        for tick in range(3):
            frame = preview.step()
            self.assertEqual(frame["tick"], tick)
            self.assertEqual(tuple(frame["joint_targets_rad"].values()), model.stand_rad)
