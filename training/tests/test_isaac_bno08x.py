"""Bridge and quaternion-contract checks; no Isaac runtime / GPU is required."""

from __future__ import annotations

import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from training.bno08x import Bno08xCfg
from training.isaac_bno08x import IsaacBno08xObserver, fused_quaternion_to_projected_gravity


class Proxy:
    def __init__(self, value):
        self.torch = value


def noiseless_cfg():
    return Bno08xCfg(
        gyro_noise_std=0.0,
        gyro_bias_std=0.0,
        tilt_noise_std_deg=0.0,
        tilt_bias_std_deg=0.0,
    )


def make_env(*, with_gravity=True):
    gyro = torch.zeros((2, 3))
    gravity = torch.tensor([[0.0, 0.0, -1.0]]).repeat(2, 1)
    quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(2, 1)
    data = SimpleNamespace(root_ang_vel_b=Proxy(gyro), root_quat_w=Proxy(quat))
    if with_gravity:
        data.projected_gravity_b = Proxy(gravity)
    robot = SimpleNamespace(data=data)
    return SimpleNamespace(physics_dt=0.005, num_envs=2, scene={"robot": robot}), gyro, gravity, quat


class QuaternionContractTests(unittest.TestCase):
    def test_xyzw_identity_roll_pitch_and_yaw(self):
        h = math.sqrt(0.5)
        # Positive 90-degree body-to-world roll, pitch, yaw, respectively.
        q = torch.tensor([[0, 0, 0, 1], [h, 0, 0, h], [0, h, 0, h], [0, 0, h, h]], dtype=torch.float64)
        expected = torch.tensor([[0, 0, -1], [0, -1, 0], [1, 0, 0], [0, 0, -1]], dtype=torch.float64)
        torch.testing.assert_close(fused_quaternion_to_projected_gravity(q), expected, atol=1e-12, rtol=0)
        torch.testing.assert_close(fused_quaternion_to_projected_gravity(-3 * q), expected, atol=1e-12, rtol=0)

    def test_invalid_quaternions_are_rejected(self):
        for q in (torch.zeros(4), torch.tensor([0.0, 0.0, float("nan"), 1.0]), torch.ones(3)):
            with self.assertRaises(ValueError):
                fused_quaternion_to_projected_gravity(q)


class ObserverTests(unittest.TestCase):
    def test_proxyarray_delay_and_read_are_explicit(self):
        env, gyro, _, _ = make_env()
        observer = IsaacBno08xObserver(env, noiseless_cfg())
        gyro[:, 0] = 1.25
        # Reads cannot advance the sensor or bypass the delayed sample.
        for _ in range(3):
            torch.testing.assert_close(observer.read()[:, :3], torch.zeros((2, 3)))
        for _ in range(3):
            observer.advance(0.005)
            torch.testing.assert_close(observer.read()[:, :3], torch.zeros((2, 3)))
        observer.advance(0.005)  # sample at 10 ms becomes visible at 20 ms
        torch.testing.assert_close(observer.read()[:, :3], gyro)
        owned = observer.read()
        owned.zero_()
        torch.testing.assert_close(observer.read()[:, :3], gyro)

    def test_selective_reset_flushes_only_reset_environment(self):
        env, gyro, _, _ = make_env()
        observer = IsaacBno08xObserver(env, noiseless_cfg())
        gyro[:, 0] = torch.tensor([2.0, 3.0])
        for _ in range(4):
            observer.advance()
        gyro[:, 0] = torch.tensor([7.0, 9.0])
        observer.reset(torch.tensor([0]))
        torch.testing.assert_close(observer.read()[:, 0], torch.tensor([7.0, 3.0]))

    def test_quaternion_fallback_and_explicit_robot(self):
        env, _, _, quat = make_env(with_gravity=False)
        quat[0] = torch.tensor([math.sqrt(0.5), 0, 0, math.sqrt(0.5)])
        observer = IsaacBno08xObserver(env, noiseless_cfg(), robot=env.scene["robot"])
        torch.testing.assert_close(observer.read()[0, 3:], torch.tensor([0.0, -1.0, 0.0]), atol=1e-6, rtol=0)

    def test_control_interval_cannot_replace_physics_ticks(self):
        env, _, _, _ = make_env()
        observer = IsaacBno08xObserver(env, noiseless_cfg())
        with self.assertRaises(ValueError):
            observer.advance(0.020)


if __name__ == "__main__":
    unittest.main()
