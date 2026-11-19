"""CPU checks for the observable contract, timing, and episode isolation.

Run: python -m unittest discover -s training/tests -p test_bno08x.py -v
"""

from dataclasses import replace
import math
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bno08x import Bno08xCfg, Bno08xObservationModel, _rotate_by_rotvec


IDEAL = Bno08xCfg(
    gyro_noise_std=0, gyro_bias_std=0, tilt_noise_std_deg=0, tilt_bias_std_deg=0,
)


def down(count: int) -> torch.Tensor:
    return torch.tensor([0.0, 0.0, -1.0]).repeat(count, 1)


class Bno08xTests(unittest.TestCase):
    def test_known_rotation_and_gravity_contract(self):
        # A positive 90-degree X rotation turns world-down into positive Y.
        rotated = _rotate_by_rotvec(down(1), torch.tensor([[math.pi / 2, 0.0, 0.0]]))
        torch.testing.assert_close(rotated, torch.tensor([[0.0, 1.0, 0.0]]), atol=2e-7, rtol=0)
        model = Bno08xObservationModel(4, 0.005, replace(IDEAL, delay_s=0))
        gravity = torch.tensor([[0., 0., -1.], [0., 1., 0.], [1., 0., 0.], [0., 0., 1.]])
        model.reset(torch.zeros(4, 3), gravity)
        torch.testing.assert_close(model.output()[:, 3:], gravity)
        # No Euler angles or yaw appear in the six-channel contract.
        self.assertEqual(tuple(model.output().shape), (4, 6))

    def test_exact_delay_step_response(self):
        model = Bno08xObservationModel(1, 0.005, IDEAL)
        model.reset(torch.zeros(1, 3), down(1))
        observed = []
        ages = []
        for tick in range(1, 9):
            model.advance(torch.tensor([[float(tick), 0., 0.]]), down(1), dt=0.005)
            observed.append(float(model.output()[0, 0]))
            ages.append(float(model.sample_age_s[0]))
        # New measurements at 10/20/30 ms arrive exactly 10 ms later.
        self.assertEqual(observed, [0., 0., 0., 2., 2., 4., 4., 6.])
        self.assertAlmostEqual(ages[3], 0.010, places=6)
        self.assertAlmostEqual(ages[4], 0.015, places=6)

    def test_sample_hold_readonly_and_clone(self):
        model = Bno08xObservationModel(1, 0.005, replace(Bno08xCfg(), delay_s=0), seed=8)
        model.reset(torch.zeros(1, 3), down(1))
        primed = model.output()
        torch.testing.assert_close(model.output(), primed, rtol=0, atol=0)
        model.advance(torch.ones(1, 3), down(1))
        torch.testing.assert_close(model.output(), primed, rtol=0, atol=0)
        model.advance(torch.ones(1, 3), down(1))
        sampled = model.output()
        self.assertGreater(float(sampled[0, 0]), 0.9)
        sampled[:] = 99
        self.assertLess(float(model.output()[0, 0]), 1.1)
        held = model.output()
        model.advance(torch.full((1, 3), 8.), down(1))
        torch.testing.assert_close(model.output(), held, rtol=0, atol=0)

    def test_partial_reset_removes_old_episode_only(self):
        model = Bno08xObservationModel(2, 0.005, IDEAL)
        reference = Bno08xObservationModel(2, 0.005, IDEAL)
        for item in (model, reference):
            item.reset(torch.zeros(2, 3), down(2))
        for tick in range(1, 4):
            gyro = torch.full((2, 3), float(tick))
            for item in (model, reference):
                item.advance(gyro, down(2))
        # Reset at an odd physics tick: its sampling phase must also restart.
        new_gravity = torch.tensor([[0., 1., 0.]])
        model.reset(torch.full((1, 3), 100.), new_gravity, env_ids=[0])
        torch.testing.assert_close(model.output()[0], torch.tensor([100., 100., 100., 0., 1., 0.]))
        torch.testing.assert_close(model.output()[1], reference.output()[1])
        self.assertEqual(float(model.sample_age_s[0]), 0.)
        for tick in range(4, 12):
            gyro = torch.full((2, 3), float(tick))
            reference.advance(gyro, down(2))
            gyro[0] = 100.
            gravity = down(2)
            gravity[0] = new_gravity[0]
            model.advance(gyro, gravity)
            # Old values must never re-emerge from any reset delay-line slot.
            torch.testing.assert_close(model.output()[0, :3], torch.full((3,), 100.))
            torch.testing.assert_close(model.output()[0, 3:], new_gravity[0])
            torch.testing.assert_close(model.output()[1], reference.output()[1])
            torch.testing.assert_close(model.sample_age_s[1], reference.sample_age_s[1])

    def test_noise_scales_and_unit_gravity(self):
        count = 12000
        cfg = Bno08xCfg(delay_s=0)
        model = Bno08xObservationModel(count, 0.005, cfg, seed=123)
        model.reset(torch.zeros(count, 3), down(count))
        bias_std = float(model.gyro_bias_b.std())
        self.assertAlmostEqual(bias_std, cfg.gyro_bias_std, delta=cfg.gyro_bias_std * .05)
        residuals = []
        for _ in range(4):
            model.advance(torch.zeros(count, 3), down(count))
            model.advance(torch.zeros(count, 3), down(count))
            residuals.append(model.output()[:, :3] - model.gyro_bias_b)
        noise_std = float(torch.cat(residuals).std())
        self.assertAlmostEqual(noise_std, cfg.gyro_noise_std, delta=cfg.gyro_noise_std * .04)
        gravity = model.output()[:, 3:]
        torch.testing.assert_close(torch.linalg.vector_norm(gravity, dim=-1), torch.ones(count), atol=2e-7, rtol=0)
        # Two tilt axes contribute to the direction error. Rotation around down
        # is unobservable, so the small-angle RMS is sqrt(2)*combined sigma.
        angle = torch.atan2(torch.linalg.vector_norm(gravity[:, :2], dim=-1), -gravity[:, 2])
        expected_rms = math.sqrt(2) * math.radians(math.hypot(cfg.tilt_noise_std_deg, cfg.tilt_bias_std_deg))
        actual_rms = float(angle.square().mean().sqrt())
        self.assertAlmostEqual(actual_rms, expected_rms, delta=expected_rms * .05)

    def test_seeded_cpu_reproducibility_with_partial_resets(self):
        models = [Bno08xObservationModel(5, 0.005, seed=456) for _ in range(2)]
        for model in models:
            model.reset(torch.zeros(5, 3), down(5))
        for tick in range(20):
            if tick == 7:
                for model in models:
                    model.reset(torch.ones(2, 3), down(2), [1, 3])
            for model in models:
                model.advance(torch.full((5, 3), tick / 10), down(5))
            torch.testing.assert_close(models[0].output(), models[1].output(), rtol=0, atol=0)
        torch.testing.assert_close(models[0].gyro_bias_b, models[1].gyro_bias_b, rtol=0, atol=0)

    def test_full_batch_partial_reset_and_empty_reset(self):
        model = Bno08xObservationModel(3, 0.005, IDEAL)
        model.reset(torch.zeros(3, 3), down(3))
        gyro = torch.arange(9, dtype=torch.float32).reshape(3, 3)
        model.reset(gyro, down(3), torch.tensor([2, 0]))
        torch.testing.assert_close(model.output()[2, :3], gyro[2])
        torch.testing.assert_close(model.output()[0, :3], gyro[0])
        torch.testing.assert_close(model.output()[1, :3], torch.zeros(3))
        before = model.output()
        model.reset(torch.empty(0, 3), torch.empty(0, 3), torch.empty(0, dtype=torch.long))
        torch.testing.assert_close(model.output(), before, rtol=0, atol=0)

    def test_rejects_inexact_timing_invalid_values_and_uninitialized_read(self):
        for dt, cfg in ((1 / 120, IDEAL), (.005, replace(IDEAL, delay_s=.007)), (.02, IDEAL)):
            with self.assertRaises(ValueError):
                Bno08xObservationModel(1, dt, cfg)
        with self.assertRaises(ValueError):
            Bno08xObservationModel(1, .005, replace(IDEAL, gyro_noise_std=-1))
        model = Bno08xObservationModel(1, .005, replace(IDEAL, validate_inputs=True))
        with self.assertRaises(RuntimeError):
            model.output()
        with self.assertRaises(ValueError):
            model.reset(torch.zeros(1, 3), torch.zeros(1, 3))
        model.reset(torch.zeros(1, 3), down(1))
        with self.assertRaises(ValueError):
            model.advance(torch.zeros(1, 3), down(1), dt=.02)
        with self.assertRaises(ValueError):
            model.advance(torch.full((1, 3), float('nan')), down(1))
        with self.assertRaises(ValueError):
            model.reset(torch.zeros(1, 3), down(1), [0, 0])


if __name__ == "__main__":
    unittest.main()
