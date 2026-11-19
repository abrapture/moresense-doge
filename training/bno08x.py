"""A small, backend-neutral model of the BNO08x observations used by the actor.

This models calibrated gyro and gravity inferred from fused orientation. It does
not emulate SH-2 firmware, raw accelerometer reports, magnetic heading, or a bus.
The six output channels are ``[gyro_b (rad/s), projected_gravity_b (unitless)]``.
The latter points DOWN and equals ``[0, 0, -1]`` for an upright robot. An actual
BNO driver must adapt its quaternion/sign/axes to this contract.

Call ``advance`` exactly once after every physics step, including decimation
substeps. Reading ``output`` never samples noise or advances time. A reset primes
the entire delay line with ONE noisy measurement of the reset state; that value
is immediately available and held until a fresh sample traverses the delay.

The default numbers are initial training assumptions, not BNO08x specifications.
Sensor and body axes are assumed aligned. No position offset is needed for these
two outputs; proper acceleration would require a separate model.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch


@dataclass(frozen=True)
class Bno08xCfg:
    sample_rate_hz: float = 100.0
    delay_s: float = 0.010
    gyro_noise_std: float = 0.005
    gyro_bias_std: float = 0.002
    tilt_noise_std_deg: float = 0.25
    tilt_bias_std_deg: float = 0.2
    # Enabling this performs finite/nonzero checks with a device synchronization
    # on every physics step. Reset always validates its inputs.
    validate_inputs: bool = False


def _rotate_by_rotvec(vector: torch.Tensor, rotvec: torch.Tensor) -> torch.Tensor:
    """Rotate vectors using a stable Rodrigues formula, including zero angles."""
    angle = torch.linalg.vector_norm(rotvec, dim=-1, keepdim=True)
    first_cross = torch.linalg.cross(rotvec, vector, dim=-1)
    sin_over_angle = torch.sinc(angle / math.pi)
    one_minus_cos_over_angle_squared = 0.5 * torch.sinc(angle / (2.0 * math.pi)).square()
    return (
        vector
        + sin_over_angle * first_cross
        + one_minus_cos_over_angle_squared * torch.linalg.cross(rotvec, first_cross, dim=-1)
    )


class Bno08xObservationModel:
    """Vectorized, independently resettable sample/hold and fixed-delay sensor.

    ``physics_dt`` must divide both the sample period and delay exactly (within
    floating point tolerance). This rejects a misleading rounded 100 Hz or
    10 ms configuration. For the defaults, physics_dt=0.005 works exactly.

    Inputs are tensors of shape ``(num_envs, 3)`` on the configured device, with
    the configured floating dtype. Gyro is expressed in body axes in rad/s.
    Gravity is a finite, nonzero down direction in the same axes and is
    normalized internally. Bias is sampled once per environment reset; white
    noise changes only at sensor sample times and remains held between samples.
    """

    def __init__(
        self,
        num_envs: int,
        physics_dt: float,
        cfg: Bno08xCfg | None = None,
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
        seed: int = 0,
    ) -> None:
        if not isinstance(num_envs, int) or isinstance(num_envs, bool) or num_envs < 1:
            raise ValueError("num_envs must be a positive integer")
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("dtype must be torch.float32 or torch.float64")
        self.cfg = cfg or Bno08xCfg()
        self.num_envs = num_envs
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.dtype = dtype
        self.physics_dt = self._positive_finite("physics_dt", physics_dt)
        sample_rate = self._positive_finite("sample_rate_hz", self.cfg.sample_rate_hz)
        self.sample_period_steps = self._exact_steps("sample period", 1.0 / sample_rate, allow_zero=False)
        self.delay_steps = self._exact_steps("delay_s", self.cfg.delay_s, allow_zero=True)
        for name in ("gyro_noise_std", "gyro_bias_std", "tilt_noise_std_deg", "tilt_bias_std_deg"):
            self._nonnegative_finite(name, getattr(self.cfg, name))
        self._tilt_noise_std = math.radians(self.cfg.tilt_noise_std_deg)
        self._tilt_bias_std = math.radians(self.cfg.tilt_bias_std_deg)
        self._generator = torch.Generator(device=self.device)
        self._generator.manual_seed(seed)
        shape = (num_envs, 3)
        self._gyro_bias_b = torch.zeros(shape, device=self.device, dtype=dtype)
        self._tilt_bias_rotvec_b = torch.zeros_like(self._gyro_bias_b)
        self._held = torch.zeros((num_envs, 6), device=self.device, dtype=dtype)
        self._output = torch.zeros_like(self._held)
        self._steps_until_sample = torch.full(
            (num_envs,), self.sample_period_steps, device=self.device, dtype=torch.long
        )
        self._held_sample_tick = torch.zeros(num_envs, device=self.device, dtype=torch.long)
        self._output_sample_tick = torch.zeros_like(self._held_sample_tick)
        self._history = torch.zeros(
            (self.delay_steps + 1, num_envs, 6), device=self.device, dtype=dtype
        )
        self._history_sample_tick = torch.zeros(
            (self.delay_steps + 1, num_envs), device=self.device, dtype=torch.long
        )
        self._tick = 0
        self._head = 0
        self._initialized_envs: set[int] = set()
        self._ready = False

    @staticmethod
    def _positive_finite(name: str, value: float) -> float:
        value = float(value)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
        return value

    @staticmethod
    def _nonnegative_finite(name: str, value: float) -> float:
        value = float(value)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
        return value

    def _exact_steps(self, name: str, seconds: float, *, allow_zero: bool) -> int:
        seconds = self._nonnegative_finite(name, seconds)
        steps = seconds / self.physics_dt
        rounded = round(steps)
        if (rounded < (0 if allow_zero else 1)) or not math.isclose(
            steps, rounded, rel_tol=1e-7, abs_tol=1e-7
        ):
            raise ValueError(
                f"{name}={seconds:g}s must be an integer multiple of "
                f"physics_dt={self.physics_dt:g}s; no timing is rounded"
            )
        return rounded

    def _check_tensor(self, value: torch.Tensor, name: str, rows: int) -> None:
        if not isinstance(value, torch.Tensor) or value.shape != (rows, 3):
            raise ValueError(f"{name} must be a tensor with shape ({rows}, 3)")
        if value.device != self.device or value.dtype != self.dtype:
            raise ValueError(f"{name} must use device={self.device}, dtype={self.dtype}")

    @staticmethod
    def _validate_values(gyro: torch.Tensor, gravity: torch.Tensor) -> None:
        if not bool(torch.isfinite(gyro).all()) or not bool(torch.isfinite(gravity).all()):
            raise ValueError("gyro and gravity must contain only finite values")
        if bool((torch.linalg.vector_norm(gravity, dim=-1) < 1e-8).any()):
            raise ValueError("projected gravity must have a nonzero direction")

    def _random(self, rows: int) -> torch.Tensor:
        return torch.randn(
            (rows, 3), generator=self._generator, device=self.device, dtype=self.dtype
        )

    def _measurement(
        self, gyro: torch.Tensor, gravity: torch.Tensor,
        gyro_bias: torch.Tensor, tilt_bias: torch.Tensor,
    ) -> torch.Tensor:
        gyro_measured = gyro + gyro_bias
        if self.cfg.gyro_noise_std:
            gyro_measured = gyro_measured + self.cfg.gyro_noise_std * self._random(gyro.shape[0])
        gravity = gravity / torch.linalg.vector_norm(gravity, dim=-1, keepdim=True).clamp_min(1e-8)
        gravity_measured = _rotate_by_rotvec(gravity, tilt_bias)
        if self._tilt_noise_std:
            gravity_measured = _rotate_by_rotvec(
                gravity_measured, self._tilt_noise_std * self._random(gyro.shape[0])
            )
        gravity_measured = gravity_measured / torch.linalg.vector_norm(
            gravity_measured, dim=-1, keepdim=True
        ).clamp_min(1e-8)
        return torch.cat((gyro_measured, gravity_measured), dim=-1)

    @torch.no_grad()
    def reset(
        self,
        gyro_b: torch.Tensor,
        projected_gravity_b: torch.Tensor,
        env_ids: torch.Tensor | Sequence[int] | None = None,
    ) -> None:
        """Prime selected environments and re-sample their episode biases.

        With env_ids, inputs may contain either all N environments or just the
        selected M rows in env_ids order. Other environments' state is untouched.
        Empty env_ids is a no-op. All environments must be reset before advance.
        """
        if env_ids is None:
            ids = torch.arange(self.num_envs, device=self.device)
        else:
            ids = torch.as_tensor(env_ids, device=self.device)
            if ids.ndim == 1 and ids.numel() == 0:
                return
            if ids.ndim != 1 or ids.dtype not in (torch.int32, torch.int64):
                raise ValueError("env_ids must be a one-dimensional integer sequence")
            ids = ids.to(dtype=torch.long)
        rows = ids.numel()
        if rows == 0:
            return
        if int(ids.min()) < 0 or int(ids.max()) >= self.num_envs or ids.unique().numel() != rows:
            raise ValueError("env_ids must be unique valid environment indices")
        if not isinstance(gyro_b, torch.Tensor) or not isinstance(projected_gravity_b, torch.Tensor):
            raise ValueError("reset inputs must be Torch tensors")
        if gyro_b.shape == (self.num_envs, 3) and projected_gravity_b.shape == (self.num_envs, 3):
            self._check_tensor(gyro_b, "gyro_b", self.num_envs)
            self._check_tensor(projected_gravity_b, "projected_gravity_b", self.num_envs)
            gyro_b, projected_gravity_b = gyro_b[ids], projected_gravity_b[ids]
        else:
            self._check_tensor(gyro_b, "gyro_b", rows)
            self._check_tensor(projected_gravity_b, "projected_gravity_b", rows)
        self._validate_values(gyro_b, projected_gravity_b)
        self._gyro_bias_b[ids] = self.cfg.gyro_bias_std * self._random(rows)
        self._tilt_bias_rotvec_b[ids] = self._tilt_bias_std * self._random(rows)
        primed = self._measurement(
            gyro_b, projected_gravity_b, self._gyro_bias_b[ids], self._tilt_bias_rotvec_b[ids]
        )
        self._held[ids] = primed
        self._output[ids] = primed
        self._history[:, ids] = primed.unsqueeze(0)
        self._held_sample_tick[ids] = self._tick
        self._output_sample_tick[ids] = self._tick
        self._history_sample_tick[:, ids] = self._tick
        self._steps_until_sample[ids] = self.sample_period_steps
        self._initialized_envs.update(ids.detach().cpu().tolist())
        self._ready = len(self._initialized_envs) == self.num_envs

    @torch.no_grad()
    def advance(
        self, gyro_b: torch.Tensor, projected_gravity_b: torch.Tensor,
        dt: float | None = None,
    ) -> None:
        """Consume the state AFTER one physics step; read output separately.

        Optional dt is a guard against calling this at the policy timestep. A
        different dt is rejected. Production inputs must stay finite/nonzero;
        enable cfg.validate_inputs to check those values on each call.
        """
        if not self._ready:
            raise RuntimeError("reset every environment before advancing the BNO model")
        if dt is not None and not math.isclose(float(dt), self.physics_dt, rel_tol=1e-7, abs_tol=1e-10):
            raise ValueError("advance dt differs from the configured physics_dt")
        self._check_tensor(gyro_b, "gyro_b", self.num_envs)
        self._check_tensor(projected_gravity_b, "projected_gravity_b", self.num_envs)
        if self.cfg.validate_inputs:
            self._validate_values(gyro_b, projected_gravity_b)
        self._tick += 1
        self._steps_until_sample -= 1
        sample_now = self._steps_until_sample == 0
        # No device-to-host synchronization occurs in the training loop. Noise
        # candidates are drawn in a batch, and only due samples enter the queue.
        candidate = self._measurement(
            gyro_b, projected_gravity_b, self._gyro_bias_b, self._tilt_bias_rotvec_b
        )
        self._held.copy_(torch.where(sample_now[:, None], candidate, self._held))
        self._held_sample_tick.copy_(torch.where(sample_now, self._tick, self._held_sample_tick))
        self._steps_until_sample.copy_(torch.where(
            sample_now, self.sample_period_steps, self._steps_until_sample
        ))
        self._head = (self._head + 1) % self._history.shape[0]
        self._history[self._head].copy_(self._held)
        self._history_sample_tick[self._head].copy_(self._held_sample_tick)
        delayed_index = (self._head - self.delay_steps) % self._history.shape[0]
        self._output.copy_(self._history[delayed_index])
        self._output_sample_tick.copy_(self._history_sample_tick[delayed_index])

    def output(self) -> torch.Tensor:
        """Return an independent [N, 6] snapshot; repeated reads are identical."""
        if not self._ready:
            raise RuntimeError("reset every environment before reading BNO observations")
        return self._output.clone()

    @property
    def sample_age_s(self) -> torch.Tensor:
        """Age of the delivered sample, including sample/hold and fixed delay."""
        return (self._tick - self._output_sample_tick).to(self.dtype) * self.physics_dt

    @property
    def time_s(self) -> float:
        return self._tick * self.physics_dt

    @property
    def gyro_bias_b(self) -> torch.Tensor:
        """Privileged diagnostic; do not add this to the actor observations."""
        return self._gyro_bias_b.clone()

    @property
    def tilt_bias_rotvec_b(self) -> torch.Tensor:
        """Privileged diagnostic in radians; not an actor observation."""
        return self._tilt_bias_rotvec_b.clone()
