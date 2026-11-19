"""Isaac Lab 3.0 bridge for the six BNO08x actor observations.

The bridge reads the backend-neutral ArticulationData ProxyArray ``.torch``
views. It deliberately imports no Isaac modules and registers no callbacks.
The task owns the lifecycle: ``reset(ids)`` after applying reset state, then
``advance(physics_dt)`` after each actual physics step and scene update. Calling
``read()`` is side-effect free, including repeated actor/critic observation reads.

Only the returned six channels belong in the actor's IMU observation term.
Do not additionally expose the ideal root angular velocity / projected gravity
to that actor. Privileged critic observations, if used, are a separate choice.
The virtual sensor axes currently coincide with the robot root body axes.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from .bno08x import Bno08xCfg, Bno08xObservationModel


def fused_quaternion_to_projected_gravity(quaternion_xyzw: torch.Tensor) -> torch.Tensor:
    """Convert a body-to-world fused quaternion to a DOWN unit vector in body.

    Isaac Lab 3.0 uses **(x, y, z, w)**, with identity ``[0, 0, 0, 1]``.
    The input may have any leading batch shape. The quaternion is normalized;
    nonfinite / zero quaternions are rejected. World up is +Z, so an upright
    sensor returns ``[0, 0, -1]``. A +90 degree roll returns ``[0, -1, 0]``.

    A real BNO driver must first convert its report's component order, mounting
    axes, and body/world rotation convention to this explicit contract. This
    helper does not assume a particular SH-2 library's quaternion field order.
    """
    if not isinstance(quaternion_xyzw, torch.Tensor):
        raise TypeError("quaternion_xyzw must be a torch.Tensor")
    if quaternion_xyzw.ndim < 1 or quaternion_xyzw.shape[-1] != 4:
        raise ValueError("quaternion_xyzw must have final dimension 4 (x, y, z, w)")
    if not quaternion_xyzw.is_floating_point():
        raise TypeError("quaternion_xyzw must have a floating dtype")
    norm = torch.linalg.vector_norm(quaternion_xyzw, dim=-1, keepdim=True)
    if not bool(torch.isfinite(quaternion_xyzw).all()) or bool((norm <= 0).any()):
        raise ValueError("quaternion_xyzw must be finite and nonzero")
    q = quaternion_xyzw / norm
    gravity_w = torch.zeros_like(q[..., :3])
    gravity_w[..., 2] = -1.0
    # Same xyzw inverse-rotation formula as isaaclab.utils.math.quat_apply_inverse.
    xyz = q[..., :3]
    twice_cross = 2.0 * torch.linalg.cross(xyz, gravity_w, dim=-1)
    return gravity_w - q[..., 3:4] * twice_cross + torch.linalg.cross(xyz, twice_cross, dim=-1)


def _torch_view(value: Any, name: str) -> torch.Tensor:
    """Unwrap an Isaac Lab 3.0 ProxyArray; accept tensors for standalone tests."""
    result = value if isinstance(value, torch.Tensor) else getattr(value, "torch", None)
    if not isinstance(result, torch.Tensor):
        raise TypeError(f"robot.data.{name} must be a torch.Tensor or expose .torch")
    return result


class IsaacBno08xObserver:
    """Explicit-lifecycle observer for an initialized Isaac Lab environment.

    Typical construction is ``IsaacBno08xObserver(env, robot=env._robot)`` for a
    direct task, or ``IsaacBno08xObserver(env)`` when ``env.scene['robot']`` exists.
    Construct after the scene has initialized its articulation state. The model
    automatically primes its delay line from that state; reset it again after
    subsequently writing the episode's initial root state.

    ``gravity_from_quaternion=True`` explicitly synthesizes the gravity direction
    from ``root_quat_w`` using a fixed world -Z reference. Otherwise the preferred
    source is ``projected_gravity_b``; a missing property falls back to the same
    quaternion conversion. A zero physical gravity setting may make the former
    undefined, so choose the quaternion option for orientation-only checks.
    """

    observation_dim = 6
    channel_order = ("gyro_x", "gyro_y", "gyro_z", "gravity_x", "gravity_y", "gravity_z")

    def __init__(
        self,
        env: Any,
        cfg: Bno08xCfg | None = None,
        *,
        asset_name: str = "robot",
        robot: Any | None = None,
        seed: int = 0,
        gravity_from_quaternion: bool = False,
    ) -> None:
        self.env = env
        self.robot = env.scene[asset_name] if robot is None else robot
        self.gravity_from_quaternion = gravity_from_quaternion
        self.physics_dt = float(env.physics_dt)
        self.num_envs = int(env.num_envs)
        gyro, gravity = self._signals()
        if gyro.shape != (self.num_envs, 3):
            raise ValueError(f"root_ang_vel_b must have shape ({self.num_envs}, 3)")
        self.model = Bno08xObservationModel(
            self.num_envs,
            self.physics_dt,
            cfg,
            device=gyro.device,
            dtype=gyro.dtype,
            seed=seed,
        )
        self.model.reset(gyro, gravity)

    def _signals(self) -> tuple[torch.Tensor, torch.Tensor]:
        data = self.robot.data
        gyro = _torch_view(data.root_ang_vel_b, "root_ang_vel_b")
        gravity_value = None if self.gravity_from_quaternion else getattr(data, "projected_gravity_b", None)
        if gravity_value is None:
            gravity = fused_quaternion_to_projected_gravity(_torch_view(data.root_quat_w, "root_quat_w"))
        else:
            gravity = _torch_view(gravity_value, "projected_gravity_b")
        return gyro, gravity

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        """Prime selected histories after their actual root state has been reset."""
        gyro, gravity = self._signals()
        self.model.reset(gyro, gravity, env_ids=env_ids)

    def advance(self, physics_dt: float | None = None) -> None:
        """Advance once AFTER physics + scene.update; never from observation reads.

        The model rejects an elapsed time different from its configured physical
        tick. A decimated action step must call this for every physics substep;
        passing the whole control interval would lose the intermediate samples.
        """
        gyro, gravity = self._signals()
        self.model.advance(gyro, gravity, dt=self.physics_dt if physics_dt is None else physics_dt)

    def read(self) -> torch.Tensor:
        """Return an owned ``(num_envs, 6)`` tensor: gyro rad/s, unit DOWN gravity."""
        return self.model.output()
