from dataclasses import dataclass
import math
from typing import Callable, Protocol

from .model import JOINT_ORDER, MAX_FORWARD_MPS, PERIOD_S, RobotModel


@dataclass(frozen=True)
class Decision:
    forward_mps: float
    reason: str

    def __post_init__(self):
        if not math.isfinite(self.forward_mps) or abs(self.forward_mps) > MAX_FORWARD_MPS:
            raise ValueError(f"forward command must be between ±{MAX_FORWARD_MPS} m/s")


@dataclass(frozen=True)
class Observation:
    positions_rad: tuple[float, ...]
    velocities_rad_s: tuple[float, ...]
    gyro_rad_s: tuple[float, float, float] = (0.0, 0.0, 0.0)
    gravity_body: tuple[float, float, float] = (0.0, 0.0, -1.0)


class Controller(Protocol):
    def targets(self, state: Observation, forward_mps: float, tick: int) -> tuple[float, ...]: ...

    def reset(self) -> None: ...


class Preview:
    def __init__(
        self, model: RobotModel, source: Callable[[int], Decision], controller: Controller | None = None,
    ):
        self.model = model
        self.source = source
        self.controller = controller
        self.observation = Observation(model.stand_rad, (0.0,) * len(JOINT_ORDER))
        self.tick = 0

    def step(self):
        decision = self.source(self.tick)
        if decision.forward_mps == 0.0:
            if self.controller is not None:
                self.controller.reset()
            requested = self.model.stand_rad
        elif self.controller is not None:
            requested = self.controller.targets(self.observation, decision.forward_mps, self.tick)
        else:
            raise ValueError("motion command without a locomotion controller")

        previous = self.observation.positions_rad
        targets = self.model.limit(requested, previous)
        velocities = tuple((new - old) / PERIOD_S for new, old in zip(targets, previous, strict=True))
        self.observation = Observation(targets, velocities)
        frame = {
            "tick": self.tick,
            "t_s": round(self.tick * PERIOD_S, 9),
            "command_vy_mps": decision.forward_mps,
            "reason": decision.reason,
            "joint_targets_rad": dict(zip(JOINT_ORDER, targets, strict=True)),
        }
        self.tick += 1
        return frame
