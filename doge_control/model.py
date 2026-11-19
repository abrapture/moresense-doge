from dataclasses import dataclass
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
CALIBRATION = Path("calibration/pose_and_limits.json")
URDF = Path("calibration/urdf/moresense_calibrated.urdf")
STAND_POSE = Path("verification/stand_pose_candidate.json")
JOINT_ORDER = tuple(
    f"leg_{leg}_{joint}_joint"
    for leg in range(1, 5)
    for joint in ("shoulder", "hip", "knee")
)
PERIOD_S = 0.02
MAX_FORWARD_MPS = 0.15
MAX_TARGET_RATE_RAD_S = 3.0


def clip(value, lower, upper):
    return min(upper, max(lower, value))


@dataclass(frozen=True)
class RobotModel:
    stand_rad: tuple[float, ...]
    bounds_rad: tuple[tuple[float, float], ...]

    @classmethod
    def load(cls, root=ROOT):
        calibration = json.loads((root / CALIBRATION).read_text(encoding="utf-8"))
        candidate = json.loads((root / STAND_POSE).read_text(encoding="utf-8"))
        if tuple(calibration["joint_order"]) != JOINT_ORDER:
            raise ValueError("unexpected calibration joint order")
        limits = calibration["joint_limits"]
        stand = tuple(candidate["initial_joint_positions_rad"][name] for name in JOINT_ORDER)
        bounds = tuple(
            (math.radians(limits[name]["lower_deg"]), math.radians(limits[name]["upper_deg"]))
            for name in JOINT_ORDER
        )
        for name, angle, (lower, upper) in zip(JOINT_ORDER, stand, bounds, strict=True):
            if not limits[name]["confirmed"] or not all(map(math.isfinite, (lower, angle, upper))):
                raise ValueError(f"unconfirmed or invalid calibration: {name}")
            if not lower <= angle <= upper or lower == upper:
                raise ValueError(f"stand is outside the joint range: {name}")
        return cls(stand, bounds)

    def limit(self, targets, previous):
        if len(targets) != len(JOINT_ORDER) or not all(map(math.isfinite, targets)):
            raise ValueError("controller must return 12 finite joint targets")
        delta = MAX_TARGET_RATE_RAD_S * PERIOD_S
        return tuple(
            clip(target, max(lower, last - delta), min(upper, last + delta))
            for target, last, (lower, upper) in zip(targets, previous, self.bounds_rad, strict=True)
        )
