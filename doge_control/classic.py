from dataclasses import dataclass
import math
from xml.etree import ElementTree

from .model import MAX_FORWARD_MPS, PERIOD_S, ROOT, URDF, clip


PHASES = (0.0, 0.5, 0.5, 0.0)
SWING_FRACTION = 0.4
FREQUENCY_HZ = 1.4
HALF_STRIDE_M = 0.018
SWING_HEIGHT_M = 0.012


def xyz(value):
    return tuple(float(component) for component in value.split())


def rpy_rotation(angles):
    roll, pitch, yaw = angles
    cr, cp, cy = map(math.cos, (roll, pitch, yaw))
    sr, sp, sy = map(math.sin, (roll, pitch, yaw))
    return (
        (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        (-sp, cp * sr, cp * cr),
    )


def dot(first, second):
    return sum(left * right for left, right in zip(first, second, strict=True))


def axis_rotation(axis, vector, angle):
    ax, ay, az = axis
    vx, vy, vz = vector
    cross = (ay * vz - az * vy, az * vx - ax * vz, ax * vy - ay * vx)
    cosine, sine = math.cos(angle), math.sin(angle)
    projection = dot(axis, vector) * (1.0 - cosine)
    return tuple(
        direction * projection + component * cosine + normal * sine
        for direction, component, normal in zip(axis, vector, cross, strict=True)
    )


@dataclass(frozen=True)
class Joint:
    origin: tuple
    rotation: tuple
    axis: tuple

    @classmethod
    def load(cls, joint):
        origin = joint.find("origin")
        axis = xyz(joint.find("axis").get("xyz"))
        length = math.sqrt(dot(axis, axis))
        return cls(
            xyz(origin.get("xyz")), rpy_rotation(xyz(origin.get("rpy"))),
            tuple(component / length for component in axis),
        )

    def map_point(self, point, angle):
        local = axis_rotation(self.axis, point, angle)
        return tuple(offset + dot(row, local) for offset, row in zip(self.origin, self.rotation, strict=True))


@dataclass(frozen=True)
class Leg:
    joints: tuple[Joint, Joint, Joint]
    foot_local: tuple[float, float, float]

    def foot_center(self, angles):
        point = self.foot_local
        for joint, angle in zip(reversed(self.joints), reversed(angles), strict=True):
            point = joint.map_point(point, angle)
        return point

    def solve(self, stand, reference, forward, lift):
        shoulder, hip, knee = stand
        reference_y, reference_z = reference[1:]
        difference_step = 1e-5
        for _ in range(8):
            pos_y, pos_z = self.foot_center((shoulder, hip, knee))[1:]
            error_y, error_z = reference_y + forward - pos_y, reference_z + lift - pos_z
            if math.hypot(error_y, error_z) < 1e-6:
                return shoulder, hip, knee
            hip_y, hip_z = self.foot_center((shoulder, hip + difference_step, knee))[1:]
            knee_y, knee_z = self.foot_center((shoulder, hip, knee + difference_step))[1:]
            jac_hip_y, jac_hip_z = (hip_y - pos_y) / difference_step, (hip_z - pos_z) / difference_step
            jac_knee_y, jac_knee_z = (knee_y - pos_y) / difference_step, (knee_z - pos_z) / difference_step
            determinant = jac_hip_y * jac_knee_z - jac_knee_y * jac_hip_z
            if abs(determinant) < 1e-8:
                break
            step_hip = (error_y * jac_knee_z - error_z * jac_knee_y) / determinant
            step_knee = (jac_hip_y * error_z - jac_hip_z * error_y) / determinant
            hip += clip(step_hip, -0.15, 0.15)
            knee += clip(step_knee, -0.15, 0.15)
        raise ValueError("classic gait target is outside the local kinematic solution")


def load_legs(urdf_path):
    robot = ElementTree.parse(urdf_path).getroot()
    joints = {joint.get("name"): Joint.load(joint) for joint in robot.findall("joint")}
    legs = []
    for leg in range(1, 5):
        link = robot.find(f"link[@name='leg_{leg}_knee_link']")
        foot, = (item for item in link.findall("collision") if item.find("geometry/cylinder") is not None)
        chain = tuple(joints[f"leg_{leg}_{joint}_joint"] for joint in ("shoulder", "hip", "knee"))
        legs.append(Leg(chain, xyz(foot.find("origin").get("xyz"))))
    return tuple(legs)


def sagittal_offset(phase, speed):
    scale = speed / MAX_FORWARD_MPS
    if phase < SWING_FRACTION:
        progress = phase / SWING_FRACTION
        return -HALF_STRIDE_M * scale * math.cos(math.pi * progress), (
            SWING_HEIGHT_M * abs(scale) * math.sin(math.pi * progress) ** 2
        )
    progress = (phase - SWING_FRACTION) / (1.0 - SWING_FRACTION)
    return HALF_STRIDE_M * scale * math.cos(math.pi * progress), 0.0


class ClassicTrot:
    def __init__(self, model, urdf_path=ROOT / URDF):
        self.legs = load_legs(urdf_path)
        self.stand = tuple(model.stand_rad[offset:offset + 3] for offset in range(0, 12, 3))
        self.reference = tuple(leg.foot_center(angles) for leg, angles in zip(self.legs, self.stand, strict=True))
        self.reset()

    def reset(self):
        self.start_tick = None

    def targets(self, state, forward_mps, tick):
        if self.start_tick is None:
            self.start_tick = tick
        cycle = (tick - self.start_tick) * PERIOD_S * FREQUENCY_HZ
        targets = []
        for offset, leg, stand, reference in zip(PHASES, self.legs, self.stand, self.reference, strict=True):
            dy, dz = sagittal_offset((cycle + offset) % 1.0, forward_mps)
            targets.extend(leg.solve(stand, reference, dy, dz))
        return tuple(targets)
