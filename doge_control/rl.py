from hashlib import sha256
import json
import math
from pathlib import Path

from .model import CALIBRATION, JOINT_ORDER, PERIOD_S, ROOT, STAND_POSE, URDF, clip


POLICY = Path("trained_policy_999/policy.pt")
POLICY_SHA256 = "21242d90b5b0d4b442f651eaf8ab3e24b6d4eba76042fcca94e4aed7c243966a"
OBSERVATIONS = 45
ACTIONS = len(JOINT_ORDER)
ACTION_SCALE_RAD = 0.25
LAYOUT = (
    ("gyro_body_xyz", 3),
    ("gravity_direction_body_xyz", 3),
    ("encoder_joint_position_minus_stand", 12),
    ("encoder_joint_velocity_finite_difference", 12),
    ("previous_clipped_action", 12),
    ("body_velocity_command_vx_vy_yaw", 3),
)
ASSETS = {
    "calibration_sha256": CALIBRATION,
    "urdf_sha256": URDF,
    "pose_candidate_sha256": STAND_POSE,
    "walk_environment_sha256": Path("training/walk_env.py"),
    "bno08x_config_sha256": Path("training/bno08x_config.json"),
    "bno08x_observer_sha256": Path("training/isaac_bno08x.py"),
    "bno08x_model_sha256": Path("training/bno08x.py"),
    "policy_sha256": POLICY,
}


def validate_snapshot(model, root=ROOT):
    metadata = json.loads((root / POLICY.with_name("metadata.json")).read_text(encoding="utf-8"))
    layout = tuple((group["name"], group["size"]) for group in metadata["observation_order"])
    if (
        metadata["format_version"] != 1 or layout != LAYOUT or
        tuple(metadata["joint_order"]) != JOINT_ORDER or
        tuple(metadata["stand_joint_positions_rad"]) != model.stand_rad or
        metadata["observation_size"] != OBSERVATIONS or metadata["action_size"] != ACTIONS or
        metadata["action_scale_rad"] != ACTION_SCALE_RAD or metadata["policy_period_s"] != PERIOD_S or
        metadata["policy_sha256"] != POLICY_SHA256
    ):
        raise ValueError("incompatible MoreSense V1 policy metadata")
    for key, asset in ASSETS.items():
        if sha256((root / asset).read_bytes()).hexdigest() != metadata[key]:
            raise ValueError(f"V1 snapshot SHA-256 mismatch: {asset}")


class RlV1:
    def __init__(self, model, predict):
        self.model = model
        self.predict = predict
        self.reset()

    @classmethod
    def load(cls, model):
        try:
            import torch
        except ImportError as error:
            raise RuntimeError("RL preview requires PyTorch (see docs/rl-v1.md)") from error
        validate_snapshot(model)
        actor = torch.jit.load(str(ROOT / POLICY), map_location="cpu").eval()

        def predict(observation):
            with torch.inference_mode():
                batch = torch.tensor([observation], dtype=torch.float32, device="cpu")
                result = actor(batch)
            if tuple(result.shape) != (1, ACTIONS):
                raise ValueError("V1 policy must return a (1, 12) action tensor")
            return result[0].tolist()

        return cls(model, predict)

    def reset(self):
        self.previous_action = (0.0,) * ACTIONS

    def targets(self, state, forward_mps, tick):
        positions = tuple(
            angle - neutral for angle, neutral in zip(state.positions_rad, self.model.stand_rad, strict=True)
        )
        observation = (
            state.gyro_rad_s + state.gravity_body + positions + state.velocities_rad_s +
            self.previous_action + (0.0, forward_mps, 0.0)
        )
        if len(observation) != OBSERVATIONS or not all(map(math.isfinite, observation)):
            raise ValueError("V1 policy expects 45 finite observations")
        action = tuple(self.predict(observation))
        if len(action) != ACTIONS or not all(map(math.isfinite, action)):
            raise ValueError("V1 policy must return 12 finite actions")
        self.previous_action = tuple(clip(value, -1.0, 1.0) for value in action)
        return tuple(
            clip(neutral + value * ACTION_SCALE_RAD, lower, upper)
            for neutral, value, (lower, upper) in zip(
                self.model.stand_rad, self.previous_action, self.model.bounds_rad, strict=True,
            )
        )
