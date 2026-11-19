"""Newton stand experiment. Controller values are provisional, not servo calibration."""
from pathlib import Path
import hashlib
import json
import math
import xml.etree.ElementTree as ET

import torch
import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

PACKAGE = Path(__file__).resolve().parent.parent
EFFORT = 0.980665
SPEED = 4.717
KP, KD = 6.0, 0.2


def read_calibration(path):
    path = Path(path).resolve()
    payload = json.loads((path / 'pose_and_limits.json').read_text(encoding='utf-8'))
    urdf = path / 'urdf/moresense_calibrated.urdf'
    root = ET.parse(urdf).getroot()
    joints = {j.get('name'): j for j in root.findall('joint') if j.get('type') != 'fixed'}
    order = payload['joint_order']
    if len(order) != 12 or len(set(order)) != 12 or set(order) != set(joints):
        raise ValueError('Calibration must contain exactly the 12 URDF joints')
    if set(order) != set(payload['initial_joint_positions_rad']) or set(order) != set(payload['joint_limits']):
        raise ValueError('Missing joint calibration')
    source = PACKAGE / 'urdf' / payload['source_urdf']
    if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest() != payload['source_sha256']:
        raise ValueError('Calibration source hash does not match this robot')
    for name in order:
        limits = payload['joint_limits'][name]
        q = float(payload['initial_joint_positions_rad'][name])
        lo, hi = math.radians(limits['lower_deg']), math.radians(limits['upper_deg'])
        if limits['confirmed'] is not True or not all(map(math.isfinite, (q, lo, hi))) or not lo <= q <= hi:
            raise ValueError(f'Unconfirmed or invalid limits/pose: {name}')
        element = joints[name].find('limit')
        if element is None or joints[name].get('type') != 'revolute' or abs(float(element.get('lower'))-lo) > 1e-6 or abs(float(element.get('upper'))-hi) > 1e-6:
            raise ValueError(f'URDF limits differ from saved calibration: {name}')
    for mesh in root.findall('.//mesh'):
        if not (urdf.parent / mesh.get('filename')).is_file():
            raise ValueError('Missing calibrated mesh: ' + mesh.get('filename'))
    return payload, urdf


def with_pose_candidate(calibration, path):
    """Validate a temporary pose without changing user calibration or its limits."""
    raw = Path(path).read_bytes()
    candidate = json.loads(raw)
    positions = candidate['initial_joint_positions_rad']
    if set(positions) != set(calibration['joint_order']):
        raise ValueError('Candidate must specify exactly the calibrated joints')
    for name, position in positions.items():
        limits = calibration['joint_limits'][name]
        value = float(position)
        if not math.isfinite(value) or not math.radians(limits['lower_deg']) <= value <= math.radians(limits['upper_deg']):
            raise ValueError(f'Candidate joint outside confirmed limits: {name}')
    root_pos = list(map(float, candidate['initial_root_position_m']))
    root_quat = list(map(float, candidate['initial_root_quaternion_wxyz']))
    if (len(root_pos) != 3 or len(root_quat) != 4 or
            not all(map(math.isfinite, root_pos + root_quat)) or not 0 <= root_pos[2] <= .5 or
            abs(sum(x*x for x in root_quat) - 1) > 1e-5):
        raise ValueError('Candidate root pose invalid')
    result = dict(calibration)
    result['initial_joint_positions_rad'] = {name: float(value) for name, value in positions.items()}
    result['initial_root_position_m'] = root_pos
    result['initial_root_quaternion_wxyz'] = root_quat
    return result, hashlib.sha256(raw).hexdigest()


@configclass
class StandSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path='/World/ground', spawn=sim_utils.CuboidCfg(
        size=(10.0, 10.0, 0.1), collision_props=sim_utils.CollisionPropertiesCfg(),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.05)))
    robot = ArticulationCfg(
        prim_path='{ENV_REGEX_NS}/Robot',
        spawn=sim_utils.UrdfFileCfg(asset_path='REPLACE', usd_dir=str(PACKAGE / 'runtime/stand_check_usd'),
            fix_base=False, merge_fixed_joints=True, collision_from_visuals=False, self_collision=False,
            robot_type='Quadruped', joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(target_type='none')),
        init_state=ArticulationCfg.InitialStateCfg(),
        actuators={'servo': IdealPDActuatorCfg(joint_names_expr=['.*'], effort_limit=EFFORT,
                                               stiffness=0.0, damping=0.0)},
    )
    contact_1 = ContactSensorCfg(prim_path='{ENV_REGEX_NS}/Robot/Geometry/base_link/leg_1_shoulder_link/leg_1_hip_link/leg_1_knee_link', update_period=0.0, filter_prim_paths_expr=['/World/ground'])
    contact_2 = ContactSensorCfg(prim_path='{ENV_REGEX_NS}/Robot/Geometry/base_link/leg_2_shoulder_link/leg_2_hip_link/leg_2_knee_link', update_period=0.0, filter_prim_paths_expr=['/World/ground'])
    contact_3 = ContactSensorCfg(prim_path='{ENV_REGEX_NS}/Robot/Geometry/base_link/leg_3_shoulder_link/leg_3_hip_link/leg_3_knee_link', update_period=0.0, filter_prim_paths_expr=['/World/ground'])
    contact_4 = ContactSensorCfg(prim_path='{ENV_REGEX_NS}/Robot/Geometry/base_link/leg_4_shoulder_link/leg_4_hip_link/leg_4_knee_link', update_period=0.0, filter_prim_paths_expr=['/World/ground'])


@configclass
class StandCfg(DirectRLEnvCfg):
    decimation = 1
    episode_length_s = 30.0
    action_space = 0
    observation_space = 24
    state_space = 0
    sim = SimulationCfg(dt=0.001, render_interval=4, physics=NewtonCfg(
        solver_cfg=MJWarpSolverCfg(njmax=512, nconmax=128), num_substeps=1, use_cuda_graph=False))
    scene = StandSceneCfg(num_envs=1, env_spacing=2.0, replicate_physics=True)


class StandEnv(DirectRLEnv):
    def __init__(self, cfg, calibration, *, kp=KP, kd=KD, **kwargs):
        self.calibration = calibration
        super().__init__(cfg, **kwargs)
        def gains(value, label):
            result = torch.as_tensor(value, device=self.device, dtype=torch.float32).reshape(-1)
            if result.numel() == 1:
                result = result.repeat(self.num_envs)
            if result.numel() != self.num_envs or not torch.isfinite(result).all() or (result < 0).any():
                raise ValueError(f'{label} must be nonnegative, finite, and scalar or one value per environment')
            return result.reshape(self.num_envs, 1)
        self.kp, self.kd = gains(kp, 'kp'), gains(kd, 'kd')
        self.robot = self.scene['robot']
        if set(self.robot.joint_names) != set(calibration['joint_order']):
            raise ValueError(f'Imported joints differ: {self.robot.joint_names}')
        self.target = torch.tensor([[calibration['initial_joint_positions_rad'][name]
                                     for name in self.robot.joint_names]], device=self.device).repeat(self.num_envs, 1)
        self.last_effort = torch.zeros_like(self.target)

    def _pre_physics_step(self, actions):
        pass

    def _apply_action(self):
        q = self.robot.data.joint_pos.torch
        v = self.robot.data.joint_vel.torch
        effort = (self.kp * (self.target - q) - self.kd * v).clamp(-EFFORT, EFFORT)
        # Reduce driving torque towards zero at the reference no-load speed;
        # braking torque remains available. This is an approximation.
        drive_cap = EFFORT * (1.0 - v.abs() / SPEED).clamp(0.0, 1.0)
        effort = torch.where(effort * v > 0, torch.sign(effort) * torch.minimum(effort.abs(), drive_cap), effort)
        self.last_effort = effort
        self.robot.set_joint_effort_target_index(target=effort)

    def _get_observations(self):
        return {'policy': torch.cat((self.robot.data.joint_pos.torch, self.robot.data.joint_vel.torch), dim=-1)}

    def _get_rewards(self):
        return torch.zeros(self.num_envs, device=self.device)

    def _get_dones(self):
        return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device), self.episode_length_buf >= self.max_episode_length

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        robot = self.scene['robot']
        pose = robot.data.default_root_pose.torch[env_ids].clone()
        pose[:, :3] = torch.tensor(self.calibration['initial_root_position_m'], device=self.device) + self.scene.env_origins[env_ids]
        w, x, y, z = self.calibration['initial_root_quaternion_wxyz']
        pose[:, 3:7] = torch.tensor([x, y, z, w], device=self.device)
        robot.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
        robot.write_root_velocity_to_sim_index(root_velocity=torch.zeros((len(env_ids), 6), device=self.device), env_ids=env_ids)
        pos = torch.tensor([[self.calibration['initial_joint_positions_rad'][n] for n in robot.joint_names]], device=self.device).repeat(len(env_ids), 1)
        robot.write_joint_position_to_sim_index(position=pos, env_ids=env_ids)
        robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(pos), env_ids=env_ids)
