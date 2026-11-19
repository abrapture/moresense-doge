"""Isaac Lab 3.0 integration task for the dog's BNO08x observations.

This is a sensor verification task, with no locomotion reward or joint controller.
The floating robot falls and rotates freely; actions must have zero dimensions.
Reuse IsaacBno08xObserver and the scene's per-physics-step update in locomotion.
"""
from pathlib import Path
import json

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

from .bno08x import Bno08xCfg
from .isaac_bno08x import IsaacBno08xObserver

PACKAGE = Path(__file__).resolve().parent.parent


def load_bno08x_cfg():
    return Bno08xCfg(**json.loads((PACKAGE / "training/bno08x_config.json").read_text(encoding="utf-8")))


class Bno08xScene(InteractiveScene):
    """Explicit sensor update after freshly refreshed articulation buffers."""

    bno08x_observer = None

    def update(self, dt):
        super().update(dt)
        if self.bno08x_observer is not None:
            self.bno08x_observer.advance(dt)


@configclass
class MoresenseImuSceneCfg(InteractiveSceneCfg):
    class_type = Bno08xScene
    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=str(PACKAGE / "urdf" / "moresense_doge_simple.urdf"),
            usd_dir=str(PACKAGE / "runtime" / "imu_check_simple_usd"),
            fix_base=False,
            merge_fixed_joints=True,
            collision_from_visuals=False,
            self_collision=False,
            robot_type="Quadruped",
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(target_type="none"),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 2.0),
            ang_vel=(0.4, 0.2, 0.1),
        ),
        actuators={},
    )


@configclass
class MoresenseImuCheckCfg(DirectRLEnvCfg):
    # One actual physics step per env.step avoids backend-owned decimation:
    # sensor sampling and delivery remain exactly 100 Hz / 10 ms.
    decimation = 1
    episode_length_s = 0.5
    action_space = 0
    observation_space = 30  # 6 IMU + 12 joint positions + 12 joint velocities.
    state_space = 0
    seed = 42
    sim = SimulationCfg(
        dt=0.005,
        render_interval=4,
        physics=NewtonCfg(
            solver_cfg=MJWarpSolverCfg(njmax=512, nconmax=128),
            num_substeps=1,
            use_cuda_graph=False,
        ),
    )
    scene = MoresenseImuSceneCfg(num_envs=8, env_spacing=2.0, replicate_physics=True)


class MoresenseImuCheckEnv(DirectRLEnv):
    """A runnable DirectRLEnv with deployable types of IMU observations."""

    def __init__(self, cfg, render_mode=None, *, bno_cfg=None, **kwargs):
        super().__init__(cfg, render_mode=render_mode, **kwargs)
        self.robot = self.scene["robot"]
        self.bno08x = IsaacBno08xObserver(
            self, cfg=bno_cfg or load_bno08x_cfg(), robot=self.robot, seed=cfg.seed
        )
        self.scene.bno08x_observer = self.bno08x

    def _pre_physics_step(self, actions):
        if actions.shape != (self.num_envs, 0):
            raise ValueError("The IMU check has no actuated actions.")

    def _apply_action(self):
        pass

    def _get_observations(self):
        # No true angular velocity / perfect gravity duplicated in the policy.
        return {"policy": torch.cat((
            self.bno08x.read(),
            self.robot.data.joint_pos.torch,
            self.robot.data.joint_vel.torch,
        ), dim=-1)}

    def _get_rewards(self):
        return torch.zeros(self.num_envs, device=self.device)

    def _get_dones(self):
        return (
            torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
            self.episode_length_buf >= self.max_episode_length,
        )

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        robot = self.scene["robot"]
        root_pose = robot.data.default_root_pose.torch[env_ids].clone()
        root_pose[:, :3] += self.scene.env_origins[env_ids]
        robot.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids)
        robot.write_root_velocity_to_sim_index(
            root_velocity=robot.data.default_root_vel.torch[env_ids].clone(), env_ids=env_ids
        )
        robot.write_joint_position_to_sim_index(
            position=robot.data.default_joint_pos.torch[env_ids].clone(), env_ids=env_ids
        )
        robot.write_joint_velocity_to_sim_index(
            velocity=robot.data.default_joint_vel.torch[env_ids].clone(), env_ids=env_ids
        )
        if hasattr(self, "bno08x"):
            self.bno08x.reset(env_ids)
