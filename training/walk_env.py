"""Flat-ground, free-body Newton locomotion task for the calibrated dog.

Actor inputs use only BNO08x-like measurements, encoder positions and finite-
difference speeds, previous actions and commanded velocity. Root velocities are
privileged simulation values used only for reward and fall detection. Servo
parameters and sensor noise are starting assumptions, not hardware calibration.
"""
import math

import torch

from .imu_check_env import Bno08xScene, load_bno08x_cfg
from .isaac_bno08x import IsaacBno08xObserver
from .stand_check_env import StandCfg, StandEnv, EFFORT, SPEED, KP, KD
from isaaclab.utils import configclass


@configclass
class WalkCfg(StandCfg):
    # 50 Hz policy. Each of the twenty 1 ms physics updates runs explicitly,
    # so both the PD controller and the 100 Hz simulated IMU get fresh state.
    decimation = 20
    episode_length_s = 12.0
    action_space = 12
    observation_space = 45
    state_space = 0
    seed = 42
    sim = StandCfg().sim.replace(render_interval=20)
    scene = StandCfg().scene.replace(class_type=Bno08xScene, num_envs=1, env_spacing=6.0)
    # Native actuator binding currently mis-shapes buffers for N > 1; the
    # existing external torque controller is retained by this switch.
    sim.use_newton_actuators = False
    # Newton handles one kinematic ground per cloned environment reliably.
    scene.ground.prim_path = '{ENV_REGEX_NS}/ground'
    scene.ground.spawn.size = (5.8, 5.8, 0.1)
    # Contact sensors are not used by the reward or actor in this task.
    scene.contact_1 = None
    scene.contact_2 = None
    scene.contact_3 = None
    scene.contact_4 = None
    action_scale = 0.25  # rad about the verified stance, before joint-limit clamp
    command_y_max = 0.20  # m/s; CAD Front is +Y, physical mounting unconfirmed
    command_zero_probability = 0.20
    command_interval_s = 4.0


class WalkEnv(StandEnv):
    """One vectorized policy controlling a floating, unassisted articulation."""

    def __init__(self, cfg, calibration, **kwargs):
        if cfg.sim.dt != 0.001 or cfg.decimation != 20:
            raise ValueError('This initial task expects physics dt=1 ms and policy dt=20 ms')
        if not (0 < cfg.action_scale <= 1 and 0 < cfg.command_y_max <= 1):
            raise ValueError('Invalid action scale or command range')
        if not 0 <= cfg.command_zero_probability <= 1:
            raise ValueError('Invalid zero command probability')
        super().__init__(cfg, calibration, kp=KP, kd=KD, **kwargs)
        # Newton may fuse decimation even with use_cuda_graph=False. Do not allow
        # it: holding a torque for 20 ms would change the controller, and the
        # sensor needs intermediate body states. Use one actual step per call.
        self.sim.physics_manager.set_decimation(1)
        self._physics_handles_decimation = False
        self.bno08x = IsaacBno08xObserver(self, cfg=load_bno08x_cfg(), robot=self.robot, seed=cfg.seed)
        self.scene.bno08x_observer = self.bno08x
        self.stand_target = self.target.clone()
        self.lower = torch.tensor([[math.radians(calibration['joint_limits'][n]['lower_deg'])
                                    for n in self.robot.joint_names]], device=self.device)
        self.upper = torch.tensor([[math.radians(calibration['joint_limits'][n]['upper_deg'])
                                    for n in self.robot.joint_names]], device=self.device)
        self.actions = torch.zeros_like(self.target)
        self.previous_actions = torch.zeros_like(self.target)
        self.commands = torch.zeros((self.num_envs, 3), device=self.device)
        self.encoder_previous = self.target.clone()
        self.command_interval_steps = max(1, round(cfg.command_interval_s / self.step_dt))
        self.last_rewards = {}

    def _sample_commands(self, env_ids):
        n = len(env_ids)
        if n == 0:
            return
        forward = (2.0 * torch.rand(n, device=self.device) - 1.0) * self.cfg.command_y_max
        forward = torch.where(torch.rand(n, device=self.device) < self.cfg.command_zero_probability, 0.0, forward)
        self.commands[env_ids] = 0.0
        self.commands[env_ids, 1] = forward

    def _pre_physics_step(self, actions):
        if actions.shape != (self.num_envs, 12):
            raise ValueError('Expected 12 actions per robot')
        self.encoder_previous.copy_(self.robot.data.joint_pos.torch)
        self.previous_actions.copy_(self.actions)
        self.actions.copy_(actions.clamp(-1., 1.))
        self.target = torch.maximum(self.lower, torch.minimum(self.upper,
                         self.stand_target + self.cfg.action_scale * self.actions))
        # Periodic command changes encourage the policy to respond, not just
        # memorize a gait. Stand is included as 20% of commands.
        ids = ((self.episode_length_buf > 0) &
               (self.episode_length_buf % self.command_interval_steps == 0)).nonzero(as_tuple=False).squeeze(-1)
        self._sample_commands(ids)

    def _get_observations(self):
        q = self.robot.data.joint_pos.torch
        # The only joint speed exposed to the policy is reconstructed from
        # successive encoder readings at the policy rate.
        qvel_est = (q - self.encoder_previous) / self.step_dt

        return {'policy': torch.cat((self.bno08x.read(), q - self.stand_target,
                                      qvel_est, self.actions, self.commands), dim=-1)}

    def _get_rewards(self):
        # Ground-truth body velocities are permitted for rewards only; they are
        # not in the actor observations and are not needed on the Raspberry Pi.
        velocity = self.robot.data.root_lin_vel_b.torch
        omega = self.robot.data.root_ang_vel_b.torch
        gravity = self.robot.data.projected_gravity_b.torch
        lin_err = (self.commands[:, :2] - velocity[:, :2]).square().sum(dim=-1)
        yaw_err = (self.commands[:, 2] - omega[:, 2]).square()
        tracking = torch.exp(-lin_err / 0.01)
        yaw_tracking = torch.exp(-yaw_err / 0.25)
        upright = gravity[:, :2].square().sum(dim=-1)
        effort = self.last_effort.square().sum(dim=-1)
        action_rate = (self.actions - self.previous_actions).square().sum(dim=-1)
        vel_z = velocity[:, 2].square()
        roll_pitch_rate = omega[:, :2].square().sum(dim=-1)
        qvel = self.robot.data.joint_vel.torch.square().sum(dim=-1)
        reward = (3.0 * tracking + 0.3 * yaw_tracking + 0.5 - 1.5 * upright
                  - 0.5 * vel_z - 0.04 * roll_pitch_rate - 0.005 * effort
                  - 0.03 * action_rate - 0.0002 * qvel)
        return reward * self.step_dt

    def _get_dones(self):
        gravity = self.robot.data.projected_gravity_b.torch
        root_z = self.robot.data.root_pos_w.torch[:, 2] - self.scene.env_origins[:, 2]
        bad = (root_z < 0.075) | (gravity[:, 2] > -0.65)
        bad |= ~torch.isfinite(self.robot.data.joint_pos.torch).all(dim=-1)
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return bad, time_out

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if not hasattr(self, 'bno08x'):
            return
        self.actions[env_ids] = 0.
        self.previous_actions[env_ids] = 0.
        self.target[env_ids] = self.stand_target[env_ids]
        self.encoder_previous[env_ids] = self.stand_target[env_ids]
        self._sample_commands(env_ids)
        self.bno08x.reset(env_ids)




