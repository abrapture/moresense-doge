"""Locomotion task with soft foot placement and straight-line objectives.

These simulator-state rewards are not actor observations. Servo parameters,
friction, and the desired footprint are modelling assumptions pending tests.
"""
import torch
from isaaclab.utils import configclass

from .walk_env import WalkCfg, WalkEnv
from .walk_reward_math import (nominal_feet_from_urdf, foot_centers_in_base,
                               expanded_footprint, footprint_penalty,
                               symmetry_penalty, heading_error, terminal_fall_penalty)


@configclass
class WalkCfgV2(WalkCfg):
    # Widen the verified footprint by 1 cm per side and lengthen by 1 cm per
    # end. These are soft reward targets, not joint commands or hard limits.
    footprint_extra_width_m = 0.010
    footprint_extra_length_m = 0.010
    foot_cost_weight = 0.8
    symmetry_cost_weight = 0.4
    heading_cost_weight = 0.6
    yaw_rate_cost_weight = 0.20
    stand_drift_cost_weight = 0.30
    termination_cost = 2.0
    command_zero_probability = 0.30


class WalkEnvV2(WalkEnv):
    def __init__(self, cfg, calibration, **kwargs):
        super().__init__(cfg, calibration, **kwargs)
        nominal, local = nominal_feet_from_urdf(cfg.scene.robot.spawn.asset_path,
                                               calibration['initial_joint_positions_rad'])
        self.nominal_feet = nominal.to(self.device)
        self.foot_offsets = local.to(self.device)
        self.desired_footprint = expanded_footprint(self.nominal_feet, cfg.footprint_extra_width_m,
                                                   cfg.footprint_extra_length_m)
        names = self.robot.body_names
        required = ['base_link'] + [f'leg_{i}_knee_link' for i in range(1, 5)]
        if any(names.count(n) != 1 for n in required):
            raise ValueError('Unable to uniquely identify base and foot-bearing links')
        self.base_index = names.index('base_link')
        self.foot_indices = [names.index(n) for n in required[1:]]
        w, x, y, z = calibration['initial_root_quaternion_wxyz']
        self.heading_reference = torch.tensor([x, y, z, w], device=self.device)

    def _get_rewards(self):
        velocity = self.robot.data.root_lin_vel_b.torch
        omega = self.robot.data.root_ang_vel_b.torch
        gravity = self.robot.data.projected_gravity_b.torch
        pose = self.robot.data.body_link_pose_w.torch
        feet = foot_centers_in_base(pose[:, self.base_index], pose, self.foot_indices, self.foot_offsets)
        command_y = self.commands[:, 1]
        stand = (command_y.abs() < 1e-5).to(velocity.dtype)
        lin_err = (self.commands[:, :2] - velocity[:, :2]).square().sum(dim=-1)
        tracking = torch.exp(-lin_err / 0.01)
        yaw_err = (self.commands[:, 2] - omega[:, 2]).square()
        yaw_tracking = torch.exp(-yaw_err / 0.25)
        upright = gravity[:, :2].square().sum(dim=-1)
        effort = self.last_effort.square().sum(dim=-1)
        action_rate = (self.actions - self.previous_actions).square().sum(dim=-1)
        vel_z = velocity[:, 2].square()
        roll_pitch_rate = omega[:, :2].square().sum(dim=-1)
        qvel = self.robot.data.joint_vel.torch.square().sum(dim=-1)
        feet_cost = footprint_penalty(feet, self.desired_footprint, command_y)
        sym_cost = symmetry_penalty(feet, self.desired_footprint, command_y)
        heading = heading_error(pose[:, self.base_index, 3:], self.heading_reference)
        # This task currently samples only straight or zero commands. If yaw
        # commands are added later, do not penalise commanded turns this way.
        straight = (self.commands[:, 2].abs() < 1e-5).to(velocity.dtype)
        heading_cost = heading.square().clamp_max(4.0) * straight
        yaw_rate_cost = omega[:, 2].square().clamp_max(4.0) * straight
        stand_drift = (velocity[:, :2].square().sum(dim=-1) / 0.01).clamp_max(4.0) * stand
        reward = (3.0 * tracking + 0.3 * yaw_tracking + 0.5 - 1.5 * upright
                  - 0.5 * vel_z - 0.04 * roll_pitch_rate - 0.005 * effort
                  - 0.03 * action_rate - 0.0002 * qvel
                  - self.cfg.foot_cost_weight * feet_cost
                  - self.cfg.symmetry_cost_weight * sym_cost
                  - self.cfg.heading_cost_weight * heading_cost
                  - self.cfg.yaw_rate_cost_weight * yaw_rate_cost
                  - self.cfg.stand_drift_cost_weight * stand_drift)
        # DirectRLEnv sets reset_terminated from _get_dones immediately before
        # computing rewards. Timeouts are separate, so they pay no fall cost.
        return reward * self.step_dt - terminal_fall_penalty(self.reset_terminated, self.cfg.termination_cost)
