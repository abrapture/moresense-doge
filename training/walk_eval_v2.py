"""Evaluate a saved policy with fixed +Y, -Y and zero commands in Newton.

The CAD Front mesh points toward body +Y; the physical mounting is unconfirmed.
Body velocities and fall flags below are *evaluation metrics*, not actor inputs.
"""
import argparse
import json
import math
from pathlib import Path

from .usd_runtime import register_kitless_importer_schemas
register_kitless_importer_schemas()

import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner
from .stand_check_env import read_calibration, with_pose_candidate
from .walk_env_v2 import WalkCfgV2, WalkEnvV2
from .walk_train_v2 import metadata, digest
from .walk_reward_math import foot_centers_in_base, heading_error

PACKAGE = Path(__file__).resolve().parent.parent


class FixedCommandWalkEnv(WalkEnvV2):
    def __init__(self, cfg, calibration, speed):
        self.fixed_speed = speed
        super().__init__(cfg, calibration)

    def _sample_commands(self, env_ids):
        # The three environments retain the same command after resets and at
        # periodic command boundaries. The actor receives the same observations
        # and sensor model as in training.
        if len(env_ids) == 0:
            return
        all_cmds = torch.tensor([[0., self.fixed_speed, 0.], [0., -self.fixed_speed, 0.],
                                 [0., 0., 0.]], device=self.device)
        self.commands[env_ids] = all_cmds[env_ids.long()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--baseline-v1', action='store_true',
                        help='evaluate an original V1 policy with V2 geometry metrics; '
                             'fixed commands and physics are unchanged')
    parser.add_argument('--calibration', type=Path, default=PACKAGE / 'calibration')
    parser.add_argument('--pose-candidate', type=Path, default=PACKAGE / 'verification/stand_pose_candidate.json')
    parser.add_argument('--seconds', type=float, default=8.)
    parser.add_argument('--speed', type=float, default=0.15)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--viser', action='store_true', help='run a Viser viewer on port 8080')
    args = parser.parse_args()
    if not (math.isfinite(args.seconds) and 0.1 <= args.seconds <= 12 and
            math.isfinite(args.speed) and 0 < args.speed <= .2):
        parser.error('require 0.1 <= seconds <= 12 and 0 < speed <= 0.2')
    checkpoint = args.checkpoint.resolve()
    run_dir = checkpoint.parent
    saved = json.loads((run_dir / 'metadata.json').read_text(encoding='utf-8'))
    runner_cfg = json.loads((run_dir / 'runner_config.json').read_text(encoding='utf-8'))
    torch.manual_seed(args.seed)
    calibration, urdf = read_calibration(args.calibration)
    calibration, candidate_sha = with_pose_candidate(calibration, args.pose_candidate)
    cfg = WalkCfgV2()
    cfg.scene.robot.spawn.asset_path = str(urdf)
    cfg.scene.num_envs = 3
    cfg.sim.device = args.device
    cfg.seed = args.seed
    cfg.episode_length_s = args.seconds + 2.0
    if args.viser:
        from isaaclab_visualizers.viser import ViserVisualizerCfg
        cfg.sim.visualizer_cfgs = [ViserVisualizerCfg(port=8080, bind_address='0.0.0.0')]
    env = FixedCommandWalkEnv(cfg, calibration, args.speed)
    wrapped = None
    try:
        current = metadata(env, cfg, calibration, args.calibration, urdf, args.pose_candidate, candidate_sha)
        keys = ['calibration_sha256', 'urdf_sha256', 'pose_candidate_sha256', 'walk_environment_sha256',
                    'bno08x_config_sha256', 'bno08x_observer_sha256', 'bno08x_model_sha256',
                    'joint_order', 'stand_joint_positions_rad', 'joint_limits_deg', 'observation_order',
                    'observation_size', 'action_size', 'output', 'actor_observation_normalization',
                    'action_scale_rad', 'policy_period_s', 'physics_period_s', 'controller_simulation_only']
        if args.baseline_v1:
            current['walk_environment_sha256'] = current['walk_v1_base_environment_sha256']
        else:
            keys.extend(('walk_reward_math_sha256', 'walk_v1_base_environment_sha256', 'walk_v2_reward_settings'))
        for key in keys:
            if saved.get(key) != current.get(key):
                raise ValueError(f'Checkpoint incompatible with current task: {key}')
        env.reset()
        wrapped = RslRlVecEnvWrapper(env, clip_actions=runner_cfg.get('clip_actions', 1.0))
        expected_commands = torch.tensor([[0., args.speed, 0.], [0., -args.speed, 0.], [0., 0., 0.]], device=env.device)
        if not torch.allclose(env.commands, expected_commands):
            raise RuntimeError('Fixed evaluation commands were not initialized correctly')
        if (env.robot.data.joint_pos.torch - env.stand_target).abs().max().item() > 1e-4:
            raise RuntimeError('Evaluation did not start from the verified stand')
        runner = OnPolicyRunner(wrapped, runner_cfg, log_dir=None, device=args.device)
        runner.load(str(checkpoint), map_location=args.device)
        runner.alg.eval_mode()
        policy = runner.alg.get_policy()
        observations = wrapped.get_observations()
        start = env.robot.data.root_pos_w.torch.clone()
        prev = start.clone()
        displacement = torch.zeros((3, 3), device=env.device)
        sum_vel = torch.zeros((3, 3), device=env.device)
        sum_error = torch.zeros(3, device=env.device)
        max_tilt = torch.zeros(3, device=env.device)
        falls = torch.zeros(3, dtype=torch.int64, device=env.device)
        nsteps = round(args.seconds / env.step_dt)
        # World yaw is measured relative to the saved initial heading and is
        # unwrapped between steps. Reset teleports are never counted as turns.
        pose0 = env.robot.data.body_link_pose_w.torch[:, env.base_index]
        previous_heading = heading_error(pose0[:, 3:], env.heading_reference)
        cumulative_yaw = torch.zeros(3, device=env.device)
        tail_heading_sum = torch.zeros(3, device=env.device)
        tail_heading_max = torch.zeros(3, device=env.device)
        tail_foot_sum = torch.zeros((3, 4, 3), device=env.device)
        tail_count = torch.zeros(3, dtype=torch.int64, device=env.device)
        for i in range(nsteps):
            with torch.no_grad():
                actions = policy(observations)
                if not torch.isfinite(actions).all():
                    raise RuntimeError('Nonfinite policy output')
                observations, _, _, _ = wrapped.step(actions)
            reset = env.reset_terminated.clone()
            falls += reset.long()
            pos = env.robot.data.root_pos_w.torch.clone()
            # Exclude teleportation caused by automatic episode reset.
            displacement += torch.where(reset[:, None], 0., pos - prev)
            prev = pos
            vel = env.robot.data.root_lin_vel_b.torch
            sum_vel += vel
            sum_error += (env.commands[:, 1] - vel[:, 1]).abs()
            grav = env.robot.data.projected_gravity_b.torch
            tilt = torch.acos((-grav[:, 2]).clamp(-1., 1.)) * (180. / math.pi)
            max_tilt = torch.maximum(max_tilt, tilt)
            pose = env.robot.data.body_link_pose_w.torch
            heading = heading_error(pose[:, env.base_index, 3:], env.heading_reference)
            diff = heading - previous_heading
            yaw_step = torch.atan2(torch.sin(diff), torch.cos(diff))
            cumulative_yaw += torch.where(reset, 0., yaw_step)
            previous_heading = heading
            if i >= nsteps // 2:
                valid = ~reset
                heading_abs = heading.abs()
                tail_heading_sum += torch.where(valid, heading_abs, 0.)
                tail_heading_max = torch.maximum(tail_heading_max, torch.where(valid, heading_abs, 0.))
                feet = foot_centers_in_base(pose[:, env.base_index], pose, env.foot_indices, env.foot_offsets)
                tail_foot_sum += torch.where(valid[:, None, None], feet, 0.)
                tail_count += valid.long()
            if (i + 1) % 50 == 0:
                print(f'WALK_EVAL_PROGRESS step={i+1}/{nsteps}', flush=True)
        result = {'checkpoint': str(checkpoint), 'checkpoint_sha256': digest(checkpoint),
                  'metadata_sha256': digest(run_dir / 'metadata.json'),
                  'checkpoint_type': 'v1_baseline' if args.baseline_v1 else 'v2',
                  'seconds': nsteps * env.step_dt, 'seed': args.seed,
                  'body_axes': 'CAD Front is +Y; actual mounting must be confirmed',
                  'sensor': 'same noisy, delayed BNO08x model used in training',
                  'geometry_measurement': ('Mean cylinder-center positions in base_link frame over the second half; '
                                           'leg order 1,2,3,4. Centers are not ground contacts or slip measures.'),
                  'heading_measurement': ('World projected +Y relative to the saved initial heading. '
                                          'Cumulative yaw is unwrapped and excludes reset teleports; '
                                          'heading deviations are wrapped to 0..180 degrees.'),
                  'second_half_sample_start_s': (nsteps // 2) * env.step_dt,
                  'stance_foot_centers_base_xyz_m': env.nominal_feet.tolist(),
                  'v2_soft_target_foot_centers_base_xyz_m': env.desired_footprint.tolist(),
                  'cases': []}
        for j, (label, command) in enumerate((('forward_cad_plus_y', args.speed),
                                               ('backward_cad_minus_y', -args.speed), ('stand', 0.))):
            count = int(tail_count[j])
            if count == 0:
                raise RuntimeError(f'No valid geometry measurements in the second half for {label}')
            result['cases'].append({'case':label, 'command_body_y_m_s':command,
                'mean_body_velocity_xyz_m_s':(sum_vel[j] / nsteps).tolist(),
                'mean_abs_body_y_tracking_error_m_s':float(sum_error[j] / nsteps),
                'cumulative_world_displacement_m_excluding_reset_jumps':displacement[j].tolist(),
                'falls':int(falls[j]), 'max_tilt_degrees':float(max_tilt[j]),
                'cumulative_world_yaw_degrees_excluding_reset_jumps':float(cumulative_yaw[j] * (180. / math.pi)),
                'second_half_mean_abs_heading_deviation_degrees':float((tail_heading_sum[j] / count) * (180. / math.pi)),
                'second_half_max_abs_heading_deviation_degrees':float(tail_heading_max[j] * (180. / math.pi)),
                'second_half_foot_centers_base_xyz_m':(tail_foot_sum[j] / count).tolist(),
                'second_half_valid_samples':count})
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        print('WALK_EVAL_RESULT ' + json.dumps(result, ensure_ascii=False), flush=True)
    finally:
        if wrapped is not None:
            wrapped.close()
        else:
            env.close()


if __name__ == '__main__':
    main()

