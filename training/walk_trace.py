"""Trace individual robot state under a checkpoint and fixed commands.

This uses the same deterministic evaluation setup and checkpoint compatibility
checks as walk_eval. Samples are post-reset if an episode terminates at that
step; interval reset counts expose any such jumps.
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
from .walk_env import WalkCfg
from .walk_eval import FixedCommandWalkEnv
from .walk_train import metadata

PACKAGE = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--calibration', type=Path, default=PACKAGE / 'calibration')
    parser.add_argument('--pose-candidate', type=Path, default=PACKAGE / 'verification/stand_pose_candidate.json')
    parser.add_argument('--seconds', type=float, default=8.)
    parser.add_argument('--speed', type=float, default=.15)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    if not (math.isfinite(args.seconds) and 0.1 <= args.seconds <= 12 and
            math.isfinite(args.speed) and 0 < args.speed <= .2):
        parser.error('require 0.1 <= seconds <= 12 and 0 < speed <= 0.2')
    checkpoint = args.checkpoint.resolve()
    saved = json.loads((checkpoint.parent / 'metadata.json').read_text(encoding='utf-8'))
    runner_cfg = json.loads((checkpoint.parent / 'runner_config.json').read_text(encoding='utf-8'))
    torch.manual_seed(args.seed)
    calibration, urdf = read_calibration(args.calibration)
    calibration, candidate_sha = with_pose_candidate(calibration, args.pose_candidate)
    cfg = WalkCfg()
    cfg.scene.robot.spawn.asset_path = str(urdf)
    cfg.scene.num_envs = 3
    cfg.sim.device = args.device
    cfg.seed = args.seed
    cfg.episode_length_s = args.seconds + 2.0
    env = FixedCommandWalkEnv(cfg, calibration, args.speed)
    wrapped = None
    try:
        current = metadata(env, cfg, calibration, args.calibration, urdf, args.pose_candidate, candidate_sha)
        for key in ('calibration_sha256', 'urdf_sha256', 'pose_candidate_sha256', 'walk_environment_sha256',
                    'bno08x_config_sha256', 'bno08x_observer_sha256', 'bno08x_model_sha256',
                    'joint_order', 'observation_order', 'action_scale_rad', 'policy_period_s'):
            if saved.get(key) != current.get(key):
                raise ValueError(f'Checkpoint incompatible with current task: {key}')
        env.reset()
        wrapped = RslRlVecEnvWrapper(env, clip_actions=runner_cfg.get('clip_actions', 1.0))
        expected = torch.tensor([[0., args.speed, 0.], [0., -args.speed, 0.], [0., 0., 0.]], device=env.device)
        if not torch.allclose(env.commands, expected):
            raise RuntimeError('Fixed evaluation commands were not initialized correctly')
        if (env.robot.data.joint_pos.torch - env.stand_target).abs().max().item() > 1e-4:
            raise RuntimeError('Evaluation did not start from the verified stand')
        runner = OnPolicyRunner(wrapped, runner_cfg, log_dir=None, device=args.device)
        runner.load(str(checkpoint), map_location=args.device)
        runner.alg.eval_mode()
        policy = runner.alg.get_policy()
        observations = wrapped.get_observations()
        nsteps = round(args.seconds / env.step_dt)
        interval = round(1.0 / env.step_dt)
        interval_falls = torch.zeros(3, dtype=torch.int64, device=env.device)
        cumulative_falls = interval_falls.clone()
        trace = []

        def sample(step):
            pos = env.robot.data.root_pos_w.torch - env.scene.env_origins
            q = env.robot.data.root_quat_w.torch
            x, y, z, w = q.unbind(dim=-1)
            yaw = torch.atan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z)) * (180. / math.pi)
            gravity = env.robot.data.projected_gravity_b.torch
            tilt = torch.acos((-gravity[:, 2]).clamp(-1., 1.)) * (180. / math.pi)
            row = {'time_s':step * env.step_dt, 'robots':[]}
            for j, label in enumerate(('forward_cad_plus_y', 'backward_cad_minus_y', 'stand')):
                row['robots'].append({
                    'case':label, 'command_body_y_m_s':float(expected[j,1]),
                    'root_position_relative_to_env_m':pos[j].tolist(),
                    'root_quaternion_xyzw':q[j].tolist(), 'yaw_deg':float(yaw[j]),
                    'tilt_deg':float(tilt[j]),
                    'root_velocity_world_m_s':env.robot.data.root_lin_vel_w.torch[j].tolist(),
                    'root_velocity_body_m_s':env.robot.data.root_lin_vel_b.torch[j].tolist(),
                    'reset_in_last_interval':int(interval_falls[j]),
                    'reset_cumulative':int(cumulative_falls[j])})
            trace.append(row)
            print('WALK_TRACE_SAMPLE ' + json.dumps(row), flush=True)
            interval_falls.zero_()

        sample(0)
        for i in range(nsteps):
            with torch.no_grad():
                actions = policy(observations)
                if not torch.isfinite(actions).all():
                    raise RuntimeError('Nonfinite policy output')
                observations, _, _, _ = wrapped.step(actions)
            reset = env.reset_terminated.clone().long()
            interval_falls += reset
            cumulative_falls += reset
            if (i + 1) % interval == 0 or i + 1 == nsteps:
                sample(i + 1)
        result = {'checkpoint':str(checkpoint), 'seconds':nsteps * env.step_dt,
                  'seed':args.seed, 'body_axes':'CAD Front is +Y; physical mounting unconfirmed',
                  'reset_semantics':'States at sampled steps are after automatic reset; counts cover previous interval',
                  'samples':trace}
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        print('WALK_TRACE_COMPLETE ' + str(args.report), flush=True)
    finally:
        if wrapped is not None:
            wrapped.close()
        else:
            env.close()


if __name__ == '__main__':
    main()
