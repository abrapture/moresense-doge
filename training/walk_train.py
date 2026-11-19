"""Train or export the flat-ground dog policy with Newton and RSL-RL."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from .usd_runtime import register_kitless_importer_schemas
register_kitless_importer_schemas()

import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner
from .stand_check_env import read_calibration, with_pose_candidate, EFFORT, SPEED, KP, KD
from .walk_env import WalkCfg, WalkEnv
from .walk_runner_config import WalkRunnerCfg

PACKAGE = Path(__file__).resolve().parent.parent


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metadata(env, cfg, calibration, calibration_path, urdf_path, candidate_path, candidate_sha):
    order = list(env.robot.joint_names)
    return {
        'format_version': 1,
        'purpose': 'experimental simulation policy; not approved for real hardware',
        'backend': 'Newton/MuJoCo Warp',
        'calibration_sha256': digest(calibration_path / 'pose_and_limits.json'),
        'urdf_sha256': digest(urdf_path),
        'pose_candidate_sha256': candidate_sha,
        'pose_candidate_path': str(candidate_path),
        'walk_environment_sha256': digest(Path(__file__).with_name('walk_env.py')),
        'bno08x_config_sha256': digest(Path(__file__).with_name('bno08x_config.json')),
        'bno08x_observer_sha256': digest(Path(__file__).with_name('isaac_bno08x.py')),
        'bno08x_model_sha256': digest(Path(__file__).with_name('bno08x.py')),
        'joint_order': order,
        'stand_joint_positions_rad': [calibration['initial_joint_positions_rad'][n] for n in order],
        'joint_limits_deg': {n: calibration['joint_limits'][n] for n in order},
        'observation_order': [
            {'name':'gyro_body_xyz', 'units':'rad/s', 'size':3},
            {'name':'gravity_direction_body_xyz', 'units':'unit vector', 'size':3},
            {'name':'encoder_joint_position_minus_stand', 'units':'rad', 'size':12, 'order':'joint_order'},
            {'name':'encoder_joint_velocity_finite_difference', 'units':'rad/s', 'size':12, 'order':'joint_order'},
            {'name':'previous_clipped_action', 'units':'dimensionless', 'size':12, 'order':'joint_order'},
            {'name':'body_velocity_command_vx_vy_yaw', 'units':'m/s,m/s,rad/s', 'size':3},
        ],
        'observation_size':45, 'action_size':12,
        'output': 'clip policy output to [-1, 1], multiply by action_scale_rad, add stand_joint_positions_rad, clip to joint_limits_deg',
        'action_scale_rad': cfg.action_scale,
        'policy_period_s': cfg.sim.dt * cfg.decimation,
        'physics_period_s': cfg.sim.dt,
        'controller_simulation_only': {'kp':KP, 'kd':KD, 'effort_limit_nm':EFFORT, 'no_load_speed_rad_s':SPEED},
        'imu_simulation': {'model':'BNO08x approximation', 'details':'training/bno08x_config.json'},
        'command_frame':'body coordinates: +Y points toward CAD Front; actual mounting must be verified',
        'actor_observation_normalization':'included inside exported policy',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--calibration', type=Path, default=PACKAGE / 'calibration')
    parser.add_argument('--pose-candidate', type=Path, default=PACKAGE / 'verification/stand_pose_candidate.json')
    parser.add_argument('--num-envs', type=int, default=64)
    parser.add_argument('--iterations', type=int, default=1000, help='additional iterations (also on resume)')
    parser.add_argument('--steps-per-env', type=int, default=24)
    parser.add_argument('--save-interval', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--log-dir', type=Path, default=None, help='host-mounted persistent run directory')
    parser.add_argument('--checkpoint', type=Path, help='resume from checkpoint in a previous run')
    parser.add_argument('--export-only', action='store_true', help='load checkpoint and export without further training')
    args = parser.parse_args()
    if args.num_envs < 1 or args.iterations < 1 or args.steps_per_env < 1 or args.save_interval < 1:
        parser.error('number of environments, iterations, steps and save interval must be positive')
    if args.export_only and not args.checkpoint:
        parser.error('--export-only requires --checkpoint')
    torch.manual_seed(args.seed)
    calibration, urdf = read_calibration(args.calibration)
    calibration, candidate_sha = with_pose_candidate(calibration, args.pose_candidate)
    cfg = WalkCfg()
    cfg.scene.robot.spawn.asset_path = str(urdf)
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    cfg.seed = args.seed
    runner_cfg = WalkRunnerCfg()
    runner_cfg.device = args.device
    runner_cfg.seed = args.seed
    runner_cfg.num_steps_per_env = args.steps_per_env
    runner_cfg.save_interval = args.save_interval
    log_dir = args.log_dir or PACKAGE / 'runtime/walk_training' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    log_dir = log_dir.resolve()
    if log_dir.exists() and any(log_dir.iterdir()):
        raise ValueError(f'Run directory already exists and is not empty: {log_dir}; choose a new directory')
    log_dir.mkdir(parents=True, exist_ok=True)
    env = WalkEnv(cfg, calibration)
    wrapped = None
    try:
        env.reset()
        initial_error = (env.robot.data.joint_pos.torch - env.stand_target).abs().max().item()
        if initial_error > 1e-4:
            raise RuntimeError(f'Imported joint positions differ from the validated stand by {initial_error} rad')
        meta = metadata(env, cfg, calibration, args.calibration, urdf, args.pose_candidate, candidate_sha)
        if args.checkpoint:
            old_file = args.checkpoint.resolve().parent / 'metadata.json'
            if not old_file.is_file():
                raise ValueError('Checkpoint metadata.json is missing; cannot verify model compatibility')
            old = json.loads(old_file.read_text(encoding='utf-8'))
            for key in ('calibration_sha256', 'urdf_sha256', 'pose_candidate_sha256', 'walk_environment_sha256',
                        'bno08x_config_sha256', 'bno08x_observer_sha256', 'bno08x_model_sha256',
                        'joint_order', 'observation_order', 'action_scale_rad', 'policy_period_s'):
                if old.get(key) != meta.get(key):
                    raise ValueError(f'Checkpoint is incompatible with current task: {key}')
        (log_dir / 'metadata.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding='utf-8')
        train_cfg = runner_cfg.to_dict()
        train_cfg['obs_groups'] = {'actor':['policy'], 'critic':['policy']}
        for model_key in ('actor', 'critic'):
            for obsolete in ('stochastic', 'init_noise_std', 'noise_std_type', 'state_dependent_std'):
                train_cfg[model_key].pop(obsolete, None)
        (log_dir / 'runner_config.json').write_text(json.dumps(train_cfg, indent=2, default=str), encoding='utf-8')
        wrapped = RslRlVecEnvWrapper(env, clip_actions=runner_cfg.clip_actions)
        runner = OnPolicyRunner(wrapped, train_cfg, log_dir=str(log_dir), device=args.device)
        if args.checkpoint:
            runner.load(str(args.checkpoint.resolve()), map_location=args.device)
        print(f'WALK_TRAIN_READY log_dir={log_dir} num_envs={args.num_envs} iterations={args.iterations}', flush=True)
        if not args.export_only:
            runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=False)
        export_dir = log_dir / 'exported'
        runner.alg.eval_mode()
        observations = wrapped.get_observations()
        with torch.no_grad():
            reference = runner.alg.get_policy()(observations).detach().cpu()
        runner.export_policy_to_jit(str(export_dir), filename='policy.pt')
        exported = torch.jit.load(str(export_dir / 'policy.pt'), map_location='cpu').eval()
        with torch.no_grad():
            prediction = exported(observations['policy'].detach().cpu())
        err = (prediction-reference).abs().max().item()
        if prediction.shape != (args.num_envs, 12) or not torch.isfinite(prediction).all() or err > 1e-5:
            raise RuntimeError(f'CPU export parity test failed; max difference {err}')
        export_metadata = dict(meta, policy_sha256=digest(export_dir / 'policy.pt'),
                               cpu_export_max_error=err, last_learning_iteration_index=runner.current_learning_iteration)
        (export_dir / 'metadata.json').write_text(json.dumps(export_metadata, indent=2, ensure_ascii=False), encoding='utf-8')
        print(f'WALK_EXPORT_VERIFIED path={export_dir / "policy.pt"} max_error={err}', flush=True)
    finally:
        if wrapped is not None:
            wrapped.close()
        else:
            env.close()


if __name__ == '__main__':
    main()
