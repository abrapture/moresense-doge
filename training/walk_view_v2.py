"""View a dog checkpoint with fixed evaluation commands.

Default mode is a fixed snapshot. --follow-latest switches to newer complete checkpoints.
Three dogs are shown with CAD-frame +Y, -Y, and zero speed commands.
"""
import argparse
import copy
import re
import json
import math
import time
from pathlib import Path

from .usd_runtime import register_kitless_importer_schemas
register_kitless_importer_schemas()
import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_visualizers.viser import ViserVisualizerCfg
from rsl_rl.runners import OnPolicyRunner
from .stand_check_env import read_calibration, with_pose_candidate
from .walk_env_v2 import WalkCfgV2
from .walk_eval_v2 import FixedCommandWalkEnv
from .walk_train_v2 import metadata

PACKAGE = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--calibration', type=Path, default=PACKAGE / 'calibration')
    parser.add_argument('--pose-candidate', type=Path, default=PACKAGE / 'verification/stand_pose_candidate.json')
    parser.add_argument('--speed', type=float, default=0.15)
    parser.add_argument('--wall-seconds', type=float, default=3600.)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--follow-latest', action='store_true', help='check for newer complete checkpoints every 30 seconds')
    args = parser.parse_args()
    if not (math.isfinite(args.wall_seconds) and 0 < args.wall_seconds <= 3600 and
            math.isfinite(args.speed) and 0 < args.speed <= .2):
        parser.error('require 0 < wall-seconds <= 3600 and 0 < speed <= 0.2')
    checkpoint = args.checkpoint.resolve()
    saved = json.loads((checkpoint.parent / 'metadata.json').read_text(encoding='utf-8'))
    runner_cfg = json.loads((checkpoint.parent / 'runner_config.json').read_text(encoding='utf-8'))
    torch.manual_seed(args.seed)
    calibration, urdf = read_calibration(args.calibration)
    calibration, candidate_sha = with_pose_candidate(calibration, args.pose_candidate)
    cfg = WalkCfgV2()
    cfg.scene.robot.spawn.asset_path = str(urdf)
    cfg.scene.num_envs = 3
    cfg.sim.device = args.device
    cfg.seed = args.seed
    cfg.sim.visualizer_cfgs = [ViserVisualizerCfg(port=8080, bind_address='0.0.0.0')]
    env = FixedCommandWalkEnv(cfg, calibration, args.speed)
    wrapped = None
    try:
        current = metadata(env, cfg, calibration, args.calibration, urdf, args.pose_candidate, candidate_sha)
        for key in ('calibration_sha256', 'urdf_sha256', 'pose_candidate_sha256', 'walk_environment_sha256',
                    'walk_reward_math_sha256', 'walk_v1_base_environment_sha256', 'walk_v2_reward_settings',
                    'bno08x_config_sha256', 'bno08x_observer_sha256', 'bno08x_model_sha256',
                    'joint_order', 'stand_joint_positions_rad', 'joint_limits_deg', 'observation_order',
                    'observation_size', 'action_size', 'output', 'actor_observation_normalization',
                    'action_scale_rad', 'policy_period_s', 'physics_period_s', 'controller_simulation_only'):
            if saved.get(key) != current.get(key):
                raise ValueError(f'Checkpoint incompatible with current task: {key}')
        env.reset()
        wrapped = RslRlVecEnvWrapper(env, clip_actions=runner_cfg.get('clip_actions', 1.0))
        expected_commands = torch.tensor([[0., args.speed, 0.], [0., -args.speed, 0.], [0., 0., 0.]], device=env.device)
        if not torch.allclose(env.commands, expected_commands):
            raise RuntimeError('Fixed view commands were not initialized correctly')
        if (env.robot.data.joint_pos.torch - env.stand_target).abs().max().item() > 1e-4:
            raise RuntimeError('View did not start from verified stand')
        runner = OnPolicyRunner(wrapped, copy.deepcopy(runner_cfg), log_dir=None, device=args.device)
        runner.load(str(checkpoint), map_location=args.device)
        runner.alg.eval_mode()
        policy = runner.alg.get_policy()
        obs = wrapped.get_observations()
        loaded_checkpoint = checkpoint
        match = re.fullmatch(r'model_(\d+)\.pt', checkpoint.name)
        loaded_iteration = int(match.group(1)) if match else -1
        seen_files = {}
        next_check = time.monotonic()
        print(f'WALK_VIEW_READY port=8080 checkpoint={checkpoint} follow_latest={args.follow_latest} '
              f'commands_body_y=+{args.speed},-{args.speed},0 wall_seconds={args.wall_seconds}', flush=True)
        deadline = time.monotonic() + args.wall_seconds
        while time.monotonic() < deadline:
            started = time.monotonic()
            if args.follow_latest and started >= next_check:
                next_check = started + 30.0
                try:
                    stable = []
                    present = set()
                    for path in checkpoint.parent.glob('model_*.pt'):
                        match = re.fullmatch(r'model_(\d+)\.pt', path.name)
                        if match is None:
                            continue
                        iteration = int(match.group(1))
                        if iteration <= loaded_iteration:
                            continue
                        try:
                            info = path.stat()
                        except OSError:
                            continue
                        signature = (info.st_size, info.st_mtime_ns)
                        present.add(path)
                        previous = seen_files.get(path)
                        if previous is not None and previous[0] == signature and started - previous[1] >= 30.0 and info.st_size > 0:
                            stable.append((iteration, path, signature))
                        elif previous is None or previous[0] != signature:
                            seen_files[path] = (signature, started)
                    seen_files = {path: state for path, state in seen_files.items() if path in present}
                    for iteration, candidate, signature in sorted(stable, key=lambda item: item[0], reverse=True):
                        try:
                            # Load into a separate runner: partial files or failed
                            # loads cannot change the currently running policy.
                            new_runner = OnPolicyRunner(wrapped, copy.deepcopy(runner_cfg), log_dir=None, device=args.device)
                            new_runner.load(str(candidate), map_location=args.device)
                            new_runner.alg.eval_mode()
                            new_policy = new_runner.alg.get_policy()
                            with torch.no_grad():
                                test = new_policy(obs)
                            info = candidate.stat()
                            if (info.st_size, info.st_mtime_ns) != signature:
                                raise RuntimeError('checkpoint changed while loading')
                            if test.shape != (env.num_envs, 12) or not torch.isfinite(test).all():
                                raise RuntimeError('checkpoint returned invalid actions')
                            runner, policy = new_runner, new_policy
                            loaded_iteration, loaded_checkpoint = iteration, candidate
                            print(f'WALK_VIEW_CHECKPOINT_LOADED checkpoint={loaded_checkpoint}', flush=True)
                            break
                        except Exception as error:
                            print(f'WALK_VIEW_CHECKPOINT_RETRY checkpoint={candidate} error={error!r}', flush=True)
                except OSError as error:
                    print(f'WALK_VIEW_SCAN_RETRY error={error!r}', flush=True)
            with torch.no_grad():
                action = policy(obs)
                if not torch.isfinite(action).all():
                    raise RuntimeError('Nonfinite policy output')
                obs, _, _, _ = wrapped.step(action)
            time.sleep(max(0., env.step_dt - (time.monotonic() - started)))
    finally:
        if wrapped is not None:
            wrapped.close()
        else:
            env.close()


if __name__ == '__main__':
    main()


