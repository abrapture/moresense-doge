"""Viser view of the provisional Newton stand experiment, not training."""
import argparse
import time
from pathlib import Path

from .usd_runtime import register_kitless_importer_schemas
register_kitless_importer_schemas()

import torch
from isaaclab_visualizers.viser import ViserVisualizerCfg
from .stand_check_env import StandCfg, StandEnv, read_calibration, with_pose_candidate, KP, KD


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--calibration', type=Path, default=Path(__file__).resolve().parent.parent / 'calibration')
    p.add_argument('--pose-candidate', type=Path)
    p.add_argument('--wall-seconds', type=float, default=600)
    p.add_argument('--episode-seconds', type=float, default=20., help='Duration of each repeated simulation episode')
    p.add_argument('--kp', type=float, default=KP)
    p.add_argument('--kd', type=float, default=KD)
    p.add_argument('--dt', type=float, default=0.001)
    args = p.parse_args()
    if not 0 < args.wall_seconds <= 3600:
        p.error('wall-seconds must be positive and at most 3600')
    if not 0 < args.episode_seconds < 29:
        p.error('episode-seconds must be positive and below 29')
    calibration, urdf = read_calibration(args.calibration)
    if args.pose_candidate:
        calibration, candidate_hash = with_pose_candidate(calibration, args.pose_candidate)
        print('STAND_POSE_CANDIDATE_SHA256 ' + candidate_hash, flush=True)
    cfg = StandCfg()
    cfg.scene.robot.spawn.asset_path = str(urdf)
    cfg.episode_length_s = args.episode_seconds
    cfg.sim.dt = args.dt
    cfg.sim.visualizer_cfgs = [ViserVisualizerCfg(port=8080, bind_address='0.0.0.0')]
    env = StandEnv(cfg, calibration, kp=args.kp, kd=args.kd)
    try:
        env.reset()
        if (env.robot.data.joint_pos.torch-env.target).abs().max().item() > 1e-4:
            raise RuntimeError('Imported joint positions do not match saved calibration')
        print(f'STAND_VIEW_READY port=8080 timeout_s={args.wall_seconds} repeated_episode_s={args.episode_seconds}', flush=True)
        deadline = time.monotonic() + args.wall_seconds
        while time.monotonic() < deadline:
            start = time.monotonic()
            obs, _, terminated, truncated, _ = env.step(torch.empty((1, 0), device=env.device))
            if terminated.any() or truncated.any():
                print(f'STAND_VIEW_EPISODE_RESET repeated_episode_s={args.episode_seconds}', flush=True)
            if not torch.isfinite(obs['policy']).all():
                raise RuntimeError('Simulation state is non-finite')
            time.sleep(max(0, env.physics_dt-(time.monotonic()-start)))
    finally:
        env.close()


if __name__ == '__main__':
    main()
