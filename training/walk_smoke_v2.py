"""Short, non-training Newton sanity check. Run 1, 8, then 512 environments.

Example: python -m moresense_doge.training.walk_smoke_v2 --num-envs 8
Does not modify any calibration, checkpoints or existing training runs.
"""
import argparse
from pathlib import Path

from .usd_runtime import register_kitless_importer_schemas
register_kitless_importer_schemas()
import torch
from .stand_check_env import read_calibration, with_pose_candidate, EFFORT
from .walk_env_v2 import WalkCfgV2, WalkEnvV2
from .walk_reward_math import nominal_feet_from_urdf, foot_centers_in_base

PACKAGE = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--calibration', type=Path, default=PACKAGE / 'calibration')
    parser.add_argument('--pose-candidate', type=Path, default=PACKAGE / 'verification/stand_pose_candidate.json')
    parser.add_argument('--num-envs', type=int, default=8)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if not 1 <= args.num_envs <= 4096 or not 1 <= args.steps <= 500:
        parser.error('require 1 <= num-envs <= 4096, 1 <= steps <= 500')
    calibration, urdf = read_calibration(args.calibration)
    calibration, _ = with_pose_candidate(calibration, args.pose_candidate)
    cfg = WalkCfgV2()
    cfg.scene.robot.spawn.asset_path = str(urdf)
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    env = WalkEnvV2(cfg, calibration)
    try:
        observation, _ = env.reset()
        if observation['policy'].shape != (args.num_envs, 45):
            raise RuntimeError(f'Unexpected actor observations: {observation["policy"].shape}')
        max_fk_error, max_effort, resets = 0., 0., 0
        # Interpolate smooth targets, not random discontinuous motor commands.
        for step in range(args.steps):
            with torch.no_grad():
                action = .2 * torch.sin(torch.tensor(step * .05, device=env.device)) * torch.ones_like(env.actions)
                obs, reward, terminated, timed_out, _ = env.step(action)
                values = (obs['policy'], reward, env.robot.data.joint_pos.torch,
                          env.robot.data.joint_vel.torch, env.last_effort)
                if any(not torch.isfinite(v).all().item() for v in values):
                    raise RuntimeError(f'Nonfinite state or reward at step {step}')
                if ((env.target < env.lower - 1e-6) | (env.target > env.upper + 1e-6)).any().item():
                    raise RuntimeError(f'Controller target exceeds calibrated limits at step {step}')
                effort = env.last_effort.abs().max().item()
                if effort > EFFORT + 1e-5:
                    raise RuntimeError(f'Torque exceeds simulation effort bound at step {step}')
                max_effort = max(max_effort, effort)
                resets += terminated.sum().item()
                if (step in {0, args.steps // 4, args.steps // 2, args.steps * 3 // 4, args.steps - 1}
                        and not (terminated[0] | timed_out[0]).item()):
                    poses = env.robot.data.body_link_pose_w.torch
                    measured = foot_centers_in_base(poses[:, env.base_index], poses,
                                                    env.foot_indices, env.foot_offsets)[0].cpu()
                    positions = env.robot.data.joint_pos.torch[0].cpu().tolist()
                    fk, _ = nominal_feet_from_urdf(urdf, dict(zip(env.robot.joint_names, positions)))
                    error = (fk - measured).abs().max().item()
                    max_fk_error = max(max_fk_error, error)
                    if error > 0.0005:
                        raise RuntimeError(f'URDF FK differs from Newton by {error:.6f} m at step {step}')
        print(f'WALK_V2_SMOKE_OK envs={args.num_envs} steps={args.steps} max_fk_error_m={max_fk_error:.8f} '
              f'max_effort_nm={max_effort:.5f} fall_resets={resets}', flush=True)
    finally:
        env.close()


if __name__ == '__main__':
    main()
