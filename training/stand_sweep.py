"""Compare provisional joint PD settings in independent Newton environments.

The shared lower-leg/foot body is sensed against the floor; contact with that
body is not proof that the actual foot cylinder touches the floor.
"""
import argparse
import itertools
import json
import math
from pathlib import Path

from .usd_runtime import register_kitless_importer_schemas
register_kitless_importer_schemas()

import torch
from .stand_check_env import StandCfg, StandEnv, read_calibration, with_pose_candidate, EFFORT
from .foot_geometry import FootGeometry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--calibration', type=Path, default=Path(__file__).resolve().parent.parent / 'calibration')
    parser.add_argument('--pose-candidate', type=Path, help='JSON pose to test without changing saved calibration')
    parser.add_argument('--kp', type=float, nargs='+', default=[2., 4., 6., 8.])
    parser.add_argument('--kd', type=float, nargs='+', default=[.02, .05, .1, .2])
    parser.add_argument('--dt', type=float, default=.001)
    parser.add_argument('--seconds', type=float, default=3.)
    parser.add_argument('--settle', type=float, default=1.)
    parser.add_argument('--report', type=Path, default=Path('verification/stand_sweep.json'))
    args = parser.parse_args()
    if (not all(math.isfinite(x) and x >= 0 for x in args.kp + args.kd)
            or not all(math.isfinite(x) for x in (args.dt, args.seconds, args.settle))
            or args.dt <= 0 or args.seconds <= 1 or not 0 <= args.settle < args.seconds - 1
            or args.seconds >= 29):
        parser.error('gains must be finite and nonnegative; require dt > 0, 0 <= settle < seconds - 1, and 1 < seconds < 29')
    pairs = list(itertools.product(args.kp, args.kd))
    if len(pairs) > 64:
        parser.error('at most 64 gain combinations')
    calibration, urdf = read_calibration(args.calibration)
    candidate_hash = None
    if args.pose_candidate:
        calibration, candidate_hash = with_pose_candidate(calibration, args.pose_candidate)
    cfg = StandCfg()
    cfg.scene.num_envs = 1
    cfg.scene.robot.spawn.asset_path = str(urdf)
    cfg.sim.dt = args.dt
    env = StandEnv(cfg, calibration, kp=pairs[0][0], kd=pairs[0][1])
    try:
        geometry = FootGeometry(urdf, env.robot.body_names)
        candidate_diagnostics = None
        if args.pose_candidate:
            candidate_diagnostics = json.loads(args.pose_candidate.read_text(encoding='utf-8')).get('diagnostics', {}).get('feet')
        results = []
        reference_feet = None
        reference_body_poses = None
        for trial, (kp, kd) in enumerate(pairs):
            env.kp.fill_(kp)
            env.kd.fill_(kd)
            env.reset()
            if (env.robot.data.joint_pos.torch - env.target).abs().max().item() > 1e-4:
                raise RuntimeError('Imported pose differs from calibrated joint angles')
            initial_feet, initial_boxes = geometry.heights(env.robot.data.body_link_pose_w.torch)
            body_poses = env.robot.data.body_link_pose_w.torch
            if reference_body_poses is None:
                reference_body_poses = body_poses.clone()
            else:
                pos_error = (body_poses[..., :3] - reference_body_poses[..., :3]).abs().max().item()
                dot = (body_poses[..., 3:] * reference_body_poses[..., 3:]).sum(dim=-1).abs()
                if pos_error > 1e-5 or (1-dot).abs().max().item() > 1e-5:
                    raise RuntimeError('Reset did not restore body-link poses')
            if env.robot.data.joint_vel.torch.abs().max().item() > 1e-5:
                raise RuntimeError('Reset did not restore zero joint velocities')
            if reference_feet is None:
                reference_feet = initial_feet.clone()
            elif (initial_feet - reference_feet).abs().max().item() > 1e-5:
                raise RuntimeError('Reset did not restore initial body-link foot heights')
            if trial == 0 and candidate_diagnostics is not None:
                expected = torch.tensor([candidate_diagnostics[f'leg_{i}_knee_link']['bottom_z_m'] for i in range(1, 5)], device=env.device)
                error = (initial_feet[0] - expected).abs().max().item()
                if error > .001:
                    raise RuntimeError(f'Initial foot heights differ from candidate FK by {error:.6f} m')
                print(f'STAND_INITIAL_FOOT_FK_ERROR_M {error}', flush=True)
            n, device = env.num_envs, env.device
            steps = round(args.seconds / env.physics_dt)
            settle_step = round(args.settle / env.physics_dt)
            tail_step = steps - round(1. / env.physics_dt)
            total = torch.zeros((n, 8), device=device)
            peak = torch.zeros((n, 4), device=device)
            foot_min = torch.full((n, 4), float('inf'), device=device)
            foot_max = torch.full((n, 4), -float('inf'), device=device)
            box_min = torch.full((n, 4), float('inf'), device=device)
            foot_sum = torch.zeros((n, 4), device=device)
            foot_near = torch.zeros((n, 4), device=device)
            box_near = torch.zeros((n, 4), device=device)
            start = env.robot.data.root_pos_w.torch[:, :2].clone()
            settled = None
            for step in range(steps):
                obs, _, terminated, truncated, _ = env.step(torch.empty((n, 0), device=device))
                if terminated.any() or truncated.any() or not torch.isfinite(obs['policy']).all():
                    raise RuntimeError('Unexpected reset or non-finite state')
                pos = env.robot.data.root_pos_w.torch[:, :2]
                if step == settle_step:
                    settled = pos.clone()
                if step >= tail_step:
                    foot_z, box_z = geometry.heights(env.robot.data.body_link_pose_w.torch)
                    foot_min = torch.minimum(foot_min, foot_z)
                    foot_max = torch.maximum(foot_max, foot_z)
                    box_min = torch.minimum(box_min, box_z)
                    foot_sum += foot_z
                    foot_near += (foot_z <= .001).float()
                    box_near += (box_z <= .001).float()
                    velocity = torch.linalg.vector_norm(env.robot.data.root_lin_vel_w.torch[:, :2], dim=1)
                    joint_err = (env.robot.data.joint_pos.torch - env.target).abs()
                    torque = env.last_effort.abs()
                    forces = []
                    for i in range(1, 5):
                        force = env.scene[f'contact_{i}'].data.normal_force_matrix_w.torch
                        if force is None or not torch.isfinite(force).all():
                            raise RuntimeError(f'Contact sensor {i} unavailable or non-finite')
                        forces.append(torch.linalg.vector_norm(force[:, 0, 0], dim=-1))
                    forces = torch.stack(forces, dim=1)
                    contacts = forces > .1
                    quat = env.robot.data.root_quat_w.torch
                    tilt = torch.rad2deg(torch.acos((1 - 2*(quat[:, 0]**2 + quat[:, 1]**2)).clamp(-1, 1)))
                    errmax = joint_err.max(dim=1).values
                    saturated = (torque >= EFFORT*.99).any(dim=1)
                    total += torch.stack((velocity, joint_err.mean(dim=1), contacts.all(dim=1).float(),
                                          saturated.float(), *forces.unbind(dim=1)), dim=1)
                    peak = torch.maximum(peak, torch.stack((velocity, errmax, tilt,
                                          torch.linalg.vector_norm(pos - settled, dim=1)), dim=1))
                if (step+1) % max(1, round(.5/env.physics_dt)) == 0:
                    print(f'STAND_SWEEP_PROGRESS {step+1}/{steps}', flush=True)
            tail_count = steps - tail_step
            mean = (total / tail_count).cpu().tolist()
            maxima = peak.cpu().tolist()
            final = env.robot.data.root_pos_w.torch[:, :2]
            final_drift = torch.linalg.vector_norm(final - start, dim=1).cpu().tolist()
            after_drift = torch.linalg.vector_norm(final - settled, dim=1).cpu().tolist()
            m, mx = mean[0], maxima[0]
            results.append({'trial':trial, 'kp':kp, 'kd':kd,
                'final_drift_from_start_m': final_drift[0], 'final_drift_from_settle_m':after_drift[0],
                'tail_mean_horizontal_speed_m_s':m[0], 'tail_max_horizontal_speed_m_s':mx[0],
                'tail_mean_abs_joint_error_rad':m[1], 'tail_max_abs_joint_error_rad':mx[1],
                'tail_all_four_lower_leg_contact_fraction':m[2], 'tail_saturated_any_joint_fraction':m[3],
                'tail_mean_lower_leg_normal_forces_n':m[4:8], 'tail_max_tilt_deg':mx[2],
                'tail_max_drift_from_settle_m':mx[3],
                'initial_foot_min_z_m':initial_feet[0].cpu().tolist(),
                'initial_lower_leg_box_min_z_m':initial_boxes[0].cpu().tolist(),
                'tail_foot_min_z_m':foot_min[0].cpu().tolist(), 'tail_foot_max_z_m':foot_max[0].cpu().tolist(),
                'tail_foot_mean_z_m':(foot_sum[0]/tail_count).cpu().tolist(),
                'tail_lower_leg_box_min_z_m':box_min[0].cpu().tolist(),
                'tail_foot_within_1mm_fraction':(foot_near[0]/tail_count).cpu().tolist(),
                'tail_lower_leg_box_within_1mm_fraction':(box_near[0]/tail_count).cpu().tolist()})
            print('STAND_SWEEP_TRIAL ' + json.dumps(results[-1]), flush=True)
        report = {'backend':'Newton/MuJoCo Warp', 'seconds':steps*env.physics_dt,
                  'pose_candidate_sha256':candidate_hash, 'execution':'sequential single environment',
                  'dt_s':env.physics_dt, 'settle_reference_s':(settle_step+1)*env.physics_dt,
                  'metrics_window_s':tail_count*env.physics_dt, 'effort_limit_nm':EFFORT,
                  'contact_measurement':'normal force on the whole lower-leg/foot link against floor, not foot-only',
                  'geometry_measurement':'minimum signed heights to horizontal ground plane from body-link poses, proximity is not proof of contact',
                  'contact_threshold_n':.1, 'self_collision':False, 'results':results}
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2), encoding='utf-8')
        print('STAND_SWEEP_REPORT ' + json.dumps(report), flush=True)
    finally:
        env.close()


if __name__ == '__main__':
    main()
