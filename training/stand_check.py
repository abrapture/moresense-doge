"""Run a provisional PD stand on Newton's floor, without RL training."""
import argparse
import json
import math
from pathlib import Path

from .usd_runtime import register_kitless_importer_schemas
register_kitless_importer_schemas()

import torch
from .stand_check_env import StandCfg, StandEnv, read_calibration, with_pose_candidate, EFFORT, SPEED, KP, KD


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--calibration', type=Path, default=Path(__file__).resolve().parent.parent / 'calibration')
    p.add_argument('--pose-candidate', type=Path)
    p.add_argument('--seconds', type=float, default=5.0)
    p.add_argument('--kp', type=float, default=KP)
    p.add_argument('--kd', type=float, default=KD)
    p.add_argument('--dt', type=float, default=0.001)
    p.add_argument('--report', type=Path, default=Path('verification/stand_check.json'))
    args = p.parse_args()
    if not math.isfinite(args.seconds) or not 0 < args.seconds < 29:
        p.error('--seconds must be positive and below 29')
    if not all(math.isfinite(x) and x > 0 for x in (args.kp, args.kd, args.dt)):
        p.error('kp, kd and dt must be positive and finite')
    calibration, urdf = read_calibration(args.calibration)
    candidate_hash = None
    if args.pose_candidate:
        calibration, candidate_hash = with_pose_candidate(calibration, args.pose_candidate)
    cfg = StandCfg()
    cfg.sim.dt = args.dt
    cfg.scene.robot.spawn.asset_path = str(urdf)
    env = StandEnv(cfg, calibration, kp=args.kp, kd=args.kd)
    try:
        env.reset()
        q0 = env.robot.data.joint_pos.torch.clone()
        error = (q0-env.target).abs().max().item()
        if error > 1e-4:
            raise ValueError(f'Imported joint pose differs from saved pose: {error} rad; joint wrapping is not accepted')
        print('STAND_INITIAL_POSE_OK max_error_rad=' + str(error), flush=True)
        start = env.robot.data.root_pos_w.torch[0].clone()
        samples = []
        contacts = {str(i): {'contact_steps':0, 'max_normal_force_n':0.0, 'sum_normal_force_n':0.0} for i in range(1, 5)}
        max_error = max_tilt = max_drift = max_effort = 0.0
        saturated_steps = 0
        steps = round(args.seconds / env.physics_dt)
        for step in range(steps):
            obs, _, terminated, truncated, _ = env.step(torch.empty((1, 0), device=env.device))
            if step == 0 or (step + 1) % 100 == 0:
                print(f'STAND_PROGRESS {step+1}/{steps} physics steps', flush=True)
            if terminated.any() or truncated.any() or not torch.isfinite(obs['policy']).all():
                raise RuntimeError('Unexpected reset or non-finite state')
            pos = env.robot.data.root_pos_w.torch[0]
            quat = env.robot.data.root_quat_w.torch[0]
            # cosine between world up and body up, Isaac Lab 3 quaternion order xyzw
            tilt = math.degrees(math.acos(max(-1., min(1., 1-2*(quat[0].item()**2 + quat[1].item()**2)))))
            drift = torch.linalg.vector_norm(pos[:2]-start[:2]).item()
            joint_error = (env.robot.data.joint_pos.torch-env.target).abs().max().item()
            torque = env.last_effort.abs().max().item()
            normal_forces = {}
            for i in range(1, 5):
                force = env.scene[f'contact_{i}'].data.normal_force_matrix_w.torch
                if force is None or not torch.isfinite(force).all():
                    raise RuntimeError(f'Contact sensor {i} is unavailable or non-finite')
                magnitude = torch.linalg.vector_norm(force[0, 0, 0]).item()
                normal_forces[str(i)] = magnitude
                item = contacts[str(i)]
                item['contact_steps'] += int(magnitude > 0.1)
                item['max_normal_force_n'] = max(item['max_normal_force_n'], magnitude)
                item['sum_normal_force_n'] += magnitude
            max_error, max_tilt, max_drift, max_effort = max(max_error,joint_error), max(max_tilt,tilt), max(max_drift,drift), max(max_effort,torque)
            saturated_steps += int(torque >= EFFORT*0.99)
            if step % max(1, round(0.1/env.physics_dt)) == 0 or step == steps-1:
                samples.append({'time_s': (step+1)*env.physics_dt, 'root_z_m': pos[2].item(),
                                'tilt_deg': tilt, 'horizontal_drift_m': drift,
                                'max_joint_error_rad': joint_error, 'max_effort_nm': torque,
                                'knee_link_floor_normal_forces_n': normal_forces})
        for item in contacts.values():
            item['contact_fraction'] = item['contact_steps']/steps
            item['mean_normal_force_n'] = item.pop('sum_normal_force_n')/steps
        report = {'backend':'Newton/MuJoCo Warp', 'purpose':'provisional stand check, not training',
                  'pose_candidate_sha256':candidate_hash,
                  'seconds':steps*env.physics_dt, 'initial_joint_error_rad':error,
                  'joint_names':env.robot.joint_names, 'max_joint_error_rad':max_error,
                  'max_tilt_deg':max_tilt, 'max_horizontal_drift_m':max_drift,
                  'max_effort_nm':max_effort, 'saturated_step_fraction':saturated_steps/steps,
                  'controller':{'kp':args.kp, 'kd':args.kd, 'physics_dt_s':args.dt, 'effort_limit_nm':EFFORT, 'reference_no_load_speed_rad_s':SPEED},
                  'contact_measurement':'filtered normal force on knee link against floor; lower leg and foot share this body',
                  'knee_link_contacts': contacts,
                  'self_collision':False, 'samples':samples}
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2), encoding='utf-8')
        print('STAND_CHECK_REPORT ' + json.dumps({k:v for k,v in report.items() if k != 'samples'}), flush=True)
    finally:
        env.close()


if __name__ == '__main__':
    main()
