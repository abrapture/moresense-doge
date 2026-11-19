"""Run the real robot's sensor task in Isaac Lab / Newton, without a training run."""
import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import torch

from .bno08x import Bno08xCfg
from .usd_runtime import register_kitless_importer_schemas

register_kitless_importer_schemas()

from .imu_check_env import MoresenseImuCheckCfg, MoresenseImuCheckEnv, load_bno08x_cfg
from .isaac_bno08x import IsaacBno08xObserver
from isaaclab.utils.math import quat_apply_inverse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--steps", type=int, default=48)
    parser.add_argument("--report", type=Path, default=Path("verification/bno08x_isaac_check.json"))
    args = parser.parse_args()
    if args.num_envs < 1 or not 8 <= args.steps < 100:
        parser.error("Use a positive environment count and 8..99 initial steps.")
    cfg = MoresenseImuCheckCfg()
    cfg.scene.num_envs = args.num_envs
    sensor_cfg = load_bno08x_cfg()
    env = MoresenseImuCheckEnv(cfg, bno_cfg=sensor_cfg)
    try:
        obs, _ = env.reset()
        assert obs["policy"].shape == (args.num_envs, 30)
        assert env.robot.num_joints == 12
        repeated = env.bno08x.read()
        assert torch.equal(repeated, env.bno08x.read()), "Reading must not resample noise."
        ideal_cfg = replace(sensor_cfg, gyro_noise_std=0, gyro_bias_std=0,
                            tilt_noise_std_deg=0, tilt_bias_std_deg=0)
        ideal = IsaacBno08xObserver(env, ideal_cfg, robot=env.robot)
        down = torch.zeros((args.num_envs, 3), device=env.device)
        down[:, 2] = -1

        def truth():
            return torch.cat((env.robot.data.root_ang_vel_b.torch,
                              quat_apply_inverse(env.robot.data.root_quat_w.torch, down)), dim=-1).clone()

        history = [truth()]
        initial_quat = env.robot.data.root_quat_w.torch.clone()
        max_norm_error = 0.0
        max_sample_hold_error = 0.0
        max_ideal_error = 0.0
        max_noisy_error = 0.0
        changed = False
        previous = repeated.clone()
        samples = []
        previous_source_tick = 0
        for step in range(args.steps):
            obs, reward, terminated, truncated, _ = env.step(torch.empty((args.num_envs, 0), device=env.device))
            imu = obs["policy"][:, :6]
            history.append(truth())
            ideal.advance(env.physics_dt)
            tick = step + 1
            source_tick = max(0, ((tick - ideal.model.delay_steps) // ideal.model.sample_period_steps)
                              * ideal.model.sample_period_steps)
            expected = history[source_tick]
            max_ideal_error = max(max_ideal_error, (ideal.read()-expected).abs().max().item())
            max_noisy_error = max(max_noisy_error, (imu-expected).abs().max().item())
            assert torch.isfinite(obs["policy"]).all()
            assert not terminated.any() and not truncated.any()
            max_norm_error = max(max_norm_error, (torch.linalg.vector_norm(imu[:, 3:], dim=-1)-1).abs().max().item())
            changed = changed or not torch.equal(imu, previous)
            if source_tick == previous_source_tick:
                max_sample_hold_error = max(max_sample_hold_error, (imu-previous).abs().max().item())
            if step < 8 or step == args.steps-1:
                samples.append({"time_s": (step+1)*env.physics_dt, "imu_env_0": imu[0].tolist()})
            previous = imu.clone()
            previous_source_tick = source_tick
        assert changed, "The physical robot must produce changing IMU reports."
        assert max_norm_error < 1e-5
        assert max_sample_hold_error == 0.0
        assert max_ideal_error < 1e-5, "Noiseless observer must match delayed physical state."
        assert max_noisy_error < 0.05, "Configured small noise must stay near delayed physical state."
        assert abs(env.bno08x.model.time_s-args.steps*env.physics_dt) < 1e-9
        rotation_change = (env.robot.data.root_quat_w.torch-initial_quat).abs().max().item()
        assert rotation_change > 1e-4, "Physical root orientation must change."
        # Reach a real DirectRLEnv auto-reset, then check the primed reset state.
        for _ in range(env.max_episode_length-args.steps):
            obs, _, terminated, truncated, _ = env.step(torch.empty((args.num_envs, 0), device=env.device))
        assert truncated.all() and not terminated.any(), "Every sensor check episode must auto-reset."
        assert env.episode_length_buf.eq(0).all()
        assert env.bno08x.model.sample_age_s.eq(0).all(), "Reset must flush the complete delay line."
        reset_error = (obs["policy"][:, :6]-truth()).abs().max().item()
        assert reset_error < 0.05, "Reset observations must describe the new physical state."
        report = {
            "passed": True,
            "purpose": "Sensor integration only; no locomotion training or contact validation.",
            "backend": "Newton / MuJoCo Warp",
            "device": env.device,
            "num_envs": args.num_envs,
            "physics_steps": env.max_episode_length,
            "delayed_state_comparison_steps": args.steps,
            "automatic_resets_checked": args.num_envs,
            "max_reset_measurement_component_error": reset_error,
            "physics_dt_s": env.physics_dt,
            "policy_observation_shape": list(obs["policy"].shape),
            "moving_joints": env.robot.num_joints,
            "sensor_cfg": asdict(sensor_cfg),
            "max_gravity_unit_norm_error": max_norm_error,
            "max_sample_hold_error": max_sample_hold_error,
            "max_noiseless_delayed_physics_error": max_ideal_error,
            "max_noisy_delayed_physics_component_error": max_noisy_error,
            "max_root_quaternion_component_change": rotation_change,
            "samples": samples,
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("BNO08X_ISAAC_CHECK_PASSED " + json.dumps({k:v for k,v in report.items() if k != "samples"}))
    finally:
        env.close()


if __name__ == "__main__":
    main()
