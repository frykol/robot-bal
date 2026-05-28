import argparse
import json
import time
from pathlib import Path

import numpy as np

from rl.envs import RaspberryBalanceRuntime
from rl.sac import SACAgent


def _profile_to_motor_scale(profile):
    if profile == "safe":
        return 0.30
    if profile == "normal":
        return 0.55
    return 0.75


def compute_online_reward(obs, action, fall_angle_rad):
    pitch = float(obs[0])
    x = float(obs[2])
    x_dot = float(obs[3])
    u = float(action[0])

    angle_term = 1.0 - min(abs(pitch) / max(fall_angle_rad, 1e-6), 1.0)
    center_term = max(0.0, 1.0 - 0.2 * abs(x))
    speed_penalty = 0.02 * abs(x_dot)
    action_penalty = 0.01 * abs(u)
    reward = angle_term + 0.2 * center_term - speed_penalty - action_penalty
    return reward


def run_online_training(args):
    calibration = {}
    if args.calibration_path.exists():
        calibration = json.loads(args.calibration_path.read_text(encoding="utf-8"))

    env = RaspberryBalanceRuntime(
        motor_scale=_profile_to_motor_scale(args.profile),
        loop_hz=args.loop_hz,
        gyro_bias_dps=float(calibration.get("gyro_bias_dps", 0.0)),
        accel_pitch_bias_rad=float(calibration.get("accel_pitch_bias_rad", 0.0)),
        fall_angle_deg=args.tilt_limit_deg,
    )

    agent = SACAgent(
        obs_dim=env.obs_dim,
        act_dim=env.act_dim,
        lr=args.lr,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
    )

    if args.resume_checkpoint and args.resume_checkpoint.exists():
        agent.load_checkpoint(str(args.resume_checkpoint), load_optimizers=True)
        print(f"Loaded SAC checkpoint: {args.resume_checkpoint}")
    else:
        agent.load_actor(str(args.actor_path))
        print(f"Loaded actor weights: {args.actor_path}")

    save_dir = args.save_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    latest_ckpt = save_dir / "sac_online_latest.pt"
    best_ckpt = save_dir / "sac_online_best.pt"

    best_avg = -np.inf
    rewards = []

    obs = env.reset()
    episode_reward = 0.0
    episode_step = 0
    episode_idx = 1
    last_save_t = time.time()

    print("Starting online SAC fine-tuning on Raspberry. Ctrl+C to stop.")
    try:
        while episode_idx <= args.episodes:
            if np.random.rand() < args.explore_prob:
                action = agent.act(obs, deterministic=False)
            else:
                action = agent.act(obs, deterministic=True)

            next_obs, _, done = env.step(action)
            reward = compute_online_reward(next_obs, action, env.fall_angle_rad)
            if done:
                reward -= args.fall_penalty

            agent.remember(obs, action, reward, next_obs, done)
            for _ in range(args.updates_per_step):
                agent.update()

            obs = next_obs
            episode_reward += reward
            episode_step += 1

            if done or episode_step >= args.max_steps:
                rewards.append(episode_reward)
                avg = float(np.mean(rewards[-args.rolling_window :]))
                print(
                    f"Ep {episode_idx:04d} | Steps {episode_step:04d} | "
                    f"Reward {episode_reward:8.2f} | Avg{args.rolling_window} {avg:8.2f}"
                )

                if avg > best_avg:
                    best_avg = avg
                    agent.save_checkpoint(str(best_ckpt))
                    print(f"New best online checkpoint: {best_ckpt} (avg={best_avg:.2f})")

                now = time.time()
                if now - last_save_t >= args.save_interval_sec:
                    agent.save_checkpoint(str(latest_ckpt))
                    print(f"Periodic checkpoint saved: {latest_ckpt}")
                    last_save_t = now

                obs = env.reset()
                episode_reward = 0.0
                episode_step = 0
                episode_idx += 1
    except KeyboardInterrupt:
        print("\nStopped by user, saving latest checkpoint...")
    finally:
        agent.save_checkpoint(str(latest_ckpt))
        env.close()
        print(f"Saved latest online checkpoint: {latest_ckpt}")
        if best_avg > -np.inf:
            print(f"Best rolling average: {best_avg:.2f} -> {best_ckpt}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--actor-path", type=Path, default=Path("artifacts/actor_best.pt"))
    parser.add_argument("--resume-checkpoint", type=Path, default=Path("artifacts/sac_online_latest.pt"))
    parser.add_argument("--calibration-path", type=Path, default=Path("artifacts/pi_calibration.json"))
    parser.add_argument("--save-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--profile", choices=["safe", "normal", "aggressive"], default="safe")
    parser.add_argument("--loop-hz", type=int, default=100)
    parser.add_argument("--tilt-limit-deg", type=float, default=20.0)
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--rolling-window", type=int, default=20)
    parser.add_argument("--fall-penalty", type=float, default=100.0)
    parser.add_argument("--explore-prob", type=float, default=0.1)
    parser.add_argument("--updates-per-step", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--buffer-size", type=int, default=30_000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save-interval-sec", type=int, default=120)
    return parser.parse_args()


if __name__ == "__main__":
    run_online_training(parse_args())

