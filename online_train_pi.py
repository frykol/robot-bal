import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

# Lighter CPU load on Raspberry Pi (avoids some crashes during backward pass).
try:
    import torch

    torch.set_num_threads(1)
except ImportError:
    torch = None

from rl.envs import RaspberryBalanceRuntime
from rl.sac import SACAgent, infer_dims_from_actor_file


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
    return angle_term + 0.2 * center_term - speed_penalty - action_penalty


def _safe_save_checkpoint(agent, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    agent.save_checkpoint(str(tmp))
    tmp.replace(path)


def run_online_training(args):
    if torch is None:
        print("PyTorch is required for online_train_pi.py", file=sys.stderr)
        sys.exit(1)

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

    resume_path = args.resume_checkpoint
    weights_path = resume_path if (resume_path is not None and resume_path.exists()) else args.actor_path

    if not weights_path.exists():
        print(f"Missing weights: {weights_path}", file=sys.stderr)
        sys.exit(1)

    ckpt_obs, ckpt_act, ckpt_hidden = infer_dims_from_actor_file(weights_path)
    if args.hidden_dim is not None and args.hidden_dim != ckpt_hidden:
        print(
            f"Error: --hidden-dim {args.hidden_dim} does not match checkpoint ({ckpt_hidden}). "
            "Remove --hidden-dim to auto-detect.",
            file=sys.stderr,
        )
        sys.exit(1)
    hidden_dim = ckpt_hidden
    if ckpt_obs != env.obs_dim or ckpt_act != env.act_dim:
        print(
            f"Warning: checkpoint dims ({ckpt_obs},{ckpt_act}) != env ({env.obs_dim},{env.act_dim})",
            file=sys.stderr,
        )
    print(f"Using hidden_dim={hidden_dim} (checkpoint actor uses {ckpt_hidden})")

    agent = SACAgent(
        obs_dim=env.obs_dim,
        act_dim=env.act_dim,
        lr=args.lr,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        hidden_dim=hidden_dim,
    )

    if resume_path is not None and resume_path.exists():
        agent.load_checkpoint(str(resume_path), load_optimizers=True)
        print(f"Loaded SAC checkpoint: {resume_path}")
    else:
        agent.load_actor_for_training(str(args.actor_path))
        print(f"Loaded actor for online training: {args.actor_path}")
        print(
            "Critics start fresh; gradient updates begin after buffer reaches "
            f"{args.batch_size} transitions."
        )

    save_dir = args.save_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    latest_ckpt = save_dir / "sac_online_latest.pt"
    best_ckpt = save_dir / "sac_online_best.pt"

    best_avg = -np.inf
    rewards = []
    updates_started = False

    obs = env.reset()
    episode_reward = 0.0
    episode_step = 0
    episode_idx = 1
    last_save_t = time.time()

    print(
        f"Online SAC on {torch.device('cpu')} | batch={args.batch_size} | "
        f"hidden={args.hidden_dim} | Ctrl+C to stop."
    )
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

            agent.remember(obs, action, reward, next_obs, float(done))

            if len(agent.buffer) >= args.batch_size:
                if not updates_started:
                    updates_started = True
                    print(
                        f"Buffer ready ({len(agent.buffer)}). "
                        "Starting SAC gradient updates..."
                    )
                for _ in range(args.updates_per_step):
                    try:
                        agent.update()
                    except Exception as exc:
                        print("\nSAC update failed:", exc, file=sys.stderr)
                        traceback.print_exc()
                        print(
                            "\nTypical on Raspberry Pi: illegal instruction from PyTorch "
                            "or out of memory. Try:\n"
                            "  --batch-size 16 --hidden-dim 64 --updates-per-step 0\n"
                            "  (collect data only) or reinstall torch from piwheels.\n",
                            file=sys.stderr,
                        )
                        raise

            obs = next_obs
            episode_reward += reward
            episode_step += 1

            if done or episode_step >= args.max_steps:
                rewards.append(episode_reward)
                avg = float(np.mean(rewards[-args.rolling_window :]))
                buf_len = len(agent.buffer)
                print(
                    f"Ep {episode_idx:04d} | Steps {episode_step:04d} | "
                    f"Reward {episode_reward:8.2f} | Avg{args.rolling_window} {avg:8.2f} | "
                    f"buf {buf_len}"
                )

                if avg > best_avg:
                    best_avg = avg
                    _safe_save_checkpoint(agent, best_ckpt)
                    print(f"New best online checkpoint: {best_ckpt} (avg={best_avg:.2f})")

                now = time.time()
                if now - last_save_t >= args.save_interval_sec:
                    _safe_save_checkpoint(agent, latest_ckpt)
                    print(f"Periodic checkpoint saved: {latest_ckpt}")
                    last_save_t = now

                obs = env.reset()
                episode_reward = 0.0
                episode_step = 0
                episode_idx += 1
    except KeyboardInterrupt:
        print("\nStopped by user, saving latest checkpoint...")
    finally:
        env.drive.stop()
        try:
            _safe_save_checkpoint(agent, latest_ckpt)
            print(f"Saved latest online checkpoint: {latest_ckpt}")
        except Exception as exc:
            print(f"Could not save checkpoint: {exc}", file=sys.stderr)
        env.close()
        if best_avg > -np.inf:
            print(f"Best rolling average: {best_avg:.2f} -> {best_ckpt}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--actor-path", type=Path, default=Path("artifacts/actor_best.pt"))
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        default=None,
        help="Resume full SAC checkpoint (use only with --resume).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from artifacts/sac_online_latest.pt if it exists.",
    )
    parser.add_argument("--calibration-path", type=Path, default=Path("artifacts/pi_calibration.json"))
    parser.add_argument("--save-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--profile", choices=["safe", "normal", "aggressive"], default="safe")
    parser.add_argument("--loop-hz", type=int, default=100)
    parser.add_argument("--tilt-limit-deg", type=float, default=20.0)
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--rolling-window", type=int, default=20)
    parser.add_argument("--fall-penalty", type=float, default=100.0)
    parser.add_argument("--explore-prob", type=float, default=0.05)
    parser.add_argument(
        "--updates-per-step",
        type=int,
        default=1,
        help="SAC gradient steps per control step (0 = inference + buffer only).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Keep low on Raspberry Pi (16-32). Was 128 and often crashed ~ep 16.",
    )
    parser.add_argument("--buffer-size", type=int, default=10_000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=None,
        help="Override network width; default: auto from actor checkpoint (usually 256).",
    )
    parser.add_argument("--save-interval-sec", type=int, default=120)
    args = parser.parse_args()

    if args.resume and args.resume_checkpoint is None:
        args.resume_checkpoint = args.save_dir / "sac_online_latest.pt"
    return args


if __name__ == "__main__":
    run_online_training(parse_args())
