import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from rl.envs import InvertedPendulumEnv
from rl.sac import SACAgent


def train(episodes, max_steps, save_path, plot_path, rolling_window):
    env = InvertedPendulumEnv()
    agent = SACAgent(obs_dim=env.obs_dim, act_dim=env.act_dim)

    rewards = []
    for ep in range(1, episodes + 1):
        obs = env.reset()
        ep_reward = 0.0
        for _ in range(max_steps):
            action = agent.act(obs, deterministic=False)
            next_obs, reward, done = env.step(action)
            agent.remember(obs, action, reward, next_obs, done)
            agent.update()
            obs = next_obs
            ep_reward += reward
            if done:
                break

        rewards.append(ep_reward)
        rolling = float(np.mean(rewards[-rolling_window:]))
        print(
            f"Ep {ep:04d} | Reward {ep_reward:8.2f} | "
            f"Avg{rolling_window} {rolling:8.2f}"
        )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    agent.save_actor(str(save_path))
    print(f"Saved actor weights: {save_path}")
    save_learning_plot(rewards, plot_path, rolling_window)


def save_learning_plot(rewards, plot_path, rolling_window):
    rewards_arr = np.array(rewards, dtype=np.float32)
    if rewards_arr.size == 0:
        return

    kernel = np.ones(rolling_window, dtype=np.float32) / float(rolling_window)
    rolling = np.convolve(rewards_arr, kernel, mode="same")
    episodes = np.arange(1, len(rewards_arr) + 1)

    plot_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(episodes, rewards_arr, label="Reward / episode", alpha=0.4)
    plt.plot(episodes, rolling, label=f"Rolling avg ({rolling_window})", linewidth=2.0)
    plt.title("SAC Learning Curve")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved learning curve: {plot_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument(
        "--save-path", type=Path, default=Path("artifacts") / "actor_sim.pt"
    )
    parser.add_argument(
        "--plot-path", type=Path, default=Path("artifacts") / "learning_curve.png"
    )
    parser.add_argument("--rolling-window", type=int, default=50)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        args.episodes,
        args.max_steps,
        args.save_path,
        args.plot_path,
        args.rolling_window,
    )

