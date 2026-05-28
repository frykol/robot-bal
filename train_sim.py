import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from rl.envs import InvertedPendulumEnv
from rl.sac import DEFAULT_HIDDEN_DIM, SACAgent


def train(
    episodes,
    max_steps,
    save_path,
    plot_path,
    rolling_window,
    save_every,
    live_plot_enabled,
    plot_update_every,
    train_fall_angle_deg,
    no_domain_randomization,
    com_height_m,
    hidden_dim,
):
    env = InvertedPendulumEnv(
        fall_angle_deg=train_fall_angle_deg,
        domain_randomization=not no_domain_randomization,
        com_height_m=com_height_m,
    )
    agent = SACAgent(obs_dim=env.obs_dim, act_dim=env.act_dim, hidden_dim=hidden_dim)

    rewards = []
    best_avg = -np.inf
    best_path = save_path.with_name("actor_best.pt")
    checkpoint_dir = save_path.parent / "checkpoints"
    live_plot = LivePlotter(
        rolling_window=rolling_window,
        enabled=live_plot_enabled,
        update_every=plot_update_every,
    )
    completed_episodes = 0
    try:
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
            completed_episodes = ep
            rolling = float(np.mean(rewards[-rolling_window:]))
            print(
                f"Ep {ep:04d} | Reward {ep_reward:8.2f} | "
                f"Avg{rolling_window} {rolling:8.2f}"
            )
            live_plot.update(rewards, force=(ep == 1))

            if rolling > best_avg:
                best_avg = rolling
                best_path.parent.mkdir(parents=True, exist_ok=True)
                agent.save_actor(str(best_path))
                print(f"New best checkpoint: {best_path} (Avg{rolling_window}={best_avg:.2f})")

            if save_every > 0 and ep % save_every == 0:
                checkpoint_path = checkpoint_dir / f"actor_ep{ep:04d}.pt"
                persist_training_state(
                    agent,
                    rewards,
                    save_path,
                    plot_path,
                    rolling_window,
                    checkpoint_path=checkpoint_path,
                )
                print(f"Checkpoint saved at episode {ep}")
    except KeyboardInterrupt:
        print("\nTraining interrupted by user (Ctrl+C). Saving current state...")
    finally:
        persist_training_state(agent, rewards, save_path, plot_path, rolling_window)
        live_plot.close()
        print(f"Saved training state after {completed_episodes} episodes.")
        if best_avg > -np.inf:
            print(f"Best rolling average: {best_avg:.2f} -> {best_path}")


def persist_training_state(
    agent, rewards, save_path, plot_path, rolling_window, checkpoint_path=None
):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    agent.save_actor(str(save_path))
    print(f"Saved actor weights: {save_path}")
    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        agent.save_actor(str(checkpoint_path))
        print(f"Saved episode checkpoint: {checkpoint_path}")
    save_learning_plot(rewards, plot_path, rolling_window)


def save_learning_plot(rewards, plot_path, rolling_window):
    rewards_arr = np.array(rewards, dtype=np.float32)
    if rewards_arr.size == 0:
        return

    rolling = compute_rolling_average(rewards_arr, rolling_window)
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


class LivePlotter:
    def __init__(self, rolling_window, enabled=True, update_every=5):
        self.rolling_window = int(max(1, rolling_window))
        self.enabled = bool(enabled)
        self.update_every = int(max(1, update_every))
        self._fig = None
        self._ax = None
        self._line_reward = None
        self._line_roll = None
        self._updates = 0

        if self.enabled:
            plt.ion()
            self._fig, self._ax = plt.subplots(figsize=(10, 5))
            self._line_reward, = self._ax.plot([], [], label="Reward / episode", alpha=0.4)
            self._line_roll, = self._ax.plot([], [], label=f"Rolling avg ({self.rolling_window})", linewidth=2.0)
            self._ax.set_title("SAC Learning Curve (Live)")
            self._ax.set_xlabel("Episode")
            self._ax.set_ylabel("Reward")
            self._ax.grid(True, alpha=0.3)
            self._ax.legend()
            self._fig.tight_layout()

    def update(self, rewards, force=False):
        if not self.enabled:
            return

        self._updates += 1
        if not force and self._updates % self.update_every != 0:
            return
        if len(rewards) == 0:
            return

        rewards_arr = np.array(rewards, dtype=np.float32)
        rolling = compute_rolling_average(rewards_arr, self.rolling_window)
        episodes = np.arange(1, len(rewards_arr) + 1)

        self._line_reward.set_data(episodes, rewards_arr)
        self._line_roll.set_data(episodes, rolling)
        self._ax.relim()
        self._ax.autoscale_view()
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()
        plt.pause(0.001)

    def close(self):
        if not self.enabled:
            return
        plt.ioff()
        plt.close(self._fig)


def compute_rolling_average(values, window):
    window = int(max(1, window))
    n = len(values)
    if n == 0:
        return np.array([], dtype=np.float32)

    rolling = np.empty(n, dtype=np.float32)
    cumsum = np.cumsum(values, dtype=np.float64)
    for i in range(n):
        start = max(0, i - window + 1)
        total = cumsum[i] - (cumsum[start - 1] if start > 0 else 0.0)
        rolling[i] = total / float(i - start + 1)
    return rolling


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
    parser.add_argument(
        "--save-every",
        type=int,
        default=50,
        help="Checkpoint interval in episodes (0 disables periodic checkpoints).",
    )
    parser.add_argument(
        "--no-live-plot",
        action="store_true",
        help="Disable live updating plot window during training.",
    )
    parser.add_argument(
        "--plot-update-every",
        type=int,
        default=5,
        help="Update live plot every N episodes.",
    )
    parser.add_argument(
        "--train-fall-angle-deg",
        type=float,
        default=30.0,
        help="Fall angle threshold used in simulation training environment.",
    )
    parser.add_argument(
        "--no-domain-randomization",
        action="store_true",
        help="Disable sim domain randomization (not recommended for transfer).",
    )
    parser.add_argument(
        "--com-height-m",
        type=float,
        default=0.11,
        help="Axle-to-center-of-mass distance in meters for training dynamics.",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=DEFAULT_HIDDEN_DIM,
        help="Liczba neuronów w warstwach ukrytych actor/critic.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        args.episodes,
        args.max_steps,
        args.save_path,
        args.plot_path,
        args.rolling_window,
        args.save_every,
        not args.no_live_plot,
        args.plot_update_every,
        args.train_fall_angle_deg,
        args.no_domain_randomization,
        args.com_height_m,
        args.hidden_dim,
    )

