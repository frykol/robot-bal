"""
Trening SAC wyłącznie w symulacji (laptop/PC).

Dla imu_raw6 / imu_raw12 obserwacje to syntetyczne odczyty BMI160 (LSB + szum),
nie I2C na Raspberry Pi. Prawdziwe IMU: calibrate_pi.py, run_policy_pi.py, online_train_pi.py.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from rl.envs import InvertedPendulumEnv
from rl.imu_obs import (
    OBS_MODE_IMU_RAW12,
    OBS_MODE_IMU_RAW6,
    OBS_MODE_PROCESSED4,
    is_raw_imu_mode,
    obs_dim_for_mode,
)
from rl.robot_mass_model import resolve_train_physics
from rl.sac import DEFAULT_HIDDEN_DIM, DEFAULT_LR, SACAgent

DEFAULT_RUNS_ROOT = Path("artifacts") / "runs"
DEFAULT_SAVE_PATH = Path("artifacts") / "actor_sim.pt"
DEFAULT_PLOT_PATH = Path("artifacts") / "learning_curve.png"
DEFAULT_DUAL_SAVE_PATH = Path("artifacts") / "actor_sim_dual.pt"
DEFAULT_DUAL_PLOT_PATH = Path("artifacts") / "learning_curve_dual.png"


def coalesce_plot_path(save_path, user_plot_path, default_plot_path):
    """Wykres zawsze obok wag runu, chyba że podano własny --plot-path."""
    if Path(user_plot_path) != Path(default_plot_path):
        return Path(user_plot_path)
    return Path(save_path).parent / "learning_curve.png"


def _lr_slug(lr):
    text = f"{lr:.0e}".replace("e-0", "e-").replace("e+0", "e+")
    return text.replace(".", "p")


def make_auto_run_name(
    hidden_dim,
    com_height_m,
    train_fall_angle_deg,
    no_domain_randomization,
    episodes,
    lr,
):
    dr = "dr" if not no_domain_randomization else "nodr"
    return (
        f"h{hidden_dim}_com{com_height_m:.3f}_fall{int(train_fall_angle_deg)}"
        f"_{dr}_lr{_lr_slug(lr)}_ep{episodes}"
    )


def resolve_run_paths(args):
    """
    Pick output directory and standard filenames for a training run.

    Priority: --run-dir > --run-name > --auto-run-name > legacy artifacts/.
    Explicit --save-path / --plot-path override only the filenames inside run dir.
    """
    if args.run_dir is not None:
        run_dir = Path(args.run_dir)
    elif args.run_name is not None:
        run_dir = DEFAULT_RUNS_ROOT / args.run_name
    elif args.auto_run_name:
        slug = make_auto_run_name(
            args.hidden_dim,
            args.com_height_m,
            args.train_fall_angle_deg,
            args.no_domain_randomization,
            args.episodes,
            args.lr,
        )
        run_dir = DEFAULT_RUNS_ROOT / slug
    else:
        return args.save_path, args.plot_path, None

    run_dir.mkdir(parents=True, exist_ok=True)
    save_path = run_dir / "actor_sim.pt"
    if args.save_path != DEFAULT_SAVE_PATH:
        save_path = Path(args.save_path)
    plot_path = coalesce_plot_path(save_path, args.plot_path, DEFAULT_PLOT_PATH)
    return save_path, plot_path, run_dir


def save_run_config(run_dir, config):
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "run_config.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, sort_keys=True)
    print(f"Saved run config: {config_path}")


def _print_physics_summary(physics):
    layout = physics.layout
    print("Physics (cart-pole):")
    print(f"  m (axle)     = {physics.m_axle_kg:.3f} kg  (motors @ z=0)")
    print(f"  M (body)     = {physics.M_body_kg:.3f} kg")
    if layout.get("battery_z_m") is not None:
        print(
            f"  stack z [m]  battery={layout['battery_z_m']:.3f}  "
            f"case={layout['case_z_m']:.3f}  rpi={layout['rpi_z_m']:.3f}"
        )
    print(f"  l (body COM) = {physics.l_body_m:.4f} m")
    print(f"  z_COM total  = {physics.z_com_full_m:.4f} m (from axle)")
    print(f"  F_max        = {physics.force_max_n:.2f} N", end="")
    if layout.get("force_from_torque_n") is not None:
        print(f"  (torque limit {layout['force_from_torque_n']:.2f} N, cap {layout.get('force_max_cap_n')})")
    else:
        print()


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
    physics,
    hidden_dim,
    lr,
    dt,
    obs_mode,
    imu_noise_std,
    action_delay_steps=0,
    max_force_delta_per_step=None,
    static_friction_force_n=0.0,
    device=None,
    run_dir=None,
):
    if run_dir is not None:
        print(f"Run output directory: {run_dir.resolve()}")
    print(
        f"Training with lr={lr}, hidden_dim={hidden_dim}, device={device}, "
        f"obs_mode={obs_mode} (obs_dim={obs_dim_for_mode(obs_mode)})"
    )
    if is_raw_imu_mode(obs_mode):
        print(
            "IMU: symulowane acc/gyro (BMI160 LSB) z dynamiki wahadła — bez I2C / Raspberry Pi."
        )
    _print_physics_summary(physics)

    env = InvertedPendulumEnv(
        fall_angle_deg=train_fall_angle_deg,
        domain_randomization=not no_domain_randomization,
        com_height_m=physics.l_body_m,
        m_nominal=physics.m_axle_kg,
        M_nominal=physics.M_body_kg,
        force_max_nominal=physics.force_max_n,
        dt=dt,
        obs_mode=obs_mode,
        imu_noise_std=imu_noise_std,
        imu_mount_height_m=float(physics.layout.get("body_height_m", 0.14)),
        action_delay_steps=action_delay_steps,
        max_force_delta_per_step=max_force_delta_per_step,
        static_friction_force_n=static_friction_force_n,
    )
    agent = SACAgent(
        obs_dim=env.obs_dim,
        act_dim=env.act_dim,
        hidden_dim=hidden_dim,
        lr=lr,
        device=device,
    )

    rewards = []
    episode_steps = []
    best_avg = -np.inf
    best_steps_avg = -1.0
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
            ep_steps = 0
            for _ in range(max_steps):
                action = agent.act(obs, deterministic=False)
                next_obs, reward, done = env.step(action)
                ep_steps += 1
                agent.remember(obs, action, reward, next_obs, done)
                agent.update()
                obs = next_obs
                ep_reward += reward
                if done:
                    break

            rewards.append(ep_reward)
            episode_steps.append(ep_steps)
            completed_episodes = ep
            rolling = float(np.mean(rewards[-rolling_window:]))
            rolling_steps = float(np.mean(episode_steps[-rolling_window:]))
            print(
                f"Ep {ep:04d} | Reward {ep_reward:8.2f} | "
                f"Avg{rolling_window} R {rolling:8.2f} S {rolling_steps:6.0f}"
            )
            live_plot.update(rewards, force=(ep == 1))

            if rolling > best_avg:
                best_avg = rolling

            if rolling_steps > best_steps_avg:
                best_steps_avg = rolling_steps
                best_path.parent.mkdir(parents=True, exist_ok=True)
                agent.save_actor(str(best_path))
                print(
                    f"New best checkpoint: {best_path} "
                    f"(Avg{rolling_window} steps={best_steps_avg:.0f})"
                )

            if save_every > 0 and ep % save_every == 0:
                checkpoint_path = checkpoint_dir / f"actor_ep{ep:04d}.pt"
                persist_training_state(
                    agent,
                    rewards,
                    save_path,
                    plot_path,
                    rolling_window,
                    checkpoint_path=checkpoint_path,
                    episode_steps=episode_steps,
                )
                print(f"Checkpoint saved at episode {ep}")
    except KeyboardInterrupt:
        print("\nTraining interrupted by user (Ctrl+C). Saving current state...")
    finally:
        persist_training_state(
            agent,
            rewards,
            save_path,
            plot_path,
            rolling_window,
            episode_steps=episode_steps,
        )
        live_plot.close()
        print(f"Saved training state after {completed_episodes} episodes.")
        if best_steps_avg >= 0:
            print(f"Best rolling steps: {best_steps_avg:.0f} -> {best_path}")
        if best_avg > -np.inf:
            print(f"Best rolling reward (informacyjnie): {best_avg:.2f}")
        print(f"Plots: {Path(plot_path).resolve()}")


def persist_training_state(
    agent,
    rewards,
    save_path,
    plot_path,
    rolling_window,
    checkpoint_path=None,
    episode_steps=None,
):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    agent.save_actor(str(save_path))
    print(f"Saved actor weights: {save_path}")
    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        agent.save_actor(str(checkpoint_path))
        print(f"Saved episode checkpoint: {checkpoint_path}")
    save_learning_plot(
        rewards, plot_path, rolling_window, episode_steps=episode_steps
    )


def save_learning_plot(rewards, plot_path, rolling_window, episode_steps=None):
    rewards_arr = np.array(rewards, dtype=np.float32)
    if rewards_arr.size == 0:
        return

    rolling = compute_rolling_average(rewards_arr, rolling_window)
    episodes = np.arange(1, len(rewards_arr) + 1)
    plot_path = Path(plot_path)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    run_label = plot_path.parent.name if plot_path.parent.name else "run"

    plt.figure(figsize=(10, 5))
    plt.plot(episodes, rewards_arr, label="Reward / episode", alpha=0.4)
    plt.plot(episodes, rolling, label=f"Rolling avg ({rolling_window})", linewidth=2.0)
    plt.title(f"SAC Learning Curve — {run_label}")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved learning curve: {plot_path.resolve()}")

    if episode_steps is not None and len(episode_steps) > 0:
        steps_arr = np.array(episode_steps, dtype=np.float32)
        rolling_steps = compute_rolling_average(steps_arr, rolling_window)
        steps_path = plot_path.with_name("learning_curve_steps.png")
        plt.figure(figsize=(10, 5))
        plt.plot(episodes, steps_arr, label="Steps / episode", alpha=0.4)
        plt.plot(
            episodes,
            rolling_steps,
            label=f"Rolling avg ({rolling_window})",
            linewidth=2.0,
        )
        plt.title(f"Episode length — {run_label}")
        plt.xlabel("Episode")
        plt.ylabel("Steps")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(steps_path, dpi=150)
        plt.close()
        print(f"Saved steps curve: {steps_path.resolve()}")


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
    # dt default is 2 ms, so 5000 steps ≈ 10 seconds per episode.
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument(
        "--save-path", type=Path, default=DEFAULT_SAVE_PATH
    )
    parser.add_argument(
        "--plot-path",
        type=Path,
        default=DEFAULT_PLOT_PATH,
        help="Domyślnie: <katalog_runu>/learning_curve.png przy --run-name/--run-dir.",
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
        "--dt",
        type=float,
        default=0.002,
        help="Krok czasowy symulacji [s]. Domyślnie 2 ms (0.002).",
    )
    parser.add_argument(
        "--obs-mode",
        choices=[OBS_MODE_PROCESSED4, OBS_MODE_IMU_RAW6, OBS_MODE_IMU_RAW12],
        default=OBS_MODE_PROCESSED4,
        help=(
            "processed4 | imu_raw6 | imu_raw12 — w simie zawsze syntetyczne LSB "
            "(nie prawdziwy I2C; na Pi użyj run_policy_pi / online_train_pi)."
        ),
    )
    parser.add_argument(
        "--imu-noise-std",
        type=float,
        default=25.0,
        help="Szum LSB na kanał (symulacja imu_raw6).",
    )
    parser.add_argument(
        "--no-domain-randomization",
        action="store_true",
        help="Disable sim domain randomization (not recommended for transfer).",
    )
    parser.add_argument(
        "--manual-com-height",
        action="store_true",
        help="Wyłącz model mas; użyj --com-height-m i stałych m/M (stary tryb).",
    )
    parser.add_argument(
        "--com-height-m",
        type=float,
        default=0.11,
        help="Ręczne l [m] (tylko z --manual-com-height).",
    )
    parser.add_argument("--body-height-m", type=float, default=0.14, help="Wysokość korpusu nad osią [m].")
    parser.add_argument("--battery-z-m", type=float, default=None, help="Wys. środka baterii [m].")
    parser.add_argument("--case-z-m", type=float, default=None, help="Wys. środka masy obudowy [m].")
    parser.add_argument("--rpi-z-m", type=float, default=None, help="Wys. środka RPi [m].")
    parser.add_argument("--motor-mass-g", type=float, default=160.0)
    parser.add_argument("--n-motors", type=int, default=2)
    parser.add_argument("--rpi-mass-g", type=float, default=55.0)
    parser.add_argument("--case-mass-g", type=float, default=466.0)
    parser.add_argument("--battery-mass-g", type=float, default=250.0)
    parser.add_argument("--wheel-radius-m", type=float, default=0.03)
    parser.add_argument("--motor-torque-nm", type=float, default=0.35, help="Moment na silnik [Nm].")
    parser.add_argument("--n-drive-motors", type=int, default=2)
    parser.add_argument(
        "--force-max",
        type=float,
        default=10.0,
        help="Górny limit siły poziomej [N] (min z limitem z momentu).",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=DEFAULT_HIDDEN_DIM,
        help="Liczba neuronów w warstwach ukrytych actor/critic.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=DEFAULT_LR,
        help="Learning rate actor/critic (Adam), np. 3e-4 lub 1e-4.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda"],
        help="Urządzenie PyTorch (domyślnie: cuda jeśli wspierane, inaczej cpu).",
    )
    parser.add_argument(
        "--action-delay-steps",
        type=int,
        default=0,
        help="Opóźnienie siły w symulacji [kroki]; 0 = wyłączone (train_sim_dual domyślnie 2).",
    )
    parser.add_argument(
        "--max-force-delta-frac",
        type=float,
        default=0.0,
        help="Slew-rate: max Δsiły na krok jako ułamek F_max; 0 = wyłączone.",
    )
    parser.add_argument(
        "--static-friction-frac",
        type=float,
        default=0.0,
        help="Martwa strefa siły (ułamek F_max); 0 = wyłączone.",
    )
    run_group = parser.add_mutually_exclusive_group()
    run_group.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Katalog na ten trening (actor_*.pt, wykres, checkpoints/, run_config.json).",
    )
    run_group.add_argument(
        "--run-name",
        type=str,
        default=None,
        help=f"Skrót: zapis do {DEFAULT_RUNS_ROOT}/<nazwa>/ (np. h64_com010).",
    )
    run_group.add_argument(
        "--auto-run-name",
        action="store_true",
        help="Katalog z nazwy z parametrów (hidden, com, fall, dr, episodes).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.com_from_masses = not args.manual_com_height
    physics = resolve_train_physics(args)
    args.com_height_m = physics.l_body_m

    save_path, plot_path, run_dir = resolve_run_paths(args)
    max_force_delta = (
        None
        if args.max_force_delta_frac <= 0
        else args.max_force_delta_frac * physics.force_max_n
    )
    static_friction = args.static_friction_frac * physics.force_max_n

    if run_dir is not None:
        save_run_config(
            run_dir,
            {
                "started_at_utc": datetime.now(timezone.utc).isoformat(),
                "run_dir": str(run_dir),
                "episodes": args.episodes,
                "max_steps": args.max_steps,
                "dt": args.dt,
                "obs_mode": args.obs_mode,
                "imu_noise_std": args.imu_noise_std,
                "rolling_window": args.rolling_window,
                "save_every": args.save_every,
                "hidden_dim": args.hidden_dim,
                "lr": args.lr,
                "com_height_m": physics.l_body_m,
                "z_com_full_m": physics.z_com_full_m,
                "m_axle_kg": physics.m_axle_kg,
                "M_body_kg": physics.M_body_kg,
                "force_max_n": physics.force_max_n,
                "physics_layout": physics.layout,
                "train_fall_angle_deg": args.train_fall_angle_deg,
                "domain_randomization": not args.no_domain_randomization,
                "save_path": str(save_path),
                "plot_path": str(plot_path),
                "best_checkpoint_metric": "rolling_mean_episode_steps",
                "action_delay_steps": args.action_delay_steps,
                "max_force_delta_per_step": max_force_delta,
                "static_friction_force_n": static_friction,
            },
        )

    train(
        args.episodes,
        args.max_steps,
        save_path,
        plot_path,
        args.rolling_window,
        args.save_every,
        not args.no_live_plot,
        args.plot_update_every,
        args.train_fall_angle_deg,
        args.no_domain_randomization,
        physics,
        args.hidden_dim,
        args.lr,
        args.dt,
        args.obs_mode,
        args.imu_noise_std,
        args.action_delay_steps,
        max_force_delta,
        static_friction,
        args.device,
        run_dir=run_dir,
    )

