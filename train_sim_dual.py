"""
Trening SAC w symulacji: 2 wyjścia akcji (kierunek × skala mocy), rzadka nagroda.

- action[0]: kierunek [-1, 1] (tanh z sieci)
- action[1]: skala mocy [0, 1] — mapowanie (tanh+1)/2 z drugiego wyjścia
- nagroda: 0 żyje; upadek: -100*(T-t)/T; suma ep.=0 → umie cały epizod
- gamma=0.999, lr liniowo 1/100 → 1/1000 przez wszystkie epizody
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from rl.envs_dual import DualActionPendulumEnv, FALL_PENALTY_MAX
from rl.imu_obs import (
    OBS_MODE_IMU_RAW12,
    OBS_MODE_IMU_RAW6,
    OBS_MODE_PROCESSED4,
    is_raw_imu_mode,
    obs_dim_for_mode,
)
from rl.robot_mass_model import resolve_train_physics
from rl.sac import DEFAULT_HIDDEN_DIM, SACAgent
from train_sim import (
    LivePlotter,
    _print_physics_summary,
    compute_rolling_average,
    persist_training_state,
    save_run_config,
)

DEFAULT_RUNS_ROOT = Path("artifacts") / "runs"
DEFAULT_GAMMA = 0.999
DEFAULT_LR_START = 0.01   # 1/100
DEFAULT_LR_END = 0.001    # 1/1000
DEFAULT_ALPHA_START = 0.4
DEFAULT_ALPHA_END = 0.05
DEFAULT_ALPHA_DECAY_EPISODES = 170
DEFAULT_HIDDEN_ACTIVATION = "tanh"
DEFAULT_BUFFER_SIZE = 50_000
DEFAULT_BATCH_SIZE = 64


def lr_for_episode(episode, total_episodes, lr_start, lr_end):
    """Liniowo: ep 1 → lr_start, ep total_episodes → lr_end."""
    if total_episodes <= 1:
        return float(lr_end)
    t = (episode - 1) / float(total_episodes - 1)
    return float(lr_start) + t * (float(lr_end) - float(lr_start))


def alpha_for_episode(episode, warmup_episodes, decay_episodes, alpha_start, alpha_end):
    """
    alpha_start through warmup; then linear decay to alpha_end over decay_episodes after warmup.
    """
    if episode <= warmup_episodes:
        return float(alpha_start)
    if decay_episodes <= 0:
        return float(alpha_end)
    idx = episode - warmup_episodes
    if idx >= decay_episodes:
        return float(alpha_end)
    if decay_episodes == 1:
        return float(alpha_end)
    t = (idx - 1) / float(decay_episodes - 1)
    return float(alpha_start) + t * (float(alpha_end) - float(alpha_start))


def make_auto_run_name(hidden_dims, com_height_m, train_fall_angle_deg, episodes, lr_start, lr_end):
    htag = "_".join(str(h) for h in hidden_dims)
    return (
        f"dual_h{htag}_com{com_height_m:.3f}_fall{int(train_fall_angle_deg)}"
        f"_g{DEFAULT_GAMMA:.3f}_lr{lr_start:.0e}-{lr_end:.0e}_ep{episodes}"
    )


def resolve_run_paths(args):
    if args.run_dir is not None:
        run_dir = Path(args.run_dir)
    elif args.run_name is not None:
        run_dir = DEFAULT_RUNS_ROOT / args.run_name
    elif args.auto_run_name:
        slug = make_auto_run_name(
            args.hidden_dims,
            args.com_height_m,
            args.train_fall_angle_deg,
            args.episodes,
            args.lr_start,
            args.lr_end,
        )
        run_dir = DEFAULT_RUNS_ROOT / slug
    else:
        return args.save_path, args.plot_path, None

    run_dir.mkdir(parents=True, exist_ok=True)
    save_path = run_dir / "actor_sim_dual.pt"
    plot_path = run_dir / "learning_curve.png"
    if args.save_path != Path("artifacts") / "actor_sim_dual.pt":
        save_path = args.save_path
    if args.plot_path != Path("artifacts") / "learning_curve.png":
        plot_path = args.plot_path
    return save_path, plot_path, run_dir


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
    hidden_dims,
    hidden_activation,
    buffer_size,
    batch_size,
    lr_start,
    lr_end,
    gamma,
    dt,
    obs_mode,
    imu_noise_std,
    fall_penalty_max,
    min_motor_power,
    init_angle_deg,
    random_warmup_episodes,
    alpha_start,
    alpha_end,
    alpha_decay_episodes,
    device=None,
    run_dir=None,
):
    if run_dir is not None:
        print(f"Run output directory: {run_dir.resolve()}")
    print(
        f"Dual-action SAC | gamma={gamma} | lr {lr_start} → {lr_end} over {episodes} ep | "
        f"hidden_dims={hidden_dims} ({hidden_activation}) | buffer={buffer_size} | "
        f"batch={batch_size} | device={device} | "
        f"obs_mode={obs_mode} (obs_dim={obs_dim_for_mode(obs_mode)}, act_dim=2)"
    )
    print(
        f"Reward: 0 na żywych krokach; upadek: "
        f"-{fall_penalty_max:.0f}*(T-t)/T (im później, tym mniej ujemne). "
        f"Suma epizodu = 0 → cały epizod bez upadku."
    )
    print(
        f"Eksploracja: ep 1–{random_warmup_episodes} tylko losowe akcje; "
        f"potem alpha {alpha_start} → {alpha_end} przez {alpha_decay_episodes} ep."
    )
    print("IMU obs: LSB → acc [g], gyro [°/s] (normalize_raw_imu_obs)")
    if obs_mode == OBS_MODE_PROCESSED4:
        print(
            "UWAGA: processed4 = sieć widzi [pitch, pitch_rate, x, x_dot] z symulacji, "
            "NIE surowe acc/gyro. Dla 2 płytek na górze użyj --obs-mode imu_raw12."
        )
    elif is_raw_imu_mode(obs_mode):
        h = physics.layout.get("body_height_m", 0.14)
        print(
            f"IMU: symulowane acc/gyro (rl/imu_obs.py), montaż na górnej ściance ~{h:.3f} m "
            f"(bez I2C)."
        )
    _print_physics_summary(physics)

    imu_h = float(physics.layout.get("body_height_m", 0.14))
    env = DualActionPendulumEnv(
        fall_angle_deg=train_fall_angle_deg,
        domain_randomization=not no_domain_randomization,
        com_height_m=physics.l_body_m,
        m_nominal=physics.m_axle_kg,
        M_nominal=physics.M_body_kg,
        force_max_nominal=physics.force_max_n,
        dt=dt,
        obs_mode=obs_mode,
        imu_noise_std=imu_noise_std,
        imu_mount_height_m=imu_h,
        imu_normalize_obs=True,
        max_episode_steps=max_steps,
        fall_penalty_max=fall_penalty_max,
        min_motor_power=min_motor_power,
        init_angle_deg=init_angle_deg,
    )
    agent = SACAgent(
        obs_dim=env.obs_dim,
        act_dim=env.act_dim,
        hidden_dims=hidden_dims,
        hidden_activation=hidden_activation,
        gamma=gamma,
        alpha=alpha_start,
        lr=lr_start,
        buffer_size=buffer_size,
        batch_size=batch_size,
        device=device,
    )
    agent.set_learning_rate(lr_start)
    agent.set_alpha(alpha_start)

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
            lr = lr_for_episode(ep, episodes, lr_start, lr_end)
            alpha = alpha_for_episode(
                ep,
                random_warmup_episodes,
                alpha_decay_episodes,
                alpha_start,
                alpha_end,
            )
            agent.set_learning_rate(lr)
            agent.set_alpha(alpha)

            obs = env.reset()
            ep_reward = 0.0
            ep_steps = 0
            use_random = ep <= random_warmup_episodes
            for _ in range(max_steps):
                if use_random:
                    action = np.random.uniform(-1.0, 1.0, size=env.act_dim).astype(np.float32)
                else:
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
            completed_episodes = ep
            rolling = float(np.mean(rewards[-rolling_window:]))
            print(
                f"Ep {ep:04d} | lr {lr:.2e} | alpha {alpha:.3f} | steps {ep_steps:4d} | "
                f"Reward {ep_reward:8.2f} | Avg{rolling_window} {rolling:8.2f}"
                + (" | warmup" if use_random else "")
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="SAC sim: direction×power, sparse reward, gamma=0.999, LR decay."
    )
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument(
        "--save-path",
        type=Path,
        default=Path("artifacts") / "actor_sim_dual.pt",
    )
    parser.add_argument(
        "--plot-path",
        type=Path,
        default=Path("artifacts") / "learning_curve_dual.png",
    )
    parser.add_argument("--rolling-window", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--no-live-plot", action="store_true")
    parser.add_argument("--plot-update-every", type=int, default=5)
    parser.add_argument("--train-fall-angle-deg", type=float, default=30.0)
    parser.add_argument("--dt", type=float, default=0.002)
    parser.add_argument(
        "--obs-mode",
        choices=[OBS_MODE_PROCESSED4, OBS_MODE_IMU_RAW6, OBS_MODE_IMU_RAW12],
        default=OBS_MODE_IMU_RAW12,
        help=(
            "Domyślnie imu_raw12: syntetyczne 2×BMI160 (górna ściana) w rl/imu_obs.py. "
            "processed4 = idealny pitch (bez acc/gyro)."
        ),
    )
    parser.add_argument("--imu-noise-std", type=float, default=25.0)
    parser.add_argument(
        "--fall-penalty-max",
        type=float,
        default=FALL_PENALTY_MAX,
        help="Maks. kara przy upadku na początku epizodu (skalowana (T-t)/T).",
    )
    parser.add_argument(
        "--min-motor-power",
        type=float,
        default=0.2,
        help="Minimalna skala mocy [0,1] z drugiego wyjścia (unika power≈0).",
    )
    parser.add_argument(
        "--init-angle-deg",
        type=float,
        default=10.0,
        help="Losowy start |pitch| ≤ ten kąt [deg] (domyślnie ±10°).",
    )
    parser.add_argument(
        "--random-warmup-episodes",
        type=int,
        default=30,
        help="Pierwsze N epizodów: wyłącznie losowa akcja (bez polityki).",
    )
    parser.add_argument(
        "--alpha-start",
        type=float,
        default=DEFAULT_ALPHA_START,
        help="Entropia SAC po warmupie (domyślnie 0.4).",
    )
    parser.add_argument(
        "--alpha-end",
        type=float,
        default=DEFAULT_ALPHA_END,
        help="Dolna granica entropii SAC po schładzaniu (domyślnie 0.05).",
    )
    parser.add_argument(
        "--alpha-decay-episodes",
        type=int,
        default=DEFAULT_ALPHA_DECAY_EPISODES,
        help="Epizody po warmupie, w których alpha maleje liniowo do alpha-end (domyślnie 170).",
    )
    parser.add_argument("--no-domain-randomization", action="store_true")
    parser.add_argument("--manual-com-height", action="store_true")
    parser.add_argument("--com-height-m", type=float, default=0.11)
    parser.add_argument("--body-height-m", type=float, default=0.14)
    parser.add_argument("--battery-z-m", type=float, default=None)
    parser.add_argument("--case-z-m", type=float, default=None)
    parser.add_argument("--rpi-z-m", type=float, default=None)
    parser.add_argument("--motor-mass-g", type=float, default=160.0)
    parser.add_argument("--n-motors", type=int, default=2)
    parser.add_argument("--rpi-mass-g", type=float, default=55.0)
    parser.add_argument("--case-mass-g", type=float, default=466.0)
    parser.add_argument("--battery-mass-g", type=float, default=250.0)
    parser.add_argument("--wheel-radius-m", type=float, default=0.03)
    parser.add_argument("--motor-torque-nm", type=float, default=0.35)
    parser.add_argument("--n-drive-motors", type=int, default=2)
    parser.add_argument("--force-max", type=float, default=10.0)
    parser.add_argument(
        "--hidden-dims",
        type=int,
        nargs=2,
        default=[32, 16],
        metavar=("H1", "H2"),
        help="Rozmiary warstw ukrytych actor/critic (domyślnie 32 16).",
    )
    parser.add_argument(
        "--hidden-activation",
        type=lambda s: str(s).lower(),
        choices=["tanh", "relu"],
        default=DEFAULT_HIDDEN_ACTIVATION,
        help="Aktywacja w warstwach ukrytych actor/critic: tanh | relu (domyślnie tanh).",
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=DEFAULT_BUFFER_SIZE,
        help="Rozmiar replay buffer (domyślnie 50000).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size SAC (domyślnie 64).",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=DEFAULT_GAMMA,
        help="Współczynnik dyskontowania (domyślnie 0.999).",
    )
    parser.add_argument(
        "--lr-start",
        type=float,
        default=DEFAULT_LR_START,
        help="Learning rate na początku (domyślnie 0.01 = 1/100).",
    )
    parser.add_argument(
        "--lr-end",
        type=float,
        default=DEFAULT_LR_END,
        help="Learning rate na ostatnim epizodzie (domyślnie 0.001 = 1/1000).",
    )
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"])
    run_group = parser.add_mutually_exclusive_group()
    run_group.add_argument("--run-dir", type=Path, default=None)
    run_group.add_argument("--run-name", type=str, default=None)
    run_group.add_argument("--auto-run-name", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.com_from_masses = not args.manual_com_height
    physics = resolve_train_physics(args)
    args.com_height_m = physics.l_body_m

    save_path, plot_path, run_dir = resolve_run_paths(args)

    if run_dir is not None:
        save_run_config(
            run_dir,
            {
                "trainer": "train_sim_dual",
                "action_layout": "direction[-1,1] * power_scale[0,1]",
                "reward": "sparse_time_scaled_fall",
                "alive_reward_per_step": 0.0,
                "fall_penalty_max": args.fall_penalty_max,
                "fall_formula": "-fall_penalty_max * (max_steps - step) / max_steps on fall; else 0",
                "episode_reward_zero_means_mastered": True,
                "min_motor_power": args.min_motor_power,
                "init_angle_deg": args.init_angle_deg,
                "imu_normalize_obs": True,
                "random_warmup_episodes": args.random_warmup_episodes,
                "alpha_start": args.alpha_start,
                "alpha_end": args.alpha_end,
                "alpha_decay_episodes": args.alpha_decay_episodes,
                "alpha_schedule": "after warmup, linear alpha_start → alpha_end over alpha_decay_episodes",
                "started_at_utc": datetime.now(timezone.utc).isoformat(),
                "run_dir": str(run_dir),
                "episodes": args.episodes,
                "max_steps": args.max_steps,
                "dt": args.dt,
                "obs_mode": args.obs_mode,
                "imu_mount_height_m": float(physics.layout.get("body_height_m", 0.14)),
                "imu_simulation": "rl/imu_obs.simulate_dual_imu_raw_reading",
                "act_dim": 2,
                "gamma": args.gamma,
                "buffer_size": args.buffer_size,
                "batch_size": args.batch_size,
                "lr_start": args.lr_start,
                "lr_end": args.lr_end,
                "lr_schedule": "linear: lr_start → lr_end over episodes",
                "hidden_dims": list(args.hidden_dims),
                "hidden_activation": args.hidden_activation,
                "com_height_m": physics.l_body_m,
                "force_max_n": physics.force_max_n,
                "train_fall_angle_deg": args.train_fall_angle_deg,
                "domain_randomization": not args.no_domain_randomization,
                "save_path": str(save_path),
                "plot_path": str(plot_path),
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
        list(args.hidden_dims),
        args.hidden_activation,
        args.buffer_size,
        args.batch_size,
        args.lr_start,
        args.lr_end,
        args.gamma,
        args.dt,
        args.obs_mode,
        args.imu_noise_std,
        args.fall_penalty_max,
        args.min_motor_power,
        args.init_angle_deg,
        args.random_warmup_episodes,
        args.alpha_start,
        args.alpha_end,
        args.alpha_decay_episodes,
        args.device,
        run_dir=run_dir,
    )
