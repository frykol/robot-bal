"""
Trening PPO w symulacji.

Przykłady:
  python train_sim_ppo.py --run-name ppo_baseline --episodes 500 --rollout-steps 2048
  python train_sim_ppo.py --dual-action --run-name ppo_cmp_v6 --obs-mode imu_raw12 \\
    --hidden-dims 48 24 --episodes 800
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from rl.envs import InvertedPendulumEnv
from rl.envs_dual import DualActionPendulumEnv, FALL_PENALTY_MAX
from rl.imu_obs import (
    OBS_MODE_IMU_RAW12,
    OBS_MODE_IMU_RAW6,
    OBS_MODE_PROCESSED4,
    is_raw_imu_mode,
    obs_dim_for_mode,
)
from rl.ppo import DEFAULT_HIDDEN_DIM, DEFAULT_LR, PPOAgent
from rl.robot_mass_model import resolve_train_physics
from train_sim import (
    DEFAULT_PLOT_PATH,
    DEFAULT_SAVE_PATH,
    LivePlotter,
    _print_physics_summary,
    coalesce_plot_path,
    compute_rolling_average,
)

DEFAULT_RUNS_ROOT = Path("artifacts") / "runs"
DEFAULT_PPO_SAVE_PATH = Path("artifacts") / "actor_sim_ppo.pt"
DEFAULT_PPO_PLOT_PATH = Path("artifacts") / "learning_curve_ppo.png"


def make_auto_run_name(hidden_dims, com_height_m, train_fall_angle_deg, episodes, lr):
    htag = "_".join(str(h) for h in hidden_dims)
    return (
        f"ppo_h{htag}_com{com_height_m:.3f}_fall{int(train_fall_angle_deg)}"
        f"_lr{lr:.0e}_ep{episodes}"
    ).replace("e-0", "e-")


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
            args.lr,
        )
        run_dir = DEFAULT_RUNS_ROOT / slug
    else:
        return args.save_path, args.plot_path, None

    run_dir.mkdir(parents=True, exist_ok=True)
    save_path = run_dir / "actor_sim_ppo.pt"
    if args.save_path != DEFAULT_PPO_SAVE_PATH:
        save_path = Path(args.save_path)
    plot_path = coalesce_plot_path(save_path, args.plot_path, DEFAULT_PPO_PLOT_PATH)
    return save_path, plot_path, run_dir


def save_run_config(run_dir, config):
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "run_config.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, sort_keys=True)
    print(f"Saved run config: {config_path}")


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
        agent.save_checkpoint(str(checkpoint_path))
        print(f"Saved episode checkpoint: {checkpoint_path}")
    save_learning_plot(rewards, plot_path, rolling_window, episode_steps=episode_steps)


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
    plt.title(f"PPO Learning Curve — {run_label}")
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
        plt.title(f"PPO episode length — {run_label}")
        plt.xlabel("Episode")
        plt.ylabel("Steps")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(steps_path, dpi=150)
        plt.close()
        print(f"Saved steps curve: {steps_path.resolve()}")


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
    hidden_dims,
    hidden_activation,
    lr,
    dt,
    obs_mode,
    imu_noise_std,
    rollout_steps,
    ppo_epochs,
    clip_eps,
    gae_lambda,
    entropy_coef,
    value_coef,
    minibatch_size,
    gamma,
    dual_action,
    fall_penalty_max,
    min_motor_power,
    init_angle_deg,
    init_angle_easy_deg,
    curriculum_episodes,
    alive_reward_per_step,
    angle_reward_scale,
    angular_rate_reward_scale,
    random_warmup_episodes,
    action_delay_steps=0,
    max_force_delta_per_step=None,
    static_friction_force_n=0.0,
    device=None,
    run_dir=None,
):
    if run_dir is not None:
        print(f"Run output directory: {run_dir.resolve()}")
    act_dim = 2 if dual_action else 1
    print(
        f"PPO training: lr={lr}, gamma={gamma}, hidden_dims={hidden_dims}, device={device}, "
        f"obs_mode={obs_mode} (obs_dim={obs_dim_for_mode(obs_mode)}), act_dim={act_dim}, "
        f"rollout_steps={rollout_steps}"
    )
    if dual_action:
        print(
            f"Reward (dual, jak SAC v6): +{alive_reward_per_step}/krok + shaping; upadek: "
            f"-{fall_penalty_max:.0f}*(T-t)/T."
        )
        print(
            f"Curriculum kąta startu: ±{init_angle_easy_deg}° → ±{init_angle_deg}° "
            f"przez {curriculum_episodes} ep."
        )
        if random_warmup_episodes > 0:
            print(f"Warmup: ep 1–{random_warmup_episodes} losowe akcje.")
    if is_raw_imu_mode(obs_mode):
        print("IMU: symulowane odczyty BMI160 w symulacji (bez I2C).")
    _print_physics_summary(physics)

    imu_h = float(physics.layout.get("body_height_m", 0.14))
    env_kw = dict(
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
        action_delay_steps=action_delay_steps,
        max_force_delta_per_step=max_force_delta_per_step,
        static_friction_force_n=static_friction_force_n,
    )
    if dual_action:
        env = DualActionPendulumEnv(
            max_episode_steps=max_steps,
            fall_penalty_max=fall_penalty_max,
            min_motor_power=min_motor_power,
            init_angle_deg=init_angle_deg,
            init_angle_easy_deg=init_angle_easy_deg,
            curriculum_episodes=curriculum_episodes,
            alive_reward_per_step=alive_reward_per_step,
            angle_reward_scale=angle_reward_scale,
            angular_rate_reward_scale=angular_rate_reward_scale,
            imu_normalize_obs=True,
            **env_kw,
        )
    else:
        env = InvertedPendulumEnv(**env_kw)

    agent = PPOAgent(
        obs_dim=env.obs_dim,
        act_dim=env.act_dim,
        hidden_dim=hidden_dim,
        hidden_dims=hidden_dims,
        hidden_activation=hidden_activation,
        lr=lr,
        gamma=gamma,
        clip_eps=clip_eps,
        gae_lambda=gae_lambda,
        entropy_coef=entropy_coef,
        value_coef=value_coef,
        ppo_epochs=ppo_epochs,
        minibatch_size=minibatch_size,
        device=device,
    )

    rewards = []
    episode_steps = []
    best_avg = -np.inf
    best_steps_avg = -1.0
    best_path = save_path.with_name("actor_best_ppo.pt")
    checkpoint_dir = save_path.parent / "checkpoints"
    live_plot = LivePlotter(
        rolling_window=rolling_window,
        enabled=live_plot_enabled,
        update_every=plot_update_every,
    )
    if live_plot.enabled and live_plot._ax is not None:
        live_plot._ax.set_title("PPO Learning Curve (Live)")

    steps_in_rollout = 0
    completed_episodes = 0

    try:
        for ep in range(1, episodes + 1):
            if dual_action:
                env.set_curriculum_episode(ep, episodes)
            use_random = dual_action and ep <= random_warmup_episodes
            obs = env.reset()
            ep_reward = 0.0
            ep_steps = 0

            for _ in range(max_steps):
                if use_random:
                    action = np.random.uniform(-1.0, 1.0, size=env.act_dim).astype(np.float32)
                else:
                    action, logp, value = agent.act(obs, deterministic=False)
                next_obs, reward, done = env.step(action)
                if not use_random:
                    agent.remember(obs, action, reward, done, logp, value)
                    steps_in_rollout += 1
                obs = next_obs
                ep_reward += reward
                ep_steps += 1

                if (not use_random) and (done or steps_in_rollout >= rollout_steps):
                    stats = agent.finish_rollout(obs, done=done)
                    steps_in_rollout = 0
                    if stats is not None:
                        print(
                            f"  PPO update | policy {stats['policy_loss']:.4f} "
                            f"| value {stats['value_loss']:.4f} "
                            f"| entropy {stats['entropy']:.4f}"
                        )

                if done:
                    break

            rewards.append(ep_reward)
            episode_steps.append(ep_steps)
            completed_episodes = ep
            rolling = float(np.mean(rewards[-rolling_window:]))
            rolling_steps = float(np.mean(episode_steps[-rolling_window:]))
            init_suffix = ""
            if dual_action:
                init_suffix = f" | init±{env._init_angle_deg:.1f}°"
            warmup_suffix = " | warmup" if use_random else ""
            print(
                f"Ep {ep:04d}{init_suffix} | steps {ep_steps:4d} | Reward {ep_reward:8.2f} | "
                f"Avg{rolling_window} R {rolling:8.2f} S {rolling_steps:6.0f}{warmup_suffix}"
            )
            live_plot.update(rewards, force=(ep == 1))

            if rolling > best_avg:
                best_avg = rolling
            if (not use_random) and rolling_steps > best_steps_avg:
                best_steps_avg = rolling_steps
                best_path.parent.mkdir(parents=True, exist_ok=True)
                agent.save_actor(str(best_path))
                print(
                    f"New best checkpoint: {best_path} "
                    f"(Avg{rolling_window} steps={best_steps_avg:.0f})"
                )

            if save_every > 0 and ep % save_every == 0:
                checkpoint_path = checkpoint_dir / f"ppo_ep{ep:04d}.pt"
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

        if steps_in_rollout > 0:
            agent.finish_rollout(obs, done=False)

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


def parse_args():
    parser = argparse.ArgumentParser(
        description="PPO training in simulation (scalar or dual-action env)."
    )
    parser.add_argument(
        "--dual-action",
        action="store_true",
        help="DualActionPendulumEnv: ta sama nagroda i akcja 2D co SAC (train_sim_dual).",
    )
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--save-path", type=Path, default=DEFAULT_PPO_SAVE_PATH)
    parser.add_argument("--plot-path", type=Path, default=DEFAULT_PPO_PLOT_PATH)
    parser.add_argument("--rolling-window", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--no-live-plot", action="store_true")
    parser.add_argument("--plot-update-every", type=int, default=5)
    parser.add_argument("--train-fall-angle-deg", type=float, default=30.0)
    parser.add_argument("--dt", type=float, default=0.002)
    parser.add_argument(
        "--obs-mode",
        choices=[OBS_MODE_PROCESSED4, OBS_MODE_IMU_RAW6, OBS_MODE_IMU_RAW12],
        default=OBS_MODE_PROCESSED4,
    )
    parser.add_argument("--imu-noise-std", type=float, default=25.0)
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
    parser.add_argument("--hidden-dim", type=int, default=DEFAULT_HIDDEN_DIM)
    parser.add_argument(
        "--hidden-dims",
        type=int,
        nargs="+",
        default=None,
        help="Rozmiary warstw ukrytych, np. --hidden-dims 48 24 (nadpisuje --hidden-dim).",
    )
    parser.add_argument(
        "--hidden-activation",
        choices=["relu", "tanh"],
        default="relu",
    )
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"])
    parser.add_argument("--rollout-steps", type=int, default=2048)
    parser.add_argument("--ppo-epochs", type=int, default=10)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--minibatch-size", type=int, default=64)
    parser.add_argument(
        "--gamma",
        type=float,
        default=None,
        help="Dyskontowanie (domyślnie 0.999 przy --dual-action, inaczej 0.99).",
    )
    parser.add_argument(
        "--fall-penalty-max",
        type=float,
        default=FALL_PENALTY_MAX,
        help="Maks. kara przy upadku (dual env, skalowana (T-t)/T).",
    )
    parser.add_argument(
        "--min-motor-power",
        type=float,
        default=0.2,
        help="Minimalna skala mocy [0,1] z drugiego wyjścia akcji (dual env).",
    )
    parser.add_argument(
        "--init-angle-deg",
        type=float,
        default=8.0,
        help="Końcowy zakres losowego startu |pitch| [deg] (dual + curriculum).",
    )
    parser.add_argument(
        "--init-angle-easy-deg",
        type=float,
        default=3.0,
        help="Początkowy zakres |pitch| przy ep. 1 [deg] (dual + curriculum).",
    )
    parser.add_argument(
        "--curriculum-episodes",
        type=int,
        default=600,
        help="Epizody, w których init_angle rośnie liniowo z easy do final (dual env).",
    )
    parser.add_argument(
        "--alive-reward-per-step",
        type=float,
        default=0.02,
        help="Stała nagroda za każdy krok bez upadku (dual env).",
    )
    parser.add_argument(
        "--angle-reward-scale",
        type=float,
        default=0.03,
        help="Kara za |theta|/theta_max na żywym kroku (dual env).",
    )
    parser.add_argument(
        "--angular-rate-reward-scale",
        type=float,
        default=0.02,
        help="Kara za |theta_dot| (dual env).",
    )
    parser.add_argument(
        "--random-warmup-episodes",
        type=int,
        default=30,
        help="Pierwsze N epizodów dual: losowe akcje bez aktualizacji PPO (jak SAC).",
    )
    parser.add_argument("--action-delay-steps", type=int, default=0)
    parser.add_argument("--max-force-delta-frac", type=float, default=0.0)
    parser.add_argument("--static-friction-frac", type=float, default=0.0)
    run_group = parser.add_mutually_exclusive_group()
    run_group.add_argument("--run-dir", type=Path, default=None)
    run_group.add_argument("--run-name", type=str, default=None)
    run_group.add_argument("--auto-run-name", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.gamma is None:
        args.gamma = 0.999 if args.dual_action else 0.99

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
        config = {
            "trainer": "train_sim_ppo",
            "algorithm": "ppo",
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "run_dir": str(run_dir),
            "dual_action": args.dual_action,
            "act_dim": 2 if args.dual_action else 1,
            "episodes": args.episodes,
            "max_steps": args.max_steps,
            "rollout_steps": args.rollout_steps,
            "ppo_epochs": args.ppo_epochs,
            "clip_eps": args.clip_eps,
            "gae_lambda": args.gae_lambda,
            "entropy_coef": args.entropy_coef,
            "value_coef": args.value_coef,
            "minibatch_size": args.minibatch_size,
            "gamma": args.gamma,
            "dt": args.dt,
            "obs_mode": args.obs_mode,
            "hidden_dim": args.hidden_dim,
            "hidden_dims": list(args.hidden_dims) if args.hidden_dims else [args.hidden_dim, args.hidden_dim],
            "hidden_activation": args.hidden_activation,
            "lr": args.lr,
            "physics_layout": physics.layout,
            "com_height_m": physics.l_body_m,
            "force_max_n": physics.force_max_n,
            "train_fall_angle_deg": args.train_fall_angle_deg,
            "domain_randomization": not args.no_domain_randomization,
            "save_path": str(save_path),
            "plot_path": str(plot_path),
            "best_checkpoint_metric": "rolling_mean_episode_steps",
        }
        if args.dual_action:
            config.update(
                {
                    "action_layout": "direction[-1,1] * power_scale[0,1]",
                    "reward": "alive_plus_shaping_plus_time_scaled_fall",
                    "alive_reward_per_step": args.alive_reward_per_step,
                    "angle_reward_scale": args.angle_reward_scale,
                    "angular_rate_reward_scale": args.angular_rate_reward_scale,
                    "fall_penalty_max": args.fall_penalty_max,
                    "fall_formula": "-fall_penalty_max * (max_steps - step) / max_steps on fall",
                    "min_motor_power": args.min_motor_power,
                    "init_angle_deg": args.init_angle_deg,
                    "init_angle_easy_deg": args.init_angle_easy_deg,
                    "curriculum_episodes": args.curriculum_episodes,
                    "random_warmup_episodes": args.random_warmup_episodes,
                    "imu_normalize_obs": True,
                    "imu_mount_height_m": float(physics.layout.get("body_height_m", 0.14)),
                    "imu_simulation": "rl/imu_obs.simulate_dual_imu_raw_reading",
                }
            )
        else:
            config["reward"] = "scalar_angle_center_fall_minus_100"
        save_run_config(run_dir, config)

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
        args.hidden_dims if args.hidden_dims else [args.hidden_dim, args.hidden_dim],
        args.hidden_activation,
        args.lr,
        args.dt,
        args.obs_mode,
        args.imu_noise_std,
        args.rollout_steps,
        args.ppo_epochs,
        args.clip_eps,
        args.gae_lambda,
        args.entropy_coef,
        args.value_coef,
        args.minibatch_size,
        args.gamma,
        args.dual_action,
        args.fall_penalty_max,
        args.min_motor_power,
        args.init_angle_deg,
        args.init_angle_easy_deg,
        args.curriculum_episodes,
        args.alive_reward_per_step,
        args.angle_reward_scale,
        args.angular_rate_reward_scale,
        args.random_warmup_episodes,
        args.action_delay_steps,
        max_force_delta,
        static_friction,
        args.device,
        run_dir=run_dir,
    )
