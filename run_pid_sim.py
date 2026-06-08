"""
Ewaluacja regulatora PID w symulacji (InvertedPendulumEnv).

Wyniki zapisywane jak w train_sim / train_sim_ppo:
  artifacts/runs/<run_name>/learning_curve.png
  artifacts/runs/<run_name>/learning_curve_steps.png
  artifacts/runs/<run_name>/run_config.json

Przykłady:
  python run_pid_sim.py --run-name pid_baseline --episodes 100
  python run_pid_sim.py --kp 120 --ki 4 --kd 6 --run-name pid_k120 --live-plot
  python run_pid_sim.py --obs-mode imu_raw6 --kp 150 --kd 8 --episodes 50
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from rl.envs import InvertedPendulumEnv
from rl.imu_obs import (
    OBS_MODE_IMU_RAW12,
    OBS_MODE_IMU_RAW6,
    OBS_MODE_PROCESSED4,
    is_raw_imu_mode,
    obs_dim_for_mode,
)
from rl.pid import BalancePIDController
from rl.robot_mass_model import resolve_train_physics
from train_sim import (
    DEFAULT_RUNS_ROOT,
    LivePlotter,
    _print_physics_summary,
    coalesce_plot_path,
    save_learning_plot,
    save_run_config,
)

DEFAULT_PLOT_PATH = Path("artifacts") / "learning_curve_pid.png"


def make_auto_run_name(kp, ki, kd, obs_mode, episodes):
    obs_tag = obs_mode.replace("_", "")
    return f"pid_kp{kp:g}_ki{ki:g}_kd{kd:g}_{obs_tag}_ep{episodes}"


def resolve_run_paths(args):
    if args.run_dir is not None:
        run_dir = Path(args.run_dir)
    elif args.run_name is not None:
        run_dir = DEFAULT_RUNS_ROOT / args.run_name
    elif args.auto_run_name:
        slug = make_auto_run_name(args.kp, args.ki, args.kd, args.obs_mode, args.episodes)
        run_dir = DEFAULT_RUNS_ROOT / slug
    else:
        return args.plot_path, None

    run_dir.mkdir(parents=True, exist_ok=True)
    plot_path = coalesce_plot_path(run_dir / "pid_eval.json", args.plot_path, DEFAULT_PLOT_PATH)
    return plot_path, run_dir


def evaluate_pid_trials(
    env,
    controller,
    *,
    episodes,
    max_steps,
    oracle_state=False,
    verbose=False,
    label=None,
):
    """Uruchom kilka epizodów i zwróć metryki (bez wykresów)."""
    rewards = []
    episode_steps = []

    for ep in range(1, episodes + 1):
        obs = env.reset()
        controller.reset()
        ep_reward = 0.0
        ep_steps = 0

        for _ in range(max_steps):
            state_arg = env.state if oracle_state else None
            action = controller.act(obs, env.force_max, env_state=state_arg)
            obs, reward, done = env.step(action)
            ep_reward += reward
            ep_steps += 1
            if done:
                break

        rewards.append(ep_reward)
        episode_steps.append(ep_steps)
        if verbose:
            prefix = f"{label} | " if label else ""
            print(f"{prefix}Ep {ep:04d} | Reward {ep_reward:8.2f} | Steps {ep_steps:5d}")

    steps_arr = np.array(episode_steps, dtype=np.float64)
    rewards_arr = np.array(rewards, dtype=np.float64)
    return {
        "episodes": int(episodes),
        "mean_reward": float(np.mean(rewards_arr)),
        "std_reward": float(np.std(rewards_arr)),
        "mean_steps": float(np.mean(steps_arr)),
        "std_steps": float(np.std(steps_arr)),
        "survival_rate": float(np.mean(steps_arr >= max_steps)),
        "rewards": rewards,
        "episode_steps": episode_steps,
    }


def make_env(args, physics):
    return InvertedPendulumEnv(
        fall_angle_deg=args.train_fall_angle_deg,
        domain_randomization=not args.no_domain_randomization,
        com_height_m=physics.l_body_m,
        m_nominal=physics.m_axle_kg,
        M_nominal=physics.M_body_kg,
        force_max_nominal=physics.force_max_n,
        dt=args.dt,
        obs_mode=args.obs_mode,
        imu_noise_std=args.imu_noise_std,
        imu_mount_height_m=float(physics.layout.get("body_height_m", 0.14)),
        action_delay_steps=args.action_delay_steps,
        max_force_delta_per_step=(
            None
            if args.max_force_delta_frac <= 0
            else args.max_force_delta_frac * physics.force_max_n
        ),
        static_friction_force_n=args.static_friction_frac * physics.force_max_n,
    )


def run_evaluation(args, physics, plot_path, run_dir=None):
    if run_dir is not None:
        print(f"Run output directory: {run_dir.resolve()}")
    print(
        f"PID eval: Kp={args.kp}, Ki={args.ki}, Kd={args.kd}, "
        f"obs_mode={args.obs_mode} (obs_dim={obs_dim_for_mode(args.obs_mode)})"
    )
    if is_raw_imu_mode(args.obs_mode):
        print("IMU: pitch/rate z syntetycznych odczytów BMI160 (jak train_sim).")
    if args.oracle_state:
        print("Oracle: kąt/pozycja z env.state (tylko processed4).")
    _print_physics_summary(physics)

    env = make_env(args, physics)
    controller = make_pid_controller(args)

    rewards = []
    episode_steps = []
    live_plot = LivePlotter(
        rolling_window=args.rolling_window,
        enabled=args.live_plot,
        update_every=args.plot_update_every,
        title="PID Learning Curve (Live)",
    )

    oracle_state = args.oracle_state and args.obs_mode == OBS_MODE_PROCESSED4
    completed = 0
    try:
        for ep in range(1, args.episodes + 1):
            obs = env.reset()
            controller.reset()
            ep_reward = 0.0
            ep_steps = 0

            for _ in range(args.max_steps):
                state_arg = env.state if oracle_state else None
                action = controller.act(obs, env.force_max, env_state=state_arg)
                obs, reward, done = env.step(action)
                ep_reward += reward
                ep_steps += 1
                if done:
                    break

            rewards.append(ep_reward)
            episode_steps.append(ep_steps)
            completed = ep
            rolling_r = float(np.mean(rewards[-args.rolling_window :]))
            rolling_s = float(np.mean(episode_steps[-args.rolling_window :]))
            print(
                f"Ep {ep:04d} | Reward {ep_reward:8.2f} | "
                f"Avg{args.rolling_window} R {rolling_r:8.2f} S {rolling_s:6.0f}"
            )
            live_plot.update(rewards, force=(ep == 1))
    except KeyboardInterrupt:
        print("\nEvaluation interrupted (Ctrl+C). Saving plots...")
    finally:
        live_plot.close()
        save_learning_plot(
            rewards,
            plot_path,
            args.rolling_window,
            episode_steps=episode_steps,
            algorithm="PID",
        )
        print(f"Completed {completed} episodes.")
        if rewards:
            print(
                f"Summary: mean reward {np.mean(rewards):.2f}, "
                f"mean steps {np.mean(episode_steps):.0f}, "
                f"survival {100.0 * np.mean(np.array(episode_steps) >= args.max_steps):.1f}%"
            )


def make_pid_controller(args):
    return BalancePIDController(
        kp=args.kp,
        ki=args.ki,
        kd=args.kd,
        kp_x=args.kp_x,
        ki_x=args.ki_x,
        kd_x=args.kd_x,
        dt=args.dt,
        integral_limit=args.integral_limit,
        obs_mode=args.obs_mode,
        imu_index=args.imu_index,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="PID evaluation in simulation")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument(
        "--plot-path",
        type=Path,
        default=DEFAULT_PLOT_PATH,
        help="Domyślnie: <run_dir>/learning_curve.png przy --run-name.",
    )
    parser.add_argument("--rolling-window", type=int, default=20)
    parser.add_argument(
        "--live-plot",
        action="store_true",
        help="Okno z wykresem na żywo podczas ewaluacji.",
    )
    parser.add_argument("--plot-update-every", type=int, default=5)
    parser.add_argument("--train-fall-angle-deg", type=float, default=30.0)
    parser.add_argument("--dt", type=float, default=0.002)
    parser.add_argument(
        "--obs-mode",
        choices=[OBS_MODE_PROCESSED4, OBS_MODE_IMU_RAW6, OBS_MODE_IMU_RAW12],
        default=OBS_MODE_PROCESSED4,
    )
    parser.add_argument("--imu-noise-std", type=float, default=25.0)
    parser.add_argument("--imu-index", type=int, default=0, help="Które IMU w imu_raw12.")
    parser.add_argument(
        "--no-domain-randomization",
        action="store_true",
        help="Wyłącz randomizację masy/szumu (łatwiejsze strojenie).",
    )
    parser.add_argument(
        "--oracle-state",
        action="store_true",
        help="Użyj env.state zamiast obs (tylko processed4; idealny pomiar).",
    )
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
    parser.add_argument("--action-delay-steps", type=int, default=0)
    parser.add_argument("--max-force-delta-frac", type=float, default=0.0)
    parser.add_argument("--static-friction-frac", type=float, default=0.0)

    pid = parser.add_argument_group("PID gains")
    pid.add_argument("--kp", type=float, default=50.0)
    pid.add_argument("--ki", type=float, default=0.0)
    pid.add_argument("--kd", type=float, default=0.0)
    pid.add_argument("--kp-x", type=float, default=0.0, help="Pozycja x [opcjonalnie].")
    pid.add_argument("--ki-x", type=float, default=0.0)
    pid.add_argument("--kd-x", type=float, default=0.0)
    pid.add_argument(
        "--integral-limit",
        type=float,
        default=5.0,
        help="Anti-windup: max |całka|; ujemne = bez limitu.",
    )

    run_group = parser.add_mutually_exclusive_group()
    run_group.add_argument("--run-dir", type=Path, default=None)
    run_group.add_argument(
        "--run-name",
        type=str,
        default=None,
        help=f"Zapis do {DEFAULT_RUNS_ROOT}/<nazwa>/ (wykresy + run_config.json).",
    )
    run_group.add_argument("--auto-run-name", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.integral_limit < 0:
        args.integral_limit = None
    args.com_from_masses = not args.manual_com_height
    physics = resolve_train_physics(args)
    args.com_height_m = physics.l_body_m

    plot_path, run_dir = resolve_run_paths(args)
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
                "algorithm": "pid",
                "started_at_utc": datetime.now(timezone.utc).isoformat(),
                "run_dir": str(run_dir),
                "episodes": args.episodes,
                "max_steps": args.max_steps,
                "dt": args.dt,
                "obs_mode": args.obs_mode,
                "imu_noise_std": args.imu_noise_std,
                "imu_index": args.imu_index,
                "rolling_window": args.rolling_window,
                "oracle_state": args.oracle_state,
                "kp": args.kp,
                "ki": args.ki,
                "kd": args.kd,
                "kp_x": args.kp_x,
                "ki_x": args.ki_x,
                "kd_x": args.kd_x,
                "integral_limit": args.integral_limit,
                "com_height_m": physics.l_body_m,
                "z_com_full_m": physics.z_com_full_m,
                "m_axle_kg": physics.m_axle_kg,
                "M_body_kg": physics.M_body_kg,
                "force_max_n": physics.force_max_n,
                "physics_layout": physics.layout,
                "train_fall_angle_deg": args.train_fall_angle_deg,
                "domain_randomization": not args.no_domain_randomization,
                "plot_path": str(plot_path),
                "action_delay_steps": args.action_delay_steps,
                "max_force_delta_per_step": max_force_delta,
                "static_friction_force_n": static_friction,
            },
        )

    run_evaluation(args, physics, plot_path, run_dir=run_dir)
