"""
Automatyczne strojenie PID — grid search po Kp, Ki, Kd w symulacji.

Wyniki w artifacts/runs/<run_name>/:
  tune_results.csv      — wszystkie kombinacje posortowane
  tune_summary.json     — TOP-N + najlepsze gainy
  run_config.json
  learning_curve.png    — ewaluacja najlepszego zestawu (dłuższa)
  learning_curve_steps.png

Przykłady:
  python tune_pid_sim.py
  python tune_pid_sim.py --run-name pid_tune_v1 --trials-episodes 30
  python tune_pid_sim.py --kp-range 30,40,50,60 --ki-range 0,1,2 --kd-range 0,2,4
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np

from rl.imu_obs import OBS_MODE_IMU_RAW12, OBS_MODE_IMU_RAW6, OBS_MODE_PROCESSED4
from rl.robot_mass_model import resolve_train_physics
from run_pid_sim import (
    DEFAULT_RUNS_ROOT,
    evaluate_pid_trials,
    make_env,
    make_pid_controller,
    save_learning_plot,
    save_run_config,
)
from train_sim import _print_physics_summary

DEFAULT_RUN_NAME = "pid_tune"


def parse_float_list(text, name):
    values = [float(x.strip()) for x in str(text).split(",") if x.strip()]
    if not values:
        raise ValueError(f"{name} must contain at least one value.")
    return values


def gains_slug(kp, ki, kd):
    return f"kp{kp:g}_ki{ki:g}_kd{kd:g}"


def rank_key(row):
    return (row["mean_steps"], row["mean_reward"], row["survival_rate"])


def run_grid_search(env, gain_grid, args, oracle_state):
    total = len(gain_grid)
    results = []

    for idx, (kp, ki, kd) in enumerate(gain_grid, start=1):
        trial_args = SimpleNamespace(**vars(args))
        trial_args.kp = kp
        trial_args.ki = ki
        trial_args.kd = kd

        controller = make_pid_controller(trial_args)
        metrics = evaluate_pid_trials(
            env,
            controller,
            episodes=args.trials_episodes,
            max_steps=args.max_steps,
            oracle_state=oracle_state,
        )
        row = {
            "kp": kp,
            "ki": ki,
            "kd": kd,
            "mean_steps": metrics["mean_steps"],
            "std_steps": metrics["std_steps"],
            "mean_reward": metrics["mean_reward"],
            "std_reward": metrics["std_reward"],
            "survival_rate": metrics["survival_rate"],
        }
        results.append(row)
        print(
            f"[{idx:4d}/{total}] Kp={kp:g} Ki={ki:g} Kd={kd:g} | "
            f"steps {metrics['mean_steps']:6.0f} ± {metrics['std_steps']:5.0f} | "
            f"reward {metrics['mean_reward']:7.1f} | "
            f"survival {100.0 * metrics['survival_rate']:5.1f}%"
        )

    results.sort(key=rank_key, reverse=True)
    return results


def save_tune_csv(path, results):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "rank",
        "kp",
        "ki",
        "kd",
        "mean_steps",
        "std_steps",
        "mean_reward",
        "std_reward",
        "survival_rate",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rank, row in enumerate(results, start=1):
            writer.writerow({"rank": rank, **row})
    print(f"Saved tune results: {path.resolve()}")


def save_heatmap(path, results, kp_values, ki_values, kd_values, kd_slice):
    """Heatmapa mean_steps dla wybranego Kd (najbliższego kd_slice)."""
    kd_pick = min(kd_values, key=lambda k: abs(k - kd_slice))
    lookup = {
        (r["kp"], r["ki"], r["kd"]): r["mean_steps"]
        for r in results
    }
    grid = np.full((len(ki_values), len(kp_values)), np.nan, dtype=np.float64)
    for i, ki in enumerate(ki_values):
        for j, kp in enumerate(kp_values):
            grid[i, j] = lookup.get((kp, ki, kd_pick), np.nan)

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(grid, aspect="auto", origin="lower", cmap="viridis")
    ax.set_xticks(range(len(kp_values)))
    ax.set_xticklabels([f"{v:g}" for v in kp_values])
    ax.set_yticks(range(len(ki_values)))
    ax.set_yticklabels([f"{v:g}" for v in ki_values])
    ax.set_xlabel("Kp")
    ax.set_ylabel("Ki")
    ax.set_title(f"PID tune — mean steps (Kd={kd_pick:g})")
    fig.colorbar(im, ax=ax, label="mean steps")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved heatmap: {path.resolve()}")


def evaluate_best(args, physics, run_dir, best_row, oracle_state):
    best_args = SimpleNamespace(**vars(args))
    best_args.kp = best_row["kp"]
    best_args.ki = best_row["ki"]
    best_args.kd = best_row["kd"]

    env = make_env(best_args, physics)
    controller = make_pid_controller(best_args)
    print(
        f"\nBest gains: Kp={best_args.kp:g} Ki={best_args.ki:g} Kd={best_args.kd:g} "
        f"— final eval ({args.final_episodes} ep.)"
    )
    metrics = evaluate_pid_trials(
        env,
        controller,
        episodes=args.final_episodes,
        max_steps=args.max_steps,
        oracle_state=oracle_state,
        verbose=False,
    )
    plot_path = run_dir / "learning_curve.png"
    save_learning_plot(
        metrics["rewards"],
        plot_path,
        args.rolling_window,
        episode_steps=metrics["episode_steps"],
        algorithm="PID (best)",
    )
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Automatic PID grid search in simulation")
    parser.add_argument(
        "--run-name",
        type=str,
        default=DEFAULT_RUN_NAME,
        help=f"Katalog wyników: {DEFAULT_RUNS_ROOT}/<nazwa>/",
    )
    parser.add_argument(
        "--trials-episodes",
        type=int,
        default=20,
        help="Epizodów na jedną kombinację gainów (szybki ranking).",
    )
    parser.add_argument(
        "--final-episodes",
        type=int,
        default=100,
        help="Epizodów dla najlepszego zestawu + wykresy.",
    )
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--rolling-window", type=int, default=20)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--train-fall-angle-deg", type=float, default=30.0)
    parser.add_argument("--dt", type=float, default=0.002)
    parser.add_argument(
        "--obs-mode",
        choices=[OBS_MODE_PROCESSED4, OBS_MODE_IMU_RAW6, OBS_MODE_IMU_RAW12],
        default=OBS_MODE_PROCESSED4,
    )
    parser.add_argument("--imu-noise-std", type=float, default=25.0)
    parser.add_argument("--imu-index", type=int, default=0)
    parser.add_argument("--no-domain-randomization", action="store_true")
    parser.add_argument("--oracle-state", action="store_true")
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
    parser.add_argument("--integral-limit", type=float, default=5.0)
    parser.add_argument("--kp-x", type=float, default=0.0)
    parser.add_argument("--ki-x", type=float, default=0.0)
    parser.add_argument("--kd-x", type=float, default=0.0)

    grid = parser.add_argument_group("grid ranges (comma-separated)")
    grid.add_argument("--kp-range", type=str, default="20,30,40,50,60,80,100")
    grid.add_argument("--ki-range", type=str, default="0,0.5,1,2,4")
    grid.add_argument("--kd-range", type=str, default="0,1,2,4,8")
    return parser.parse_args()


def print_top_table(results, top_n):
    print(f"\nTOP {top_n}:")
    print(f"{'#':>3}  {'Kp':>6} {'Ki':>6} {'Kd':>6}  {'steps':>7}  {'reward':>8}  {'surv%':>6}")
    for rank, row in enumerate(results[:top_n], start=1):
        print(
            f"{rank:3d}  {row['kp']:6g} {row['ki']:6g} {row['kd']:6g}  "
            f"{row['mean_steps']:7.0f}  {row['mean_reward']:8.1f}  "
            f"{100.0 * row['survival_rate']:6.1f}"
        )


if __name__ == "__main__":
    args = parse_args()
    if args.integral_limit < 0:
        args.integral_limit = None
    args.com_from_masses = not args.manual_com_height
    physics = resolve_train_physics(args)
    args.com_height_m = physics.l_body_m

    kp_values = parse_float_list(args.kp_range, "kp-range")
    ki_values = parse_float_list(args.ki_range, "ki-range")
    kd_values = parse_float_list(args.kd_range, "kd-range")
    gain_grid = list(product(kp_values, ki_values, kd_values))

    run_dir = DEFAULT_RUNS_ROOT / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    oracle_state = args.oracle_state and args.obs_mode == OBS_MODE_PROCESSED4

    print(f"PID auto-tune → {run_dir.resolve()}")
    print(
        f"Grid: {len(kp_values)}×{len(ki_values)}×{len(kd_values)} = "
        f"{len(gain_grid)} kombinacji × {args.trials_episodes} ep."
    )
    _print_physics_summary(physics)

    max_force_delta = (
        None
        if args.max_force_delta_frac <= 0
        else args.max_force_delta_frac * physics.force_max_n
    )
    static_friction = args.static_friction_frac * physics.force_max_n

    save_run_config(
        run_dir,
        {
            "algorithm": "pid_tune",
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "run_dir": str(run_dir),
            "trials_episodes": args.trials_episodes,
            "final_episodes": args.final_episodes,
            "max_steps": args.max_steps,
            "dt": args.dt,
            "obs_mode": args.obs_mode,
            "domain_randomization": not args.no_domain_randomization,
            "oracle_state": args.oracle_state,
            "kp_range": kp_values,
            "ki_range": ki_values,
            "kd_range": kd_values,
            "n_combinations": len(gain_grid),
            "physics_layout": physics.layout,
            "force_max_n": physics.force_max_n,
            "action_delay_steps": args.action_delay_steps,
            "max_force_delta_per_step": max_force_delta,
            "static_friction_force_n": static_friction,
        },
    )

    env = make_env(args, physics)
    try:
        results = run_grid_search(env, gain_grid, args, oracle_state)
    except KeyboardInterrupt:
        print("\nTune interrupted (Ctrl+C).")
        raise SystemExit(1)

    best = results[0]
    save_tune_csv(run_dir / "tune_results.csv", results)

    summary = {
        "best": best,
        "top_n": results[: args.top_n],
        "ranking_metric": "mean_steps, then mean_reward, then survival_rate",
    }
    summary_path = run_dir / "tune_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Saved tune summary: {summary_path.resolve()}")

    if len(kp_values) > 1 and len(ki_values) > 1:
        save_heatmap(
            run_dir / "tune_heatmap_steps.png",
            results,
            kp_values,
            ki_values,
            kd_values,
            best["kd"],
        )

    print_top_table(results, args.top_n)
    evaluate_best(args, physics, run_dir, best, oracle_state)

    print(
        f"\nDone. Best: Kp={best['kp']:g} Ki={best['ki']:g} Kd={best['kd']:g} "
        f"(mean steps {best['mean_steps']:.0f})"
    )
    print(f"Full results: {run_dir.resolve()}")
