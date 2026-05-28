import argparse
import csv
import json
import time
from pathlib import Path

from rl.envs import RaspberryBalanceRuntime
from rl.sac import SACAgent, infer_dims_from_actor_file


def _profile_to_motor_scale(profile):
    if profile == "safe":
        return 0.30
    if profile == "normal":
        return 0.55
    return 0.75


def run(
    actor_path,
    deterministic=True,
    profile="safe",
    loop_hz=100,
    tilt_limit_deg=25.0,
    log_path=None,
    calibration_path=None,
):
    calibration = {}
    if calibration_path is not None and calibration_path.exists():
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))

    env = RaspberryBalanceRuntime(
        motor_scale=_profile_to_motor_scale(profile),
        loop_hz=loop_hz,
        gyro_bias_dps=float(calibration.get("gyro_bias_dps", 0.0)),
        accel_pitch_bias_rad=float(calibration.get("accel_pitch_bias_rad", 0.0)),
        fall_angle_deg=tilt_limit_deg,
    )
    _, _, hidden_dim = infer_dims_from_actor_file(Path(actor_path))
    agent = SACAgent(obs_dim=env.obs_dim, act_dim=env.act_dim, hidden_dim=hidden_dim)
    agent.load_actor(actor_path)

    csv_file = None
    writer = None
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        csv_file = open(log_path, "w", newline="", encoding="utf-8")
        writer = csv.writer(csv_file)
        writer.writerow(
            ["timestamp", "pitch_rad", "pitch_rate_rad_s", "x_m", "x_dot_m_s", "action"]
        )

    tilt_limit_rad = tilt_limit_deg * 3.141592653589793 / 180.0
    obs = env.reset()
    print(
        f"Running policy on Raspberry runtime (profile={profile}, "
        f"motor_scale={env.motor_scale:.2f}). Ctrl+C to stop."
    )
    try:
        while True:
            action = agent.act(obs, deterministic=deterministic)
            obs, _, done = env.step(action)
            if writer is not None:
                writer.writerow(
                    [time.time(), float(obs[0]), float(obs[1]), float(obs[2]), float(obs[3]), float(action[0])]
                )
            if abs(float(obs[0])) > tilt_limit_rad:
                done = True
            if done:
                print("Safety stop: tilt threshold exceeded.")
                env.drive.stop()
                obs = env.reset()
    except KeyboardInterrupt:
        pass
    finally:
        if csv_file is not None:
            csv_file.close()
        env.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--actor-path", default="artifacts/actor_sim.pt")
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--profile", choices=["safe", "normal", "aggressive"], default="safe")
    parser.add_argument("--loop-hz", type=int, default=100)
    parser.add_argument("--tilt-limit-deg", type=float, default=25.0)
    parser.add_argument("--log-path", type=Path, default=Path("logs") / "run_latest.csv")
    parser.add_argument(
        "--calibration-path", type=Path, default=Path("artifacts") / "pi_calibration.json"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        args.actor_path,
        deterministic=not args.stochastic,
        profile=args.profile,
        loop_hz=args.loop_hz,
        tilt_limit_deg=args.tilt_limit_deg,
        log_path=args.log_path,
        calibration_path=args.calibration_path,
    )

