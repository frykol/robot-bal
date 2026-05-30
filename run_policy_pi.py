import argparse
import csv
import json
import time
from pathlib import Path

from rl.pi_runtime import RaspberryBalanceRuntime
from rl.imu_obs import (
    OBS_MODE_IMU_RAW12,
    OBS_MODE_IMU_RAW6,
    OBS_MODE_PROCESSED4,
    features_from_obs,
    obs_dim_for_mode,
)
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
    obs_mode=OBS_MODE_IMU_RAW12,
    imu_bus_ids=(1, 3),
):
    calibration = {}
    if calibration_path is not None and calibration_path.exists():
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))

    ckpt_obs, ckpt_act, hidden_dims = infer_dims_from_actor_file(Path(actor_path))
    action_layout = "dual" if ckpt_act == 2 else "scalar"
    if ckpt_obs != obs_dim_for_mode(obs_mode):
        raise ValueError(
            f"Actor obs_dim={ckpt_obs} != --obs-mode {obs_mode} "
            f"(obs_dim={obs_dim_for_mode(obs_mode)})."
        )

    env = RaspberryBalanceRuntime(
        motor_scale=_profile_to_motor_scale(profile),
        loop_hz=loop_hz,
        imu_bus_ids=imu_bus_ids,
        imu_calibration=calibration,
        fall_angle_deg=tilt_limit_deg,
        obs_mode=obs_mode,
        action_layout=action_layout,
    )
    if ckpt_act != env.act_dim:
        raise ValueError(
            f"Actor act_dim={ckpt_act} != runtime act_dim={env.act_dim}."
        )
    agent = SACAgent(obs_dim=env.obs_dim, act_dim=env.act_dim, hidden_dims=hidden_dims)
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
        f"Running policy on Raspberry (obs_mode={obs_mode}, action={action_layout}, "
        f"hidden_dims={hidden_dims}, motor_scale={env.motor_scale:.2f}). Ctrl+C to stop."
    )
    try:
        while True:
            action = agent.act(obs, deterministic=deterministic)
            obs, _, done = env.step(action)
            if writer is not None:
                pitch, pitch_rate, x_m, x_dot = features_from_obs(
                    obs, obs_mode, calibration=calibration
                )
                row_action = (
                    f"{float(action[0]):.4f},{float(action[1]):.4f}"
                    if len(action) > 1
                    else f"{float(action[0]):.4f}"
                )
                writer.writerow(
                    [time.time(), pitch, pitch_rate, x_m, x_dot, row_action]
                )
            pitch, _, _, _ = features_from_obs(obs, obs_mode, calibration=calibration)
            if abs(pitch) > tilt_limit_rad:
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
    parser.add_argument(
        "--obs-mode",
        choices=[OBS_MODE_PROCESSED4, OBS_MODE_IMU_RAW6, OBS_MODE_IMU_RAW12],
        default=OBS_MODE_IMU_RAW12,
        help="processed4 | imu_raw6 (1 czujnik) | imu_raw12 (2 czujniki I2C).",
    )
    parser.add_argument(
        "--imu-bus-ids",
        type=int,
        nargs=2,
        default=[1, 3],
        metavar=("BUS_A", "BUS_B"),
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
        obs_mode=args.obs_mode,
        imu_bus_ids=tuple(args.imu_bus_ids),
    )

