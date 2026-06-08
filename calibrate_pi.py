import argparse
import json
import select
import sys
import time
from pathlib import Path

import numpy as np

from rl.imu_obs import (
    OBS_MODE_IMU_RAW12,
    OBS_MODE_IMU_RAW12_ENC1,
    OBS_MODE_IMU_RAW12_ENC2,
    OBS_MODE_IMU_RAW6_ENC1,
    OBS_MODE_IMU_RAW6,
    OBS_MODE_IMU_RAW6_ENC2,
    pitch_rad_from_raw_obs,
)
from rl.pi_runtime import RaspberryBalanceRuntime


def _preview_tilt_until_enter(env, hz=20):
    print(
        "Podgląd nachylenia (ustaw robota). "
        "Enter — koniec podglądu i start kalibracji (robot nieruchomo)..."
    )
    env.reset()
    period = 1.0 / float(hz)
    while True:
        obs = env._get_obs()
        if env.obs_mode == OBS_MODE_IMU_RAW12:
            p0 = np.degrees(pitch_rad_from_raw_obs(obs, imu_index=0))
            if getattr(env, "duplicate_imu12_obs", False):
                print(
                    f"\rIMU bus {env.imu_bus_ids[0]} pitch {p0:+6.2f}° (×2 w obs)   ",
                    end="",
                    flush=True,
                )
            else:
                p1 = np.degrees(pitch_rad_from_raw_obs(obs, imu_index=1))
                print(
                    f"\rIMU bus {env.imu_bus_ids[0]} pitch {p0:+6.2f}° | "
                    f"bus {env.imu_bus_ids[1]} pitch {p1:+6.2f}°   ",
                    end="",
                    flush=True,
                )
        else:
            pitch_deg = float(np.degrees(obs[0]))
            acc_deg = float(np.degrees(env._read_pitch_from_acc()))
            rate_deg_s = float(np.degrees(obs[1]))
            print(
                f"\rnachylenie: {pitch_deg:+7.2f}°  "
                f"| acc: {acc_deg:+7.2f}°  "
                f"| gyro: {rate_deg_s:+7.2f}°/s   ",
                end="",
                flush=True,
            )

        if select.select([sys.stdin], [], [], 0)[0]:
            try:
                sys.stdin.readline()
            except EOFError:
                raise KeyboardInterrupt from None
            print()
            break
        time.sleep(period)


def main(samples, sample_dt, output_path, preview_hz, obs_mode, imu_bus_id, dual_physical_imu, imu_bus_ids):
    env = RaspberryBalanceRuntime(
        motor_scale=0.0,
        loop_hz=100,
        obs_mode=obs_mode,
        imu_primary_bus_id=imu_bus_id,
        dual_physical_imu=dual_physical_imu,
        imu_bus_ids=imu_bus_ids,
    )
    try:
        _preview_tilt_until_enter(env, hz=preview_hz)
        print("Trzymaj robota nieruchomo podczas kalibracji...")
        result = env.calibrate_imu(samples=samples, sample_dt=sample_dt)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Saved calibration: {output_path}")
        print(json.dumps(result, indent=2))
        # Warn if any IMU has suspiciously large accel bias.
        sensors = result.get("imensors") or [
            {
                "bus_id": imu_bus_id,
                "accel_pitch_bias_rad": result.get("accel_pitch_bias_rad", 0.0),
            }
        ]
        for s in sensors:
            bias_deg = float(np.degrees(s.get("accel_pitch_bias_rad", 0.0)))
            if abs(bias_deg) > 15.0:
                bid = s.get("bus_id", "?")
                print(
                    f"Uwaga: duży bias acc na IMU bus {bid} ({bias_deg:.1f}°) — "
                    "robot mógł się ruszać albo IMU jest pod innym kątem."
                )
    finally:
        env.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--sample-dt", type=float, default=0.005)
    parser.add_argument(
        "--preview-hz",
        type=int,
        default=20,
        help="Odświeżanie podglądu nachylenia przed kalibracją.",
    )
    parser.add_argument(
        "--output-path", type=Path, default=Path("artifacts") / "pi_calibration.json"
    )
    parser.add_argument(
        "--obs-mode",
        default=OBS_MODE_IMU_RAW12,
        choices=[
            "processed4",
            "imu_raw6",
            "imu_raw12",
            OBS_MODE_IMU_RAW6_ENC1,
            OBS_MODE_IMU_RAW12_ENC1,
            OBS_MODE_IMU_RAW6_ENC2,
            OBS_MODE_IMU_RAW12_ENC2,
        ],
    )
    parser.add_argument("--imu-bus-id", type=int, default=1)
    parser.add_argument("--dual-imu", action="store_true")
    parser.add_argument("--imu-bus-ids", type=int, nargs=2, default=[1, 3])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        args.samples,
        args.sample_dt,
        args.output_path,
        args.preview_hz,
        args.obs_mode,
        args.imu_bus_id,
        args.dual_imu,
        tuple(args.imu_bus_ids) if args.dual_imu else None,
    )
