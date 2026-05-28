import argparse
import json
import select
import sys
import time
from pathlib import Path

import numpy as np

from rl.envs import RaspberryBalanceRuntime


def _preview_tilt_until_enter(env, hz=20):
    print(
        "Podgląd nachylenia (ustaw robota). "
        "Enter — koniec podglądu i start kalibracji (robot nieruchomo)..."
    )
    env.reset()
    period = 1.0 / float(hz)
    while True:
        obs = env._get_obs()
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


def main(samples, sample_dt, output_path, preview_hz):
    env = RaspberryBalanceRuntime(motor_scale=0.0, loop_hz=100)
    try:
        _preview_tilt_until_enter(env, hz=preview_hz)
        print("Trzymaj robota nieruchomo podczas kalibracji...")
        result = env.calibrate_imu(samples=samples, sample_dt=sample_dt)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Saved calibration: {output_path}")
        print(json.dumps(result, indent=2))
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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.samples, args.sample_dt, args.output_path, args.preview_hz)
