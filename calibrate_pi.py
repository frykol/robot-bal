import argparse
import json
from pathlib import Path

from rl.envs import RaspberryBalanceRuntime


def main(samples, sample_dt, output_path):
    env = RaspberryBalanceRuntime(motor_scale=0.0, loop_hz=100)
    try:
        print("Keep robot still during calibration...")
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
        "--output-path", type=Path, default=Path("artifacts") / "pi_calibration.json"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.samples, args.sample_dt, args.output_path)

