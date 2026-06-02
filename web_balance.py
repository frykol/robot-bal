"""
Web UI: domyślnie balans AI; przełącznik Manual / AI na stronie.

Uruchomienie na Raspberry Pi:
  python web_balance.py --actor-path artifacts/runs/dual_h32_16_v6/actor_best.pt --profile safe
"""

import argparse
import json
import threading
import time
from pathlib import Path

import uvicorn

from rl.imu_obs import (
    OBS_MODE_IMU_RAW12,
    OBS_MODE_IMU_RAW6,
    OBS_MODE_IMU_RAW12_ENC1,
    OBS_MODE_IMU_RAW6_ENC1,
    OBS_MODE_IMU_RAW12_ENC2,
    OBS_MODE_IMU_RAW6_ENC2,
    OBS_MODE_PROCESSED4,
    features_from_obs,
    obs_dim_for_mode,
)
from rl.pi_runtime import RaspberryBalanceRuntime
from rl.sac import SACAgent, infer_dims_from_actor_file
from web.web_server import create_app


def _load_sensor_recorder():
    import importlib
    import sys
    from pathlib import Path

    pkg_dir = Path(__file__).resolve().parent / "web"
    if (pkg_dir / "sensor_recorder").is_dir():
        print(
            "Błąd: istnieje katalog web/sensor_recorder/ — usuń go "
            "(powinien być tylko plik web/sensor_recorder.py).",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        mod = importlib.import_module("web.sensor_recorder")
    except SyntaxError as exc:
        print(
            f"Błąd składni web/sensor_recorder.py (linia {exc.lineno}): {exc}\n"
            "Sprawdź: python3 -m py_compile web/sensor_recorder.py",
            file=sys.stderr,
        )
        sys.exit(1)

    cls = getattr(mod, "SensorRecorder", None)
    if cls is None:
        print(
            "web/sensor_recorder.py nie eksportuje SensorRecorder.\n"
            f"Plik: {getattr(mod, '__file__', '?')}\n"
            "Sprawdź: git pull && rm -rf web/__pycache__\n"
            "        grep class SensorRecorder web/sensor_recorder.py",
            file=sys.stderr,
        )
        sys.exit(1)
    return cls


SensorRecorder = _load_sensor_recorder()


class BalanceControl:
    """Współdzielony stan między wątkiem robota a WebSocket."""

    MODES = ("ai", "manual")

    def __init__(self, default_mode="ai"):
        if default_mode not in self.MODES:
            raise ValueError(f"default_mode must be one of {self.MODES}")
        self._lock = threading.Lock()
        self._mode = default_mode
        self._manual_power = 0.0

    def set_mode(self, mode):
        mode = str(mode).lower()
        if mode not in self.MODES:
            raise ValueError(f"Unknown mode: {mode}")
        with self._lock:
            self._mode = mode
            if mode == "ai":
                self._manual_power = 0.0
        print(f"Control mode -> {mode}")

    def set_manual_power(self, value):
        value = max(-1.0, min(1.0, float(value)))
        with self._lock:
            if self._mode != "manual":
                return
            self._manual_power = value
        print(f"Manual power -> {value:.2f}")

    def snapshot(self):
        with self._lock:
            return self._mode, self._manual_power

    def stop_manual(self):
        with self._lock:
            self._manual_power = 0.0


def _profile_to_motor_scale(profile):
    if profile == "safe":
        return 0.30
    if profile == "normal":
        return 0.55
    return 0.75


def _apply_manual_drive(env, power, motor_scale):
    pwm = min(abs(power) * motor_scale, 1.0)
    if power > 0:
        env.drive.forward(pwm)
    elif power < 0:
        env.drive.backward(pwm)
    else:
        env.drive.stop()


def _offer_sensor_sample(recorder, env):
    """Tylko kolejka — zapis CSV w wątku SensorRecorder."""
    if recorder is not None and recorder.recording:
        recorder.offer(env.peek_sensor_snapshot())


def robot_loop(env, agent, control, obs_mode, calibration, tilt_limit_rad, deterministic, telemetry=None):
    obs = env.reset()
    last_mode = None

    try:
        while True:
            mode, manual_power = control.snapshot()

            if mode != last_mode:
                env.drive.stop()
                if mode == "ai":
                    obs = env._get_obs()
                else:
                    control.stop_manual()
                last_mode = mode

            if mode == "ai":
                action = agent.act(obs, deterministic=deterministic)
                obs, _, done = env.step(action)
                _offer_sensor_sample(telemetry, env)
                pitch, _, _, _ = features_from_obs(obs, obs_mode, calibration=calibration)
                if done or abs(pitch) > tilt_limit_rad:
                    print("Safety stop: tilt threshold exceeded.")
                    env.drive.stop()
                    obs = env.reset()
            else:
                _apply_manual_drive(env, manual_power, env.motor_scale)
                time.sleep(env.loop_dt)
                obs = env._get_obs()
                _offer_sensor_sample(telemetry, env)
                pitch, pitch_rate, x_m, x_dot = features_from_obs(
                    obs, obs_mode, calibration=calibration
                )
                e1, e2 = env.drive.get_encoder_steps()
                print(
                    f"MANUAL pwr:{manual_power:.2f} | "
                    f"pitch:{pitch:.3f} rate:{pitch_rate:.3f} | "
                    f"ENC M1:{e1} M2:{e2} | x:{x_m:.3f}"
                )

    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(
        description="Web UI: AI balance (default) or manual motor slider."
    )
    parser.add_argument(
        "--actor-path",
        type=Path,
        default=Path("artifacts/runs/dual_h32_16_v6/actor_best.pt"),
    )
    parser.add_argument("--profile", choices=["safe", "normal", "aggressive"], default="safe")
    parser.add_argument("--loop-hz", type=int, default=100)
    parser.add_argument("--tilt-limit-deg", type=float, default=25.0)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument(
        "--calibration-path",
        type=Path,
        default=Path("artifacts") / "pi_calibration.json",
    )
    parser.add_argument(
        "--obs-mode",
        choices=[
            OBS_MODE_PROCESSED4,
            OBS_MODE_IMU_RAW6,
            OBS_MODE_IMU_RAW12,
            OBS_MODE_IMU_RAW6_ENC1,
            OBS_MODE_IMU_RAW12_ENC1,
            OBS_MODE_IMU_RAW6_ENC2,
            OBS_MODE_IMU_RAW12_ENC2,
        ],
        default=OBS_MODE_IMU_RAW12,
    )
    parser.add_argument(
        "--imu-bus-id",
        type=int,
        default=1,
        help="BMI160 do odczytu (domyślnie bus 1; imu_raw12 = ten sam sygnał 2× w obs).",
    )
    parser.add_argument(
        "--dual-imu",
        action="store_true",
        help="Dwa fizyczne IMU (--imu-bus-ids 1 3); domyślnie jeden czujnik powielony.",
    )
    parser.add_argument(
        "--imu-bus-ids",
        type=int,
        nargs=2,
        default=[1, 3],
        metavar=("BUS_A", "BUS_B"),
        help="Tylko z --dual-imu.",
    )
    parser.add_argument(
        "--default-mode",
        choices=["ai", "manual"],
        default="ai",
        help="Tryb startowy (domyślnie AI balansuje).",
    )
    parser.add_argument(
        "--record-logs-dir",
        type=Path,
        default=Path("logs") / "sensor_recordings",
        help="Katalog CSV z nagraniami IMU + enkoderów.",
    )
    args = parser.parse_args()

    calibration = {}
    if args.calibration_path.exists():
        calibration = json.loads(args.calibration_path.read_text(encoding="utf-8"))

    ckpt_obs, ckpt_act, hidden_dims = infer_dims_from_actor_file(args.actor_path)
    action_layout = "dual" if ckpt_act == 2 else "scalar"
    if ckpt_obs != obs_dim_for_mode(args.obs_mode):
        raise ValueError(
            f"Actor obs_dim={ckpt_obs} != --obs-mode {args.obs_mode} "
            f"(obs_dim={obs_dim_for_mode(args.obs_mode)})."
        )

    env = RaspberryBalanceRuntime(
        motor_scale=_profile_to_motor_scale(args.profile),
        loop_hz=args.loop_hz,
        imu_primary_bus_id=args.imu_bus_id,
        dual_physical_imu=args.dual_imu,
        imu_bus_ids=tuple(args.imu_bus_ids) if args.dual_imu else None,
        imu_calibration=calibration,
        fall_angle_deg=args.tilt_limit_deg,
        obs_mode=args.obs_mode,
        action_layout=action_layout,
    )
    if ckpt_act != env.act_dim:
        raise ValueError(
            f"Actor act_dim={ckpt_act} != runtime act_dim={env.act_dim}."
        )

    agent = SACAgent(
        obs_dim=env.obs_dim,
        act_dim=env.act_dim,
        hidden_dims=hidden_dims,
        device="cpu",
    )
    agent.load_actor(str(args.actor_path))

    control = BalanceControl(default_mode=args.default_mode)
    telemetry = SensorRecorder(logs_dir=args.record_logs_dir)
    tilt_limit_rad = args.tilt_limit_deg * 3.141592653589793 / 180.0

    imu_note = (
        f"bus {env.imu_bus_ids[0]} ×2 w obs"
        if env.duplicate_imu12_obs
        else f"dual buses {env.imu_bus_ids}"
    )
    print(
        f"web_balance | obs_mode={args.obs_mode} | IMU {imu_note} | "
        f"action={action_layout} | hidden_dims={hidden_dims} | "
        f"motor_scale={env.motor_scale:.2f} | default_mode={args.default_mode}"
    )

    robot_thread = threading.Thread(
        target=robot_loop,
        args=(
            env,
            agent,
            control,
            args.obs_mode,
            calibration,
            tilt_limit_rad,
            not args.stochastic,
            telemetry,
        ),
        daemon=True,
    )
    robot_thread.start()

    def on_disconnect():
        control.set_mode("ai")
        control.stop_manual()
        if telemetry.recording:
            telemetry.stop()

    app = create_app(
        on_motor_power_change=control.set_manual_power,
        on_mode_change=control.set_mode,
        default_mode=args.default_mode,
        on_disconnect=on_disconnect,
        telemetry_hub=telemetry,
    )

    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
