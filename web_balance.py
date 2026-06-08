"""
Web UI: balans AI, regulator PID lub manual na stronie.

Uruchomienie na Raspberry Pi:
  python web_balance.py --actor-path artifacts/runs/dual_h32_16_v6/actor_best.pt --profile safe
  python web_balance.py --default-mode pid --pid-kp 12 --pid-ki 0 --pid-kd 0 --profile safe
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
    OBS_MODE_PROCESSED4,
    features_from_obs,
    obs_dim_for_mode,
)
from rl.pid import BalancePIDController
from rl.pi_runtime import RaspberryBalanceRuntime
from rl.sac import SACAgent, infer_dims_from_actor_file
from web.web_server import create_app

DEFAULT_PID_KP = 12.0
DEFAULT_PID_KI = 0.0
DEFAULT_PID_KD = 0.0
DEFAULT_PID_FORCE_MAX_N = 10.0


class BalanceControl:
    """Współdzielony stan między wątkiem robota a WebSocket."""

    MODES = ("ai", "pid", "manual")

    def __init__(
        self,
        default_mode="ai",
        pid_kp=DEFAULT_PID_KP,
        pid_ki=DEFAULT_PID_KI,
        pid_kd=DEFAULT_PID_KD,
    ):
        if default_mode not in self.MODES:
            raise ValueError(f"default_mode must be one of {self.MODES}")
        self._lock = threading.Lock()
        self._mode = default_mode
        self._manual_power = 0.0
        self._pid_kp = float(pid_kp)
        self._pid_ki = float(pid_ki)
        self._pid_kd = float(pid_kd)

    def set_mode(self, mode):
        mode = str(mode).lower()
        if mode not in self.MODES:
            raise ValueError(f"Unknown mode: {mode}")
        with self._lock:
            self._mode = mode
            if mode != "manual":
                self._manual_power = 0.0
        print(f"Control mode -> {mode}")

    def set_manual_power(self, value):
        value = max(-1.0, min(1.0, float(value)))
        with self._lock:
            if self._mode != "manual":
                return
            self._manual_power = value
        print(f"Manual power -> {value:.2f}")

    def set_pid_gains(self, kp=None, ki=None, kd=None):
        with self._lock:
            if kp is not None:
                self._pid_kp = float(kp)
            if ki is not None:
                self._pid_ki = float(ki)
            if kd is not None:
                self._pid_kd = float(kd)
            kp, ki, kd = self._pid_kp, self._pid_ki, self._pid_kd
        print(f"PID gains -> Kp={kp:g} Ki={ki:g} Kd={kd:g}")

    def snapshot(self):
        with self._lock:
            return {
                "mode": self._mode,
                "manual_power": self._manual_power,
                "pid_kp": self._pid_kp,
                "pid_ki": self._pid_ki,
                "pid_kd": self._pid_kd,
            }

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


def robot_loop(
    env,
    agent,
    control,
    obs_mode,
    calibration,
    tilt_limit_rad,
    deterministic,
    pid_force_max_n,
    pid_dt,
):
    obs = env.reset()
    last_mode = None
    pid = BalancePIDController(
        kp=DEFAULT_PID_KP,
        ki=DEFAULT_PID_KI,
        kd=DEFAULT_PID_KD,
        dt=pid_dt,
        obs_mode=obs_mode,
    )
    last_pid_gains = None

    try:
        while True:
            snap = control.snapshot()
            mode = snap["mode"]
            manual_power = snap["manual_power"]
            pid_gains = (snap["pid_kp"], snap["pid_ki"], snap["pid_kd"])

            if mode != last_mode:
                env.drive.stop()
                if mode in ("ai", "pid"):
                    obs = env._get_obs()
                if mode == "pid":
                    pid.reset()
                else:
                    control.stop_manual()
                last_mode = mode
                last_pid_gains = None

            if pid_gains != last_pid_gains:
                pid.kp, pid.ki, pid.kd = pid_gains
                pid.reset()
                last_pid_gains = pid_gains

            if mode == "ai":
                action = agent.act(obs, deterministic=deterministic)
                obs, _, done = env.step(action)
                pitch, _, _, _ = features_from_obs(obs, obs_mode, calibration=calibration)
                if done or abs(pitch) > tilt_limit_rad:
                    print("Safety stop: tilt threshold exceeded.")
                    env.drive.stop()
                    obs = env.reset()
            elif mode == "pid":
                action = pid.act(obs, pid_force_max_n)
                obs, _, done = env.step(action)
                pitch, pitch_rate, x_m, x_dot = features_from_obs(
                    obs, obs_mode, calibration=calibration
                )
                if done or abs(pitch) > tilt_limit_rad:
                    print("PID safety stop: tilt threshold exceeded.")
                    env.drive.stop()
                    obs = env.reset()
                    pid.reset()
                else:
                    print(
                        f"PID Kp={pid.kp:g} Ki={pid.ki:g} Kd={pid.kd:g} | "
                        f"pitch:{pitch:.3f} rate:{pitch_rate:.3f} | x:{x_m:.3f}"
                    )
            else:
                _apply_manual_drive(env, manual_power, env.motor_scale)
                time.sleep(env.loop_dt)
                obs = env._get_obs()
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
        choices=[OBS_MODE_PROCESSED4, OBS_MODE_IMU_RAW6, OBS_MODE_IMU_RAW12],
        default=OBS_MODE_IMU_RAW12,
    )
    parser.add_argument("--imu-bus-ids", type=int, nargs=2, default=[1, 3])
    parser.add_argument(
        "--default-mode",
        choices=["ai", "pid", "manual"],
        default="ai",
        help="Tryb startowy (domyślnie AI balansuje).",
    )
    parser.add_argument("--pid-kp", type=float, default=DEFAULT_PID_KP)
    parser.add_argument("--pid-ki", type=float, default=DEFAULT_PID_KI)
    parser.add_argument("--pid-kd", type=float, default=DEFAULT_PID_KD)
    parser.add_argument(
        "--pid-force-max-n",
        type=float,
        default=DEFAULT_PID_FORCE_MAX_N,
        help="Skala siły w PID (jak force_max w symulacji).",
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
        imu_bus_ids=tuple(args.imu_bus_ids),
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

    control = BalanceControl(
        default_mode=args.default_mode,
        pid_kp=args.pid_kp,
        pid_ki=args.pid_ki,
        pid_kd=args.pid_kd,
    )
    tilt_limit_rad = args.tilt_limit_deg * 3.141592653589793 / 180.0
    pid_dt = 1.0 / float(args.loop_hz)

    print(
        f"web_balance | obs_mode={args.obs_mode} | action={action_layout} | "
        f"hidden_dims={hidden_dims} | motor_scale={env.motor_scale:.2f} | "
        f"default_mode={args.default_mode} | "
        f"pid Kp={args.pid_kp:g} Ki={args.pid_ki:g} Kd={args.pid_kd:g}"
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
            args.pid_force_max_n,
            pid_dt,
        ),
        daemon=True,
    )
    robot_thread.start()

    def on_disconnect():
        control.set_mode("ai")
        control.stop_manual()

    app = create_app(
        on_motor_power_change=control.set_manual_power,
        on_mode_change=control.set_mode,
        on_pid_gains_change=control.set_pid_gains,
        default_mode=args.default_mode,
        default_pid_kp=args.pid_kp,
        default_pid_ki=args.pid_ki,
        default_pid_kd=args.pid_kd,
        on_disconnect=on_disconnect,
    )

    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
