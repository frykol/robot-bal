"""Raspberry Pi deployment runtime (real I2C IMU + motors). Not used by train_sim."""

import copy
import time

import numpy as np

from rl.envs import _accel_pitch_raw_rad, _angle_diff_rad, _wrap_angle_rad
from rl.envs_dual import parse_dual_action
from rl.imu_obs import (
    DEFAULT_IMU_BUS_IDS,
    OBS_MODE_IMU_RAW12,
    OBS_MODE_PROCESSED4,
    is_raw_imu_mode,
    load_imu_calibration,
    normalize_raw_imu_obs,
    obs_dim_for_mode,
    pitch_rad_for_safety,
)


class RaspberryBalanceRuntime:
    """
    Runtime environment for deployment/inference on Raspberry Pi.
    Exposes the same (obs, action) interface as training env.
    """

    def __init__(
        self,
        bus_id=1,
        imu_bus_ids=None,
        imu_primary_bus_id=1,
        dual_physical_imu=False,
        motor_scale=0.8,
        loop_hz=100,
        pitch_alpha=0.98,
        gyro_lsb_per_dps=None,
        encoder_step_to_m=0.0005,
        imu_calibration=None,
        gyro_bias_dps=0.0,
        accel_pitch_bias_rad=0.0,
        fall_angle_deg=25.0,
        obs_mode=OBS_MODE_PROCESSED4,
        action_layout="scalar",
        min_motor_power=0.2,
    ):
        from hardware.accelerometer import BMI160, GYR_LSB_PER_DPS
        from hardware.drive_module import DriveModule

        if gyro_lsb_per_dps is None:
            gyro_lsb_per_dps = GYR_LSB_PER_DPS

        self.obs_mode = str(obs_mode)
        use_dual = bool(dual_physical_imu) and self.obs_mode == OBS_MODE_IMU_RAW12
        if imu_bus_ids is None:
            imu_bus_ids = DEFAULT_IMU_BUS_IDS if use_dual else (int(imu_primary_bus_id),)
        if use_dual:
            self.imu_bus_ids = tuple(int(b) for b in imu_bus_ids)[:2]
            self.imus = [BMI160(bus_id=b) for b in self.imu_bus_ids]
            self.duplicate_imu12_obs = False
        else:
            primary = int(imu_bus_ids[0] if imu_bus_ids else imu_primary_bus_id)
            self.imu_bus_ids = (primary,)
            self.imus = [BMI160(bus_id=primary)]
            self.duplicate_imu12_obs = self.obs_mode == OBS_MODE_IMU_RAW12
        self.imu = self.imus[0]
        self.drive = DriveModule()
        self.motor_scale = float(motor_scale)
        self.loop_dt = 1.0 / float(loop_hz)
        self.pitch_alpha = float(pitch_alpha)
        self.gyro_lsb_per_dps = float(gyro_lsb_per_dps)
        self.encoder_step_to_m = float(encoder_step_to_m)
        cal = imu_calibration if imu_calibration is not None else {}
        if not cal and (gyro_bias_dps or accel_pitch_bias_rad):
            cal = {
                "gyro_bias_dps": gyro_bias_dps,
                "accel_pitch_bias_rad": accel_pitch_bias_rad,
            }
        self.imu_calibration = cal
        n_imu_cal = 2 if self.obs_mode == OBS_MODE_IMU_RAW12 else max(1, len(self.imus))
        self.imu_sensor_biases = load_imu_calibration(cal, n_imus=n_imu_cal)
        self.gyro_bias_dps = self.imu_sensor_biases[0]["gyro_bias_dps"]
        self.accel_pitch_bias_rad = self.imu_sensor_biases[0]["accel_pitch_bias_rad"]
        self.fall_angle_rad = np.radians(float(fall_angle_deg))

        self.pitch = 0.0
        self.pitch_rate = 0.0
        self.x_est = 0.0
        self.x_dot_est = 0.0
        self._last_t = time.time()

        self.obs_dim = obs_dim_for_mode(self.obs_mode)
        self.action_layout = str(action_layout)
        if self.action_layout not in ("scalar", "dual"):
            raise ValueError("action_layout must be 'scalar' or 'dual'")
        self.min_motor_power = float(min_motor_power)
        self.act_dim = 2 if self.action_layout == "dual" else 1
        self._last_sensor_snapshot = None

    def _read_imu_raw_one(self, imu):
        ax, ay, az = imu.read_acc()
        gx, gy, gz = imu.read_gyro()
        return np.array([ax, ay, az, gx, gy, gz], dtype=np.float32)

    def _store_sensor_snapshot(self, imus):
        e1, e2 = self.drive.get_encoder_steps()
        self._last_sensor_snapshot = {
            "t": time.time(),
            "imus": imus,
            "enc": [int(e1), int(e2)],
        }

    def _imu_dict_from_device(self, bus_id=None):
        imu = self.imu
        bid = int(self.imu_bus_ids[0] if bus_id is None else bus_id)
        ax, ay, az = imu.read_acc()
        gx, gy, gz = imu.read_gyro()
        return {
            "bus_id": bid,
            "acc": [int(ax), int(ay), int(az)],
            "gyro": [int(gx), int(gy), int(gz)],
        }

    def _snapshot_imu_list(self):
        if self.duplicate_imu12_obs:
            one = self._imu_dict_from_device()
            return [one, copy.deepcopy(one)]
        return [self._imu_dict_from_device(b) for b in self.imu_bus_ids]

    def read_sensor_snapshot(self):
        """Raw BMI160 LSB + encoder counts (extra I2C read — avoid in hot loop)."""
        imus = self._snapshot_imu_list()
        self._store_sensor_snapshot(imus)
        return dict(self._last_sensor_snapshot)

    def peek_sensor_snapshot(self):
        """Ostatni odczyt z pętli sterowania + świeże enkodery (bez ponownego I2C IMU)."""
        if self._last_sensor_snapshot is None:
            return self.read_sensor_snapshot()
        e1, e2 = self.drive.get_encoder_steps()
        snap = {
            "t": time.time(),
            "imus": self._last_sensor_snapshot["imus"],
            "enc": [int(e1), int(e2)],
        }
        return snap

    def _read_imu_raw(self):
        if self.duplicate_imu12_obs:
            chunk = self._read_imu_raw_one(self.imu)
            one = self._imu_dict_from_device()
            imus = [one, copy.deepcopy(one)]
            self._store_sensor_snapshot(imus)
            raw = np.concatenate([chunk, chunk]).astype(np.float32)
            return normalize_raw_imu_obs(raw)

        imus = []
        parts = []
        for bus_id, imu in zip(self.imu_bus_ids, self.imus[:2]):
            chunk = self._read_imu_raw_one(imu)
            parts.append(chunk)
            ax, ay, az = int(chunk[0]), int(chunk[1]), int(chunk[2])
            gx, gy, gz = int(chunk[3]), int(chunk[4]), int(chunk[5])
            imus.append(
                {
                    "bus_id": int(bus_id),
                    "acc": [ax, ay, az],
                    "gyro": [gx, gy, gz],
                }
            )
        if len(parts) == 1:
            parts.append(parts[0].copy())
            imus.append(dict(imus[0]))
        self._store_sensor_snapshot(imus[:2])
        raw = np.concatenate(parts[:2]).astype(np.float32)
        return normalize_raw_imu_obs(raw)

    def _pitch_rad_for_done(self, obs):
        if is_raw_imu_mode(self.obs_mode):
            return pitch_rad_for_safety(
                obs,
                self.obs_mode,
                self.accel_pitch_bias_rad,
                self.imu_calibration,
            )
        return float(obs[0])

    def _read_pitch_rate_rad(self):
        gx, _, _ = self.imu.read_gyro()
        deg_s = (gx / self.gyro_lsb_per_dps) - self.gyro_bias_dps
        return np.deg2rad(deg_s)

    def _read_pitch_from_acc(self):
        ax, _, az = self.imu.read_acc()
        return _accel_pitch_raw_rad(ax, az) - self.accel_pitch_bias_rad

    def reset(self):
        self.drive.stop()
        self.drive.reset_encoders()
        self.x_est = 0.0
        self.x_dot_est = 0.0
        self._last_t = time.time()
        self.pitch = self._read_pitch_from_acc()
        self.pitch_rate = self._read_pitch_rate_rad()
        return self._get_obs()

    def _get_obs(self):
        if is_raw_imu_mode(self.obs_mode):
            return self._read_imu_raw()

        now = time.time()
        dt = max(1e-4, now - self._last_t)
        self._last_t = now

        self.pitch_rate = self._read_pitch_rate_rad()
        pitch_acc = self._read_pitch_from_acc()
        pitch_gyro = _wrap_angle_rad(self.pitch + self.pitch_rate * dt)
        acc_blend = (1.0 - self.pitch_alpha) * _angle_diff_rad(pitch_acc, pitch_gyro)
        self.pitch = _wrap_angle_rad(pitch_gyro + acc_blend)

        enc_left, enc_right = self.drive.get_encoder_steps()
        steps_avg = 0.5 * (enc_left + enc_right)
        meters = steps_avg * self.encoder_step_to_m
        self.x_dot_est = (meters - self.x_est) / dt
        self.x_est = meters

        self._store_sensor_snapshot(self._snapshot_imu_list())

        return np.array(
            [self.pitch, self.pitch_rate, self.x_est, self.x_dot_est], dtype=np.float32
        )

    def _motor_cmd_from_action(self, action):
        if self.act_dim == 2:
            direction, power = parse_dual_action(action, min_power=self.min_motor_power)
            return float(direction * power)
        return float(np.clip(action[0], -1.0, 1.0))

    def step(self, action):
        cmd = self._motor_cmd_from_action(action)
        pwm = min(abs(cmd) * self.motor_scale, 1.0)
        if cmd >= 0:
            self.drive.forward(pwm)
        else:
            self.drive.backward(pwm)

        time.sleep(self.loop_dt)
        obs = self._get_obs()
        done = abs(self._pitch_rad_for_done(obs)) > self.fall_angle_rad
        reward = 0.0
        return obs, reward, done

    def close(self):
        self.drive.close()

    def calibrate_imu(self, samples=500, sample_dt=0.005):
        """
        Estimate IMU biases while the robot is stationary (each BMI160 separately).
        Returns a dict with biases that can be reused in runtime args.
        """
        imensors = []
        for idx, imu in enumerate(self.imus):
            gyro_dps = []
            accel_pitch = []
            for _ in range(samples):
                gx, _, _ = imu.read_gyro()
                ax, _, az = imu.read_acc()
                gyro_dps.append(gx / self.gyro_lsb_per_dps)
                accel_pitch.append(_accel_pitch_raw_rad(ax, az))
                time.sleep(sample_dt)
            imensors.append(
                {
                    "bus_id": int(self.imu_bus_ids[idx]),
                    "gyro_bias_dps": float(np.mean(gyro_dps)),
                    "accel_pitch_bias_rad": float(np.mean(accel_pitch)),
                }
            )

        self.imu_sensor_biases = load_imu_calibration({"imensors": imensors}, n_imus=len(imensors))
        self.imu_calibration = {
            "imu_bus_ids": list(self.imu_bus_ids),
            "imensors": imensors,
            "gyro_bias_dps": imensors[0]["gyro_bias_dps"],
            "accel_pitch_bias_rad": imensors[0]["accel_pitch_bias_rad"],
        }
        self.gyro_bias_dps = imensors[0]["gyro_bias_dps"]
        self.accel_pitch_bias_rad = imensors[0]["accel_pitch_bias_rad"]
        return self.imu_calibration
