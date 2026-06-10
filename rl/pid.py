"""Classical PID balance controller for simulation and deployment."""

from __future__ import annotations

import numpy as np

from rl.imu_obs import (
    OBS_MODE_IMU_RAW12_ENC1,
    OBS_MODE_IMU_RAW12_ENC2,
    OBS_MODE_IMU_RAW6_ENC1,
    OBS_MODE_IMU_RAW6_ENC2,
    OBS_MODE_PROCESSED4,
    RAW_IMU_CHANNELS,
    is_raw_imu_mode,
    pitch_rad_from_raw_obs,
    pitch_rate_rad_from_raw_obs,
)


class BalancePIDController:
    """
    Cascaded balance PID:
      - inner/primary loop on pitch angle (target 0 rad)
      - optional outer loop on cart position (target x = 0)

    Output is normalized drive command in [-1, 1] (force / force_max).
    """

    def __init__(
        self,
        kp=50.0,
        ki=0.0,
        kd=0.0,
        kp_x=0.0,
        ki_x=0.0,
        kd_x=0.0,
        dt=0.002,
        integral_limit=5.0,
        obs_mode=OBS_MODE_PROCESSED4,
        imu_index=0,
        gyro_bias_dps=0.0,
        accel_pitch_bias_rad=0.0,
    ):
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.kp_x = float(kp_x)
        self.ki_x = float(ki_x)
        self.kd_x = float(kd_x)
        self.dt = float(dt)
        self.integral_limit = (
            None if integral_limit is None else float(integral_limit)
        )
        self.obs_mode = str(obs_mode)
        self.imu_index = int(imu_index)
        self.gyro_bias_dps = float(gyro_bias_dps)
        self.accel_pitch_bias_rad = float(accel_pitch_bias_rad)
        self.reset()

    def reset(self):
        self._i_theta = 0.0
        self._i_x = 0.0

    def _clip_integral(self, value):
        if self.integral_limit is None:
            return value
        return float(np.clip(value, -self.integral_limit, self.integral_limit))

    def _state_from_obs(self, obs, env_state=None):
        if env_state is not None:
            x, x_dot, theta, theta_dot = [float(v) for v in env_state]
            return theta, theta_dot, x, x_dot

        obs = np.asarray(obs, dtype=np.float64)
        if self.obs_mode == OBS_MODE_PROCESSED4:
            return float(obs[0]), float(obs[1]), float(obs[2]), float(obs[3])

        if is_raw_imu_mode(self.obs_mode):
            theta = pitch_rad_from_raw_obs(
                obs,
                accel_pitch_bias_rad=self.accel_pitch_bias_rad,
                imu_index=self.imu_index,
            )
            theta_dot = pitch_rate_rad_from_raw_obs(
                obs,
                gyro_bias_dps=self.gyro_bias_dps,
                imu_index=self.imu_index,
            )
            x_m, x_dot = 0.0, 0.0
            if self.obs_mode in (OBS_MODE_IMU_RAW6_ENC1, OBS_MODE_IMU_RAW12_ENC1):
                base = RAW_IMU_CHANNELS if self.obs_mode == OBS_MODE_IMU_RAW6_ENC1 else 2 * RAW_IMU_CHANNELS
                x_m = float(obs[base])
            elif self.obs_mode in (OBS_MODE_IMU_RAW6_ENC2, OBS_MODE_IMU_RAW12_ENC2):
                base = RAW_IMU_CHANNELS if self.obs_mode == OBS_MODE_IMU_RAW6_ENC2 else 2 * RAW_IMU_CHANNELS
                x_m = float(obs[base])
                x_dot = float(obs[base + 1])
            return theta, theta_dot, x_m, x_dot

        raise ValueError(f"Unsupported obs_mode for PID: {self.obs_mode}")

    def compute(self, obs, force_max_n, env_state=None):
        force_max_n = max(float(force_max_n), 1e-6)
        theta, theta_dot, x, x_dot = self._state_from_obs(obs, env_state=env_state)

        # theta > 0 → dodatnia siła (jak na Segwayu / wahadle w env).
        e_theta = theta
        self._i_theta = self._clip_integral(self._i_theta + e_theta * self.dt)
        u_theta = (
            self.kp * e_theta
            + self.ki * self._i_theta
            - self.kd * theta_dot
        )

        u_x = 0.0
        if self.kp_x or self.ki_x or self.kd_x:
            e_x = -x
            self._i_x = self._clip_integral(self._i_x + e_x * self.dt)
            u_x = self.kp_x * e_x + self.ki_x * self._i_x - self.kd_x * x_dot

        force_n = u_theta + u_x
        action = float(np.clip(force_n / force_max_n, -1.0, 1.0))
        return np.array([action], dtype=np.float32)

    def act(self, obs, force_max_n, env_state=None):
        return self.compute(obs, force_max_n, env_state=env_state)
