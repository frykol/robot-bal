import numpy as np

from rl.imu_obs import (
    OBS_MODE_IMU_RAW12,
    OBS_MODE_IMU_RAW6,
    OBS_MODE_PROCESSED4,
    obs_dim_for_mode,
    simulate_dual_imu_raw_reading,
    simulate_imu_raw_reading,
)


def _wrap_angle_rad(angle):
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def _angle_diff_rad(target, source):
    return _wrap_angle_rad(target - source)


def _accel_pitch_raw_rad(ax, az):
    # Upright with gravity on -Z (typical mount): ax≈0, az<0 → pitch≈0, not ±π.
    return float(np.arctan2(-ax, -az + 1e-6))


class InvertedPendulumEnv:
    """
    Lightweight dynamics environment for fast SAC pretraining.
    State: [x, x_dot, theta, theta_dot]
    Observation modes:
      - processed4: [theta, theta_dot, x, x_dot]
      - imu_raw6: one BMI160 [ax, ay, az, gx, gy, gz] LSB
      - imu_raw12: two BMI160 [imu0(6), imu1(6)] LSB — simulated in train_sim (no I2C)
    Action: normalized in [-1, 1], internally scaled to force.
    """

    def __init__(
        self,
        fall_angle_deg=30.0,
        domain_randomization=True,
        com_height_m=0.11,
        dt=0.002,
        obs_mode=OBS_MODE_PROCESSED4,
        imu_noise_std=25.0,
        m_nominal=None,
        M_nominal=None,
        force_max_nominal=None,
    ):
        self.g = 9.81

        # Default: motors 2x160g at axle; body Pi+case+battery (see rl/robot_mass_model.py).
        self.m_nominal = 0.320 if m_nominal is None else float(m_nominal)
        self.M_nominal = 0.771 if M_nominal is None else float(M_nominal)
        self.l_nominal = float(com_height_m)
        self.dt = float(dt)
        self.force_max_nominal = 10.0 if force_max_nominal is None else float(force_max_nominal)
        self.theta_max = np.radians(float(fall_angle_deg))
        self.domain_randomization = bool(domain_randomization)

        self.state = None
        self.m = self.m_nominal
        self.M = self.M_nominal
        self.l = self.l_nominal
        self.force_max = self.force_max_nominal
        self.obs_mode = str(obs_mode)
        self.imu_noise_std = float(imu_noise_std)
        self.obs_dim = obs_dim_for_mode(self.obs_mode)
        self.act_dim = 1
        n_imu = 2 if self.obs_mode == OBS_MODE_IMU_RAW12 else 1
        self._gyro_bias_lsb = np.zeros((n_imu, 3), dtype=np.float64)
        self._accel_bias_lsb = np.zeros((n_imu, 3), dtype=np.float64)
        self._imu_theta_offset_rad = np.zeros(n_imu, dtype=np.float64)
        self._last_x_ddot = 0.0
        self.reset()

    def reset(self):
        self._resample_dynamics()
        self.state = np.array([0.0, 0.0, np.radians(np.random.uniform(-2, 2)), 0.0])
        if np.random.rand() < 0.05:
            self.state[3] += np.random.uniform(-0.5, 0.5)
        self._last_x_ddot = 0.0
        return self._to_obs(self.state)

    def _resample_dynamics(self):
        if not self.domain_randomization:
            self.m = self.m_nominal
            self.M = self.M_nominal
            self.l = self.l_nominal
            self.force_max = self.force_max_nominal
            self._gyro_bias_lsb[:] = 0.0
            self._accel_bias_lsb[:] = 0.0
            self._imu_theta_offset_rad[:] = 0.0
            return

        # Sim-to-real randomization for better transfer robustness.
        self.m = self.m_nominal * np.random.uniform(0.9, 1.1)
        self.M = self.M_nominal * np.random.uniform(0.9, 1.1)
        self.l = self.l_nominal * np.random.uniform(0.9, 1.1)
        self.force_max = self.force_max_nominal * np.random.uniform(0.85, 1.15)
        n_imu = self._gyro_bias_lsb.shape[0]
        self._gyro_bias_lsb = np.random.uniform(-80.0, 80.0, size=(n_imu, 3))
        self._accel_bias_lsb = np.random.uniform(-200.0, 200.0, size=(n_imu, 3))
        self._imu_theta_offset_rad = np.random.uniform(-0.02, 0.02, size=n_imu)

    def _to_obs(self, state, x_ddot=0.0):
        x, x_dot, theta, theta_dot = state
        if self.obs_mode == OBS_MODE_IMU_RAW12:
            return simulate_dual_imu_raw_reading(
                theta,
                theta_dot,
                x_ddot,
                self._gyro_bias_lsb,
                self._accel_bias_lsb,
                self.imu_noise_std,
                self._imu_theta_offset_rad,
            )
        if self.obs_mode == OBS_MODE_IMU_RAW6:
            return simulate_imu_raw_reading(
                theta,
                theta_dot,
                x_ddot,
                self._gyro_bias_lsb[0],
                self._accel_bias_lsb[0],
                self.imu_noise_std,
            )
        return np.array([theta, theta_dot, x, x_dot], dtype=np.float32)

    def step(self, action):
        x, x_dot, theta, theta_dot = self.state
        force = float(np.clip(action[0], -1.0, 1.0)) * self.force_max

        sin_t = np.sin(theta)
        cos_t = np.cos(theta)
        total_mass = self.M + self.m

        temp = (force + self.M * self.l * theta_dot**2 * sin_t) / total_mass
        theta_ddot = (self.g * sin_t - cos_t * temp) / (
            self.l * (4.0 / 3.0 - self.M * cos_t**2 / total_mass)
        )
        x_ddot = temp - self.M * self.l * theta_ddot * cos_t / total_mass

        x += x_dot * self.dt
        x_dot += x_ddot * self.dt
        theta += theta_dot * self.dt
        theta_dot += theta_ddot * self.dt

        self.state = np.array([x, x_dot, theta, theta_dot])
        self._last_x_ddot = float(x_ddot)
        done = bool(abs(theta) > self.theta_max)

        if done:
            reward = -100.0
        else:
            angle_term = 1.0 - (abs(theta) / self.theta_max)
            center_term = max(0.0, 1.0 - 0.25 * abs(x))
            reward = angle_term + 0.2 * center_term

        return self._to_obs(self.state, x_ddot=self._last_x_ddot), reward, done

