import time

import numpy as np

from hardware.accelerometer import BMI160
from hardware.drive_module import DriveModule


class InvertedPendulumEnv:
    """
    Lightweight dynamics environment for fast SAC pretraining.
    State: [x, x_dot, theta, theta_dot]
    Observation used by policy: [theta, theta_dot, x, x_dot]
    Action: normalized in [-1, 1], internally scaled to force.
    """

    def __init__(self, fall_angle_deg=30.0, domain_randomization=True):
        self.g = 9.81

        # Component masses from user-provided data:
        # motors: 2 x 160g, rpi: 55g, case: 466g, battery: 250g.
        self.m_nominal = 0.320  # wheels+motors [kg]
        self.M_nominal = 0.771  # body [kg]

        self.l_nominal = 0.15
        self.dt = 0.01
        self.force_max_nominal = 10.0
        self.theta_max = np.radians(float(fall_angle_deg))
        self.domain_randomization = bool(domain_randomization)

        self.state = None
        self.m = self.m_nominal
        self.M = self.M_nominal
        self.l = self.l_nominal
        self.force_max = self.force_max_nominal
        self.obs_dim = 4
        self.act_dim = 1
        self.reset()

    def reset(self):
        self._resample_dynamics()
        self.state = np.array([0.0, 0.0, np.radians(np.random.uniform(-2, 2)), 0.0])
        if np.random.rand() < 0.05:
            self.state[3] += np.random.uniform(-0.5, 0.5)
        return self._to_obs(self.state)

    def _resample_dynamics(self):
        if not self.domain_randomization:
            self.m = self.m_nominal
            self.M = self.M_nominal
            self.l = self.l_nominal
            self.force_max = self.force_max_nominal
            return

        # Sim-to-real randomization for better transfer robustness.
        self.m = self.m_nominal * np.random.uniform(0.9, 1.1)
        self.M = self.M_nominal * np.random.uniform(0.9, 1.1)
        self.l = self.l_nominal * np.random.uniform(0.9, 1.1)
        self.force_max = self.force_max_nominal * np.random.uniform(0.85, 1.15)

    def _to_obs(self, state):
        x, x_dot, theta, theta_dot = state
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
        done = bool(abs(theta) > self.theta_max)

        if done:
            reward = -100.0
        else:
            angle_term = 1.0 - (abs(theta) / self.theta_max)
            center_term = max(0.0, 1.0 - 0.25 * abs(x))
            reward = angle_term + 0.2 * center_term

        return self._to_obs(self.state), reward, done


class RaspberryBalanceRuntime:
    """
    Runtime environment for deployment/inference on Raspberry Pi.
    Exposes the same (obs, action) interface as training env.
    """

    def __init__(
        self,
        bus_id=1,
        motor_scale=0.8,
        loop_hz=100,
        pitch_alpha=0.98,
        gyro_lsb_per_dps=131.0,
        encoder_step_to_m=0.0005,
        gyro_bias_dps=0.0,
        accel_pitch_bias_rad=0.0,
        fall_angle_deg=25.0,
    ):
        self.imu = BMI160(bus_id=bus_id)
        self.drive = DriveModule()
        self.motor_scale = float(motor_scale)
        self.loop_dt = 1.0 / float(loop_hz)
        self.pitch_alpha = float(pitch_alpha)
        self.gyro_lsb_per_dps = float(gyro_lsb_per_dps)
        self.encoder_step_to_m = float(encoder_step_to_m)
        self.gyro_bias_dps = float(gyro_bias_dps)
        self.accel_pitch_bias_rad = float(accel_pitch_bias_rad)
        self.fall_angle_rad = np.radians(float(fall_angle_deg))

        self.pitch = 0.0
        self.pitch_rate = 0.0
        self.x_est = 0.0
        self.x_dot_est = 0.0
        self._last_t = time.time()

        self.obs_dim = 4
        self.act_dim = 1

    def _read_pitch_rate_rad(self):
        gx, _, _ = self.imu.read_gyro()
        deg_s = (gx / self.gyro_lsb_per_dps) - self.gyro_bias_dps
        return np.deg2rad(deg_s)

    def _read_pitch_from_acc(self):
        ax, _, az = self.imu.read_acc()
        # Complementary estimate from acceleration vector.
        return np.arctan2(ax, az + 1e-6) - self.accel_pitch_bias_rad

    def reset(self):
        self.drive.stop()
        self.drive.reset_encoders()
        self.pitch = 0.0
        self.pitch_rate = 0.0
        self.x_est = 0.0
        self.x_dot_est = 0.0
        self._last_t = time.time()
        return self._get_obs()

    def _get_obs(self):
        now = time.time()
        dt = max(1e-4, now - self._last_t)
        self._last_t = now

        self.pitch_rate = self._read_pitch_rate_rad()
        pitch_acc = self._read_pitch_from_acc()
        self.pitch = self.pitch_alpha * (self.pitch + self.pitch_rate * dt) + (
            1.0 - self.pitch_alpha
        ) * pitch_acc

        enc_left, enc_right = self.drive.get_encoder_steps()
        steps_avg = 0.5 * (enc_left + enc_right)
        meters = steps_avg * self.encoder_step_to_m
        self.x_dot_est = (meters - self.x_est) / dt
        self.x_est = meters

        return np.array(
            [self.pitch, self.pitch_rate, self.x_est, self.x_dot_est], dtype=np.float32
        )

    def step(self, action):
        cmd = float(np.clip(action[0], -1.0, 1.0))
        pwm = min(abs(cmd) * self.motor_scale, 1.0)
        if cmd >= 0:
            self.drive.forward(pwm)
        else:
            self.drive.backward(pwm)

        time.sleep(self.loop_dt)
        obs = self._get_obs()
        done = abs(obs[0]) > self.fall_angle_rad
        reward = 0.0
        return obs, reward, done

    def close(self):
        self.drive.close()

    def calibrate_imu(self, samples=500, sample_dt=0.005):
        """
        Estimate IMU biases while the robot is stationary.
        Returns a dict with biases that can be reused in runtime args.
        """
        gyro_dps = []
        accel_pitch = []
        for _ in range(samples):
            gx, _, _ = self.imu.read_gyro()
            ax, _, az = self.imu.read_acc()
            gyro_dps.append(gx / self.gyro_lsb_per_dps)
            accel_pitch.append(np.arctan2(ax, az + 1e-6))
            time.sleep(sample_dt)

        self.gyro_bias_dps = float(np.mean(gyro_dps))
        self.accel_pitch_bias_rad = float(np.mean(accel_pitch))
        return {
            "gyro_bias_dps": self.gyro_bias_dps,
            "accel_pitch_bias_rad": self.accel_pitch_bias_rad,
        }

