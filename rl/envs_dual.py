"""Inverted pendulum sim with 2D action: direction [-1,1] × motor power scale [0,1]."""

import numpy as np

from rl.envs import InvertedPendulumEnv

FALL_PENALTY_MAX = 100.0


def parse_dual_action(action, min_power=0.2):
    """
    SAC actor outputs tanh in [-1, 1] per dimension.
    dim0 → direction; dim1 → power scale in [0, 1].
    """
    direction = float(np.clip(action[0], -1.0, 1.0))
    power = float(np.clip((action[1] + 1.0) * 0.5, 0.0, 1.0))
    if min_power > 0.0:
        power = max(float(min_power), power)
    return direction, power


def fall_reward_at_step(step_index, max_episode_steps, penalty_max=FALL_PENALTY_MAX):
    """
    Kara tylko przy upadku. Im później w epizodzie, tym mniejsza (bliżej 0).
    step_index: numer kroku 1..max_episode_steps w momencie upadku.
    """
    if max_episode_steps <= 0:
        return 0.0
    frac_remaining = (max_episode_steps - step_index) / float(max_episode_steps)
    frac_remaining = float(np.clip(frac_remaining, 0.0, 1.0))
    return -float(penalty_max) * frac_remaining


class DualActionPendulumEnv(InvertedPendulumEnv):
    """
    Action: [direction, power_scale] → force = direction * power_scale * F_max.

    Reward:
      - żywy krok: 0
      - upadek: -fall_penalty_max * (T - t) / T  (wcześniej ≈ -100, pod koniec ep. → 0)
      - cały epizod bez upadku: suma = 0  → „już umie”
    """

    def __init__(
        self,
        max_episode_steps=5000,
        fall_penalty_max=FALL_PENALTY_MAX,
        min_motor_power=0.2,
        init_angle_deg=10.0,
        **kwargs,
    ):
        self.max_episode_steps = int(max_episode_steps)
        self.fall_penalty_max = float(fall_penalty_max)
        self.min_motor_power = float(min_motor_power)
        self.init_angle_deg = float(init_angle_deg)
        self._step_count = 0
        super().__init__(**kwargs)
        self.act_dim = 2

    def reset(self):
        self._resample_dynamics()
        self._step_count = 0
        lo = -self.init_angle_deg
        hi = self.init_angle_deg
        self.state = np.array(
            [0.0, 0.0, np.radians(np.random.uniform(lo, hi)), 0.0], dtype=np.float64
        )
        if np.random.rand() < 0.05:
            self.state[3] += np.random.uniform(-0.3, 0.3)
        self._last_x_ddot = 0.0
        self._last_theta_ddot = 0.0
        return self._to_obs(self.state)

    def step(self, action):
        self._step_count += 1
        direction, power = parse_dual_action(action, min_power=self.min_motor_power)
        force = direction * power * self.force_max

        x, x_dot, theta, theta_dot = self.state
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
        self._last_theta_ddot = float(theta_ddot)
        done = bool(abs(theta) > self.theta_max)
        if done:
            reward = fall_reward_at_step(
                self._step_count, self.max_episode_steps, self.fall_penalty_max
            )
        else:
            reward = 0.0

        return self._to_obs(self.state, x_ddot=self._last_x_ddot), reward, done
