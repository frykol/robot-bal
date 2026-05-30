"""Inverted pendulum sim with 2D action: direction [-1,1] × motor power scale [0, 1]."""

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


def init_angle_deg_for_episode(
    episode,
    easy_deg,
    final_deg,
    curriculum_episodes,
):
    """Liniowy curriculum: ep 1 → easy_deg, ep >= curriculum_episodes → final_deg."""
    easy = float(easy_deg)
    final = float(final_deg)
    if curriculum_episodes <= 0:
        return final
    if episode <= 1:
        return easy
    if episode >= curriculum_episodes:
        return final
    t = (float(episode) - 1.0) / float(curriculum_episodes - 1)
    return easy + t * (final - easy)


class DualActionPendulumEnv(InvertedPendulumEnv):
    """
    Action: [direction, power_scale] → force = direction * power_scale * F_max.

    Reward (żywy krok):
      - alive_reward_per_step
      - minus shaping za |theta|, |theta_dot|
    Upadek: -fall_penalty_max * (T - t) / T
    """

    def __init__(
        self,
        max_episode_steps=5000,
        fall_penalty_max=FALL_PENALTY_MAX,
        min_motor_power=0.2,
        init_angle_deg=10.0,
        init_angle_easy_deg=3.0,
        curriculum_episodes=400,
        alive_reward_per_step=0.02,
        angle_reward_scale=0.03,
        angular_rate_reward_scale=0.02,
        max_pitch_rate_rad_s=5.0,
        **kwargs,
    ):
        self.max_episode_steps = int(max_episode_steps)
        self.fall_penalty_max = float(fall_penalty_max)
        self.min_motor_power = float(min_motor_power)
        self.init_angle_final_deg = float(init_angle_deg)
        self.init_angle_easy_deg = float(init_angle_easy_deg)
        self.curriculum_episodes = int(curriculum_episodes)
        self._init_angle_deg = self.init_angle_easy_deg
        self.alive_reward_per_step = float(alive_reward_per_step)
        self.angle_reward_scale = float(angle_reward_scale)
        self.angular_rate_reward_scale = float(angular_rate_reward_scale)
        self.max_pitch_rate_rad_s = float(max(1e-6, max_pitch_rate_rad_s))
        self._step_count = 0
        super().__init__(**kwargs)
        self.act_dim = 2

    def set_curriculum_episode(self, episode, total_episodes=None):
        """Wywołaj przed reset() w danym epizodzie treningu."""
        _ = total_episodes
        self._init_angle_deg = init_angle_deg_for_episode(
            episode,
            self.init_angle_easy_deg,
            self.init_angle_final_deg,
            self.curriculum_episodes,
        )

    def _shaping_reward(self, theta, theta_dot):
        reward = self.alive_reward_per_step
        reward -= self.angle_reward_scale * (abs(theta) / self.theta_max)
        rate_term = min(1.0, abs(theta_dot) / self.max_pitch_rate_rad_s)
        reward -= self.angular_rate_reward_scale * rate_term
        return float(reward)

    def reset(self):
        self._resample_dynamics()
        self._reset_actuation()
        self._step_count = 0
        lo = -self._init_angle_deg
        hi = self._init_angle_deg
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
        desired_force = direction * power * self.force_max
        force = self._actuate_force(desired_force)
        theta, theta_dot = self._integrate_dynamics(force)
        done = bool(abs(theta) > self.theta_max)
        if done:
            reward = fall_reward_at_step(
                self._step_count, self.max_episode_steps, self.fall_penalty_max
            )
        else:
            reward = self._shaping_reward(theta, theta_dot)

        return self._to_obs(self.state, x_ddot=self._last_x_ddot), reward, done
