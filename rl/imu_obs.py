"""IMU observation helpers (BMI160-style raw LSB per sensor)."""

import numpy as np

OBS_MODE_PROCESSED4 = "processed4"
OBS_MODE_IMU_RAW6 = "imu_raw6"
OBS_MODE_IMU_RAW12 = "imu_raw12"

RAW_IMU_CHANNELS = 6  # ax, ay, az, gx, gy, gz per BMI160

# BMI160 ±2g → 16384 LSB / g; ±250 °/s → 131.2 LSB/(°/s) (must match hardware driver).
ACC_LSB_PER_G = 16384.0
ACC_LSB_PER_MS2 = ACC_LSB_PER_G / 9.81
GYR_LSB_PER_DPS = 131.2

DEFAULT_IMU_BUS_IDS = (1, 3)


def obs_dim_for_mode(obs_mode):
    if obs_mode == OBS_MODE_IMU_RAW12:
        return 2 * RAW_IMU_CHANNELS
    if obs_mode == OBS_MODE_IMU_RAW6:
        return RAW_IMU_CHANNELS
    return 4


def is_raw_imu_mode(obs_mode):
    return obs_mode in (OBS_MODE_IMU_RAW6, OBS_MODE_IMU_RAW12)


def normalize_raw_imu_obs(obs):
    """
    Scale simulated/real BMI160 LSB to network-friendly units:
    acc → g, gyro → °/s (matches typical fusion code).
  """
    out = np.asarray(obs, dtype=np.float32).copy()
    for i in range(0, out.size, RAW_IMU_CHANNELS):
        out[i : i + 3] /= ACC_LSB_PER_G
        out[i + 3 : i + 6] /= GYR_LSB_PER_DPS
    return out


def rad_s_to_gyro_lsb(rate_rad_s):
    return float(np.rad2deg(rate_rad_s) * GYR_LSB_PER_DPS)


def ms2_to_acc_lsb(accel_ms2):
    return float(accel_ms2 * ACC_LSB_PER_MS2)


def gyro_lsb_to_rad_s(lsb):
    return float(np.deg2rad(lsb / GYR_LSB_PER_DPS))


def pitch_rad_from_imu_slice(obs_slice, accel_pitch_bias_rad=0.0):
    ax, _, az = float(obs_slice[0]), float(obs_slice[1]), float(obs_slice[2])
    return float(np.arctan2(-ax, -az + 1e-6)) - float(accel_pitch_bias_rad)


def pitch_rate_rad_from_imu_slice(obs_slice, gyro_bias_dps=0.0):
    gx = float(obs_slice[3])
    return gyro_lsb_to_rad_s(gx) - float(np.deg2rad(gyro_bias_dps))


def pitch_rad_from_raw_obs(obs, accel_pitch_bias_rad=0.0, imu_index=0):
    start = imu_index * RAW_IMU_CHANNELS
    sl = obs[start : start + RAW_IMU_CHANNELS]
    return pitch_rad_from_imu_slice(sl, accel_pitch_bias_rad)


def pitch_rate_rad_from_raw_obs(obs, gyro_bias_dps=0.0, imu_index=0):
    start = imu_index * RAW_IMU_CHANNELS
    sl = obs[start : start + RAW_IMU_CHANNELS]
    return pitch_rate_rad_from_imu_slice(sl, gyro_bias_dps)


def simulate_imu_raw_reading(
    theta,
    theta_dot,
    x_ddot,
    gyro_bias_lsb,
    accel_bias_lsb,
    noise_std,
    imu_height_m=0.0,
    theta_ddot=0.0,
):
    """
    Synthetic BMI160 reading for pitch-about-X mount (same convention as Raspberry runtime).

    Body frame: X = pitch axis, Z = along body toward top when upright (az < 0 at rest).
    imu_height_m: distance from wheel axle to sensor along the body (top wall ≈ body_height).
    Adds rigid-body tangential / centripetal terms at that height.
    """
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    h = float(imu_height_m)
    td = float(theta_ddot)
    td2 = float(theta_dot) ** 2

    ax_ms2 = x_ddot + 9.81 * sin_t + h * td
    ay_ms2 = 0.0
    az_ms2 = -9.81 * cos_t - h * td2

    gx_lsb = rad_s_to_gyro_lsb(theta_dot)
    gy_lsb = 0.0
    gz_lsb = 0.0

    ax = ms2_to_acc_lsb(ax_ms2) + accel_bias_lsb[0]
    ay = ms2_to_acc_lsb(ay_ms2) + accel_bias_lsb[1]
    az = ms2_to_acc_lsb(az_ms2) + accel_bias_lsb[2]

    if noise_std > 0.0:
        ax += np.random.randn() * noise_std
        ay += np.random.randn() * noise_std
        az += np.random.randn() * noise_std
        gx_lsb += np.random.randn() * noise_std
        gy_lsb += np.random.randn() * noise_std
        gz_lsb += np.random.randn() * noise_std

    gx_lsb += gyro_bias_lsb[0]
    gy_lsb += gyro_bias_lsb[1]
    gz_lsb += gyro_bias_lsb[2]

    return np.array([ax, ay, az, gx_lsb, gy_lsb, gz_lsb], dtype=np.float32)


def simulate_dual_imu_raw_reading(
    theta,
    theta_dot,
    x_ddot,
    gyro_bias_lsb_pair,
    accel_bias_lsb_pair,
    noise_std,
    theta_offsets_rad,
    imu_heights_m,
    theta_ddot=0.0,
):
    """
    Two BMI160 on the top wall (e.g. I2C bus 1 and 3): independent bias/noise,
    slightly different mount height / small pitch offset per board.
    """
    imu0 = simulate_imu_raw_reading(
        theta + float(theta_offsets_rad[0]),
        theta_dot,
        x_ddot,
        gyro_bias_lsb_pair[0],
        accel_bias_lsb_pair[0],
        noise_std,
        imu_height_m=float(imu_heights_m[0]),
        theta_ddot=theta_ddot,
    )
    imu1 = simulate_imu_raw_reading(
        theta + float(theta_offsets_rad[1]),
        theta_dot,
        x_ddot,
        gyro_bias_lsb_pair[1],
        accel_bias_lsb_pair[1],
        noise_std,
        imu_height_m=float(imu_heights_m[1]),
        theta_ddot=theta_ddot,
    )
    return np.concatenate([imu0, imu1]).astype(np.float32)


def load_imu_calibration(calibration, n_imus=2):
    """Return list of dicts with gyro_bias_dps and accel_pitch_bias_rad per IMU."""
    default = {"gyro_bias_dps": 0.0, "accel_pitch_bias_rad": 0.0}
    if calibration.get("imensors"):
        sensors = [
            {
                "gyro_bias_dps": float(s.get("gyro_bias_dps", 0.0)),
                "accel_pitch_bias_rad": float(s.get("accel_pitch_bias_rad", 0.0)),
            }
            for s in calibration["imensors"]
        ]
        while len(sensors) < n_imus:
            sensors.append(dict(default))
        return sensors[:n_imus]
    return [
        {
            "gyro_bias_dps": float(calibration.get("gyro_bias_dps", 0.0)),
            "accel_pitch_bias_rad": float(calibration.get("accel_pitch_bias_rad", 0.0)),
        }
        for _ in range(n_imus)
    ]


def features_from_obs(obs, obs_mode, accel_pitch_bias_rad=0.0, gyro_bias_dps=0.0, calibration=None):
    """Map policy observation to (pitch, pitch_rate, x, x_dot) for reward / logging."""
    if obs_mode == OBS_MODE_IMU_RAW12:
        sensors = load_imu_calibration(calibration or {}, n_imus=2)
        p0 = pitch_rad_from_raw_obs(obs, sensors[0]["accel_pitch_bias_rad"], imu_index=0)
        p1 = pitch_rad_from_raw_obs(obs, sensors[1]["accel_pitch_bias_rad"], imu_index=1)
        r0 = pitch_rate_rad_from_raw_obs(obs, sensors[0]["gyro_bias_dps"], imu_index=0)
        r1 = pitch_rate_rad_from_raw_obs(obs, sensors[1]["gyro_bias_dps"], imu_index=1)
        return 0.5 * (p0 + p1), 0.5 * (r0 + r1), 0.0, 0.0
    if obs_mode == OBS_MODE_IMU_RAW6:
        return (
            pitch_rad_from_raw_obs(obs, accel_pitch_bias_rad, imu_index=0),
            pitch_rate_rad_from_raw_obs(obs, gyro_bias_dps, imu_index=0),
            0.0,
            0.0,
        )
    return float(obs[0]), float(obs[1]), float(obs[2]), float(obs[3])


def pitch_rad_for_safety(obs, obs_mode, accel_pitch_bias_rad=0.0, calibration=None):
    if obs_mode == OBS_MODE_IMU_RAW12:
        sensors = load_imu_calibration(calibration or {}, n_imus=2)
        p0 = pitch_rad_from_raw_obs(obs, sensors[0]["accel_pitch_bias_rad"], imu_index=0)
        p1 = pitch_rad_from_raw_obs(obs, sensors[1]["accel_pitch_bias_rad"], imu_index=1)
        return 0.5 * (p0 + p1)
    if is_raw_imu_mode(obs_mode):
        return pitch_rad_from_raw_obs(obs, accel_pitch_bias_rad, imu_index=0)
    return float(obs[0])
