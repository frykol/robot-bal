# robot-bal

This repository now includes a full SAC pipeline for a balancing robot:

- fast pretraining in a virtual inverted pendulum environment,
- reusable SAC agent implementation,
- Raspberry Pi runtime using existing hardware abstraction (`BMI160`, `DriveModule`),
- model export for deployment.

## Files added

- `rl/sac.py` - SAC agent, actor, critics, replay buffer.
- `rl/envs.py` - symulacja treningowa (`InvertedPendulumEnv`; syntetyczne IMU dla `imu_raw*`)
- `rl/pi_runtime.py` - runtime na Raspberry Pi (prawdziwe BMI160 + silniki)
- `train_sim.py` - simulation pretraining (SAC).
- `train_sim_ppo.py` - simulation pretraining (PPO, clipped objective + GAE).
- `rl/ppo.py` - PPO agent (actor-critic, on-policy rollouts).
- `train_sim_dual.py` - dual action (direction × motor power), sparse reward, γ=0.999, LR decay.
- `rl/envs_dual.py` - `DualActionPendulumEnv` (2D action, reward 0 / -100 on fall).
- `export_actor.py` - TorchScript export for deterministic deployment.
- `run_policy_pi.py` - runs learned policy on Raspberry Pi.
- `calibrate_pi.py` - stationary IMU calibration on Raspberry Pi.
- `online_train_pi.py` - online SAC fine-tuning directly on Raspberry Pi.

## Mass model used in simulation

Hardware masses (default):

- motors: `2 x 160g` at the wheel axle (bottom),
- body: Raspberry `55g` + case `466g` + battery `250g` → `M = 0.771 kg`,
- `m = 0.320 kg` at the axle.

By default `train_sim.py` uses `rl/robot_mass_model.py`: component heights along the
body box give **body COM height `l`** and **drive force limit** from motor torque
`F ≤ n_motors * τ / r_wheel` (capped by `--force-max`).

Tune geometry: `--body-height-m`, `--battery-z-m`, `--motor-torque-nm`, `--wheel-radius-m`.
Legacy fixed COM: `--manual-com-height --com-height-m 0.11`.

Batch comparison runs:

```bash
bash scripts/sim_comparison_runs.sh
```

## Train in virtual environment

```bash
python train_sim.py --episodes 1000 --max-steps 1000
```

PPO baseline (scalar action, `InvertedPendulumEnv`):

```bash
python train_sim_ppo.py --run-name ppo_baseline --episodes 500 --rollout-steps 2048
```

PPO with same reward/action as SAC dual (`DualActionPendulumEnv`):

```bash
python train_sim_ppo.py --dual-action --run-name ppo_cmp_v6_dual \
  --obs-mode imu_raw12 --hidden-dims 48 24 --episodes 800 --gamma 0.999
```

Outputs: `artifacts/runs/<run_name>/actor_sim_ppo.pt`, `actor_best_ppo.pt`, `learning_curve.png`.

Compare multiple runs in separate folders:

```bash
python train_sim.py --run-name h16_baseline --hidden-dim 16 --com-height-m 0.11
python train_sim.py --run-name h64_low_com --hidden-dim 64 --com-height-m 0.09
python train_sim.py --run-name h32_lr1e4 --hidden-dim 32 --lr 1e-4
python train_sim.py --auto-run-name --hidden-dim 32 --com-height-m 0.10 --lr 3e-4
```

Each run directory (under `artifacts/runs/` by default) contains:
`actor_sim.pt`, `actor_best.pt`, `learning_curve.png`, `checkpoints/`, `run_config.json`.

This saves actor weights to `artifacts/actor_sim.pt` when no `--run-dir` / `--run-name` is used.
It also saves a learning plot to `artifacts/learning_curve.png`.
Best rolling-average checkpoint is saved to `artifacts/actor_best.pt`.
Periodic checkpoints are saved to `artifacts/checkpoints/`.

Optional plotting args:

```bash
python train_sim.py --plot-path artifacts/my_curve.png --rolling-window 100
```

Set training fall angle and randomization behavior:

```bash
python train_sim.py --train-fall-angle-deg 25
python train_sim.py --no-domain-randomization
```

Dual-action sparse-reward training (laptop only):

```bash
python train_sim_dual.py --run-name dual_sparse --hidden-dim 32 --device cpu
# action: [direction in [-1,1], power scale in [0,1]] → force = direction * scale * F_max
# reward: 0 alive; fall -100*(T-t)/T; episode sum 0 = full episode without fall
```

## Export policy

```bash
python export_actor.py --weights-path artifacts/actor_sim.pt --output-path artifacts/policy.ts
```

## Run on Raspberry Pi

```bash
python run_policy_pi.py --actor-path artifacts/actor_sim.pt --profile safe
```

Important before first real run:

- tune `gyro` and `encoder` scaling in `rl/envs.py` (`_read_pitch_rate_rad`, `_get_obs`),
- verify motor polarity and safe PWM range (`motor_scale`),
- keep an external safety stop during initial tests.

### Calibration and safety

First run IMU calibration while robot is not moving:

```bash
python calibrate_pi.py --output-path artifacts/pi_calibration.json
```

Runtime now supports:

- motor profiles: `safe`, `normal`, `aggressive`,
- safety cutoff: `--tilt-limit-deg` (default 25),
- CSV logs: `--log-path logs/run_latest.csv`.

Example:

```bash
python run_policy_pi.py \
  --actor-path artifacts/actor_sim.pt \
  --profile safe \
  --tilt-limit-deg 20 \
  --log-path logs/run_001.csv
```

### Online fine-tuning on Raspberry Pi

Start from a pretrained actor and continue learning on robot:

```bash
python online_train_pi.py \
  --actor-path artifacts/actor_best.pt \
  --profile safe \
  --tilt-limit-deg 15
```

Resume from previous online checkpoint:

```bash
python online_train_pi.py --resume-checkpoint artifacts/sac_online_latest.pt
```

## Suggested sim-to-real path

1. Train in `InvertedPendulumEnv` until stable average rewards.
2. Run low-power closed-loop tests on Raspberry with deterministic policy.
3. Log real trajectories (`pitch`, `pitch_rate`, encoder speed, action).
4. Update simulation parameters/noise from logs and retrain.
5. Repeat until robust.

