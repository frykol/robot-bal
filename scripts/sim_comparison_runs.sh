#!/usr/bin/env bash
# Porównanie runów SAC w symulacji (uruchom z katalogu repo: bash scripts/sim_comparison_runs.sh)
# Każdy run zapisuje się do artifacts/runs/<nazwa>/ (wykres + run_config.json + actor_best.pt)

set -euo pipefail
cd "$(dirname "$0")/.."

EPISODES="${EPISODES:-400}"
COMMON=(--episodes "$EPISODES" --hidden-dim 32 --lr 3e-4 --save-every 50)

echo "=== 1) Bazowy: model mas (silniki na dole), body 14 cm ==="
python train_sim.py --run-name cmp01_baseline_massmodel "${COMMON[@]}"

echo "=== 2) Niski środek ciężkości: niska bateria w korpusie ==="
python train_sim.py --run-name cmp02_low_com --body-height-m 0.12 --battery-z-m 0.03 "${COMMON[@]}"

echo "=== 3) Wysoki środek ciężkości: wysoka bateria / wysoki korpus ==="
python train_sim.py --run-name cmp03_high_com --body-height-m 0.18 --battery-z-m 0.12 --rpi-z-m 0.15 "${COMMON[@]}"

echo "=== 4) Stary ręczny COM 0.11 m (bez modelu mas) ==="
python train_sim.py --run-name cmp04_manual_com011 --manual-com-height --com-height-m 0.11 "${COMMON[@]}"

echo "=== 5) Słabsze silniki (mniejszy moment → mniejsza F_max) ==="
python train_sim.py --run-name cmp05_low_torque --motor-torque-nm 0.20 "${COMMON[@]}"

echo "=== 6) Mocniejsze silniki (wyższy limit momentu) ==="
python train_sim.py --run-name cmp06_high_torque --motor-torque-nm 0.50 --force-max 15 "${COMMON[@]}"

echo "=== 7) Wolniejsze uczenie lr=1e-4 ==="
python train_sim.py --run-name cmp07_lr1e4 --lr 1e-4 "${COMMON[@]}"

echo "=== 8) Szybsze uczenie lr=1e-3 ==="
python train_sim.py --run-name cmp08_lr1e3 --lr 1e-3 "${COMMON[@]}"

echo "Gotowe. Porównaj: artifacts/runs/cmp*/learning_curve.png i run_config.json"
