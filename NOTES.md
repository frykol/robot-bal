# robot-bal — notatki kontekstu (RL + Pi)

Skrót decyzji i wniosków z treningu / deployu. Użyj `@NOTES.md` w nowym czacie Cursora na laptopie albo po `git pull`.

---

## Projekt

- **Cel:** balans wózka na Raspberry Pi, trening w symulacji (SAC).
- **Algorytm:** SAC (off-policy, replay) — nie PPO.
- **Akcja dual:** `[kierunek ∈ [-1,1], moc ∈ [0,1]]` → siła = kierunek × moc × F_max (`train_sim_dual.py`, `rl/envs_dual.py`).
- **Obserwacja na Pi / docelowo:** `imu_raw12` (2× BMI160, bus **1** i **3**).
- **Artefakty runów:** `artifacts/runs/<run_name>/` — `actor_best.pt`, `actor_sim_dual.pt`, `run_config.json`, wykresy.

---

## Najlepsze checkpointy (stan na ~2026-05)

| Run | Sieć | Uwagi |
|-----|------|--------|
| **`dual_h32_16_v6/actor_best.pt`** | 48→24, relu | **Lider w sim** — eval deterministyczny ±8°: pełne 5000 kroków. Deploy / test na Pi. |
| `dual_h32_16_v4/actor_best.pt` | 32→16, relu | Szczyt ~ep 386 (~3540 kroków rolling, init ~±9.8°). Potem regres przy ±10°. |
| `dual_h32_16_v7/actor_best.pt` | 32→16 | Słabszy niż v6; backup. |
| v3 | — | Nie uczył się (~90 kroków, reward ~−102). |

**Zawsze bierz `actor_best.pt`**, nie ostatni `actor_sim_dual.pt` z końcówki runu (tam często regres).

---

## Trening — co działa / co psuje

### Działa (v4 do ~380, v6)

- `obs_mode imu_raw12`, `hidden_dims 32 16` lub **48 24** (v6 lepszy).
- `hidden_activation relu`.
- Shaped reward + curriculum kąta startu (±3° → finał).
- `dt=0.002`, `max_steps=5000`, `gamma=0.999`.
- Best checkpoint po **rolling mean episode steps** (50 ep.), nie po reward.

### Psuje (powtarzalny wzorzec „V” na krzywej)

1. **LR rosnący** (`3e-4 → 1e-3`) — po szczycie polityka się rozjeżdża. **Używać LR malejący:** `--lr-start 3e-4 --lr-end 1e-4`.
2. **Curriculum za szybki do ±10°** — regres rolling od ~ep 400 (v4). **Cap finału:** `--init-angle-deg 8`, `--curriculum-episodes 600–800`.
3. **Alpha → 0.05 za wcześnie** — mało eksploracji. Lepiej `--alpha-end 0.12–0.15`, `--alpha-decay-episodes 400+`.
4. **Długi trening po best** — końcówka runu psuje `actor_sim_dual`; `actor_best` zostaje zamrożony.

### Propozycja następnego runu (v8+)

```bash
python train_sim_dual.py \
  --run-name dual_h48_24_v8 \
  --hidden-dims 48 24 \
  --hidden-activation relu \
  --obs-mode imu_raw12 \
  --episodes 800 \
  --no-live-plot \
  --lr-start 3e-4 --lr-end 1e-4 \
  --init-angle-deg 8 \
  --init-angle-easy-deg 3 \
  --curriculum-episodes 600 \
  --alpha-start 0.35 --alpha-end 0.15 --alpha-decay-episodes 400 \
  --batch-size 128
```

Opcjonalnie później w kodzie: `--resume-from actor_best`, `--update-every N`, early stop gdy brak nowego best przez N ep.

---

## Deploy na Raspberry Pi

### Wymagania

- **`--obs-mode`** musi pasować do checkpointu (v6/v7 → `imu_raw12`).
- **Kalibracja:** `python calibrate_pi.py` → `artifacts/pi_calibration.json`.
- **Profil mocy:** zaczynać od `--profile safe` (motor_scale 0.30).

### Uruchomienie polityki

```bash
python run_policy_pi.py \
  --actor-path artifacts/runs/dual_h32_16_v6/actor_best.pt \
  --obs-mode imu_raw12 \
  --profile safe \
  --calibration-path artifacts/pi_calibration.json
```

### Web: AI + Manual

```bash
python web_balance.py \
  --actor-path artifacts/runs/dual_h32_16_v6/actor_best.pt \
  --obs-mode imu_raw12 \
  --profile safe \
  --calibration-path artifacts/pi_calibration.json
```

- Domyślnie **AI** balansuje; przyciski **AI / Manual** na `http://<ip-pi>:8000`.
- **Nagraj dane** — CSV z surowymi LSB acc/gyro (oba BMI160) + enkodery; 3 wykresy na żywo podczas nagrywania (`logs/sensor_recordings/`).
- Stary test tylko suwakiem: `python web_test.py`.

### Sim → real — typowe przyczyny porażki

- Zły `obs_mode` vs checkpoint.
- Brak kalibracji IMU / zła oś / znak.
- Za agresywny PWM (`--profile safe`).
- Dual action: `min_motor_power=0.2` w sim — na Pi to też w `pi_runtime`.
- Encodery / `encoder_step_to_m` — estymacja pozycji może się rozjeżdżać.

---

## Hardware (`hardware/drive_module.py`)

- **Motor1:** GPIO 20, 21.
- **Motor2:** GPIO **24, 23** (forward=24, backward=23) — zamiana 23↔24 naprawia niespójny kierunek kół.
- **`forward()` / `backward()`** — odwrócone względem starych pinów (celowo pod okablowanie); oba silniki dostają ten sam kierunek logiczny.
- Enkodery: M1 (5, 6), M2 (13, 19).

`BMI160` importuje `rl.imu_obs.GYR_LSB_PER_DPS` — na Pi potrzebny **numpy** (nie torch) przy `web_test`.

---

## Branch / repo

- Główny merge: **`feat/model` → `main`** (SAC, dual env, Pi runtime, run artifacts, web).
- Merge zawiera dużo `artifacts/` — rozważyć `.gitignore` na checkpointy w przyszłości.

---

## Przydatne komendy

```bash
# Trening dual (skrót)
python train_sim_dual.py --run-name MY_RUN --hidden-dims 48 24 --hidden-activation relu \
  --obs-mode imu_raw12 --episodes 800 --no-live-plot

# Eksport TorchScript (opcjonalnie)
python export_actor.py --weights artifacts/runs/.../actor_best.pt --output artifacts/policy.pt

# Online fine-tune na Pi (osobny skrypt)
python online_train_pi.py --actor-path artifacts/runs/.../actor_best.pt
```

---

## Metryki w logu treningu

- **steps** — długość epizodu (max 5000 ≈ 10 s przy dt=0.002).
- **Reward** — suma epizodu (sparse + shaping + fall penalty).
- **Avg50 S** — rolling średnia kroków; **to wybiera `actor_best`**.
- Pojedyncze epy 5000 kroków + słabe epy = duża wariancja; patrz na **trend** i **actor_best**, nie ostatni ep.

---

## Otwarte / TODO (opcjonalnie)

- [ ] `--resume-from` w `train_sim_dual.py`
- [ ] `eval_dual.py` — eval checkpointów przy wybranym init±
- [ ] `--update-every N` (mniej update’ów SAC na krok sim)
- [ ] `--realistic-actuation` dopiero po stabilnym ±8° w sim
- [ ] `.gitignore` dla `artifacts/**/*.pt` (zostawić przykładowy run w docs)

---

*Ostatnia aktualizacja notatek: 2026-05-30 — kontekst z sesji treningowej v3–v7, deploy, web_balance, hardware pins.*
