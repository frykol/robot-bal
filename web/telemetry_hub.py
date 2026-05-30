"""Thread-safe sensor logging and chart buffer for web_balance."""

from __future__ import annotations

import csv
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path


class TelemetryHub:
    def __init__(self, logs_dir=None, max_chart_points=600):
        self.logs_dir = Path(logs_dir or Path("logs") / "sensor_recordings")
        self.max_chart_points = int(max_chart_points)
        self._lock = threading.Lock()
        self.recording = False
        self._record_path: Path | None = None
        self._csv_file = None
        self._csv_writer = None
        self._t0: float | None = None
        self._chart: deque[dict] = deque(maxlen=self.max_chart_points)

    def set_recording(self, enabled: bool) -> dict:
        enabled = bool(enabled)
        with self._lock:
            if enabled and not self.recording:
                self._start_recording_locked()
            elif not enabled and self.recording:
                self._stop_recording_locked()
            return self.status_locked()

    def status(self) -> dict:
        with self._lock:
            return self.status_locked()

    def status_locked(self) -> dict:
        return {
            "recording": self.recording,
            "path": str(self._record_path) if self._record_path else None,
            "points": len(self._chart),
        }

    def _start_recording_locked(self):
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self._record_path = self.logs_dir / f"sensors_{stamp}.csv"
        self._csv_file = open(self._record_path, "w", newline="", encoding="utf-8")
        self._csv_writer = None
        self._t0 = None
        self._chart.clear()
        self.recording = True
        print(f"Sensor recording started: {self._record_path}")

    def _stop_recording_locked(self):
        self.recording = False
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None
        self._csv_writer = None
        path = self._record_path
        self._record_path = None
        self._t0 = None
        if path is not None:
            print(f"Sensor recording stopped: {path}")

    def push(self, sample: dict):
        """sample from RaspberryBalanceRuntime.read_sensor_snapshot()."""
        with self._lock:
            if not self.recording:
                return

            t = float(sample["t"])
            if self._t0 is None:
                self._t0 = t
            t_rel = t - self._t0

            imus = sample.get("imus") or []
            enc = sample.get("enc") or [0, 0]

            row_for_chart = {
                "t_rel": t_rel,
                "imus": imus,
                "enc": [int(enc[0]), int(enc[1])],
            }
            self._chart.append(row_for_chart)

            if self._csv_writer is None:
                header = ["t_unix", "t_rel_s", "enc_m1", "enc_m2"]
                for i, imu in enumerate(imus):
                    bus = imu.get("bus_id", i)
                    prefix = f"bus{bus}"
                    header.extend(
                        [
                            f"{prefix}_ax",
                            f"{prefix}_ay",
                            f"{prefix}_az",
                            f"{prefix}_gx",
                            f"{prefix}_gy",
                            f"{prefix}_gz",
                        ]
                    )
                self._csv_writer = csv.writer(self._csv_file)
                self._csv_writer.writerow(header)

            row = [f"{t:.6f}", f"{t_rel:.6f}", int(enc[0]), int(enc[1])]
            for imu in imus:
                acc = imu["acc"]
                gyro = imu["gyro"]
                row.extend(
                    [
                        int(acc[0]),
                        int(acc[1]),
                        int(acc[2]),
                        int(gyro[0]),
                        int(gyro[1]),
                        int(gyro[2]),
                    ]
                )
            self._csv_writer.writerow(row)

    def chart_payload(self) -> dict:
        with self._lock:
            if not self._chart:
                return {"type": "telemetry", "recording": self.recording, "series": None}

            points = list(self._chart)
            t_rel = [p["t_rel"] for p in points]
            n_imu = max((len(p["imus"]) for p in points), default=0)

            acc = []
            gyro = []
            for i in range(n_imu):
                acc.append(
                    {
                        "x": [p["imus"][i]["acc"][0] for p in points],
                        "y": [p["imus"][i]["acc"][1] for p in points],
                        "z": [p["imus"][i]["acc"][2] for p in points],
                        "bus_id": points[-1]["imus"][i].get("bus_id", i),
                    }
                )
                gyro.append(
                    {
                        "x": [p["imus"][i]["gyro"][0] for p in points],
                        "y": [p["imus"][i]["gyro"][1] for p in points],
                        "z": [p["imus"][i]["gyro"][2] for p in points],
                        "bus_id": points[-1]["imus"][i].get("bus_id", i),
                    }
                )

            enc_m1 = [p["enc"][0] for p in points]
            enc_m2 = [p["enc"][1] for p in points]

            return {
                "type": "telemetry",
                "recording": self.recording,
                "path": str(self._record_path) if self._record_path else None,
                "series": {
                    "t_rel": t_rel,
                    "acc": acc,
                    "gyro": gyro,
                    "enc": {"m1": enc_m1, "m2": enc_m2},
                },
            }
