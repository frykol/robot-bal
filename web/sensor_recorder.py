"""
Background sensor CSV logger — WebSocket tylko start/stop, zapis w osobnym wątku.

Pętla robota wrzuca próbki do kolejki (bez dodatkowego I2C); wątek zapisu pisze CSV
i bufor wykresu.
"""

from __future__ import annotations

import csv
import queue
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

CHART_WS_MAX_POINTS = 180


class SensorRecorder:
    def __init__(self, logs_dir=None, max_chart_points=400, sample_queue_size=256):
        self.logs_dir = Path(logs_dir or Path("logs") / "sensor_recordings")
        self.max_chart_points = int(max_chart_points)
        self._sample_queue: queue.Queue = queue.Queue(maxsize=int(sample_queue_size))
        self._cmd_queue: queue.Queue = queue.Queue()
        self._chart: deque[dict] = deque(maxlen=self.max_chart_points)
        self._chart_lock = threading.Lock()
        self.recording = False
        self._record_path: Path | None = None
        self._last_saved_path: str | None = None
        self._csv_file = None
        self._csv_writer = None
        self._t0: float | None = None
        self._worker = threading.Thread(target=self._worker_loop, name="sensor-recorder", daemon=True)
        self._worker.start()

    def start(self) -> dict:
        done = threading.Event()
        self._cmd_queue.put(("start", done))
        done.wait(timeout=10.0)
        return self.status()

    def stop(self) -> dict:
        done = threading.Event()
        self._cmd_queue.put(("stop", done))
        done.wait(timeout=10.0)
        status = self.status()
        if self._last_saved_path:
            status["path"] = self._last_saved_path
        return status

    def offer(self, sample: dict | None) -> None:
        if not self.recording or sample is None:
            return
        try:
            self._sample_queue.put_nowait(sample)
        except queue.Full:
            pass

    def status(self) -> dict:
        path = self._record_path
        if path is None and self._last_saved_path:
            path = Path(self._last_saved_path)
        return {
            "recording": self.recording,
            "path": str(path) if path else None,
            "points": len(self._chart),
            "queue_size": self._sample_queue.qsize(),
        }

    def chart_payload(self) -> dict:
        with self._chart_lock:
            recording = self.recording
            if not recording:
                return {"type": "telemetry", "recording": False, "series": None}
            points = list(self._chart)
            path = str(self._record_path) if self._record_path else None

        if not points:
            return {"type": "telemetry", "recording": True, "path": path, "series": None}

        points = _downsample(points, CHART_WS_MAX_POINTS)
        t_rel = [p["t_rel"] for p in points]
        n_imu = max((len(p["imus"]) for p in points), default=0)

        acc, gyro = [], []
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

        return {
            "type": "telemetry",
            "recording": True,
            "path": path,
            "series": {
                "t_rel": t_rel,
                "acc": acc,
                "gyro": gyro,
                "enc": {
                    "m1": [p["enc"][0] for p in points],
                    "m2": [p["enc"][1] for p in points],
                },
            },
        }

    def _worker_loop(self) -> None:
        while True:
            try:
                cmd = self._cmd_queue.get_nowait()
            except queue.Empty:
                cmd = None

            if cmd is not None:
                kind, done = cmd
                if kind == "start":
                    self._open_recording()
                    done.set()
                elif kind == "stop":
                    self._close_recording()
                    done.set()
                continue

            try:
                sample = self._sample_queue.get(timeout=0.05)
            except queue.Empty:
                continue

            if self.recording:
                self._append_sample(sample)

    def _open_recording(self) -> None:
        self._drain_sample_queue()
        with self._chart_lock:
            self._chart.clear()
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self._record_path = self.logs_dir / f"sensors_{stamp}.csv"
        self._csv_file = open(self._record_path, "w", newline="", encoding="utf-8")
        self._csv_writer = None
        self._t0 = None
        self.recording = True
        print(f"Sensor recording started: {self._record_path}")

    def _close_recording(self) -> None:
        self.recording = False
        deadline = time.time() + 3.0
        while time.time() < deadline:
            try:
                sample = self._sample_queue.get_nowait()
                self._append_sample(sample)
            except queue.Empty:
                if self._sample_queue.empty():
                    break
                time.sleep(0.01)

        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None
        self._csv_writer = None
        if self._record_path is not None:
            self._last_saved_path = str(self._record_path)
            print(f"Sensor recording stopped: {self._record_path}")
        self._record_path = None
        self._t0 = None

    def _drain_sample_queue(self) -> None:
        while True:
            try:
                self._sample_queue.get_nowait()
            except queue.Empty:
                break

    def _append_sample(self, sample: dict) -> None:
        t = float(sample["t"])
        if self._t0 is None:
            self._t0 = t
        t_rel = t - self._t0
        imus = sample.get("imus") or []
        enc = sample.get("enc") or [0, 0]

        row_chart = {"t_rel": t_rel, "imus": imus, "enc": [int(enc[0]), int(enc[1])]}
        with self._chart_lock:
            self._chart.append(row_chart)

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

    # Alias for web_server / telemetry_hub API
    def set_recording(self, enabled: bool) -> dict:
        return self.start() if enabled else self.stop()


def _downsample(points: list[dict], max_points: int) -> list[dict]:
    n = len(points)
    if n <= max_points:
        return points
    step = max(1, n // max_points)
    return points[::step]
