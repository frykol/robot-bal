import asyncio
import json

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse


class _WsManager:
    def __init__(self):
        self._connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self._connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self._connections:
            self._connections.remove(websocket)

    async def broadcast_json(self, payload: dict):
        dead = []
        text = json.dumps(payload)
        for ws in self._connections:
            try:
                await ws.send_text(text)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


def _manual_only_html():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Robot Control</title>
    <style>
        body { background: #111; color: white; font-family: Arial, sans-serif; text-align: center; padding: 30px; }
        h1 { margin-bottom: 40px; }
        .value { font-size: 60px; margin-top: 30px; margin-bottom: 30px; }
        button { font-size: 30px; padding: 15px 40px; border: none; border-radius: 15px; cursor: pointer; }
        .status { margin-top: 30px; font-size: 20px; }
        .slider-container { height: 400px; display: flex; justify-content: center; align-items: center; }
        .vertical-slider {
            width: 350px; height: 60px; transform: rotate(-90deg);
            appearance: none; -webkit-appearance: none; touch-action: none;
        }
    </style>
</head>
<body>
    <h1>Robot Motor Control</h1>
    <div class="slider-container">
        <input id="slider" class="vertical-slider" type="range" min="-1" max="1" step="0.01" value="0">
    </div>
    <div class="value" id="value">0.00</div>
    <button onclick="stopMotor()">STOP</button>
    <div class="status" id="status">Connecting...</div>
    <script>
        const statusDiv = document.getElementById("status");
        const ws = new WebSocket(`ws://${window.location.hostname}:8000/ws`);
        ws.onopen = () => { statusDiv.innerText = "Connected"; };
        ws.onerror = () => { statusDiv.innerText = "WebSocket error"; };
        ws.onclose = () => { statusDiv.innerText = "Disconnected"; };
        const slider = document.getElementById("slider");
        const value = document.getElementById("value");
        slider.oninput = () => {
            const v = Number(slider.value);
            value.innerText = v.toFixed(2);
            if (ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ type: "motor_power", value: v }));
            }
        };
        function stopMotor() {
            slider.value = 0;
            value.innerText = "0.00";
            if (ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ type: "motor_power", value: 0 }));
            }
        }
    </script>
</body>
</html>
"""


def _balance_html(default_mode):
    default_mode = default_mode if default_mode in ("ai", "manual") else "ai"
    return f"""
<!DOCTYPE html>
<html lang="pl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Robot Balance</title>
    <style>
        body {{ background: #111; color: white; font-family: Arial, sans-serif; text-align: center; padding: 24px; }}
        h1 {{ margin-bottom: 16px; }}
        .mode-row {{ display: flex; gap: 16px; justify-content: center; margin: 24px 0; flex-wrap: wrap; }}
        .mode-btn {{
            font-size: 28px; padding: 18px 36px; border: 3px solid #444; border-radius: 16px;
            cursor: pointer; background: #222; color: #ccc; min-width: 140px;
        }}
        .mode-btn.active {{ border-color: #4caf50; background: #1b3a1f; color: #fff; }}
        .mode-btn.manual.active {{ border-color: #ff9800; background: #3a2a10; }}
        .panel {{ opacity: 0.35; pointer-events: none; transition: opacity 0.2s; }}
        .panel.enabled {{ opacity: 1; pointer-events: auto; }}
        .value {{ font-size: 48px; margin: 20px 0; }}
        button.stop {{ font-size: 24px; padding: 12px 32px; border: none; border-radius: 12px; cursor: pointer; }}
        .status {{ margin-top: 24px; font-size: 18px; color: #aaa; }}
        .hint {{ font-size: 16px; color: #888; margin-top: 8px; }}
        .slider-container {{ height: 320px; display: flex; justify-content: center; align-items: center; }}
        .vertical-slider {{
            width: 300px; height: 56px; transform: rotate(-90deg);
            appearance: none; -webkit-appearance: none; touch-action: none;
        }}
        .record-row {{ margin: 20px 0; }}
        .record-btn {{
            font-size: 20px; padding: 12px 28px; border: none; border-radius: 12px;
            cursor: pointer; background: #b71c1c; color: #fff;
        }}
        .record-btn.active {{ background: #2e7d32; }}
        .record-path {{ font-size: 14px; color: #8bc34a; margin-top: 8px; word-break: break-all; }}
        .record-stats {{ font-size: 14px; color: #888; margin-top: 6px; }}
        .record-opt {{ display: block; margin-top: 12px; font-size: 15px; color: #aaa; cursor: pointer; }}
        #chartPanel {{ display: none; max-width: 960px; margin: 24px auto; text-align: left; }}
        #chartPanel.visible {{ display: block; }}
        .chart-box {{
            background: #1a1a1a; border-radius: 12px; padding: 12px; margin-bottom: 16px;
        }}
        .chart-box h3 {{ margin: 0 0 8px 4px; font-size: 16px; color: #bbb; }}
        .chart-canvas {{ width: 100%; height: 220px; }}
    </style>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
</head>
<body>
    <h1>Balans robota</h1>
    <div class="mode-row">
        <button id="btnAi" class="mode-btn" onclick="setMode('ai')">AI</button>
        <button id="btnManual" class="mode-btn manual" onclick="setMode('manual')">Manual</button>
    </div>
    <div class="status" id="modeLabel">Tryb: ...</div>
    <div class="hint" id="hint">Domyślnie robot balansuje sam (AI).</div>

    <div id="manualPanel" class="panel">
        <div class="slider-container">
            <input id="slider" class="vertical-slider" type="range" min="-1" max="1" step="0.01" value="0">
        </div>
        <div class="value" id="value">0.00</div>
        <button class="stop" onclick="stopMotor()">STOP</button>
    </div>

    <div class="status" id="status">Łączenie...</div>

    <div class="record-row">
        <button id="btnRecord" class="record-btn" onclick="toggleRecording()">Nagraj dane</button>
        <label class="record-opt">
            <input type="checkbox" id="chkLiveCharts" onchange="onLiveChartsChange()">
            Podgląd wykresów (wolniejsze)
        </label>
        <div class="record-path" id="recordPath"></div>
        <div class="record-stats" id="recordStats"></div>
    </div>

    <div id="chartPanel">
        <div class="chart-box">
            <h3>Accelerometer (LSB) — xyz</h3>
            <canvas id="chartAcc" class="chart-canvas"></canvas>
        </div>
        <div class="chart-box">
            <h3>Gyroscope (LSB) — xyz</h3>
            <canvas id="chartGyro" class="chart-canvas"></canvas>
        </div>
        <div class="chart-box">
            <h3>Encoders (steps)</h3>
            <canvas id="chartEnc" class="chart-canvas"></canvas>
        </div>
    </div>

    <script>
        let currentMode = "{default_mode}";
        const statusDiv = document.getElementById("status");
        const modeLabel = document.getElementById("modeLabel");
        const manualPanel = document.getElementById("manualPanel");
        const btnAi = document.getElementById("btnAi");
        const btnManual = document.getElementById("btnManual");
        const slider = document.getElementById("slider");
        const value = document.getElementById("value");
        const btnRecord = document.getElementById("btnRecord");
        const recordPath = document.getElementById("recordPath");
        const chartPanel = document.getElementById("chartPanel");

        let recording = false;
        let recordPending = false;
        let liveCharts = false;
        let lastChartDrawMs = 0;
        let chartRafPending = false;
        let pendingSeries = null;
        let charts = {{ acc: null, gyro: null, enc: null }};

        const ws = new WebSocket(`ws://${{window.location.hostname}}:8000/ws`);
        const chkLiveCharts = document.getElementById("chkLiveCharts");
        const recordStats = document.getElementById("recordStats");

        function renderMode() {{
            btnAi.classList.toggle("active", currentMode === "ai");
            btnManual.classList.toggle("active", currentMode === "manual");
            manualPanel.classList.toggle("enabled", currentMode === "manual");
            modeLabel.innerText = currentMode === "ai"
                ? "Tryb: AI (balans automatyczny)"
                : "Tryb: Manual (suwak)";
        }}

        function setMode(mode) {{
            currentMode = mode;
            renderMode();
            if (ws.readyState === WebSocket.OPEN) {{
                ws.send(JSON.stringify({{ type: "set_mode", mode: mode }}));
            }}
            if (mode === "manual") {{
                stopMotor();
            }}
        }}

        function axisColors(prefix) {{
            return {{
                x: prefix + " rgb(244,67,54)",
                y: prefix + " rgb(76,175,80)",
                z: prefix + " rgb(33,150,243)",
            }};
        }}

        function makeChart(canvasId, yTitle) {{
            const ctx = document.getElementById(canvasId).getContext("2d");
            return new Chart(ctx, {{
                type: "line",
                data: {{ labels: [], datasets: [] }},
                options: {{
                    animation: false,
                    responsive: true,
                    maintainAspectRatio: false,
                    interaction: {{ mode: "index", intersect: false }},
                    scales: {{
                        x: {{ title: {{ display: true, text: "t [s]" }} }},
                        y: {{ title: {{ display: true, text: yTitle }} }},
                    }},
                    plugins: {{ legend: {{ labels: {{ boxWidth: 12, font: {{ size: 11 }} }} }} }},
                }},
            }});
        }}

        function ensureCharts() {{
            if (!charts.acc) {{
                charts.acc = makeChart("chartAcc", "acc LSB");
                charts.gyro = makeChart("chartGyro", "gyro LSB");
                charts.enc = makeChart("chartEnc", "steps");
            }}
        }}

        function imuDatasets(imus) {{
            const out = [];
            const pal = axisColors("");
            for (let i = 0; i < imus.length; i++) {{
                const bus = imus[i].bus_id ?? i;
                const data = imus[i];
                for (const axis of ["x", "y", "z"]) {{
                    out.push({{
                        label: `bus${{bus}} ${{axis}}`,
                        data: data[axis],
                        borderColor: pal[axis],
                        borderDash: i > 0 ? [4, 3] : [],
                        pointRadius: 0,
                        borderWidth: 1.2,
                        tension: 0.05,
                    }});
                }}
            }}
            return out;
        }}

        function flushCharts() {{
            chartRafPending = false;
            const series = pendingSeries;
            pendingSeries = null;
            if (!series || !recording || !liveCharts) return;
            const now = performance.now();
            if (now - lastChartDrawMs < 900) return;
            lastChartDrawMs = now;
            ensureCharts();
            const labels = series.t_rel;
            charts.acc.data.labels = labels;
            charts.acc.data.datasets = imuDatasets(series.acc);
            charts.acc.update("none");
            charts.gyro.data.labels = labels;
            charts.gyro.data.datasets = imuDatasets(series.gyro);
            charts.gyro.update("none");
            charts.enc.data.labels = labels;
            charts.enc.data.datasets = [
                {{
                    label: "M1",
                    data: series.enc.m1,
                    borderColor: "rgb(255,152,0)",
                    pointRadius: 0,
                    borderWidth: 1.5,
                }},
                {{
                    label: "M2",
                    data: series.enc.m2,
                    borderColor: "rgb(156,39,176)",
                    pointRadius: 0,
                    borderWidth: 1.5,
                }},
            ];
            charts.enc.update("none");
        }}

        function scheduleChartUpdate(series) {{
            if (!liveCharts || !recording) return;
            pendingSeries = series;
            if (chartRafPending) return;
            chartRafPending = true;
            requestAnimationFrame(flushCharts);
        }}

        function destroyCharts() {{
            for (const key of ["acc", "gyro", "enc"]) {{
                if (charts[key]) {{
                    charts[key].destroy();
                    charts[key] = null;
                }}
            }}
        }}

        function onLiveChartsChange() {{
            liveCharts = chkLiveCharts.checked;
            chartPanel.classList.toggle("visible", recording && liveCharts);
            if (!liveCharts) destroyCharts();
            if (ws.readyState === WebSocket.OPEN) {{
                ws.send(JSON.stringify({{ type: "set_live_charts", enabled: liveCharts }}));
            }}
        }}

        function renderRecording(on, path, stats) {{
            recording = on;
            recordPending = false;
            btnRecord.disabled = false;
            btnRecord.classList.toggle("active", on);
            btnRecord.innerText = on ? "Zatrzymaj nagrywanie" : "Nagraj dane";
            chartPanel.classList.toggle("visible", on && liveCharts);
            chkLiveCharts.disabled = on;
            if (!on) {{
                destroyCharts();
                lastChartDrawMs = 0;
                recordStats.innerText = "";
            }}
            if (path) {{
                recordPath.innerText = "Zapis: " + path;
            }}
            if (stats && on) {{
                let s = "Wiersze CSV: " + (stats.rows ?? "?");
                if (stats.dropped > 0) s += " | pominięte: " + stats.dropped;
                if (stats.queue_size > 0) s += " | kolejka: " + stats.queue_size;
                recordStats.innerText = s;
            }}
        }}

        function toggleRecording() {{
            if (recordPending) return;
            const next = !recording;
            recordPending = true;
            btnRecord.disabled = false;
            renderRecording(next, null, null);
            btnRecord.innerText = next ? "Zatrzymaj nagrywanie" : "Nagraj dane";
            if (ws.readyState === WebSocket.OPEN) {{
                ws.send(JSON.stringify({{
                    type: "set_recording",
                    enabled: next,
                    live_charts: chkLiveCharts.checked,
                }}));
            }} else {{
                recordPending = false;
                btnRecord.disabled = false;
                btnRecord.innerText = "Nagraj dane";
                statusDiv.innerText = "Brak połączenia WebSocket";
            }}
        }}

        ws.onopen = () => {{
            statusDiv.innerText = "Połączono";
            setMode(currentMode);
        }};
        ws.onerror = () => {{ statusDiv.innerText = "Błąd WebSocket"; }};
        ws.onclose = () => {{ statusDiv.innerText = "Rozłączono"; }};

        ws.onmessage = (ev) => {{
            let data;
            try {{ data = JSON.parse(ev.data); }} catch (_) {{ return; }}
            if (data.type === "record_status") {{
                if (typeof data.live_charts === "boolean") {{
                    liveCharts = data.live_charts;
                    chkLiveCharts.checked = liveCharts;
                }}
                renderRecording(!!data.recording, data.path, data);
            }} else if (data.type === "record_stats") {{
                if (recording) renderRecording(true, null, data);
            }} else if (data.type === "telemetry") {{
                if (!data.recording || !liveCharts) return;
                scheduleChartUpdate(data.series);
            }}
        }};

        slider.oninput = () => {{
            if (currentMode !== "manual") return;
            const v = Number(slider.value);
            value.innerText = v.toFixed(2);
            if (ws.readyState === WebSocket.OPEN) {{
                ws.send(JSON.stringify({{ type: "motor_power", value: v }}));
            }}
        }};

        function stopMotor() {{
            slider.value = 0;
            value.innerText = "0.00";
            if (ws.readyState === WebSocket.OPEN) {{
                ws.send(JSON.stringify({{ type: "motor_power", value: 0 }}));
            }}
        }}

        renderMode();
    </script>
</body>
</html>
"""


def create_app(
    on_motor_power_change,
    on_mode_change=None,
    default_mode="manual",
    on_disconnect=None,
    telemetry_hub=None,
):
    """
    on_motor_power_change(value): suwak manual (-1..1).
    on_mode_change(mode): opcjonalnie "ai" | "manual" — włącza UI z przełącznikiem trybów.
    """
    app = FastAPI()
    balance_ui = on_mode_change is not None
    ws_manager = _WsManager()

    if balance_ui:
        html = _balance_html(default_mode)
    else:
        html = _manual_only_html()

    async def _recording_sideband_loop():
        """Lekki status + opcjonalny podgląd wykresów (bez blokowania WS)."""
        while True:
            if telemetry_hub is not None and telemetry_hub.recording:
                st = telemetry_hub.status()
                await ws_manager.broadcast_json(
                    {
                        "type": "record_stats",
                        "rows": st["rows"],
                        "dropped": st["dropped"],
                        "queue_size": st["queue_size"],
                    }
                )
                if telemetry_hub.live_charts:
                    payload = await asyncio.to_thread(telemetry_hub.chart_payload)
                    if payload.get("series"):
                        await ws_manager.broadcast_json(payload)
                await asyncio.sleep(1.0)
            else:
                await asyncio.sleep(0.5)

    async def _finish_recording(enabled: bool):
        if enabled:
            status = await asyncio.to_thread(telemetry_hub.start)
        else:
            status = await asyncio.to_thread(telemetry_hub.stop)
        await ws_manager.broadcast_json({"type": "record_status", **status})

    @app.on_event("startup")
    async def _startup():
        if telemetry_hub is not None:
            asyncio.create_task(_recording_sideband_loop())

    @app.get("/")
    async def index():
        return HTMLResponse(html)

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        await ws_manager.connect(websocket)
        print("WebSocket connected")

        if telemetry_hub is not None:
            await websocket.send_text(json.dumps({"type": "record_status", **telemetry_hub.status()}))

        try:
            while True:
                msg = await websocket.receive_text()
                data = json.loads(msg)

                if data.get("type") == "motor_power":
                    value = max(-1.0, min(1.0, float(data.get("value", 0))))
                    print(f"Motor power -> {value:.2f}")
                    on_motor_power_change(value)

                elif data.get("type") == "set_mode" and balance_ui:
                    mode = str(data.get("mode", "ai")).lower()
                    if mode in ("ai", "manual"):
                        print(f"Mode -> {mode}")
                        on_mode_change(mode)
                        if mode == "ai":
                            on_motor_power_change(0.0)

                elif data.get("type") == "set_live_charts" and telemetry_hub is not None:
                    telemetry_hub.set_live_charts(bool(data.get("enabled", False)))

                elif data.get("type") == "set_recording" and telemetry_hub is not None:
                    enabled = bool(data.get("enabled", False))
                    if "live_charts" in data:
                        telemetry_hub.set_live_charts(bool(data["live_charts"]))
                    status = (
                        telemetry_hub.request_start()
                        if enabled
                        else telemetry_hub.request_stop()
                    )
                    await websocket.send_text(
                        json.dumps({"type": "record_status", **status})
                    )
                    asyncio.create_task(_finish_recording(enabled))

        except WebSocketDisconnect:
            print("WebSocket disconnected")
            ws_manager.disconnect(websocket)
            if on_disconnect is not None:
                on_disconnect()
            else:
                on_motor_power_change(0.0)

        except Exception as e:
            print("WebSocket error:", e)
            ws_manager.disconnect(websocket)
            if on_disconnect is not None:
                on_disconnect()
            else:
                on_motor_power_change(0.0)

    return app
