import json

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse


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
    </style>
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

    <script>
        let currentMode = "{default_mode}";
        const statusDiv = document.getElementById("status");
        const modeLabel = document.getElementById("modeLabel");
        const manualPanel = document.getElementById("manualPanel");
        const btnAi = document.getElementById("btnAi");
        const btnManual = document.getElementById("btnManual");
        const slider = document.getElementById("slider");
        const value = document.getElementById("value");

        const ws = new WebSocket(`ws://${{window.location.hostname}}:8000/ws`);

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

        ws.onopen = () => {{
            statusDiv.innerText = "Połączono";
            setMode(currentMode);
        }};
        ws.onerror = () => {{ statusDiv.innerText = "Błąd WebSocket"; }};
        ws.onclose = () => {{ statusDiv.innerText = "Rozłączono"; }};

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
):
    """
    on_motor_power_change(value): suwak manual (-1..1).
    on_mode_change(mode): opcjonalnie "ai" | "manual" — włącza UI z przełącznikiem trybów.
    """
    app = FastAPI()
    balance_ui = on_mode_change is not None

    if balance_ui:
        html = _balance_html(default_mode)
    else:
        html = _manual_only_html()

    @app.get("/")
    async def index():
        return HTMLResponse(html)

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        await websocket.accept()
        print("WebSocket connected")

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

        except WebSocketDisconnect:
            print("WebSocket disconnected")
            if on_disconnect is not None:
                on_disconnect()
            else:
                on_motor_power_change(0.0)

        except Exception as e:
            print("WebSocket error:", e)
            if on_disconnect is not None:
                on_disconnect()
            else:
                on_motor_power_change(0.0)

    return app
