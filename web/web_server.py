import json

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse


def create_app(on_motor_power_change):
    app = FastAPI()

    html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Robot Control</title>

    <style>
        body {
            background: #111;
            color: white;
            font-family: Arial, sans-serif;
            text-align: center;
            padding: 30px;
        }

        h1 {
            margin-bottom: 40px;
        }

        input[type=range] {
            width: 90%;
            height: 60px;
        }

        .value {
            font-size: 60px;
            margin-top: 30px;
            margin-bottom: 30px;
        }

        button {
            font-size: 30px;
            padding: 15px 40px;
            border: none;
            border-radius: 15px;
            cursor: pointer;
        }

        .status {
            margin-top: 30px;
            font-size: 20px;
        }
        .slider-container {
    height: 400px;
    display: flex;
    justify-content: center;
    align-items: center;
}

.vertical-slider {
    width: 350px;
    height: 60px;

    transform: rotate(-90deg);

    appearance: none;
    -webkit-appearance: none;
touch-action: none;
}
    </style>
</head>

<body>

    <h1>Robot Motor Control</h1>

<div class="slider-container">
    <input
        id="slider"
        class="vertical-slider"
        type="range"
        min="-1"
        max="1"
        step="0.01"
        value="0"
    >
</div>
    <div class="value" id="value">0.00</div>

    <button onclick="stopMotor()">STOP</button>

    <div class="status" id="status">
        Connecting...
    </div>

    <script>
        const statusDiv = document.getElementById("status");

        const ws = new WebSocket(
            `ws://${window.location.hostname}:8000/ws`
        );

        ws.onopen = () => {
            console.log("WebSocket connected");
            statusDiv.innerText = "Connected";
        };

        ws.onerror = (e) => {
            console.log("WebSocket error", e);
            statusDiv.innerText = "WebSocket error";
        };

        ws.onclose = () => {
            console.log("WebSocket closed");
            statusDiv.innerText = "Disconnected";
        };

        const slider = document.getElementById("slider");
        const value = document.getElementById("value");

        slider.oninput = () => {
            const v = Number(slider.value);

            value.innerText = v.toFixed(2);

            if (ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({
                    type: "motor_power",
                    value: v
                }));
            }
        };

        function stopMotor() {
            slider.value = 0;
            value.innerText = "0.00";

            if (ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({
                    type: "motor_power",
                    value: 0
                }));
            }
        }
    </script>

</body>
</html>
"""

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
                    value = float(data.get("value", 0))

                    value = max(-1.0, min(1.0, value))

                    print(f"Motor power -> {value:.2f}")

                    on_motor_power_change(value)

        except WebSocketDisconnect:
            print("WebSocket disconnected")
            on_motor_power_change(0.0)

        except Exception as e:
            print("WebSocket error:", e)
            on_motor_power_change(0.0)

    return app
