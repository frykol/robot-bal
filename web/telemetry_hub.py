"""Facade for web_balance — delegates to background SensorRecorder."""

from web.sensor_recorder import SensorRecorder

# Re-export for imports: from web.telemetry_hub import TelemetryHub
TelemetryHub = SensorRecorder
