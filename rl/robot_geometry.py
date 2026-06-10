"""
Physical layout for the real robot — shared with simulation mass model.

Heights in JSON can be given:
  - from the wheel axle upward (height_reference: "wheel_axle", default)
  - from the floor upward (height_reference: "floor") → converted as z_axle = z_floor - wheel_radius_m

Used on Pi to derive COM height, drive force limit, and PID gain scaling.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rl.robot_mass_model import DynamicsParams, RobotMassLayout, compute_dynamics_params

DEFAULT_GEOMETRY_PATH = Path("artifacts") / "robot_geometry.json"

# v6-like sim robot used as PID reference when scaling gains.
DEFAULT_PID_REFERENCE = {
    "com_height_m": 0.063,
    "force_max_n": 10.0,
    "kp": 12.0,
    "ki": 0.0,
    "kd": 0.0,
    "kp_x": 0.0,
    "ki_x": 0.0,
    "kd_x": 0.0,
}


def _floor_to_axle(z_floor: float, wheel_radius_m: float) -> float:
    return float(z_floor) - float(wheel_radius_m)


def _maybe_axle_height(
    geom: dict,
    axle_key: str,
    floor_key: str,
    wheel_radius_m: float,
) -> float | None:
    if axle_key in geom and geom[axle_key] is not None:
        return float(geom[axle_key])
    if floor_key in geom and geom[floor_key] is not None:
        return _floor_to_axle(float(geom[floor_key]), wheel_radius_m)
    return None


def layout_from_geometry_dict(geom: dict) -> RobotMassLayout:
    """Build RobotMassLayout from a geometry JSON object."""
    ref = str(geom.get("height_reference", "wheel_axle")).lower()
    wheel_radius_m = float(geom.get("wheel_radius_m", 0.03))

    def z_axle(axle_key: str, floor_key: str, default=None):
        if ref == "floor":
            v = _maybe_axle_height(geom, axle_key, floor_key, wheel_radius_m)
        else:
            v = geom.get(axle_key)
            if v is not None:
                v = float(v)
        return v if v is not None else default

    body_height_m = z_axle("body_height_m", "body_top_from_floor_m")
    if body_height_m is None:
        body_height_m = 0.14

    return RobotMassLayout(
        motor_mass_kg=float(geom.get("motor_mass_kg", geom.get("motor_mass_g", 160.0) / 1000.0)),
        n_motors=int(geom.get("n_motors", 2)),
        rpi_mass_kg=float(geom.get("rpi_mass_kg", geom.get("rpi_mass_g", 55.0) / 1000.0)),
        case_mass_kg=float(geom.get("case_mass_kg", geom.get("case_mass_g", 466.0) / 1000.0)),
        battery_mass_kg=float(
            geom.get("battery_mass_kg", geom.get("battery_mass_g", 250.0) / 1000.0)
        ),
        motor_z_m=float(geom.get("motor_z_m", 0.0)),
        body_height_m=float(body_height_m),
        battery_z_m=z_axle("battery_z_m", "battery_height_from_floor_m"),
        case_z_m=z_axle("case_z_m", "case_height_from_floor_m"),
        rpi_z_m=z_axle("rpi_z_m", "rpi_height_from_floor_m"),
        wheel_radius_m=wheel_radius_m,
        motor_torque_nm=float(geom.get("motor_torque_nm", 0.35)),
        n_drive_motors=int(geom.get("n_drive_motors", 2)),
        force_max_cap_n=geom.get("force_max_cap_n", geom.get("force_max_n", 10.0)),
    )


def physics_from_geometry_dict(geom: dict) -> DynamicsParams:
    layout = layout_from_geometry_dict(geom)
    physics = compute_dynamics_params(layout)
    imu_z = _maybe_axle_height(
        geom,
        "imu_z_m",
        "imu_height_from_floor_m",
        layout.wheel_radius_m,
    )
    if imu_z is None:
        imu_z = float(physics.layout.get("body_height_m", layout.body_height_m))
    physics.layout["imu_z_m"] = float(imu_z)
    physics.layout["height_reference"] = geom.get("height_reference", "wheel_axle")
    if geom.get("ground_clearance_m") is not None:
        physics.layout["ground_clearance_m"] = float(geom["ground_clearance_m"])
    return physics


def load_robot_geometry(path: Path | str | None) -> dict | None:
    path = Path(path) if path is not None else DEFAULT_GEOMETRY_PATH
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def pid_reference_from_geometry(geom: dict) -> dict:
    ref = dict(DEFAULT_PID_REFERENCE)
    user_ref = geom.get("pid_reference") or geom.get("pid") or {}
    for key in ref:
        if key in user_ref and user_ref[key] is not None:
            ref[key] = float(user_ref[key])
    return ref


def pid_gain_scale(physics: DynamicsParams, pid_ref: dict) -> float:
    """
    Scale CLI/reference PID gains for a different COM height and drive limit.

    Higher COM → larger gravitational torque → scale Kp/Ki up.
    Lower force_max → scale up so normalized command stays meaningful.
    """
    l_ref = max(float(pid_ref["com_height_m"]), 1e-4)
    f_ref = max(float(pid_ref["force_max_n"]), 1e-4)
    l = max(float(physics.l_body_m), 1e-4)
    f_max = max(float(physics.force_max_n), 1e-4)
    return (l / l_ref) * (f_ref / f_max)


def resolve_pid_for_robot(
    geom: dict,
    *,
    kp: float,
    ki: float,
    kd: float,
    kp_x: float = 0.0,
    ki_x: float = 0.0,
    kd_x: float = 0.0,
    force_max_n: float,
    auto_scale: bool = True,
    use_geometry_force_max: bool = True,
) -> dict[str, Any]:
    """
    Return PID gains and force_max_n for deployment from geometry + user base gains.
    """
    physics = physics_from_geometry_dict(geom)
    pid_ref = pid_reference_from_geometry(geom)

    out_force = float(physics.force_max_n if use_geometry_force_max else force_max_n)
    out_kp, out_ki, out_kd = float(kp), float(ki), float(kd)
    out_kp_x, out_ki_x, out_kd_x = float(kp_x), float(ki_x), float(kd_x)

    scale = 1.0
    if auto_scale:
        scale = pid_gain_scale(physics, pid_ref)
        out_kp *= scale
        out_ki *= scale
        out_kd *= scale
        out_kp_x *= scale
        out_ki_x *= scale
        out_kd_x *= scale

    if geom.get("use_geometry_pid_gains"):
        out_kp = float(pid_ref["kp"]) * (scale if auto_scale else 1.0)
        out_ki = float(pid_ref["ki"]) * (scale if auto_scale else 1.0)
        out_kd = float(pid_ref["kd"]) * (scale if auto_scale else 1.0)
        out_kp_x = float(pid_ref["kp_x"]) * (scale if auto_scale else 1.0)
        out_ki_x = float(pid_ref["ki_x"]) * (scale if auto_scale else 1.0)
        out_kd_x = float(pid_ref["kd_x"]) * (scale if auto_scale else 1.0)

    return {
        "kp": out_kp,
        "ki": out_ki,
        "kd": out_kd,
        "kp_x": out_kp_x,
        "ki_x": out_ki_x,
        "kd_x": out_kd_x,
        "force_max_n": out_force,
        "gain_scale": scale,
        "physics": physics,
        "pid_reference": pid_ref,
    }


def print_geometry_pid_summary(resolved: dict) -> None:
    physics = resolved["physics"]
    layout = physics.layout
    print("Robot geometry → PID:")
    print(f"  COM height l={physics.l_body_m:.4f} m  |  force_max={physics.force_max_n:.2f} N")
    print(
        f"  stack z [m]  battery={layout.get('battery_z_m', 0):.3f}  "
        f"case={layout.get('case_z_m', 0):.3f}  rpi={layout.get('rpi_z_m', 0):.3f}  "
        f"imu={layout.get('imu_z_m', 0):.3f}"
    )
    ref = resolved["pid_reference"]
    print(
        f"  PID ref (l={ref['com_height_m']:.3f} m, F={ref['force_max_n']:.1f} N): "
        f"Kp={ref['kp']:g} Ki={ref['ki']:g} Kd={ref['kd']:g}"
    )
    print(f"  gain_scale={resolved['gain_scale']:.3f}")
    print(
        f"  deployed PID: Kp={resolved['kp']:g} Ki={resolved['ki']:g} "
        f"Kd={resolved['kd']:g}  force_max_n={resolved['force_max_n']:.2f}"
    )
    if resolved["kp_x"] or resolved["ki_x"] or resolved["kd_x"]:
        print(
            f"  position loop: Kp_x={resolved['kp_x']:g} "
            f"Ki_x={resolved['ki_x']:g} Kd_x={resolved['kd_x']:g}"
        )
