"""
Mass layout for the balancing robot (motors at the bottom of the body box).

Components (user-provided):
  - 2 x motor 160 g at wheel axle (low)
  - Raspberry Pi 55 g
  - case 466 g
  - battery 250 g

The cart-pole model uses:
  m  — mass at the axle (motors + wheels),
  M  — upper body mass (Pi + case + battery),
  l  — height of the body COM above the axle (meters),
  F_max — horizontal drive limit from motor torque at the wheel.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class RobotMassLayout:
    motor_mass_kg: float = 0.16
    n_motors: int = 2
    rpi_mass_kg: float = 0.055
    case_mass_kg: float = 0.466
    battery_mass_kg: float = 0.250
    motor_z_m: float = 0.0
    body_height_m: float = 0.14
    battery_z_m: float | None = None
    case_z_m: float | None = None
    rpi_z_m: float | None = None
    wheel_radius_m: float = 0.03
    motor_torque_nm: float = 0.35
    n_drive_motors: int = 2
    force_max_cap_n: float | None = 10.0


@dataclass
class DynamicsParams:
    m_axle_kg: float
    M_body_kg: float
    l_body_m: float
    z_com_full_m: float
    force_max_n: float
    layout: dict


def _z_positions(layout: RobotMassLayout) -> tuple[float, float, float]:
    h = layout.body_height_m
    z_battery = layout.battery_z_m if layout.battery_z_m is not None else 0.30 * h
    z_case = layout.case_z_m if layout.case_z_m is not None else 0.50 * h
    z_rpi = layout.rpi_z_m if layout.rpi_z_m is not None else 0.72 * h
    return z_battery, z_case, z_rpi


def compute_dynamics_params(layout: RobotMassLayout) -> DynamicsParams:
    z_battery, z_case, z_rpi = _z_positions(layout)

    m_axle = layout.n_motors * layout.motor_mass_kg
    M_body = layout.rpi_mass_kg + layout.case_mass_kg + layout.battery_mass_kg
    if M_body <= 0:
        raise ValueError("Body mass must be positive.")

    l_body = (
        layout.battery_mass_kg * z_battery
        + layout.case_mass_kg * z_case
        + layout.rpi_mass_kg * z_rpi
    ) / M_body

    total_mass = m_axle + M_body
    z_com_full = (m_axle * layout.motor_z_m + M_body * l_body) / total_mass

    if layout.wheel_radius_m <= 0:
        raise ValueError("wheel_radius_m must be positive.")

    force_from_torque = (
        layout.n_drive_motors * layout.motor_torque_nm / layout.wheel_radius_m
    )
    if layout.force_max_cap_n is None:
        force_max = force_from_torque
    else:
        force_max = min(force_from_torque, layout.force_max_cap_n)

    layout_dict = asdict(layout)
    layout_dict["battery_z_m"] = z_battery
    layout_dict["case_z_m"] = z_case
    layout_dict["rpi_z_m"] = z_rpi
    layout_dict["force_from_torque_n"] = force_from_torque

    return DynamicsParams(
        m_axle_kg=m_axle,
        M_body_kg=M_body,
        l_body_m=l_body,
        z_com_full_m=z_com_full,
        force_max_n=force_max,
        layout=layout_dict,
    )


def layout_from_train_args(args) -> RobotMassLayout:
    return RobotMassLayout(
        motor_mass_kg=args.motor_mass_g / 1000.0,
        n_motors=args.n_motors,
        rpi_mass_kg=args.rpi_mass_g / 1000.0,
        case_mass_kg=args.case_mass_g / 1000.0,
        battery_mass_kg=args.battery_mass_g / 1000.0,
        body_height_m=args.body_height_m,
        battery_z_m=args.battery_z_m,
        case_z_m=args.case_z_m,
        rpi_z_m=args.rpi_z_m,
        wheel_radius_m=args.wheel_radius_m,
        motor_torque_nm=args.motor_torque_nm,
        n_drive_motors=args.n_drive_motors,
        force_max_cap_n=args.force_max,
    )


def resolve_train_physics(args) -> DynamicsParams:
    if args.com_from_masses:
        return compute_dynamics_params(layout_from_train_args(args))
    total_m = 0.320 + 0.771
    return DynamicsParams(
        m_axle_kg=0.320,
        M_body_kg=0.771,
        l_body_m=float(args.com_height_m),
        z_com_full_m=float(args.com_height_m),
        force_max_n=float(args.force_max),
        layout={"mode": "manual_com_height_m", "com_height_m": args.com_height_m},
    )
