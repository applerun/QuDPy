"""参数数据结构。"""

from __future__ import annotations

from dataclasses import dataclass

from .fields import FieldConfig
from .normalization import PhysicalParams


@dataclass(frozen=True)
class OpticalBlochParameters:
    """内部求解参数。"""

    t_start: float = 0.0
    t_final: float = 120.0
    t_end: float | None = None
    dt: float = 0.01
    hbar: float = 1.0
    epsilon_1: float = 0.0
    detuning: float = 0.0
    dipole: float = 0.08
    field_amplitude: float = 1.0
    omega_drive: float = 1.0
    gamma1: float = 0.0
    gamma_phi: float = 0.0
    gamma2: float = 0.0
    pulse_center: float | None = None
    pulse_sigma: float | None = None
    fields: tuple[FieldConfig, ...] | None = None
    tlist: object | None = None
    times_fs: object | None = None


@dataclass(frozen=True)
class PhysicalParameterSweep:
    """基于真实物理量的参数扫描设置。"""

    base_params: PhysicalParams
    field_MV_per_cm_values: tuple[float, ...] = ()
    laser_energy_eV_values: tuple[float, ...] = ()


@dataclass(frozen=True)
class ParameterSweep:
    """旧的内部参数扫描设置，保留为兼容接口。"""

    t_final: float = 120.0
    dt: float = 0.01
    hbar: float = 1.0
    epsilon_1: float = 0.0
    dipole: float = 0.08
    omega_drive: float = 1.0
    field_amplitudes: tuple[float, ...] = (0.5, 1.0, 2.0)
    detunings: tuple[float, ...] = (0.0,)


__all__ = ["OpticalBlochParameters", "PhysicalParameterSweep", "ParameterSweep"]
