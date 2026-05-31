"""内部 solver 参数数据结构。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .fields import FieldConfig
from .normalization import NLevelPhysicalParams


@dataclass(frozen=True)
class OpticalBlochParameters:
    """内部 N-level solver 参数。

    这里的矩阵和速率已经是 solver code unit。用户侧不应直接构造 code-unit
    输入；普通示例应先构造 `NLevelPhysicalParams`，再经 `ParaNormalizer`
    转换。
    """

    t_start: float = 0.0
    t_final: float = 120.0
    t_end: float | None = None
    dt: float = 0.01
    hbar: float = 1.0
    energies: tuple[float, ...] = (0.0, 1.0)
    dipole_matrix: tuple[tuple[complex, ...], ...] = ((0.0, 1.0), (1.0, 0.0))
    coupling_matrix: tuple[tuple[complex, ...], ...] | None = None
    omega_drive: float = 1.0
    relaxation_channels: tuple[dict[str, Any], ...] = ()
    pure_dephasing_channels: tuple[dict[str, Any], ...] = ()
    fields: tuple[FieldConfig, ...] | None = None
    tlist: object | None = None
    times_fs: object | None = None
    pulse_center: float | None = None
    pulse_sigma: float | None = None
    basis: tuple[str, ...] | None = None

    # 旧内部字段只作为兼容 fallback；新路径不依赖这些标量。
    epsilon_1: float = 0.0
    detuning: float = 0.0
    dipole: float = 1.0
    field_amplitude: float = 1.0
    gamma1: float = 0.0
    gamma_phi: float = 0.0
    gamma2: float = 0.0


@dataclass(frozen=True)
class PhysicalParameterSweep:
    base_params: NLevelPhysicalParams
    field_MV_per_cm_values: tuple[float, ...] = ()
    laser_energy_eV_values: tuple[float, ...] = ()


@dataclass(frozen=True)
class ParameterSweep:
    """旧 code-unit sweep 兼容结构；普通用户示例不再使用。"""

    t_final: float = 120.0
    dt: float = 0.01
    hbar: float = 1.0
    energies: tuple[float, ...] = (0.0, 1.0)
    dipole_matrix: tuple[tuple[complex, ...], ...] = ((0.0, 1.0), (1.0, 0.0))
    omega_drive: float = 1.0
    field_amplitudes: tuple[float, ...] = (0.5, 1.0, 2.0)
    detunings: tuple[float, ...] = (0.0,)


def as_complex_matrix(value) -> np.ndarray:
    return np.asarray(value, dtype=np.complex128)


__all__ = ["OpticalBlochParameters", "PhysicalParameterSweep", "ParameterSweep", "as_complex_matrix"]
