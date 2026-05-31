"""参数数据结构。

本模块只定义数据容器，不做单位换算、不构造 Hamiltonian，也不调用
QuTiP。用户侧物理系统使用 `NLevelPhysicalParams`；solver 内部参数使用
`NLevelSolverParams`。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RelaxationChannel:
    """population relaxation 通道。

    物理约定：`C_{to <- from} = sqrt(rate) |to><from|`。
    可以用 `T1_fs` 或 `rate_fs_inv` 指定速率；二者都给出时，
    normalizer 会优先使用 `rate_fs_inv`。
    """

    name: str
    from_level: int
    to_level: int
    T1_fs: float | None = None
    rate_fs_inv: float | None = None


@dataclass(frozen=True)
class PureDephasingChannel:
    """level projector pure dephasing 通道。

    物理约定：`C_level^phi = sqrt(rate) |level><level|`。
    可以用 `Tphi_fs` 或 `rate_fs_inv` 指定速率；二者都给出时，
    normalizer 会优先使用 `rate_fs_inv`。
    """

    name: str
    level: int
    Tphi_fs: float | None = None
    rate_fs_inv: float | None = None


@dataclass(frozen=True)
class NLevelPhysicalParams:
    """用户侧 N-level 物理系统输入。

    所有普通输入保持真实物理单位：`energies_eV` 用 eV，
    `dipole_matrix_D` 用 Debye，`field_MV_per_cm` 用 MV/cm，时间用 fs。
    `dipole_matrix_D` 是沿选定 optical polarization 投影后的偶极矩矩阵。
    two-level system 也是普通 N=2 system，不再作为核心层特殊标量模型。
    """

    energies_eV: tuple[float, ...]
    dipole_matrix_D: tuple[tuple[float, ...], ...]
    field_MV_per_cm: float
    laser_energy_eV: float
    t_start_fs: float
    t_end_fs: float
    dt_fs: float
    basis: tuple[str, ...] | None = None
    relaxation_channels: tuple[RelaxationChannel, ...] = ()
    pure_dephasing_channels: tuple[PureDephasingChannel, ...] = ()
    pulse_center_fs: float | None = None
    pulse_sigma_fs: float | None = None

    @property
    def dimension(self) -> int:
        return len(self.energies_eV)

    @property
    def energy_gap_eV(self) -> float:
        """N=2 教学示例常用的 0->1 能隙；核心模型仍以 `energies_eV` 为准。"""
        if self.dimension < 2:
            raise ValueError("energy_gap_eV requires at least two levels.")
        return float(self.energies_eV[1] - self.energies_eV[0])


@dataclass
class SolverParams:
    """归一化后的 solver 参数摘要。

    这里包含 fs^-1 和 code-unit 两套量，供 solver 构造内部参数以及
    `debug_meta.json` 调试使用。普通用户不应直接构造这个对象。
    """

    time_scale_fs: float
    energies_fs_inv: np.ndarray
    energies_code: np.ndarray
    dipole_matrix_D: np.ndarray
    coupling_matrix_fs_inv: np.ndarray
    coupling_matrix_code: np.ndarray
    relaxation_channels_fs_inv: tuple[dict[str, Any], ...]
    pure_dephasing_channels_fs_inv: tuple[dict[str, Any], ...]
    relaxation_channels_code: tuple[dict[str, Any], ...]
    pure_dephasing_channels_code: tuple[dict[str, Any], ...]
    omega_L_fs_inv: float
    omega_L: float
    t_start: float
    t_end: float
    dt: float
    tlist: np.ndarray
    pulse_center: float | None = None
    pulse_sigma: float | None = None
    pulse_center_fs: float | None = None
    pulse_sigma_fs: float | None = None

    @property
    def omega_eg_fs_inv(self) -> float:
        return float(self.energies_fs_inv[1] - self.energies_fs_inv[0]) if len(self.energies_fs_inv) >= 2 else 0.0

    @property
    def omega_eg(self) -> float:
        return self.omega_eg_fs_inv * self.time_scale_fs

    @property
    def detuning_fs_inv(self) -> float:
        return self.omega_eg_fs_inv - self.omega_L_fs_inv

    @property
    def detuning(self) -> float:
        return self.detuning_fs_inv * self.time_scale_fs

    @property
    def rabi_fs_inv(self) -> float:
        if self.coupling_matrix_fs_inv.shape[0] < 2:
            return 0.0
        return float(self.coupling_matrix_fs_inv[0, 1].real)

    @property
    def rabi(self) -> float:
        return self.rabi_fs_inv * self.time_scale_fs

    @property
    def gamma1_fs_inv(self) -> float:
        for channel in self.relaxation_channels_fs_inv:
            if channel.get("from_level") == 1 and channel.get("to_level") == 0:
                return float(channel["rate_fs_inv"])
        return 0.0

    @property
    def gamma_phi_fs_inv(self) -> float:
        if len(self.pure_dephasing_channels_fs_inv) == 1:
            return float(self.pure_dephasing_channels_fs_inv[0]["rate_fs_inv"])
        if len(self.pure_dephasing_channels_fs_inv) >= 2:
            rates = [float(item["rate_fs_inv"]) for item in self.pure_dephasing_channels_fs_inv[:2]]
            return 0.5 * sum(rates)
        return 0.0

    @property
    def gamma2_fs_inv(self) -> float:
        return self.gamma_phi_fs_inv + 0.5 * self.gamma1_fs_inv

    @property
    def gamma1(self) -> float:
        return self.gamma1_fs_inv * self.time_scale_fs

    @property
    def gamma_phi(self) -> float:
        return self.gamma_phi_fs_inv * self.time_scale_fs

    @property
    def gamma2(self) -> float:
        return self.gamma2_fs_inv * self.time_scale_fs


@dataclass(frozen=True)
class NLevelSolverParams:
    """内部 N-level solver 参数。

    这里的矩阵、频率、时间和速率已经是 solver code unit。普通用户侧
    示例应先构造 `NLevelPhysicalParams`，再经 `ParaNormalizer` 转换。
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
    fields: tuple[Any, ...] | None = None
    tlist: object | None = None
    times_fs: object | None = None
    pulse_center: float | None = None
    pulse_sigma: float | None = None
    basis: tuple[str, ...] | None = None

    # 临时兼容字段：旧 code-unit helper 可能还会读取这些标量。
    # 新的 solver/model 路径不依赖它们，后续可逐步删除。
    epsilon_1: float = 0.0
    detuning: float = 0.0
    dipole: float = 1.0
    field_amplitude: float = 1.0
    gamma1: float = 0.0
    gamma_phi: float = 0.0
    gamma2: float = 0.0


# 临时兼容别名：旧脚本可能仍从 utils 导入 `OpticalBlochParameters`。
# 推荐新代码使用 `NLevelSolverParams`，普通用户侧入口使用 `NLevelPhysicalParams`。
OpticalBlochParameters = NLevelSolverParams


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


__all__ = [
    "NLevelPhysicalParams",
    "RelaxationChannel",
    "PureDephasingChannel",
    "SolverParams",
    "NLevelSolverParams",
    "OpticalBlochParameters",
    "PhysicalParameterSweep",
    "ParameterSweep",
    "as_complex_matrix",
]
