"""真实物理 N-level 系统到内部 solver code unit 的归一化工具。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional

import numpy as np


@dataclass(frozen=True)
class RelaxationChannel:
    """人口弛豫通道：C_{to <- from} = sqrt(rate) |to><from|。"""

    name: str
    from_level: int
    to_level: int
    T1_fs: float | None = None
    rate_fs_inv: float | None = None


@dataclass(frozen=True)
class PureDephasingChannel:
    """能级投影退相干通道：C_level^phi = sqrt(rate) |level><level|。"""

    name: str
    level: int
    Tphi_fs: float | None = None
    rate_fs_inv: float | None = None


@dataclass(frozen=True)
class NLevelPhysicalParams:
    """用户侧 N-level 物理输入。

    所有普通输入保持真实物理单位：eV、Debye、MV/cm、fs。
    `dipole_matrix_D` 是沿选定 optical polarization 投影后的偶极矩矩阵。
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
        """N=2 示例层常用的 0->1 能隙；核心模型仍以 `energies_eV` 为准。"""
        if self.dimension < 2:
            raise ValueError("energy_gap_eV requires at least two levels.")
        return float(self.energies_eV[1] - self.energies_eV[0])


@dataclass
class SolverParams:
    """内部 solver 参数。

    code unit 只供 Hamiltonian/c_ops 构造和 debug metadata 使用。
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
    pulse_center: Optional[float] = None
    pulse_sigma: Optional[float] = None
    pulse_center_fs: Optional[float] = None
    pulse_sigma_fs: Optional[float] = None

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


class ParaNormalizer:
    """把真实物理单位转换为 solver code unit 的归一化器。"""

    HBAR_J_S = 1.054571817e-34
    E_CHARGE_C = 1.602176634e-19
    FS_TO_S = 1e-15
    DEBYE_TO_C_M = 3.33564e-30
    MV_PER_CM_TO_V_PER_M = 1e8

    EV_TO_FS_INV = (E_CHARGE_C / HBAR_J_S) * FS_TO_S
    DIPOLE_FIELD_TO_RABI_FS_INV = (
        DEBYE_TO_C_M * MV_PER_CM_TO_V_PER_M / HBAR_J_S
    ) * FS_TO_S

    def __init__(self, time_scale_fs: Optional[float] = None, auto_scale: bool = True):
        self.user_time_scale_fs = time_scale_fs
        self.auto_scale = auto_scale
        self.last_physical: Optional[NLevelPhysicalParams] = None
        self.last_solver: Optional[SolverParams] = None

    @classmethod
    def energy_eV_to_fs_inv(cls, energy_eV: float | np.ndarray) -> float | np.ndarray:
        return np.asarray(energy_eV, dtype=float) * cls.EV_TO_FS_INV

    @classmethod
    def fs_inv_to_energy_eV(cls, omega_fs_inv: float) -> float:
        return omega_fs_inv / cls.EV_TO_FS_INV

    @classmethod
    def rate_from_time_fs(cls, T_fs: Optional[float]) -> float:
        if T_fs is None:
            return 0.0
        if T_fs <= 0:
            raise ValueError("时间常数必须为正。")
        return 1.0 / T_fs

    @classmethod
    def rabi_fs_inv_from_mu_and_field(cls, projected_dipole_D: float, field_MV_per_cm: float) -> float:
        return projected_dipole_D * field_MV_per_cm * cls.DIPOLE_FIELD_TO_RABI_FS_INV

    @classmethod
    def coupling_matrix_fs_inv_from_mu_and_field(
        cls,
        dipole_matrix_D: np.ndarray,
        field_MV_per_cm: float,
    ) -> np.ndarray:
        return np.asarray(dipole_matrix_D, dtype=np.complex128) * field_MV_per_cm * cls.DIPOLE_FIELD_TO_RABI_FS_INV

    def normalize(self, p: NLevelPhysicalParams) -> SolverParams:
        self._validate_physical_params(p)
        energies_fs_inv = np.asarray(self.energy_eV_to_fs_inv(np.asarray(p.energies_eV, dtype=float)), dtype=float)
        omega_L_fs_inv = float(self.energy_eV_to_fs_inv(p.laser_energy_eV))
        coupling_matrix_fs_inv = self.coupling_matrix_fs_inv_from_mu_and_field(
            np.asarray(p.dipole_matrix_D, dtype=np.complex128),
            p.field_MV_per_cm,
        )
        relaxation_fs = tuple(self._relaxation_channel_to_rate_dict(channel) for channel in p.relaxation_channels)
        dephasing_fs = tuple(self._pure_dephasing_channel_to_rate_dict(channel) for channel in p.pure_dephasing_channels)

        rate_candidates = [float(abs(value)) for value in coupling_matrix_fs_inv.ravel() if abs(value) > 0]
        rate_candidates.extend(abs(float(ch["rate_fs_inv"])) for ch in relaxation_fs if ch["rate_fs_inv"] > 0)
        rate_candidates.extend(abs(float(ch["rate_fs_inv"])) for ch in dephasing_fs if ch["rate_fs_inv"] > 0)
        if len(energies_fs_inv) >= 2:
            rate_candidates.append(abs(float(energies_fs_inv[1] - energies_fs_inv[0] - omega_L_fs_inv)))
        if p.pulse_sigma_fs is not None and p.pulse_sigma_fs > 0:
            rate_candidates.append(1.0 / p.pulse_sigma_fs)
        time_scale_fs = self._choose_time_scale_fs(rate_candidates, energies_fs_inv)

        t_start = p.t_start_fs / time_scale_fs
        t_end = p.t_end_fs / time_scale_fs
        dt = p.dt_fs / time_scale_fs
        tlist = self._build_tlist(t_start, t_end, dt)
        pulse_center = None if p.pulse_center_fs is None else p.pulse_center_fs / time_scale_fs
        pulse_sigma = None if p.pulse_sigma_fs is None else p.pulse_sigma_fs / time_scale_fs

        solver = SolverParams(
            time_scale_fs=time_scale_fs,
            energies_fs_inv=energies_fs_inv,
            energies_code=energies_fs_inv * time_scale_fs,
            dipole_matrix_D=np.asarray(p.dipole_matrix_D, dtype=np.complex128),
            coupling_matrix_fs_inv=coupling_matrix_fs_inv,
            coupling_matrix_code=coupling_matrix_fs_inv * time_scale_fs,
            relaxation_channels_fs_inv=relaxation_fs,
            pure_dephasing_channels_fs_inv=dephasing_fs,
            relaxation_channels_code=tuple(self._scale_rate_dict(item, time_scale_fs) for item in relaxation_fs),
            pure_dephasing_channels_code=tuple(self._scale_rate_dict(item, time_scale_fs) for item in dephasing_fs),
            omega_L_fs_inv=omega_L_fs_inv,
            omega_L=omega_L_fs_inv * time_scale_fs,
            t_start=t_start,
            t_end=t_end,
            dt=dt,
            tlist=tlist,
            pulse_center=pulse_center,
            pulse_sigma=pulse_sigma,
            pulse_center_fs=p.pulse_center_fs,
            pulse_sigma_fs=p.pulse_sigma_fs,
        )
        self.last_physical = p
        self.last_solver = solver
        return solver

    def _choose_time_scale_fs(self, candidates: list[float], energies_fs_inv: np.ndarray) -> float:
        if self.user_time_scale_fs is not None:
            if self.user_time_scale_fs <= 0:
                raise ValueError("time_scale_fs 必须为正。")
            return self.user_time_scale_fs
        if not self.auto_scale:
            return 1.0
        positive = [value for value in candidates if value > 0]
        if not positive:
            nonzero_energies = [abs(float(value)) for value in energies_fs_inv if abs(value) > 0]
            positive.extend(nonzero_energies)
        if not positive:
            return 1.0
        return 1.0 / max(positive)

    def _relaxation_channel_to_rate_dict(self, channel: RelaxationChannel) -> dict[str, Any]:
        rate = channel.rate_fs_inv if channel.rate_fs_inv is not None else self.rate_from_time_fs(channel.T1_fs)
        return {
            "name": channel.name,
            "from_level": channel.from_level,
            "to_level": channel.to_level,
            "T1_fs": channel.T1_fs,
            "rate_fs_inv": float(rate),
        }

    def _pure_dephasing_channel_to_rate_dict(self, channel: PureDephasingChannel) -> dict[str, Any]:
        rate = channel.rate_fs_inv if channel.rate_fs_inv is not None else self.rate_from_time_fs(channel.Tphi_fs)
        return {
            "name": channel.name,
            "level": channel.level,
            "Tphi_fs": channel.Tphi_fs,
            "rate_fs_inv": float(rate),
        }

    @staticmethod
    def _scale_rate_dict(channel: dict[str, Any], time_scale_fs: float) -> dict[str, Any]:
        scaled = dict(channel)
        scaled["rate_code"] = float(channel["rate_fs_inv"]) * time_scale_fs
        return scaled

    @staticmethod
    def _build_tlist(t_start: float, t_end: float, dt: float) -> np.ndarray:
        n = int(np.floor((t_end - t_start) / dt)) + 1
        return t_start + np.arange(n) * dt

    def _validate_physical_params(self, p: NLevelPhysicalParams) -> None:
        n = len(p.energies_eV)
        if n < 2:
            raise ValueError("N-level system 至少需要两个能级。")
        dipole = np.asarray(p.dipole_matrix_D, dtype=np.complex128)
        if dipole.shape != (n, n):
            raise ValueError("dipole_matrix_D 必须是 N x N，并与 energies_eV 长度一致。")
        if p.basis is not None and len(p.basis) != n:
            raise ValueError("basis 长度必须与 energies_eV 一致。")
        if p.t_end_fs <= p.t_start_fs:
            raise ValueError("t_end_fs 必须大于 t_start_fs。")
        if p.dt_fs <= 0:
            raise ValueError("dt_fs 必须为正。")
        if p.pulse_sigma_fs is not None and p.pulse_sigma_fs <= 0:
            raise ValueError("pulse_sigma_fs 必须为正。")
        for channel in p.relaxation_channels:
            if not (0 <= channel.from_level < n and 0 <= channel.to_level < n):
                raise ValueError(f"relaxation channel {channel.name} 的 level index 超界。")
            if channel.T1_fs is not None and channel.T1_fs <= 0:
                raise ValueError(f"relaxation channel {channel.name} 的 T1_fs 必须为正。")
            if channel.rate_fs_inv is not None and channel.rate_fs_inv < 0:
                raise ValueError(f"relaxation channel {channel.name} 的 rate_fs_inv 不能为负。")
        for channel in p.pure_dephasing_channels:
            if not 0 <= channel.level < n:
                raise ValueError(f"pure_dephasing channel {channel.name} 的 level index 超界。")
            if channel.Tphi_fs is not None and channel.Tphi_fs <= 0:
                raise ValueError(f"pure_dephasing channel {channel.name} 的 Tphi_fs 必须为正。")
            if channel.rate_fs_inv is not None and channel.rate_fs_inv < 0:
                raise ValueError(f"pure_dephasing channel {channel.name} 的 rate_fs_inv 不能为负。")

    def denormalize_time_array(self, t_code_array: np.ndarray, solver: Optional[SolverParams] = None) -> np.ndarray:
        s = self._require_solver(solver)
        return np.asarray(t_code_array, dtype=float) * s.time_scale_fs

    def _require_solver(self, solver: Optional[SolverParams]) -> SolverParams:
        if solver is not None:
            return solver
        if self.last_solver is None:
            raise RuntimeError("还没有调用 normalize()，无法反归一化。")
        return self.last_solver

    def summary_dict(
        self,
        physical: Optional[NLevelPhysicalParams] = None,
        solver: Optional[SolverParams] = None,
    ) -> dict[str, Any]:
        p = physical if physical is not None else self.last_physical
        s = solver if solver is not None else self.last_solver
        if p is None or s is None:
            raise RuntimeError("没有可用的 physical / solver 参数。")
        return {
            "physical": asdict(p),
            "conversion_constants": {
                "EV_TO_FS_INV": self.EV_TO_FS_INV,
                "DIPOLE_FIELD_TO_RABI_FS_INV": self.DIPOLE_FIELD_TO_RABI_FS_INV,
            },
            "solver_scales": {"time_scale_fs": s.time_scale_fs},
            "solver_params_fs_inv": {
                "energies_fs_inv": s.energies_fs_inv,
                "omega_L_fs_inv": s.omega_L_fs_inv,
                "detuning_fs_inv": s.detuning_fs_inv,
                "coupling_matrix_fs_inv": s.coupling_matrix_fs_inv,
                "relaxation_channels_fs_inv": s.relaxation_channels_fs_inv,
                "pure_dephasing_channels_fs_inv": s.pure_dephasing_channels_fs_inv,
            },
            "solver_params_code": {
                "energies_code": s.energies_code,
                "omega_L_code": s.omega_L,
                "detuning_code": s.detuning,
                "coupling_matrix_code": s.coupling_matrix_code,
                "relaxation_channels_code": s.relaxation_channels_code,
                "pure_dephasing_channels_code": s.pure_dephasing_channels_code,
                "t_start": s.t_start,
                "t_end": s.t_end,
                "dt": s.dt,
                "tlist": s.tlist,
            },
        }


__all__ = [
    "ParaNormalizer",
    "NLevelPhysicalParams",
    "RelaxationChannel",
    "PureDephasingChannel",
    "SolverParams",
]
