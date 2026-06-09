"""真实物理单位到 solver code unit 的归一化工具。

本模块只负责单位换算和归一化：不定义主参数 dataclass，不构造
Hamiltonian，也不生成 collapse operator。
"""

from __future__ import annotations

from dataclasses import asdict, fields as dataclass_fields
from typing import Any, Optional

import numpy as np

from sjh_learn.utils.fields.lab_fields import FieldPhyRoot, make_code_field_adapter
from sjh_learn.utils.core.parameters import NLevelPhysicalParams, PureDephasingChannel, RelaxationChannel, SolverParams


class ParaNormalizer:
    """把 `NLevelPhysicalParams` 转换为 solver 可用的 code-unit 参数。"""

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
            positive.extend(abs(float(value)) for value in energies_fs_inv if abs(value) > 0)
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
        # `dipole_matrix_D` 表示物理偶极矩算符，必须是 Hermitian；这允许
        # complex transition dipole，但会拒绝非共轭的跃迁矩阵元和虚数对角元。
        if not np.allclose(dipole, dipole.conj().T):
            raise ValueError("dipole_matrix_D 必须是 Hermitian；transition dipole 需要满足 mu_ij = conj(mu_ji)。")
        if p.basis is not None and len(p.basis) != n:
            raise ValueError("basis 长度必须与 energies_eV 一致。")
        if p.t_end_fs <= p.t_start_fs:
            raise ValueError("t_end_fs 必须大于 t_start_fs。")
        if p.dt_fs <= 0:
            raise ValueError("dt_fs 必须为正。")
        if p.pulse_sigma_fs is not None and p.pulse_sigma_fs <= 0:
            raise ValueError("pulse_sigma_fs 必须为正。")
        if p.field is not None and not isinstance(p.field, FieldPhyRoot):
            raise TypeError("field must be None or a FieldPhyRoot instance.")
        if p.input_description is not None and not isinstance(p.input_description, str):
            raise TypeError("input_description must be None or str.")
        if p.input_metadata is not None and not isinstance(p.input_metadata, dict):
            raise TypeError("input_metadata must be None or dict.")
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

    def denormalize_time(self, t_code: float, solver: Optional[SolverParams] = None) -> float:
        """把 solver code time 转回物理时间 fs。"""

        return float(self.denormalize_time_array(np.asarray(t_code, dtype=float), solver))

    @staticmethod
    def normalize_field_MV_per_cm(
        E_MV_per_cm: float | np.ndarray,
        *,
        reference_field_MV_per_cm: float,
    ) -> float | np.ndarray:
        """把真实电场 MV/cm 归一化为 solver field callable 使用的无量纲 code signal。

        N-level coupling matrix 已经包含 `mu * E0 / hbar`，因此进入 Hamiltonian
        的 field callable 应表示 `E(t) / E0`。`reference_field_MV_per_cm`
        就是这个 E0；为 0 时直接报错，避免静默产生无物理意义的归一化。
        """

        if reference_field_MV_per_cm == 0:
            raise ValueError("reference_field_MV_per_cm must be nonzero.")
        values = np.asarray(E_MV_per_cm, dtype=float) / float(reference_field_MV_per_cm)
        if values.ndim == 0:
            return float(values)
        return values

    def make_code_field(
        self,
        field_phy: FieldPhyRoot,
        solver: Optional[SolverParams] = None,
        *,
        reference_field_MV_per_cm: float | None = None,
    ):
        """把用户侧物理电场转换成 solver 内部 code-unit callable。

        这里显式完成 `t_code -> t_fs` 和 `E_MV_per_cm -> E_code`。生成的对象
        是内部 adapter，不是用户侧输入 API。
        """

        if not isinstance(field_phy, FieldPhyRoot):
            raise TypeError("field_phy must be a FieldPhyRoot instance.")
        s = self._require_solver(solver)
        reference = reference_field_MV_per_cm
        if reference is None:
            physical = getattr(self, "last_physical", None)
            if physical is None:
                raise RuntimeError("reference_field_MV_per_cm is required without last_physical.")
            reference = physical.field_MV_per_cm
        return make_code_field_adapter(
            field_phy,
            self,
            s,
            reference_field_MV_per_cm=float(reference),
        )

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
        physical_payload = {
            item.name: getattr(p, item.name)
            for item in dataclass_fields(p)
            if item.name != "field"
        }
        physical_payload["field"] = None if p.field is None else p.field.to_dict()
        return {
            "physical": physical_payload,
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


__all__ = ["ParaNormalizer"]
