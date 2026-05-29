"""真实物理量到内部求解参数的归一化工具。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class PhysicalParams:
    """用户输入的真实物理参数。"""

    energy_gap_eV: float
    laser_energy_eV: float
    dipole_D: float
    field_MV_per_cm: float

    t_start_fs: float
    t_end_fs: float
    dt_fs: float

    T1_fs: Optional[float] = None
    T2_fs: Optional[float] = None
    Tphi_fs: Optional[float] = None

    pulse_center_fs: Optional[float] = None
    pulse_sigma_fs: Optional[float] = None


@dataclass
class SolverParams:
    """内部求解参数。"""

    time_scale_fs: float

    omega_eg_fs_inv: float
    omega_L_fs_inv: float
    detuning_fs_inv: float
    rabi_fs_inv: float
    gamma1_fs_inv: float
    gamma_phi_fs_inv: float
    gamma2_fs_inv: float

    omega_eg: float
    omega_L: float
    detuning: float
    rabi: float
    gamma1: float
    gamma_phi: float
    gamma2: float

    t_start: float
    t_end: float
    dt: float
    tlist: np.ndarray

    pulse_center: Optional[float] = None
    pulse_sigma: Optional[float] = None

    pulse_center_fs: Optional[float] = None
    pulse_sigma_fs: Optional[float] = None


class ParaNormalizer:
    """参数归一化器。"""

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
        self.last_physical: Optional[PhysicalParams] = None
        self.last_solver: Optional[SolverParams] = None

    @classmethod
    def energy_eV_to_fs_inv(cls, energy_eV: float) -> float:
        return energy_eV * cls.EV_TO_FS_INV

    @classmethod
    def fs_inv_to_energy_eV(cls, omega_fs_inv: float) -> float:
        return omega_fs_inv / cls.EV_TO_FS_INV

    @classmethod
    def rate_from_time_fs(cls, T_fs: Optional[float]) -> float:
        if T_fs is None:
            return 0.0
        if T_fs <= 0:
            raise ValueError("时间常数必须为正数。")
        return 1.0 / T_fs

    @classmethod
    def rabi_fs_inv_from_mu_and_field(cls, dipole_D: float, field_MV_per_cm: float) -> float:
        return dipole_D * field_MV_per_cm * cls.DIPOLE_FIELD_TO_RABI_FS_INV

    def normalize(self, p: PhysicalParams) -> SolverParams:
        self._validate_physical_params(p)

        omega_eg_fs_inv = self.energy_eV_to_fs_inv(p.energy_gap_eV)
        omega_L_fs_inv = self.energy_eV_to_fs_inv(p.laser_energy_eV)
        detuning_fs_inv = omega_eg_fs_inv - omega_L_fs_inv
        rabi_fs_inv = self.rabi_fs_inv_from_mu_and_field(p.dipole_D, p.field_MV_per_cm)

        gamma1_fs_inv = self.rate_from_time_fs(p.T1_fs)
        if p.Tphi_fs is not None:
            gamma_phi_fs_inv = self.rate_from_time_fs(p.Tphi_fs)
        elif p.T2_fs is not None:
            gamma2_input = self.rate_from_time_fs(p.T2_fs)
            gamma_phi_fs_inv = max(0.0, gamma2_input - 0.5 * gamma1_fs_inv)
        else:
            gamma_phi_fs_inv = 0.0
        gamma2_fs_inv = gamma_phi_fs_inv + 0.5 * gamma1_fs_inv

        time_scale_fs = self._choose_time_scale_fs(
            user_time_scale_fs=self.user_time_scale_fs,
            omega_eg_fs_inv=omega_eg_fs_inv,
            omega_L_fs_inv=omega_L_fs_inv,
            detuning_fs_inv=detuning_fs_inv,
            rabi_fs_inv=rabi_fs_inv,
            gamma1_fs_inv=gamma1_fs_inv,
            gamma_phi_fs_inv=gamma_phi_fs_inv,
            pulse_sigma_fs=p.pulse_sigma_fs,
        )

        omega_eg = omega_eg_fs_inv * time_scale_fs
        omega_L = omega_L_fs_inv * time_scale_fs
        detuning = detuning_fs_inv * time_scale_fs
        rabi = rabi_fs_inv * time_scale_fs
        gamma1 = gamma1_fs_inv * time_scale_fs
        gamma_phi = gamma_phi_fs_inv * time_scale_fs
        gamma2 = gamma2_fs_inv * time_scale_fs

        t_start = p.t_start_fs / time_scale_fs
        t_end = p.t_end_fs / time_scale_fs
        dt = p.dt_fs / time_scale_fs
        tlist = self._build_tlist(t_start, t_end, dt)

        pulse_center = None
        pulse_sigma = None
        if p.pulse_center_fs is not None:
            pulse_center = p.pulse_center_fs / time_scale_fs
        if p.pulse_sigma_fs is not None:
            pulse_sigma = p.pulse_sigma_fs / time_scale_fs

        solver = SolverParams(
            time_scale_fs=time_scale_fs,
            omega_eg_fs_inv=omega_eg_fs_inv,
            omega_L_fs_inv=omega_L_fs_inv,
            detuning_fs_inv=detuning_fs_inv,
            rabi_fs_inv=rabi_fs_inv,
            gamma1_fs_inv=gamma1_fs_inv,
            gamma_phi_fs_inv=gamma_phi_fs_inv,
            gamma2_fs_inv=gamma2_fs_inv,
            omega_eg=omega_eg,
            omega_L=omega_L,
            detuning=detuning,
            rabi=rabi,
            gamma1=gamma1,
            gamma_phi=gamma_phi,
            gamma2=gamma2,
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

    def _validate_physical_params(self, p: PhysicalParams) -> None:
        if p.energy_gap_eV <= 0:
            raise ValueError("energy_gap_eV 必须为正。")
        if p.t_end_fs <= p.t_start_fs:
            raise ValueError("t_end_fs 必须大于 t_start_fs。")
        if p.dt_fs <= 0:
            raise ValueError("dt_fs 必须为正。")
        if p.T1_fs is not None and p.T1_fs <= 0:
            raise ValueError("T1_fs 必须为正。")
        if p.T2_fs is not None and p.T2_fs <= 0:
            raise ValueError("T2_fs 必须为正。")
        if p.Tphi_fs is not None and p.Tphi_fs <= 0:
            raise ValueError("Tphi_fs 必须为正。")
        if p.pulse_sigma_fs is not None and p.pulse_sigma_fs <= 0:
            raise ValueError("pulse_sigma_fs 必须为正。")

    def _choose_time_scale_fs(
        self,
        user_time_scale_fs: Optional[float],
        omega_eg_fs_inv: float,
        omega_L_fs_inv: float,
        detuning_fs_inv: float,
        rabi_fs_inv: float,
        gamma1_fs_inv: float,
        gamma_phi_fs_inv: float,
        pulse_sigma_fs: Optional[float],
    ) -> float:
        if user_time_scale_fs is not None:
            if user_time_scale_fs <= 0:
                raise ValueError("time_scale_fs 必须为正。")
            return user_time_scale_fs
        if not self.auto_scale:
            return 1.0

        candidates = []
        for value in [
            abs(detuning_fs_inv),
            abs(rabi_fs_inv),
            abs(gamma1_fs_inv),
            abs(gamma_phi_fs_inv),
        ]:
            if value > 0:
                candidates.append(value)
        if pulse_sigma_fs is not None and pulse_sigma_fs > 0:
            candidates.append(1.0 / pulse_sigma_fs)
        if not candidates and abs(omega_eg_fs_inv) > 0:
            candidates.append(abs(omega_eg_fs_inv))
        if not candidates:
            return 1.0
        dominant_rate_fs_inv = max(candidates)
        return 1.0 / dominant_rate_fs_inv

    @staticmethod
    def _build_tlist(t_start: float, t_end: float, dt: float) -> np.ndarray:
        n = int(np.floor((t_end - t_start) / dt)) + 1
        return t_start + np.arange(n) * dt

    def denormalize_time(self, t_code: float, solver: Optional[SolverParams] = None) -> float:
        s = self._require_solver(solver)
        return t_code * s.time_scale_fs

    def denormalize_time_array(self, t_code_array: np.ndarray, solver: Optional[SolverParams] = None) -> np.ndarray:
        s = self._require_solver(solver)
        return np.asarray(t_code_array) * s.time_scale_fs

    def denormalize_rate_to_fs_inv(self, rate_code: float, solver: Optional[SolverParams] = None) -> float:
        s = self._require_solver(solver)
        return rate_code / s.time_scale_fs

    def denormalize_energy_to_eV(self, omega_code: float, solver: Optional[SolverParams] = None) -> float:
        s = self._require_solver(solver)
        omega_fs_inv = omega_code / s.time_scale_fs
        return self.fs_inv_to_energy_eV(omega_fs_inv)

    def _require_solver(self, solver: Optional[SolverParams]) -> SolverParams:
        if solver is not None:
            return solver
        if self.last_solver is None:
            raise RuntimeError("还没有调用 normalize()，无法反变换。")
        return self.last_solver

    def summary_dict(
        self,
        physical: Optional[PhysicalParams] = None,
        solver: Optional[SolverParams] = None,
    ) -> Dict[str, Any]:
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
                "omega_eg_fs_inv": s.omega_eg_fs_inv,
                "omega_L_fs_inv": s.omega_L_fs_inv,
                "detuning_fs_inv": s.detuning_fs_inv,
                "rabi_fs_inv": s.rabi_fs_inv,
                "gamma1_fs_inv": s.gamma1_fs_inv,
                "gamma_phi_fs_inv": s.gamma_phi_fs_inv,
                "gamma2_fs_inv": s.gamma2_fs_inv,
            },
            "solver_params_code": {
                "omega_eg": s.omega_eg,
                "omega_L": s.omega_L,
                "detuning": s.detuning,
                "rabi": s.rabi,
                "gamma1": s.gamma1,
                "gamma_phi": s.gamma_phi,
                "gamma2": s.gamma2,
                "t_start": s.t_start,
                "t_end": s.t_end,
                "dt": s.dt,
            },
            "time_ranges": {
                "t_start_fs": p.t_start_fs,
                "t_end_fs": p.t_end_fs,
                "dt_fs": p.dt_fs,
                "t_start_code": s.t_start,
                "t_end_code": s.t_end,
                "dt_code": s.dt,
                "n_steps": len(s.tlist),
            },
            "pulse": {
                "pulse_center_fs": s.pulse_center_fs,
                "pulse_sigma_fs": s.pulse_sigma_fs,
                "pulse_center": s.pulse_center,
                "pulse_sigma": s.pulse_sigma,
            },
            "sanity_check_ratios": {
                "Omega_over_omega_eg": abs(s.rabi_fs_inv / s.omega_eg_fs_inv) if s.omega_eg_fs_inv != 0 else None,
                "Delta_over_omega_eg": abs(s.detuning_fs_inv / s.omega_eg_fs_inv) if s.omega_eg_fs_inv != 0 else None,
                "gamma1_over_omega_eg": abs(s.gamma1_fs_inv / s.omega_eg_fs_inv) if s.omega_eg_fs_inv != 0 else None,
                "gamma_phi_over_omega_eg": abs(s.gamma_phi_fs_inv / s.omega_eg_fs_inv) if s.omega_eg_fs_inv != 0 else None,
                "gamma2_over_omega_eg": abs(s.gamma2_fs_inv / s.omega_eg_fs_inv) if s.omega_eg_fs_inv != 0 else None,
            },
        }

    def summary_text(
        self,
        physical: Optional[PhysicalParams] = None,
        solver: Optional[SolverParams] = None,
    ) -> str:
        p = physical if physical is not None else self.last_physical
        s = solver if solver is not None else self.last_solver
        if p is None or s is None:
            raise RuntimeError("没有可用的 physical / solver 参数。")

        lines = [
            "=== ParaNormalizer Summary ===",
            "",
            "[Physical input]",
            f"energy_gap_eV      = {p.energy_gap_eV}",
            f"laser_energy_eV    = {p.laser_energy_eV}",
            f"dipole_D           = {p.dipole_D}",
            f"field_MV_per_cm    = {p.field_MV_per_cm}",
            f"t_start_fs         = {p.t_start_fs}",
            f"t_end_fs           = {p.t_end_fs}",
            f"dt_fs              = {p.dt_fs}",
            f"T1_fs              = {p.T1_fs}",
            f"T2_fs              = {p.T2_fs}",
            f"Tphi_fs            = {p.Tphi_fs}",
            "",
            "[Converted physical solver units: fs^-1]",
            f"omega_eg_fs_inv    = {s.omega_eg_fs_inv:.6g}",
            f"omega_L_fs_inv     = {s.omega_L_fs_inv:.6g}",
            f"detuning_fs_inv    = {s.detuning_fs_inv:.6g}",
            f"rabi_fs_inv        = {s.rabi_fs_inv:.6g}",
            f"gamma1_fs_inv      = {s.gamma1_fs_inv:.6g}",
            f"gamma_phi_fs_inv   = {s.gamma_phi_fs_inv:.6g}",
            f"gamma2_fs_inv      = {s.gamma2_fs_inv:.6g}",
            "",
            "[Internal code units]",
            f"time_scale_fs      = {s.time_scale_fs:.6g}",
            f"omega_eg           = {s.omega_eg:.6g}",
            f"omega_L            = {s.omega_L:.6g}",
            f"detuning           = {s.detuning:.6g}",
            f"rabi               = {s.rabi:.6g}",
            f"gamma1             = {s.gamma1:.6g}",
            f"gamma_phi          = {s.gamma_phi:.6g}",
            f"gamma2             = {s.gamma2:.6g}",
            f"t_start            = {s.t_start:.6g}",
            f"t_end              = {s.t_end:.6g}",
            f"dt                 = {s.dt:.6g}",
            f"n_steps            = {len(s.tlist)}",
            "",
            "[Time ranges]",
            f"real time range(fs)= [{p.t_start_fs:.6g}, {p.t_end_fs:.6g}]",
            f"code time range    = [{s.t_start:.6g}, {s.t_end:.6g}]",
            "",
            "[Pulse]",
            f"pulse_center_fs    = {s.pulse_center_fs}",
            f"pulse_sigma_fs     = {s.pulse_sigma_fs}",
            f"pulse_center       = {s.pulse_center}",
            f"pulse_sigma        = {s.pulse_sigma}",
            "",
            "[Sanity ratios]",
            f"Omega / omega_eg   = {abs(s.rabi_fs_inv / s.omega_eg_fs_inv) if s.omega_eg_fs_inv != 0 else None}",
            f"Delta / omega_eg   = {abs(s.detuning_fs_inv / s.omega_eg_fs_inv) if s.omega_eg_fs_inv != 0 else None}",
            f"gamma1             = {s.gamma1_fs_inv:.6g}",
            f"gamma_phi          = {s.gamma_phi_fs_inv:.6g}",
            f"gamma2             = {s.gamma2_fs_inv:.6g}",
            "",
            "[Notes]",
            "1. 密度矩阵 rho、本征布居、相干本身是无量纲的，不需要反变换。",
            "2. 时间轴需要乘回 time_scale_fs 才恢复到 fs。",
            "3. 如果你在旋转系 / RWA 下建模，auto_scale 通常更合理。",
        ]
        return "\n".join(lines)


__all__ = ["ParaNormalizer", "PhysicalParams", "SolverParams"]
