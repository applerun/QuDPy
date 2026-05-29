"""结果容器。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from typing import TYPE_CHECKING

import numpy as np
from qutip import Qobj

from .normalization import PhysicalParams, SolverParams
from .parameters import OpticalBlochParameters

if TYPE_CHECKING:
    from .multilevel import MultiLevelParameters


@dataclass
class OpticalBlochResult:
    """保存单个参数点下的全部计算结果。"""

    parameters: OpticalBlochParameters
    epsilon_2: float
    times: np.ndarray
    times_fs: np.ndarray | None
    lab_states: list[Qobj]
    rotating_states: list[Qobj]
    rwa_states: list[Qobj]
    physical_params: PhysicalParams | None = None
    solver_params: SolverParams | None = None
    sanity_checks: dict[str, Any] = field(default_factory=dict)

    def density_array(self, frame: str) -> np.ndarray:
        state_map = {
            "lab": self.lab_states,
            "rotating": self.rotating_states,
            "rwa": self.rwa_states,
        }
        try:
            states = state_map[frame]
        except KeyError as exc:
            raise ValueError(f"未知参考系: {frame}") from exc
        return np.stack([state.full() for state in states], axis=0)

    def components(self, frame: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        density = self.density_array(frame)
        return density[:, 0, 0], density[:, 1, 1], density[:, 0, 1], density[:, 1, 0]

    def max_trace_error(self, frame: str = "lab") -> float:
        rho11, rho22, _rho12, _rho21 = self.components(frame)
        return float(np.max(np.abs((rho11 + rho22) - 1.0)))

    def max_hermiticity_error(self, frame: str = "lab") -> float:
        _rho11, _rho22, rho12, rho21 = self.components(frame)
        return float(np.max(np.abs(rho21 - np.conjugate(rho12))))

    def summary_dict(self) -> dict[str, str]:
        rho11, rho22, rho12, rho21 = self.components("lab")
        summary = {
            "epsilon_1": f"{self.parameters.epsilon_1:.6f}",
            "epsilon_2": f"{self.epsilon_2:.6f}",
            "rho11_lab(final)": f"{rho11[-1].real:.6f}",
            "rho22_lab(final)": f"{rho22[-1].real:.6f}",
            "rho12_lab(final)": f"{rho12[-1].real:.6f} + {rho12[-1].imag:.6f}i",
            "rho21_lab(final)": f"{rho21[-1].real:.6f} + {rho21[-1].imag:.6f}i",
            "max trace error": f"{self.max_trace_error('lab'):.3e}",
            "max Hermitian error": f"{self.max_hermiticity_error('lab'):.3e}",
        }
        if self.times_fs is not None:
            summary["time_range_fs"] = f"{self.times_fs[0]:.6f} -> {self.times_fs[-1]:.6f}"
        return summary

    def parameter_summary_dict(self) -> dict[str, Any]:
        summary: dict[str, Any] = {
            "result_summary": self.summary_dict(),
            "sanity_checks": self.sanity_checks,
        }
        if self.physical_params is not None:
            summary["physical_params"] = self.physical_params.__dict__
        if self.solver_params is not None:
            summary["solver_params_fs_inv"] = {
                "omega_eg_fs_inv": self.solver_params.omega_eg_fs_inv,
                "omega_L_fs_inv": self.solver_params.omega_L_fs_inv,
                "detuning_fs_inv": self.solver_params.detuning_fs_inv,
                "rabi_fs_inv": self.solver_params.rabi_fs_inv,
                "gamma1_fs_inv": self.solver_params.gamma1_fs_inv,
                "gamma_phi_fs_inv": self.solver_params.gamma_phi_fs_inv,
                "gamma2_fs_inv": self.solver_params.gamma2_fs_inv,
            }
            summary["solver_params_code"] = {
                "omega_eg": self.solver_params.omega_eg,
                "omega_L": self.solver_params.omega_L,
                "detuning": self.solver_params.detuning,
                "rabi": self.solver_params.rabi,
                "gamma1": self.solver_params.gamma1,
                "gamma_phi": self.solver_params.gamma_phi,
                "gamma2": self.solver_params.gamma2,
            }
            summary["time_scale_fs"] = self.solver_params.time_scale_fs
            summary["tlist_ranges"] = {
                "t_start_fs": float(self.times_fs[0]) if self.times_fs is not None else None,
                "t_end_fs": float(self.times_fs[-1]) if self.times_fs is not None else None,
                "t_start_code": float(self.times[0]),
                "t_end_code": float(self.times[-1]),
            }
            summary["sanity_ratios"] = {
                "Omega_over_omega_eg": (
                    abs(self.solver_params.rabi_fs_inv / self.solver_params.omega_eg_fs_inv)
                    if self.solver_params.omega_eg_fs_inv != 0
                    else None
                ),
                "Delta_over_omega_eg": (
                    abs(self.solver_params.detuning_fs_inv / self.solver_params.omega_eg_fs_inv)
                    if self.solver_params.omega_eg_fs_inv != 0
                    else None
                ),
                "gamma1": self.solver_params.gamma1_fs_inv,
                "gamma_phi": self.solver_params.gamma_phi_fs_inv,
                "gamma2": self.solver_params.gamma2_fs_inv,
            }
        return summary

    def plot_times_and_label(self) -> tuple[np.ndarray, str]:
        if self.times_fs is not None:
            return self.times_fs, "Time (fs)"
        return self.times, "Time"


@dataclass
class MultiLevelResult:
    """保存多能级实验室系求解结果。"""

    parameters: "MultiLevelParameters"
    times: np.ndarray
    times_fs: np.ndarray | None
    lab_states: list[Qobj]
    sanity_checks: dict[str, Any] = field(default_factory=dict)

    def density_array(self) -> np.ndarray:
        return np.stack([state.full() for state in self.lab_states], axis=0)

    def dimension(self) -> int:
        return int(self.density_array().shape[1])

    def populations(self) -> np.ndarray:
        density = self.density_array()
        return np.stack([density[:, index, index] for index in range(density.shape[1])], axis=1)

    def matrix_element(self, i: int, j: int) -> np.ndarray:
        density = self.density_array()
        return density[:, i, j]

    def matrix_elements(self, pairs: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> dict[str, np.ndarray]:
        return {f"rho_{i + 1}{j + 1}": self.matrix_element(i, j) for i, j in pairs}

    def selected_elements(self, elements: dict[str, tuple[int, int]]) -> dict[str, np.ndarray]:
        return {label: self.matrix_element(i, j) for label, (i, j) in elements.items()}

    def max_trace_error(self) -> float:
        density = self.density_array()
        traces = np.trace(density, axis1=1, axis2=2)
        return float(np.max(np.abs(traces - 1.0)))

    def max_hermiticity_error(self) -> float:
        density = self.density_array()
        return float(np.max(np.abs(density - np.conjugate(np.swapaxes(density, 1, 2)))))

    def summary_dict(self) -> dict[str, Any]:
        populations = self.populations()
        summary: dict[str, Any] = {
            "dimension": self.dimension(),
            "final_populations": [float(value.real) for value in populations[-1]],
            "max trace error": f"{self.max_trace_error():.3e}",
            "max Hermitian error": f"{self.max_hermiticity_error():.3e}",
        }
        if self.times_fs is not None:
            summary["time_range_fs"] = f"{self.times_fs[0]:.6f} -> {self.times_fs[-1]:.6f}"
        else:
            summary["time_range_code"] = f"{self.times[0]:.6f} -> {self.times[-1]:.6f}"
        return summary

    def parameter_summary_dict(self) -> dict[str, Any]:
        dipole = np.asarray(self.parameters.dipole_matrix, dtype=np.complex128)
        return {
            "result_summary": self.summary_dict(),
            "sanity_checks": self.sanity_checks,
            "parameters": {
                "energies": list(self.parameters.energies),
                "fields": [field.__dict__ for field in self.parameters.fields],
                "collapse_channels": [channel.__dict__ for channel in self.parameters.collapse_channels],
                "t_start": self.parameters.t_start,
                "t_end": self.parameters.t_end,
                "dt": self.parameters.dt,
                "times_fs_available": self.times_fs is not None,
                "dipole_matrix": [
                    [{"real": float(value.real), "imag": float(value.imag)} for value in row] for row in dipole
                ],
            },
        }

    def plot_times_and_label(self) -> tuple[np.ndarray, str]:
        if self.times_fs is not None:
            return self.times_fs, "Time (fs)"
        return self.times, "Time"


__all__ = ["OpticalBlochResult", "MultiLevelResult"]
