"""结果容器。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any
from typing import TYPE_CHECKING

import numpy as np
from qutip import Qobj

from .normalization import PhysicalParams, SolverParams
from .parameters import OpticalBlochParameters

if TYPE_CHECKING:
    from .multilevel import MultiLevelParameters


def _require_pandas():
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("pandas is required for dataframe export. Install pandas in this environment.") from exc
    return pd


def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, Qobj):
        return {
            "qobj_shape": list(value.shape),
            "data": _complex_matrix_to_json(value.full()),
        }
    if isinstance(value, np.ndarray):
        if np.iscomplexobj(value):
            return _complex_matrix_to_json(value)
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _complex_matrix_to_json(value: Any) -> list:
    array = np.asarray(value, dtype=np.complex128)
    return [[{"real": float(item.real), "imag": float(item.imag)} for item in row] for row in array]


def _time_axis_fs(times: np.ndarray, times_fs: np.ndarray | None) -> np.ndarray:
    if times_fs is not None:
        return np.asarray(times_fs, dtype=float)
    return np.asarray(times, dtype=float)


@dataclass
class DynamicsResult:
    """Single trajectory result from one simulation mode or derived view."""

    mode: str
    times: np.ndarray
    times_fs: np.ndarray | None
    states: list[Qobj]
    parameters: Any
    physical_params: PhysicalParams | None = None
    solver_params: SolverParams | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    source_mode: str | None = None
    sanity_checks: dict[str, Any] = field(default_factory=dict)

    def density_array(self) -> np.ndarray:
        return np.stack([state.full() for state in self.states], axis=0)

    def dimension(self) -> int:
        return int(self.density_array().shape[1])

    def populations(self) -> np.ndarray:
        density = self.density_array()
        return np.stack([density[:, index, index] for index in range(density.shape[1])], axis=1)

    def matrix_element(self, i: int, j: int) -> np.ndarray:
        return self.density_array()[:, i, j]

    def matrix_elements(self, pairs: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> dict[str, np.ndarray]:
        return {f"rho{i + 1}{j + 1}": self.matrix_element(i, j) for i, j in pairs}

    def selected_elements(self, elements: dict[str, tuple[int, int]]) -> dict[str, np.ndarray]:
        return {label: self.matrix_element(i, j) for label, (i, j) in elements.items()}

    def components(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self.dimension() != 2:
            raise ValueError("components() is only defined for two-level results.")
        density = self.density_array()
        return density[:, 0, 0], density[:, 1, 1], density[:, 0, 1], density[:, 1, 0]

    def max_trace_error(self) -> float:
        density = self.density_array()
        traces = np.trace(density, axis1=1, axis2=2)
        return float(np.max(np.abs(traces - 1.0)))

    def max_hermiticity_error(self) -> float:
        density = self.density_array()
        return float(np.max(np.abs(density - np.conjugate(np.swapaxes(density, 1, 2)))))

    def summary_dict(self) -> dict[str, Any]:
        summary: dict[str, Any] = {
            "mode": self.mode,
            "dimension": self.dimension(),
            "max trace error": f"{self.max_trace_error():.3e}",
            "max Hermitian error": f"{self.max_hermiticity_error():.3e}",
        }
        populations = self.populations()
        summary["final_populations"] = [float(value.real) for value in populations[-1]]
        if self.dimension() == 2:
            rho11, rho22, rho12, rho21 = self.components()
            summary.update(
                {
                    "rho11(final)": f"{rho11[-1].real:.6f}",
                    "rho22(final)": f"{rho22[-1].real:.6f}",
                    "rho12(final)": f"{rho12[-1].real:.6f} + {rho12[-1].imag:.6f}i",
                    "rho21(final)": f"{rho21[-1].real:.6f} + {rho21[-1].imag:.6f}i",
                }
            )
        times = _time_axis_fs(self.times, self.times_fs)
        summary["time_range_fs"] = f"{times[0]:.6f} -> {times[-1]:.6f}"
        return summary

    def metadata_dict(self) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "result_type": "DynamicsResult",
            "mode": self.mode,
            "source_mode": self.source_mode,
            "summary": self.summary_dict(),
            "sanity_checks": self.sanity_checks,
            "metadata": self.metadata,
            "density_matrix_unit": "dimensionless",
            "population_unit": "dimensionless",
            "coherence_unit": "dimensionless",
            "time_axis_saved_as": "time_fs",
            "time_axis_unit": "fs",
            "time_fs_is_physical": self.times_fs is not None,
            "n_time_points": int(len(self.times)),
            "dimension": self.dimension(),
            "parameters": self.parameters,
        }
        if self.physical_params is not None:
            metadata["physical_params"] = self.physical_params
        if self.solver_params is not None:
            metadata["solver_params"] = self.solver_params
        return _json_safe(metadata)

    def parameter_summary_dict(self) -> dict[str, Any]:
        return self.metadata_dict()

    def to_npz_dict(self) -> dict[str, np.ndarray]:
        return {
            "time_fs": _time_axis_fs(self.times, self.times_fs),
            "time_code": np.asarray(self.times, dtype=float),
            "density": self.density_array(),
        }

    def components_dataframe(self):
        if self.dimension() != 2:
            raise ValueError("components_dataframe() is only defined for two-level results.")
        pd = _require_pandas()
        rho11, rho22, rho12, rho21 = self.components()
        return pd.DataFrame(
            {
                "time_fs": _time_axis_fs(self.times, self.times_fs),
                "rho11": rho11.real,
                "rho22": rho22.real,
                "Re_rho12": rho12.real,
                "Im_rho12": rho12.imag,
                "Re_rho21": rho21.real,
                "Im_rho21": rho21.imag,
            }
        )

    def populations_dataframe(self):
        pd = _require_pandas()
        populations = self.populations()
        data: dict[str, Any] = {"time_fs": _time_axis_fs(self.times, self.times_fs)}
        for index in range(populations.shape[1]):
            data[f"rho{index + 1}{index + 1}"] = populations[:, index].real
        return pd.DataFrame(data)

    def selected_elements_dataframe(self, elements: dict[str, tuple[int, int]]):
        pd = _require_pandas()
        data: dict[str, Any] = {"time_fs": _time_axis_fs(self.times, self.times_fs)}
        for label, values in self.selected_elements(elements).items():
            data[f"Re_{label}"] = values.real
            data[f"Im_{label}"] = values.imag
        return pd.DataFrame(data)

    def plot_times_and_label(self) -> tuple[np.ndarray, str]:
        if self.times_fs is not None:
            return self.times_fs, "Time (fs)"
        return self.times, "Time"


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

    def available_frames(self) -> tuple[str, ...]:
        return ("lab", "rotating", "rwa")

    def to_npz_dict(self) -> dict[str, np.ndarray]:
        times_fs = _time_axis_fs(self.times, self.times_fs)
        payload: dict[str, np.ndarray] = {
            "time_fs": times_fs,
            "time_code": np.asarray(self.times, dtype=float),
        }
        for frame in self.available_frames():
            payload[f"density_{frame}"] = self.density_array(frame)
        return payload

    def components(self, frame: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        density = self.density_array(frame)
        return density[:, 0, 0], density[:, 1, 1], density[:, 0, 1], density[:, 1, 0]

    def components_dataframe(self, frame: str):
        pd = _require_pandas()
        rho11, rho22, rho12, rho21 = self.components(frame)
        return pd.DataFrame(
            {
                "time_fs": _time_axis_fs(self.times, self.times_fs),
                "rho11": rho11.real,
                "rho22": rho22.real,
                "Re_rho12": rho12.real,
                "Im_rho12": rho12.imag,
                "Re_rho21": rho21.real,
                "Im_rho21": rho21.imag,
            }
        )

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

    def metadata_dict(self) -> dict[str, Any]:
        metadata = self.parameter_summary_dict()
        metadata.update(
            {
                "result_type": "OpticalBlochResult",
                "available_frames": list(self.available_frames()),
                "density_matrix_unit": "dimensionless",
                "population_unit": "dimensionless",
                "coherence_unit": "dimensionless",
                "time_axis_saved_as": "time_fs",
                "time_axis_unit": "fs",
                "time_fs_is_physical": self.times_fs is not None,
                "n_time_points": int(len(self.times)),
            }
        )
        return _json_safe(metadata)

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

    def to_npz_dict(self) -> dict[str, np.ndarray]:
        return {
            "time_fs": _time_axis_fs(self.times, self.times_fs),
            "time_code": np.asarray(self.times, dtype=float),
            "density_lab": self.density_array(),
        }

    def dimension(self) -> int:
        return int(self.density_array().shape[1])

    def populations(self) -> np.ndarray:
        density = self.density_array()
        return np.stack([density[:, index, index] for index in range(density.shape[1])], axis=1)

    def populations_dataframe(self):
        pd = _require_pandas()
        populations = self.populations()
        data: dict[str, Any] = {"time_fs": _time_axis_fs(self.times, self.times_fs)}
        for index in range(populations.shape[1]):
            data[f"rho{index + 1}{index + 1}"] = populations[:, index].real
        return pd.DataFrame(data)

    def matrix_element(self, i: int, j: int) -> np.ndarray:
        density = self.density_array()
        return density[:, i, j]

    def matrix_elements(self, pairs: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> dict[str, np.ndarray]:
        return {f"rho_{i + 1}{j + 1}": self.matrix_element(i, j) for i, j in pairs}

    def selected_elements(self, elements: dict[str, tuple[int, int]]) -> dict[str, np.ndarray]:
        return {label: self.matrix_element(i, j) for label, (i, j) in elements.items()}

    def selected_elements_dataframe(self, elements: dict[str, tuple[int, int]]):
        pd = _require_pandas()
        data: dict[str, Any] = {"time_fs": _time_axis_fs(self.times, self.times_fs)}
        for label, values in self.selected_elements(elements).items():
            data[f"Re_{label}"] = values.real
            data[f"Im_{label}"] = values.imag
        return pd.DataFrame(data)

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

    def metadata_dict(self) -> dict[str, Any]:
        metadata = self.parameter_summary_dict()
        metadata.update(
            {
                "result_type": "MultiLevelResult",
                "available_frames": ["lab"],
                "density_matrix_unit": "dimensionless",
                "population_unit": "dimensionless",
                "coherence_unit": "dimensionless",
                "time_axis_saved_as": "time_fs",
                "time_axis_unit": "fs",
                "time_fs_is_physical": self.times_fs is not None,
                "n_time_points": int(len(self.times)),
                "dimension": self.dimension(),
            }
        )
        return _json_safe(metadata)

    def plot_times_and_label(self) -> tuple[np.ndarray, str]:
        if self.times_fs is not None:
            return self.times_fs, "Time (fs)"
        return self.times, "Time"


__all__ = ["DynamicsResult", "OpticalBlochResult", "MultiLevelResult"]
