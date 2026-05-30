"""Single-trajectory quantum dynamics result container."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any

import numpy as np
from qutip import Qobj

from .normalization import PhysicalParams, SolverParams


def _require_pandas():
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("pandas is required for dataframe export. Install pandas in this environment.") from exc
    return pd


def _complex_matrix_to_json(value: Any) -> list:
    array = np.asarray(value, dtype=np.complex128)
    return [[{"real": float(item.real), "imag": float(item.imag)} for item in row] for row in array]


def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, Qobj):
        return {"qobj_shape": list(value.shape), "data": _complex_matrix_to_json(value.full())}
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


def _time_axis_fs(times: np.ndarray, times_fs: np.ndarray | None) -> np.ndarray:
    if times_fs is not None:
        return np.asarray(times_fs, dtype=float)
    return np.asarray(times, dtype=float)


def _solver_params_fs_inv_dict(solver: SolverParams) -> dict[str, Any]:
    return {
        "time_scale_fs": solver.time_scale_fs,
        "omega_eg_fs_inv": solver.omega_eg_fs_inv,
        "omega_L_fs_inv": solver.omega_L_fs_inv,
        "detuning_fs_inv": solver.detuning_fs_inv,
        "rabi_fs_inv": solver.rabi_fs_inv,
        "gamma1_fs_inv": solver.gamma1_fs_inv,
        "gamma_phi_fs_inv": solver.gamma_phi_fs_inv,
        "gamma2_fs_inv": solver.gamma2_fs_inv,
    }


def _solver_params_code_dict(solver: SolverParams) -> dict[str, Any]:
    return {
        "omega_eg_code": solver.omega_eg,
        "omega_L_code": solver.omega_L,
        "detuning_code": solver.detuning,
        "rabi_code": solver.rabi,
        "gamma1_code": solver.gamma1,
        "gamma_phi_code": solver.gamma_phi,
        "gamma2_code": solver.gamma2,
        "t_start_code": solver.t_start,
        "t_end_code": solver.t_end,
        "dt_code": solver.dt,
        "tlist_code": solver.tlist,
        "pulse_center_code": solver.pulse_center,
        "pulse_sigma_code": solver.pulse_sigma,
    }


@dataclass
class DynamicsResult:
    """One result object for one trajectory from one simulation mode or derived view."""

    mode: str
    times: np.ndarray
    times_fs: np.ndarray | None
    states: list[Qobj]
    parameters: Any
    physical_params: PhysicalParams | None = None
    solver_params: SolverParams | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    source_mode: str | None = None
    drive: Any | None = None
    drive_dict: dict[str, Any] | None = None
    drive_expr: str | None = None
    drive_name: str | None = None
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

    def drive_code_values(self, times: np.ndarray | None = None) -> np.ndarray | None:
        if self.drive is None:
            return None
        sample_times = self.times if times is None else np.asarray(times, dtype=float)
        return np.asarray(self.drive(sample_times), dtype=float)

    def drive_fs_inv_values(self, times: np.ndarray | None = None) -> np.ndarray | None:
        drive_code = self.drive_code_values(times)
        if drive_code is None or self.solver_params is None or self.mode != "rwa":
            return None
        return np.asarray(drive_code, dtype=float) / float(self.solver_params.time_scale_fs)

    def field_MV_per_cm_values(
        self,
        times: np.ndarray | None = None,
        *,
        times_fs: np.ndarray | None = None,
    ) -> np.ndarray | None:
        if self.mode != "lab_exact" or self.physical_params is None or self.solver_params is None:
            return None
        if times_fs is None:
            if times is None:
                if self.times_fs is None:
                    return None
                sample_times_fs = np.asarray(self.times_fs, dtype=float)
            elif self.times_fs is not None and len(times) == len(self.times_fs):
                sample_times_fs = np.asarray(self.times_fs, dtype=float)
            else:
                return None
        else:
            sample_times_fs = np.asarray(times_fs, dtype=float)
        phase = float(getattr(self.drive, "phase", 0.0)) if self.drive is not None else 0.0
        return 2.0 * float(self.physical_params.field_MV_per_cm) * np.cos(
            float(self.solver_params.omega_L_fs_inv) * sample_times_fs + phase
        )

    def drive_values(self, times: np.ndarray | None = None) -> np.ndarray | None:
        return self.drive_code_values(times)

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
            "mode": self.mode,
            "dimension": self.dimension(),
            "final_populations": [float(value.real) for value in populations[-1]],
            "max trace error": f"{self.max_trace_error():.3e}",
            "max Hermitian error": f"{self.max_hermiticity_error():.3e}",
            "time_range_fs": f"{_time_axis_fs(self.times, self.times_fs)[0]:.6f} -> "
            f"{_time_axis_fs(self.times, self.times_fs)[-1]:.6f}",
        }
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
        return summary

    def metadata_dict(self) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "result_type": "DynamicsResult",
            "mode": self.mode,
            "source_mode": self.source_mode,
            "summary": self.summary_dict(),
            "sanity_checks": self.sanity_checks,
            "result_metadata": self.metadata,
            "drive_dict": self.drive_dict,
            "drive_expr": self.drive_expr,
            "drive_name": self.drive_name,
            "density_matrix_unit": "dimensionless",
            "population_unit": "dimensionless",
            "coherence_unit": "dimensionless",
            "time_axis_saved_as": "time_fs",
            "time_axis_unit": "fs",
            "time_fs_is_physical": self.times_fs is not None,
            "n_time_points": int(len(self.times)),
            "dimension": self.dimension(),
            "parameters_code": self.parameters,
        }
        if self.physical_params is not None:
            metadata["physical_params"] = self.physical_params
        if self.solver_params is not None:
            metadata["solver_params_fs_inv"] = _solver_params_fs_inv_dict(self.solver_params)
            metadata["solver_params_code"] = _solver_params_code_dict(self.solver_params)
        return _json_safe(metadata)

    def parameter_summary_dict(self) -> dict[str, Any]:
        return self.metadata_dict()

    def to_npz_dict(self) -> dict[str, np.ndarray]:
        payload = {
            "time_fs": _time_axis_fs(self.times, self.times_fs),
            "time_code": np.asarray(self.times, dtype=float),
            "density": self.density_array(),
        }
        drive_code = self.drive_code_values()
        drive_fs_inv = self.drive_fs_inv_values()
        field_MV_per_cm = self.field_MV_per_cm_values()
        if drive_code is not None:
            payload["drive_code"] = drive_code
        if drive_fs_inv is not None:
            payload["drive_fs_inv"] = drive_fs_inv
        if field_MV_per_cm is not None:
            payload["field_MV_per_cm"] = field_MV_per_cm
        return payload

    def _add_drive_columns(self, data: dict[str, Any]) -> dict[str, Any]:
        drive_code = self.drive_code_values()
        drive_fs_inv = self.drive_fs_inv_values()
        field_MV_per_cm = self.field_MV_per_cm_values()
        if drive_code is not None:
            data["drive_code"] = drive_code
        if drive_fs_inv is not None:
            data["drive_fs_inv"] = drive_fs_inv
        if field_MV_per_cm is not None:
            data["field_MV_per_cm"] = field_MV_per_cm
        return data

    def components_dataframe(self):
        if self.dimension() != 2:
            raise ValueError("components_dataframe() is only defined for two-level results.")
        pd = _require_pandas()
        rho11, rho22, rho12, rho21 = self.components()
        data = {
            "time_fs": _time_axis_fs(self.times, self.times_fs),
            "rho11": rho11.real,
            "rho22": rho22.real,
            "Re_rho12": rho12.real,
            "Im_rho12": rho12.imag,
            "abs_rho12": np.abs(rho12),
            "Re_rho21": rho21.real,
            "Im_rho21": rho21.imag,
        }
        self._add_drive_columns(data)
        return pd.DataFrame(data)

    def populations_dataframe(self):
        pd = _require_pandas()
        populations = self.populations()
        data: dict[str, Any] = {"time_fs": _time_axis_fs(self.times, self.times_fs)}
        for index in range(populations.shape[1]):
            data[f"rho{index + 1}{index + 1}"] = populations[:, index].real
        self._add_drive_columns(data)
        return pd.DataFrame(data)

    def selected_elements_dataframe(self, elements: dict[str, tuple[int, int]]):
        pd = _require_pandas()
        data: dict[str, Any] = {"time_fs": _time_axis_fs(self.times, self.times_fs)}
        for label, values in self.selected_elements(elements).items():
            data[f"Re_{label}"] = values.real
            data[f"Im_{label}"] = values.imag
        self._add_drive_columns(data)
        return pd.DataFrame(data)

    def plot_times_and_label(self) -> tuple[np.ndarray, str]:
        if self.times_fs is not None:
            return self.times_fs, "Time (fs)"
        return self.times, "Time"


__all__ = ["DynamicsResult"]
