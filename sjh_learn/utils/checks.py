"""简单的物理 sanity checks。"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
from qutip import Qobj, mesolve

from .fields import FieldConfig
from .model import (
    build_c_ops,
    build_lab_hamiltonian,
    coherent_superposition_density_matrix,
    excited_density_matrix,
    initial_density_matrix,
)
from .parameters import OpticalBlochParameters
from .results import OpticalBlochResult


def _default_tlist(parameters: OpticalBlochParameters) -> np.ndarray:
    if parameters.tlist is not None:
        return np.asarray(parameters.tlist, dtype=float)
    t_end = parameters.t_final if parameters.t_end is None else parameters.t_end
    return np.arange(parameters.t_start, t_end + 0.5 * parameters.dt, parameters.dt)


def _simulate_lab_for_check(
    parameters: OpticalBlochParameters,
    rho0: Qobj,
    field_amplitude_override: float = 0.0,
) -> tuple[np.ndarray, list[Qobj]]:
    times = _default_tlist(parameters)
    fields = (
        FieldConfig(
            amplitude=field_amplitude_override,
            omega=parameters.omega_drive,
            phase=0.0,
            envelope="constant" if parameters.pulse_sigma is None else "gaussian",
            center=0.0 if parameters.pulse_center is None else parameters.pulse_center,
            sigma=parameters.pulse_sigma,
        ),
    )
    result = mesolve(
        H=build_lab_hamiltonian(parameters),
        rho0=rho0,
        tlist=times,
        c_ops=build_c_ops(parameters),
        e_ops=[],
        args={"fields": fields},
    )
    return times, list(result.states)


def _simulate_relaxation_sanity(parameters: OpticalBlochParameters) -> dict[str, Any]:
    aux_parameters = replace(parameters, gamma_phi=0.0)
    times, states = _simulate_lab_for_check(aux_parameters, excited_density_matrix(), 0.0)
    rho11 = np.array([state.full()[0, 0] for state in states])
    rho22 = np.array([state.full()[1, 1] for state in states])
    return {
        "rho22_initial": float(rho22[0].real),
        "rho22_final": float(rho22[-1].real),
        "rho11_initial": float(rho11[0].real),
        "rho11_final": float(rho11[-1].real),
        "passed": bool(rho22[-1].real < rho22[0].real and rho11[-1].real > rho11[0].real),
        "time_points": len(times),
    }


def _simulate_pure_dephasing_sanity(parameters: OpticalBlochParameters) -> dict[str, Any]:
    aux_parameters = replace(parameters, gamma1=0.0)
    times, states = _simulate_lab_for_check(aux_parameters, coherent_superposition_density_matrix(), 0.0)
    rho11 = np.array([state.full()[0, 0] for state in states])
    rho22 = np.array([state.full()[1, 1] for state in states])
    rho12 = np.array([state.full()[0, 1] for state in states])
    return {
        "population_change_max": float(
            max(
                np.max(np.abs(rho11.real - rho11.real[0])),
                np.max(np.abs(rho22.real - rho22.real[0])),
            )
        ),
        "coherence_abs_initial": float(abs(rho12[0])),
        "coherence_abs_final": float(abs(rho12[-1])),
        "passed": bool(
            np.max(np.abs(rho11.real - rho11.real[0])) < 1e-6
            and np.max(np.abs(rho22.real - rho22.real[0])) < 1e-6
            and abs(rho12[-1]) < abs(rho12[0])
        ),
        "time_points": len(times),
    }


def _simulate_closed_system_sanity(parameters: OpticalBlochParameters) -> dict[str, Any]:
    aux_parameters = replace(parameters, gamma1=0.0, gamma_phi=0.0)
    times, states = _simulate_lab_for_check(aux_parameters, initial_density_matrix(), 0.0)
    rho11 = np.array([state.full()[0, 0] for state in states])
    rho22 = np.array([state.full()[1, 1] for state in states])
    variation = max(
        np.max(np.abs(rho11.real - rho11.real[0])),
        np.max(np.abs(rho22.real - rho22.real[0])),
    )
    return {
        "population_change_max": float(variation),
        "passed": bool(variation < 1e-8),
        "time_points": len(times),
    }


def evaluate_sanity_checks(result: OpticalBlochResult) -> dict[str, Any]:
    rho11, rho22, rho12, rho21 = result.components("lab")
    checks: dict[str, Any] = {
        "trace_error_small": {
            "value": result.max_trace_error("lab"),
            "threshold": 1e-8,
            "passed": bool(result.max_trace_error("lab") < 1e-8),
        },
        "hermiticity_error_small": {
            "value": result.max_hermiticity_error("lab"),
            "threshold": 1e-8,
            "passed": bool(result.max_hermiticity_error("lab") < 1e-8),
        },
        "zero_field_closed_system_auxiliary": _simulate_closed_system_sanity(result.parameters),
    }

    if result.parameters.gamma_phi > 0:
        checks["pure_dephasing_auxiliary"] = _simulate_pure_dephasing_sanity(result.parameters)
    if result.parameters.gamma1 > 0:
        checks["population_relaxation_auxiliary"] = _simulate_relaxation_sanity(result.parameters)

    checks["coherence_norm_final"] = {
        "rho12_abs_final": float(abs(rho12[-1])),
        "rho21_abs_final": float(abs(rho21[-1])),
    }
    return checks


__all__ = ["evaluate_sanity_checks"]
