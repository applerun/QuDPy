"""简单的物理 sanity checks。"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
from qutip import Qobj, mesolve

from .fields import CarrierField, GaussianCarrierField
from .model import (
    build_c_ops,
    build_lab_hamiltonian,
    coherent_superposition_density_matrix,
    excited_density_matrix,
    initial_density_matrix,
)
from .parameters import OpticalBlochParameters
from .results import DynamicsResult


def _default_tlist(parameters: OpticalBlochParameters) -> np.ndarray:
    if parameters.tlist is not None:
        return np.asarray(parameters.tlist, dtype=float)
    t_end = parameters.t_final if parameters.t_end is None else parameters.t_end
    return np.arange(parameters.t_start, t_end + 0.5 * parameters.dt, parameters.dt)


def _two_level_elements_from_states(states: list[Qobj]) -> dict[str, np.ndarray]:
    density = np.stack([state.full() for state in states], axis=0)
    if density.shape[1:] != (2, 2):
        raise ValueError("This sanity check is only defined for two-level density matrices.")
    return {
        "rho_00": density[:, 0, 0],
        "rho_11": density[:, 1, 1],
        "rho_01": density[:, 0, 1],
        "rho_10": density[:, 1, 0],
    }


def _two_level_elements_from_result(result: DynamicsResult) -> dict[str, np.ndarray]:
    if result.dimension() != 2:
        raise ValueError("evaluate_sanity_checks() is only defined for two-level results.")
    return {
        "rho_00": result.matrix_element(0, 0),
        "rho_11": result.matrix_element(1, 1),
        "rho_01": result.matrix_element(0, 1),
        "rho_10": result.matrix_element(1, 0),
    }


def _simulate_lab_for_check(
    parameters: OpticalBlochParameters,
    rho0: Qobj,
    field_amplitude_override: float = 0.0,
) -> tuple[np.ndarray, list[Qobj]]:
    times = _default_tlist(parameters)
    fields = (
        (CarrierField(
            amplitude=field_amplitude_override,
            omega=parameters.omega_drive,
            phase=0.0,
        ) if parameters.pulse_sigma is None else GaussianCarrierField(
            amplitude=field_amplitude_override,
            omega=parameters.omega_drive,
            phase=0.0,
            center=0.0 if parameters.pulse_center is None else parameters.pulse_center,
            sigma=parameters.pulse_sigma,
        )),
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
    elements = _two_level_elements_from_states(states)
    rho_00 = elements["rho_00"]
    rho_11 = elements["rho_11"]
    return {
        "rho_11_initial": float(rho_11[0].real),
        "rho_11_final": float(rho_11[-1].real),
        "rho_00_initial": float(rho_00[0].real),
        "rho_00_final": float(rho_00[-1].real),
        "passed": bool(rho_11[-1].real < rho_11[0].real and rho_00[-1].real > rho_00[0].real),
        "time_points": len(times),
    }


def _simulate_pure_dephasing_sanity(parameters: OpticalBlochParameters) -> dict[str, Any]:
    aux_parameters = replace(parameters, gamma1=0.0)
    times, states = _simulate_lab_for_check(aux_parameters, coherent_superposition_density_matrix(), 0.0)
    elements = _two_level_elements_from_states(states)
    rho_00 = elements["rho_00"]
    rho_11 = elements["rho_11"]
    rho_01 = elements["rho_01"]
    return {
        "population_change_max": float(
            max(
                np.max(np.abs(rho_00.real - rho_00.real[0])),
                np.max(np.abs(rho_11.real - rho_11.real[0])),
            )
        ),
        "coherence_abs_initial": float(abs(rho_01[0])),
        "coherence_abs_final": float(abs(rho_01[-1])),
        "passed": bool(
            np.max(np.abs(rho_00.real - rho_00.real[0])) < 1e-6
            and np.max(np.abs(rho_11.real - rho_11.real[0])) < 1e-6
            and abs(rho_01[-1]) < abs(rho_01[0])
        ),
        "time_points": len(times),
    }


def _simulate_closed_system_sanity(parameters: OpticalBlochParameters) -> dict[str, Any]:
    aux_parameters = replace(parameters, gamma1=0.0, gamma_phi=0.0)
    times, states = _simulate_lab_for_check(aux_parameters, initial_density_matrix(), 0.0)
    elements = _two_level_elements_from_states(states)
    rho_00 = elements["rho_00"]
    rho_11 = elements["rho_11"]
    variation = max(
        np.max(np.abs(rho_00.real - rho_00.real[0])),
        np.max(np.abs(rho_11.real - rho_11.real[0])),
    )
    return {
        "population_change_max": float(variation),
        "passed": bool(variation < 1e-8),
        "time_points": len(times),
    }


def evaluate_sanity_checks(result: DynamicsResult) -> dict[str, Any]:
    elements = _two_level_elements_from_result(result)
    rho_01 = elements["rho_01"]
    rho_10 = elements["rho_10"]
    max_trace_error = result.max_trace_error()
    max_hermiticity_error = result.max_hermiticity_error()
    checks: dict[str, Any] = {
        "trace_error_small": {
            "value": max_trace_error,
            "threshold": 1e-8,
            "passed": bool(max_trace_error < 1e-8),
        },
        "hermiticity_error_small": {
            "value": max_hermiticity_error,
            "threshold": 1e-8,
            "passed": bool(max_hermiticity_error < 1e-8),
        },
        "zero_field_closed_system_auxiliary": _simulate_closed_system_sanity(result.parameters),
    }

    if result.parameters.gamma_phi > 0:
        checks["pure_dephasing_auxiliary"] = _simulate_pure_dephasing_sanity(result.parameters)
    if result.parameters.gamma1 > 0:
        checks["population_relaxation_auxiliary"] = _simulate_relaxation_sanity(result.parameters)

    checks["coherence_norm_final"] = {
        "rho_01_abs_final": float(abs(rho_01[-1])),
        "rho_10_abs_final": float(abs(rho_10[-1])),
    }
    return checks


__all__ = ["evaluate_sanity_checks"]
