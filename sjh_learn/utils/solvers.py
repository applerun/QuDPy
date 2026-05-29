"""基于 QuTiP mesolve 的求解流程。"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from qutip import Qobj, mesolve

from .checks import evaluate_sanity_checks
from .fields import FieldConfig
from .model import (
    build_c_ops,
    build_lab_hamiltonian,
    build_rwa_hamiltonian,
    compute_energy_gap,
    initial_density_matrix,
    parameter_fields,
)
from .normalization import ParaNormalizer, PhysicalParams, SolverParams
from .parameters import OpticalBlochParameters, ParameterSweep, PhysicalParameterSweep
from .results import OpticalBlochResult


def _default_tlist(parameters: OpticalBlochParameters) -> np.ndarray:
    if parameters.tlist is not None:
        return np.asarray(parameters.tlist, dtype=float)
    t_end = parameters.t_final if parameters.t_end is None else parameters.t_end
    return np.arange(parameters.t_start, t_end + 0.5 * parameters.dt, parameters.dt)


def simulate_lab_frame(
    parameters: OpticalBlochParameters,
    rho0: Qobj | None = None,
    field_amplitude_override: float | None = None,
) -> tuple[np.ndarray, list[Qobj]]:
    times = _default_tlist(parameters)
    fields = parameter_fields(parameters)
    if field_amplitude_override is not None:
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
        rho0=initial_density_matrix() if rho0 is None else rho0,
        tlist=times,
        c_ops=build_c_ops(parameters),
        e_ops=[],
        args={"fields": fields},
    )
    return times, list(result.states)


def simulate_rwa_frame(
    parameters: OpticalBlochParameters,
    times: np.ndarray,
    rho0: Qobj | None = None,
) -> list[Qobj]:
    result = mesolve(
        H=build_rwa_hamiltonian(parameters),
        rho0=initial_density_matrix() if rho0 is None else rho0,
        tlist=times,
        c_ops=build_c_ops(parameters),
        e_ops=[],
    )
    return list(result.states)


def rotating_frame_unitary(time: float, omega_drive: float) -> Qobj:
    return Qobj(
        np.array(
            [
                [1.0, 0.0],
                [0.0, np.exp(-1j * omega_drive * time)],
            ],
            dtype=np.complex128,
        )
    )


def rotate_density_trajectory(times: np.ndarray, lab_states: list[Qobj], omega_drive: float) -> list[Qobj]:
    rotated_states: list[Qobj] = []
    for time, rho_lab in zip(times, lab_states):
        unitary = rotating_frame_unitary(time, omega_drive)
        rotated_states.append(unitary.dag() * rho_lab * unitary)
    return rotated_states


def optical_params_from_solver(
    solver: SolverParams,
    physical: PhysicalParams | None = None,
    normalizer: ParaNormalizer | None = None,
) -> OpticalBlochParameters:
    times_fs = None
    if normalizer is not None:
        times_fs = normalizer.denormalize_time_array(solver.tlist, solver)
    elif physical is not None:
        times_fs = np.linspace(physical.t_start_fs, physical.t_end_fs, len(solver.tlist))

    return OpticalBlochParameters(
        t_start=solver.t_start,
        t_end=solver.t_end,
        dt=solver.dt,
        t_final=solver.t_end,
        hbar=1.0,
        epsilon_1=0.0,
        detuning=solver.detuning,
        dipole=1.0,
        field_amplitude=solver.rabi,
        omega_drive=solver.omega_L,
        gamma1=solver.gamma1,
        gamma_phi=solver.gamma_phi,
        gamma2=solver.gamma2,
        pulse_center=solver.pulse_center,
        pulse_sigma=solver.pulse_sigma,
        fields=None,
        tlist=solver.tlist,
        times_fs=times_fs,
    )


def run_case(parameters: OpticalBlochParameters) -> OpticalBlochResult:
    epsilon_2 = parameters.epsilon_1 + compute_energy_gap(
        detuning=parameters.detuning,
        omega_drive=parameters.omega_drive,
        hbar=parameters.hbar,
    )
    times, lab_states = simulate_lab_frame(parameters)
    rotating_states = rotate_density_trajectory(times, lab_states, parameters.omega_drive)
    rwa_states = simulate_rwa_frame(parameters, times)
    result = OpticalBlochResult(
        parameters=parameters,
        epsilon_2=epsilon_2,
        times=times,
        times_fs=parameters.times_fs,
        lab_states=lab_states,
        rotating_states=rotating_states,
        rwa_states=rwa_states,
    )
    result.sanity_checks = evaluate_sanity_checks(result)
    return result


def run_physical_case(
    physical_params: PhysicalParams,
    normalizer: ParaNormalizer | None = None,
) -> OpticalBlochResult:
    local_normalizer = ParaNormalizer() if normalizer is None else normalizer
    solver = local_normalizer.normalize(physical_params)
    parameters = optical_params_from_solver(solver=solver, physical=physical_params, normalizer=local_normalizer)
    result = run_case(parameters)
    result.physical_params = physical_params
    result.solver_params = solver
    result.sanity_checks = evaluate_sanity_checks(result)
    return result


def run_parameter_sweep(sweep: ParameterSweep) -> list[OpticalBlochResult]:
    results: list[OpticalBlochResult] = []
    for detuning in sweep.detunings:
        for field_amplitude in sweep.field_amplitudes:
            parameters = OpticalBlochParameters(
                t_final=sweep.t_final,
                dt=sweep.dt,
                hbar=sweep.hbar,
                epsilon_1=sweep.epsilon_1,
                detuning=detuning,
                dipole=sweep.dipole,
                field_amplitude=field_amplitude,
                omega_drive=sweep.omega_drive,
            )
            results.append(run_case(parameters))
    return results


def run_physical_parameter_sweep(
    sweep: PhysicalParameterSweep,
    normalizer: ParaNormalizer | None = None,
) -> list[OpticalBlochResult]:
    field_values = sweep.field_MV_per_cm_values or (sweep.base_params.field_MV_per_cm,)
    laser_values = sweep.laser_energy_eV_values or (sweep.base_params.laser_energy_eV,)

    results: list[OpticalBlochResult] = []
    for laser_energy_eV in laser_values:
        for field_MV_per_cm in field_values:
            physical_params = replace(
                sweep.base_params,
                laser_energy_eV=laser_energy_eV,
                field_MV_per_cm=field_MV_per_cm,
            )
            results.append(run_physical_case(physical_params, normalizer=normalizer))
    return results


__all__ = [
    "simulate_lab_frame",
    "simulate_rwa_frame",
    "rotating_frame_unitary",
    "rotate_density_trajectory",
    "optical_params_from_solver",
    "run_case",
    "run_physical_case",
    "run_parameter_sweep",
    "run_physical_parameter_sweep",
]
