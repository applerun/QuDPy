"""基于 QuTiP mesolve 的单轨迹求解流程。"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from qutip import Qobj, mesolve

from .checks import evaluate_sanity_checks
from .fields.solver_inputs import (
    CodeCarrierField,
    CodeConstantDrive,
    CodeGaussianCarrierField,
    CodeGaussianDrive,
)
from .model import build_c_ops, build_lab_hamiltonian, build_rwa_hamiltonian, initial_density_matrix, parameter_fields
from .normalization import ParaNormalizer
from .parameters import NLevelPhysicalParams, NLevelSolverParams, ParameterSweep, PhysicalParameterSweep, SolverParams
from .results import DynamicsResult


def _default_tlist(parameters: NLevelSolverParams) -> np.ndarray:
    if parameters.tlist is not None:
        return np.asarray(parameters.tlist, dtype=float)
    t_end = parameters.t_final if parameters.t_end is None else parameters.t_end
    return np.arange(parameters.t_start, t_end + 0.5 * parameters.dt, parameters.dt)


def _mesolve_options(parameters: NLevelSolverParams) -> dict[str, float]:
    return {"max_step": float(parameters.dt)}


def _rho0(parameters: NLevelSolverParams, rho0: Qobj | None) -> Qobj:
    return initial_density_matrix(len(parameters.energies)) if rho0 is None else rho0


def simulate_lab_frame(
    parameters: NLevelSolverParams,
    rho0: Qobj | None = None,
    amplitude_code_override: float | None = None,
) -> tuple[np.ndarray, list[Qobj]]:
    times = _default_tlist(parameters)
    fields = parameter_fields(parameters)
    if amplitude_code_override is not None:
        fields = (
            CodeCarrierField(
                amplitude_code=amplitude_code_override,
                omega_code=parameters.omega_drive,
            )
            if parameters.pulse_sigma is None
            else CodeGaussianCarrierField(
                amplitude_code=amplitude_code_override,
                omega_code=parameters.omega_drive,
                center_code=0.0 if parameters.pulse_center is None else parameters.pulse_center,
                sigma_code=parameters.pulse_sigma,
            ),
        )
    result = mesolve(
        H=build_lab_hamiltonian(parameters),
        rho0=_rho0(parameters, rho0),
        tlist=times,
        c_ops=build_c_ops(parameters),
        e_ops=[],
        args={"fields": fields},
        options=_mesolve_options(parameters),
    )
    return times, list(result.states)


def default_rwa_drive(parameters: NLevelSolverParams) -> CodeConstantDrive | CodeGaussianDrive:
    # RWA Hamiltonian 已经携带 coupling_matrix；drive 只表示慢包络 f(t)。
    if parameters.pulse_sigma is None:
        return CodeConstantDrive(name="rwa_cw_envelope", amplitude_code=1.0)
    return CodeGaussianDrive(
        name="rwa_gaussian_envelope",
        amplitude_code=1.0,
        center_code=0.0 if parameters.pulse_center is None else parameters.pulse_center,
        sigma_code=parameters.pulse_sigma,
    )


def simulate_rwa_frame(
    parameters: NLevelSolverParams,
    times: np.ndarray | None = None,
    rho0: Qobj | None = None,
    drive: CodeConstantDrive | CodeGaussianDrive | None = None,
) -> list[Qobj]:
    if times is None:
        times = _default_tlist(parameters)
    local_drive = default_rwa_drive(parameters) if drive is None else drive
    result = mesolve(
        H=build_rwa_hamiltonian(parameters),
        rho0=_rho0(parameters, rho0),
        tlist=times,
        c_ops=build_c_ops(parameters),
        e_ops=[],
        args={"drive": local_drive},
        options=_mesolve_options(parameters),
    )
    return list(result.states)


def _basic_sanity_checks(result: DynamicsResult) -> dict[str, object]:
    return {
        "trace_error_small": {
            "value": result.max_trace_error(),
            "threshold": 1e-8,
            "passed": bool(result.max_trace_error() < 1e-8),
        },
        "hermiticity_error_small": {
            "value": result.max_hermiticity_error(),
            "threshold": 1e-8,
            "passed": bool(result.max_hermiticity_error() < 1e-8),
        },
    }


def run_lab_case(parameters: NLevelSolverParams, rho0: Qobj | None = None) -> DynamicsResult:
    times, states = simulate_lab_frame(parameters, rho0=rho0)
    fields = parameter_fields(parameters)
    drive = fields[0] if len(fields) == 1 else None
    result = DynamicsResult(
        mode="lab_exact",
        times=times,
        times_fs=parameters.times_fs,
        states=states,
        parameters=parameters,
        metadata={"energies_code": parameters.energies},
        drive=drive,
        drive_dict=drive.to_dict() if drive is not None and hasattr(drive, "to_dict") else None,
        drive_expr=drive.to_expr() if drive is not None and hasattr(drive, "to_expr") else None,
        drive_name=getattr(drive, "name", None),
    )
    result.sanity_checks = evaluate_sanity_checks(result)
    return result


def run_rwa_case(
    parameters: NLevelSolverParams,
    rho0: Qobj | None = None,
    drive: CodeConstantDrive | CodeGaussianDrive | None = None,
) -> DynamicsResult:
    times = _default_tlist(parameters)
    local_drive = default_rwa_drive(parameters) if drive is None else drive
    states = simulate_rwa_frame(parameters, times=times, rho0=rho0, drive=local_drive)
    result = DynamicsResult(
        mode="rwa",
        times=times,
        times_fs=parameters.times_fs,
        states=states,
        parameters=parameters,
        metadata={"energies_code": parameters.energies},
        drive=local_drive,
        drive_dict=local_drive.to_dict(),
        drive_expr=local_drive.to_expr(),
        drive_name=local_drive.name,
    )
    result.sanity_checks = _basic_sanity_checks(result)
    return result


def rotating_frame_unitary(time: float, omega_drive: float) -> Qobj:
    return Qobj(np.diag([1.0, np.exp(-1j * omega_drive * time)]).astype(np.complex128))


def rotate_density_trajectory(times: np.ndarray, states: list[Qobj], omega_drive: float) -> list[Qobj]:
    rotated_states: list[Qobj] = []
    for time, rho_lab in zip(times, states):
        unitary = rotating_frame_unitary(time, omega_drive)
        rotated_states.append(unitary.dag() * rho_lab * unitary)
    return rotated_states


def make_rotating_view(lab_result: DynamicsResult) -> DynamicsResult:
    if lab_result.mode != "lab_exact":
        raise ValueError("make_rotating_view expects a lab_exact DynamicsResult.")
    if lab_result.dimension() != 2:
        raise ValueError("rotating_view 当前只用于 N=2 lab_exact 后处理。")
    states = rotate_density_trajectory(
        np.asarray(lab_result.times, dtype=float),
        lab_result.states,
        lab_result.parameters.omega_drive,
    )
    result = DynamicsResult(
        mode="rotating_view",
        times=lab_result.times,
        times_fs=lab_result.times_fs,
        states=states,
        parameters=lab_result.parameters,
        physical_params=lab_result.physical_params,
        solver_params=lab_result.solver_params,
        metadata=dict(lab_result.metadata),
        source_mode=lab_result.mode,
    )
    result.sanity_checks = _basic_sanity_checks(result)
    return result


def optical_params_from_solver(
    solver: SolverParams,
    physical: NLevelPhysicalParams | None = None,
    normalizer: ParaNormalizer | None = None,
) -> NLevelSolverParams:
    if normalizer is not None:
        times_fs = normalizer.denormalize_time_array(solver.tlist, solver)
    elif physical is not None:
        times_fs = np.linspace(physical.t_start_fs, physical.t_end_fs, len(solver.tlist))
    else:
        times_fs = None

    return NLevelSolverParams(
        t_start=solver.t_start,
        t_end=solver.t_end,
        dt=solver.dt,
        t_final=solver.t_end,
        hbar=1.0,
        energies=tuple(float(value) for value in solver.energies_code),
        dipole_matrix=tuple(tuple(complex(item) for item in row) for row in solver.coupling_matrix_code),
        coupling_matrix=tuple(tuple(complex(item) for item in row) for row in solver.coupling_matrix_code),
        omega_drive=solver.omega_L,
        relaxation_channels=solver.relaxation_channels_code,
        pure_dephasing_channels=solver.pure_dephasing_channels_code,
        detuning=solver.detuning,
        pulse_center=solver.pulse_center,
        pulse_sigma=solver.pulse_sigma,
        fields=None,
        tlist=solver.tlist,
        times_fs=times_fs,
        basis=None if physical is None else physical.basis,
    )


def run_case(parameters: NLevelSolverParams) -> DynamicsResult:
    return run_lab_case(parameters)


def run_physical_case(
    physical_params: NLevelPhysicalParams,
    normalizer: ParaNormalizer | None = None,
) -> DynamicsResult:
    local_normalizer = ParaNormalizer() if normalizer is None else normalizer
    solver = local_normalizer.normalize(physical_params)
    parameters = optical_params_from_solver(solver=solver, physical=physical_params, normalizer=local_normalizer)
    result = run_lab_case(parameters)
    result.physical_params = physical_params
    result.solver_params = solver
    result.sanity_checks = evaluate_sanity_checks(result)
    return result


def run_parameter_sweep(sweep: ParameterSweep) -> list[DynamicsResult]:
    results: list[DynamicsResult] = []
    for detuning in sweep.detunings:
        for amplitude_scale in sweep.field_amplitudes:
            energies = (sweep.energies[0], sweep.omega_drive + detuning)
            parameters = NLevelSolverParams(
                t_final=sweep.t_final,
                dt=sweep.dt,
                hbar=sweep.hbar,
                energies=energies,
                dipole_matrix=sweep.dipole_matrix,
                coupling_matrix=tuple(tuple(amplitude_scale * complex(item) for item in row) for row in sweep.dipole_matrix),
                omega_drive=sweep.omega_drive,
            )
            results.append(run_case(parameters))
    return results


def run_physical_parameter_sweep(
    sweep: PhysicalParameterSweep,
    normalizer: ParaNormalizer | None = None,
) -> list[DynamicsResult]:
    field_values = sweep.field_MV_per_cm_values or (sweep.base_params.field_MV_per_cm,)
    laser_values = sweep.laser_energy_eV_values or (sweep.base_params.laser_energy_eV,)
    results: list[DynamicsResult] = []
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
    "default_rwa_drive",
    "rotating_frame_unitary",
    "rotate_density_trajectory",
    "optical_params_from_solver",
    "run_lab_case",
    "run_rwa_case",
    "run_case",
    "run_physical_case",
    "run_parameter_sweep",
    "run_physical_parameter_sweep",
    "make_rotating_view",
]
