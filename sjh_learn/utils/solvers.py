"""基于 QuTiP mesolve 的求解流程。"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from qutip import Qobj, mesolve

from .checks import evaluate_sanity_checks
from .fields import CarrierField, ConstantDrive, GaussianCarrierField, GaussianDrive
from .model import (
    build_c_ops,
    build_lab_hamiltonian,
    compute_energy_gap,
    initial_density_matrix,
    parameter_fields,
)
from .normalization import ParaNormalizer, PhysicalParams, SolverParams
from .parameters import OpticalBlochParameters, ParameterSweep, PhysicalParameterSweep
from .results import DynamicsResult


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
        rho0=initial_density_matrix() if rho0 is None else rho0,
        tlist=times,
        c_ops=build_c_ops(parameters),
        e_ops=[],
        args={"fields": fields},
    )
    return times, list(result.states)


def simulate_rwa_frame(
    parameters: OpticalBlochParameters,
    times: np.ndarray | None = None,
    rho0: Qobj | None = None,
    drive: ConstantDrive | GaussianDrive | None = None,
) -> list[Qobj]:
    if times is None:
        times = _default_tlist(parameters)
    if drive is None:
        drive = default_rwa_drive(parameters)
    h_static = Qobj(np.array([[0.0, 0.0], [0.0, parameters.detuning]], dtype=np.complex128))
    h_coupling = Qobj(np.array([[0.0, -1.0], [-1.0, 0.0]], dtype=np.complex128))
    result = mesolve(
        H=[h_static, [h_coupling, lambda t, args: float(args["drive"](t))]],
        rho0=initial_density_matrix() if rho0 is None else rho0,
        tlist=times,
        c_ops=build_c_ops(parameters),
        e_ops=[],
        args={"drive": drive},
    )
    return list(result.states)


def default_rwa_drive(parameters: OpticalBlochParameters) -> ConstantDrive | GaussianDrive:
    amplitude = parameters.dipole * parameters.field_amplitude
    if parameters.pulse_sigma is None:
        return ConstantDrive(name="rwa_cw_drive", amplitude=amplitude)
    return GaussianDrive(
        name="rwa_gaussian_drive",
        amplitude=amplitude,
        center=0.0 if parameters.pulse_center is None else parameters.pulse_center,
        sigma=parameters.pulse_sigma,
    )


def _energy_upper(parameters: OpticalBlochParameters) -> float:
    return parameters.epsilon_1 + compute_energy_gap(
        detuning=parameters.detuning,
        omega_drive=parameters.omega_drive,
        hbar=parameters.hbar,
    )


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


def run_lab_case(parameters: OpticalBlochParameters, rho0: Qobj | None = None) -> DynamicsResult:
    times, states = simulate_lab_frame(parameters, rho0=rho0)
    fields = parameter_fields(parameters)
    drive = fields[0] if len(fields) == 1 else None
    field_expr = fields[0].to_expr() if len(fields) == 1 and hasattr(fields[0], "to_expr") else None
    result = DynamicsResult(
        mode="lab_exact",
        times=times,
        times_fs=parameters.times_fs,
        states=states,
        parameters=parameters,
        metadata={"epsilon_2": _energy_upper(parameters)},
        drive=drive,
        drive_dict=drive.to_dict() if drive is not None and hasattr(drive, "to_dict") else None,
        drive_expr=field_expr,
        drive_name=getattr(drive, "name", None),
    )
    result.sanity_checks = evaluate_sanity_checks(result)
    return result


def run_rwa_case(
    parameters: OpticalBlochParameters,
    rho0: Qobj | None = None,
    drive: ConstantDrive | GaussianDrive | None = None,
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
        metadata={"epsilon_2": _energy_upper(parameters)},
        drive=local_drive,
        drive_dict=local_drive.to_dict(),
        drive_expr=local_drive.to_expr(),
        drive_name=local_drive.name,
    )
    result.sanity_checks = _basic_sanity_checks(result)
    return result


def make_rotating_view(lab_result: DynamicsResult) -> DynamicsResult:
    if lab_result.mode != "lab_exact":
        raise ValueError("make_rotating_view expects a lab_exact DynamicsResult.")
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


def rotate_density_trajectory(times: np.ndarray, states: list[Qobj], omega_drive: float) -> list[Qobj]:
    rotated_states: list[Qobj] = []
    for time, rho_lab in zip(times, states):
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


def run_case(parameters: OpticalBlochParameters) -> DynamicsResult:
    """Compatibility entrypoint. Prefer run_lab_case/run_rwa_case for new code."""
    return run_lab_case(parameters)


def run_physical_case(
    physical_params: PhysicalParams,
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
    "make_rotating_view",
]
