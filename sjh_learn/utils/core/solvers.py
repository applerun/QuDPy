"""基于 QuTiP mesolve 的单轨迹求解流程。"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path

import numpy as np
from qutip import Qobj, mesolve

from sjh_learn.utils.checks import evaluate_sanity_checks
from sjh_learn.utils.fields.solver_inputs import (
    CodeCompositeField,
    CodeConstantDrive,
    CodeGaussianDrive,
)
from sjh_learn.utils.fields import FieldPhyRoot, default_field_from_physical_params
from sjh_learn.utils.core.model import build_c_ops, build_lab_hamiltonian, build_rwa_hamiltonian, initial_density_matrix, parameter_fields
from sjh_learn.utils.core.normalization import ParaNormalizer
from sjh_learn.utils.core.parameters import NLevelPhysicalParams, NLevelSolverParams, ParameterSweep, PhysicalParameterSweep, SolverParams
from sjh_learn.utils.core.results import DynamicsResult


def _default_tlist(parameters: NLevelSolverParams) -> np.ndarray:
    if parameters.tlist is not None:
        return np.asarray(parameters.tlist, dtype=float)
    t_end = parameters.t_final if parameters.t_end is None else parameters.t_end
    return np.arange(parameters.t_start, t_end + 0.5 * parameters.dt, parameters.dt)


def _mesolve_options(parameters: NLevelSolverParams) -> dict[str, float]:
    return {"max_step": float(parameters.dt)}


def _rho0(parameters: NLevelSolverParams, rho0: Qobj | None) -> Qobj:
    return initial_density_matrix(len(parameters.energies)) if rho0 is None else rho0


def _default_rwa_drive(parameters: NLevelSolverParams) -> CodeConstantDrive | CodeGaussianDrive:
    # RWA Hamiltonian 已经携带 coupling_matrix；drive 只表示慢包络 f(t)。
    if parameters.pulse_sigma is None:
        return CodeConstantDrive(name="rwa_cw_envelope", amplitude_code=1.0)
    return CodeGaussianDrive(
        name="rwa_gaussian_envelope",
        amplitude_code=1.0,
        center_code=0.0 if parameters.pulse_center is None else parameters.pulse_center,
        sigma_code=parameters.pulse_sigma,
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


def _run_lab_case(
    parameters: NLevelSolverParams,
    rho0: Qobj | None = None,
    *,
    physical_params: NLevelPhysicalParams | None = None,
    solver_params: SolverParams | None = None,
) -> DynamicsResult:
    times = _default_tlist(parameters)
    fields = parameter_fields(parameters)
    solver_result = mesolve(
        H=build_lab_hamiltonian(parameters),
        rho0=_rho0(parameters, rho0),
        tlist=times,
        c_ops=build_c_ops(parameters),
        e_ops=[],
        args={"fields": fields},
        options=_mesolve_options(parameters),
    )
    states = list(solver_result.states)
    fields = parameter_fields(parameters)
    drive = fields[0] if len(fields) == 1 else CodeCompositeField(fields=fields)
    result = DynamicsResult(
        mode="lab_exact",
        times=times,
        times_fs=parameters.times_fs,
        states=states,
        parameters=parameters,
        physical_params=physical_params,
        solver_params=solver_params,
        metadata={"energies_code": parameters.energies},
        drive=drive,
        drive_dict=drive.to_dict() if drive is not None and hasattr(drive, "to_dict") else None,
        drive_expr=drive.to_expr() if drive is not None and hasattr(drive, "to_expr") else None,
        drive_name=getattr(drive, "name", None),
    )
    result.sanity_checks = evaluate_sanity_checks(result)
    return result


def _run_rwa_case(
    parameters: NLevelSolverParams,
    rho0: Qobj | None = None,
    drive: CodeConstantDrive | CodeGaussianDrive | None = None,
) -> DynamicsResult:
    times = _default_tlist(parameters)
    local_drive = _default_rwa_drive(parameters) if drive is None else drive
    solver_result = mesolve(
        H=build_rwa_hamiltonian(parameters),
        rho0=_rho0(parameters, rho0),
        tlist=times,
        c_ops=build_c_ops(parameters),
        e_ops=[],
        args={"drive": local_drive},
        options=_mesolve_options(parameters),
    )
    states = list(solver_result.states)
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


def _rotating_frame_unitary(time: float, omega_drive: float) -> Qobj:
    return Qobj(np.diag([1.0, np.exp(-1j * omega_drive * time)]).astype(np.complex128))


def _rotate_density_trajectory(times: np.ndarray, states: list[Qobj], omega_drive: float) -> list[Qobj]:
    rotated_states: list[Qobj] = []
    for time, rho_lab in zip(times, states):
        unitary = _rotating_frame_unitary(time, omega_drive)
        rotated_states.append(unitary.dag() * rho_lab * unitary)
    return rotated_states


def _bound_physical_field(
    physical: NLevelPhysicalParams,
    normalizer: ParaNormalizer,
    solver: SolverParams,
):
    if physical.field is None:
        field = default_field_from_physical_params(physical, normalizer)
    elif isinstance(physical.field, FieldPhyRoot):
        field = physical.field
    else:
        raise TypeError("NLevelPhysicalParams.field must be None or a FieldPhyRoot instance.")
    return normalizer.make_code_field(
        field,
        solver,
        reference_field_MV_per_cm=physical.field_MV_per_cm,
    )


def make_rotating_view(lab_result: DynamicsResult) -> DynamicsResult:
    if lab_result.mode != "lab_exact":
        raise ValueError("make_rotating_view expects a lab_exact DynamicsResult.")
    if lab_result.dimension() != 2:
        raise ValueError("rotating_view 当前只用于 N=2 lab_exact 后处理。")
    states = _rotate_density_trajectory(
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


def _optical_params_from_solver(
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

    fields = None
    if physical is not None and normalizer is not None:
        fields = (_bound_physical_field(physical, normalizer, solver),)

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
        fields=fields,
        tlist=solver.tlist,
        times_fs=times_fs,
        basis=None if physical is None else physical.basis,
    )



def run_case(
    physical_params: NLevelPhysicalParams,
    normalizer: ParaNormalizer | None = None,
    rho0: Qobj | None = None,
    *,
    load_ckp: str | Path | None = None,
    save_ckp: str | Path | None = None,
    force_run: bool = False,
) -> DynamicsResult:
    load_path = None if load_ckp is None else Path(load_ckp)
    if load_path is not None and load_path.exists() and not force_run:
        print(f"Loading checkpoint: {load_path}")
        return DynamicsResult.from_ckp(load_path)
    if load_path is not None and force_run:
        print(f"force_run=True, running simulation and ignoring checkpoint: {load_path}")
    elif load_path is not None:
        print(f"Checkpoint not found, running simulation: {load_path}")

    if physical_params.solver_mode == "rwa" and physical_params.field is not None:
        raise ValueError(
            "RWA mode currently derives its internal envelope from NLevelPhysicalParams pulse parameters, "
            "not from an explicit FieldPhyRoot. Leave field=None or use lab_exact for custom physical fields."
        )
    local_normalizer = ParaNormalizer() if normalizer is None else normalizer
    solver = local_normalizer.normalize(physical_params)
    parameters = _optical_params_from_solver(solver=solver, physical=physical_params, normalizer=local_normalizer)
    if physical_params.solver_mode == "lab_exact":
        result = _run_lab_case(parameters, rho0=rho0, physical_params=physical_params, solver_params=solver)
    if physical_params.solver_mode == "rwa":
        result = _run_rwa_case(parameters, rho0=rho0)
        result.physical_params = physical_params
        result.solver_params = solver
    if physical_params.solver_mode not in {"lab_exact", "rwa"}:
        raise ValueError(f"Unsupported solver_mode: {physical_params.solver_mode!r}. Expected 'lab_exact' or 'rwa'.")

    checkpoint_save_path = save_ckp if save_ckp is not None else load_ckp
    if checkpoint_save_path is not None:
        save_path = Path(checkpoint_save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Saving checkpoint: {save_path}")
        result.save_ckp(save_path)
    return result


def run_cases(
    physical_params_list: Iterable[NLevelPhysicalParams],
    normalizer: ParaNormalizer | None = None,
) -> list[DynamicsResult]:
    return [run_case(physical_params, normalizer=normalizer) for physical_params in physical_params_list]


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
            results.append(_run_lab_case(parameters))
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
            results.append(run_case(physical_params, normalizer=normalizer))
    return results


__all__ = [
    "run_case",
    "run_cases",
    "make_rotating_view",
]
