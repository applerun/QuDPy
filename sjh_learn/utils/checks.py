"""物理 sanity checks。

本模块只做结果一致性检查，不定义新的物理模型。N>2 结果使用通用
trace / Hermiticity / population 检查；只对 N=2 保留教学用的辅助检查。
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
from qutip import Qobj, mesolve

from .fields.solver_inputs import CodeCarrierField, CodeGaussianCarrierField
from sjh_learn.utils.core.model import (
    build_c_ops,
    build_lab_hamiltonian,
    coherent_superposition_density_matrix,
    excited_density_matrix,
    initial_density_matrix,
)
from sjh_learn.utils.core.parameters import NLevelSolverParams
from sjh_learn.utils.core.results import DynamicsResult


def _default_tlist(parameters: NLevelSolverParams) -> np.ndarray:
    if parameters.tlist is not None:
        return np.asarray(parameters.tlist, dtype=float)
    t_end = parameters.t_final if parameters.t_end is None else parameters.t_end
    return np.arange(parameters.t_start, t_end + 0.5 * parameters.dt, parameters.dt)


def _two_level_elements_from_states(states: list[Qobj]) -> dict[str, np.ndarray]:
    density = np.stack([state.full() for state in states], axis=0)
    if density.shape[1:] != (2, 2):
        raise ValueError("This sanity check is only defined for N=2 density matrices.")
    return {
        "rho_00": density[:, 0, 0],
        "rho_11": density[:, 1, 1],
        "rho_01": density[:, 0, 1],
        "rho_10": density[:, 1, 0],
    }


def _two_level_elements_from_result(result: DynamicsResult) -> dict[str, np.ndarray]:
    if result.dimension() != 2:
        raise ValueError("This helper is only defined for N=2 results.")
    return {
        "rho_00": result.matrix_element(0, 0),
        "rho_11": result.matrix_element(1, 1),
        "rho_01": result.matrix_element(0, 1),
        "rho_10": result.matrix_element(1, 0),
    }


def _simulate_lab_for_check(
    parameters: NLevelSolverParams,
    rho0: Qobj,
    amplitude_code_override: float = 0.0,
) -> tuple[np.ndarray, list[Qobj]]:
    times = _default_tlist(parameters)
    fields = (
        CodeCarrierField(
            amplitude_code=amplitude_code_override,
            omega_code=parameters.omega_drive,
            phase=0.0,
        )
        if parameters.pulse_sigma is None
        else CodeGaussianCarrierField(
            amplitude_code=amplitude_code_override,
            omega_code=parameters.omega_drive,
            phase=0.0,
            center_code=0.0 if parameters.pulse_center is None else parameters.pulse_center,
            sigma_code=parameters.pulse_sigma,
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


def _has_positive_channel(channels: tuple[dict[str, Any], ...]) -> bool:
    return any(float(channel.get("rate_code", channel.get("rate", 0.0))) > 0 for channel in channels)


def _lab_field_export_matches_solver_input(result: DynamicsResult) -> dict[str, Any]:
    if result.mode != "lab_exact":
        return {"passed": True, "skipped": "not_lab_exact"}
    if result.physical_params is None or result.drive is None:
        return {"passed": True, "skipped": "missing_physical_params_or_solver_field_callable"}

    # 这里检查输出/分析层使用的物理电场是否直接来自 solver 实际 field callable。
    # solver field 是无量纲 code-unit 时间函数；乘以 field_MV_per_cm 后才是 MV/cm。
    expected = np.asarray(result.drive(result.times), dtype=float) * float(result.physical_params.field_MV_per_cm)
    exported = result.field_MV_per_cm_values()
    if exported is None:
        return {"passed": False, "reason": "field_MV_per_cm_values returned None for lab_exact result."}
    max_difference = float(np.max(np.abs(np.asarray(exported, dtype=float) - expected)))
    return {
        "max_difference_MV_per_cm": max_difference,
        "threshold_MV_per_cm": 1e-12,
        "passed": bool(max_difference < 1e-12),
        "source": "solver_field_callable_times_physical_field_MV_per_cm",
    }


def _simulate_relaxation_sanity(parameters: NLevelSolverParams) -> dict[str, Any]:
    # 辅助检查只移除 pure dephasing channel，保留 relaxation channel 的 N-level 构造。
    aux_parameters = replace(parameters, pure_dephasing_channels=())
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


def _simulate_pure_dephasing_sanity(parameters: NLevelSolverParams) -> dict[str, Any]:
    # 辅助检查只移除 population relaxation channel，避免回到旧的全局 gamma 标量路径。
    aux_parameters = replace(parameters, relaxation_channels=())
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


def _simulate_closed_system_sanity(parameters: NLevelSolverParams) -> dict[str, Any]:
    aux_parameters = replace(parameters, relaxation_channels=(), pure_dephasing_channels=())
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
    max_trace_error = result.max_trace_error()
    max_hermiticity_error = result.max_hermiticity_error()
    populations = result.populations()
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
        "dimension": result.dimension(),
        "population_sum_final": float(np.sum(populations[-1].real)),
    }
    checks["lab_field_export_matches_solver_input"] = _lab_field_export_matches_solver_input(result)

    if result.dimension() != 2:
        checks["two_level_auxiliary_checks"] = "skipped_for_dimension_not_equal_to_2"
        return checks

    elements = _two_level_elements_from_result(result)
    rho_01 = elements["rho_01"]
    rho_10 = elements["rho_10"]
    checks["zero_field_closed_system_auxiliary"] = _simulate_closed_system_sanity(result.parameters)

    if _has_positive_channel(result.parameters.pure_dephasing_channels):
        checks["pure_dephasing_auxiliary"] = _simulate_pure_dephasing_sanity(result.parameters)
    if _has_positive_channel(result.parameters.relaxation_channels):
        checks["population_relaxation_auxiliary"] = _simulate_relaxation_sanity(result.parameters)

    checks["coherence_norm_final"] = {
        "rho_01_abs_final": float(abs(rho_01[-1])),
        "rho_10_abs_final": float(abs(rho_10[-1])),
    }
    return checks


def n2_mainline_equivalence_check(physical_params, normalizer=None) -> dict[str, float]:
    """检查 N=2 physical mainline 与显式 normalize+solver 路径的一致性。

    这个检查替代旧的 solver-ready multilevel 对照路径：两边都走
    `NLevelPhysicalParams -> ParaNormalizer -> NLevelSolverParams -> model.py`
    主线，只是一个使用 `run_physical_case`，另一个显式展开归一化与
    `run_lab_case`。因此它用于防止主线封装层和底层求解层发生偏离。
    """

    if physical_params.dimension != 2:
        raise ValueError("n2_mainline_equivalence_check requires an N=2 physical system.")

    from sjh_learn.utils.core.normalization import ParaNormalizer
    from sjh_learn.utils.core.solvers import optical_params_from_solver, run_lab_case, run_physical_case

    local_normalizer = ParaNormalizer(time_scale_fs=1.0, auto_scale=False) if normalizer is None else normalizer
    wrapped = run_physical_case(physical_params, normalizer=local_normalizer)

    solver = local_normalizer.normalize(physical_params)
    explicit_params = optical_params_from_solver(
        solver=solver,
        physical=physical_params,
        normalizer=local_normalizer,
    )
    explicit = run_lab_case(explicit_params)

    density_wrapped = wrapped.density_array()
    density_explicit = explicit.density_array()
    if density_wrapped.shape != density_explicit.shape or not np.allclose(wrapped.times, explicit.times):
        raise ValueError("N=2 mainline equivalence check time grids do not match.")

    rho_00_diff = float(np.max(np.abs(density_wrapped[:, 0, 0] - density_explicit[:, 0, 0])))
    rho_11_diff = float(np.max(np.abs(density_wrapped[:, 1, 1] - density_explicit[:, 1, 1])))
    rho_01_diff = float(np.max(np.abs(density_wrapped[:, 0, 1] - density_explicit[:, 0, 1])))
    rho_10_diff = float(np.max(np.abs(density_wrapped[:, 1, 0] - density_explicit[:, 1, 0])))
    return {
        "rho_00_max_difference": rho_00_diff,
        "rho_11_max_difference": rho_11_diff,
        "rho_01_max_difference": rho_01_diff,
        "rho_10_max_difference": rho_10_diff,
        "overall_max_difference": max(rho_00_diff, rho_11_diff, rho_01_diff, rho_10_diff),
    }


__all__ = ["evaluate_sanity_checks", "n2_mainline_equivalence_check"]
