"""最小可运行的多能级实验室系求解框架。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from qutip import Qobj, basis, mesolve

from .fields import FieldConfig, total_electric_field_value
from .model import compute_energy_gap, initial_density_matrix, parameter_fields
from .parameters import OpticalBlochParameters
from .results import MultiLevelResult


@dataclass(frozen=True)
class CollapseChannel:
    """多能级耗散通道定义。"""

    name: str
    rate: float
    operator: list[list[complex]] | np.ndarray | None = None
    from_level: int | None = None
    to_level: int | None = None
    dephase_level: int | None = None
    kind: str = "custom"


@dataclass(frozen=True)
class MultiLevelParameters:
    """多能级实验室系求解参数。"""

    energies: tuple[float, ...]
    dipole_matrix: list[list[float]] | np.ndarray
    t_start: float
    t_end: float
    dt: float
    fields: tuple[FieldConfig, ...]
    hbar: float = 1.0
    level_labels: tuple[str, ...] | None = None
    collapse_channels: tuple[CollapseChannel, ...] = ()
    tlist: np.ndarray | None = None
    times_fs: np.ndarray | None = None


def _validate_multilevel_parameters(parameters: MultiLevelParameters) -> int:
    n_levels = len(parameters.energies)
    if n_levels < 2:
        raise ValueError("多能级系统至少需要两个能级。")
    dipole = np.asarray(parameters.dipole_matrix, dtype=np.complex128)
    if dipole.shape != (n_levels, n_levels):
        raise ValueError("dipole_matrix 的形状必须与 energies 一致。")
    if parameters.dt <= 0:
        raise ValueError("dt 必须为正。")
    if parameters.t_end <= parameters.t_start:
        raise ValueError("t_end 必须大于 t_start。")
    if not parameters.fields:
        raise ValueError("多能级求解至少需要一个 FieldConfig。")
    return n_levels


def _default_tlist(parameters: MultiLevelParameters) -> np.ndarray:
    if parameters.tlist is not None:
        return np.asarray(parameters.tlist, dtype=float)
    return np.arange(parameters.t_start, parameters.t_end + 0.5 * parameters.dt, parameters.dt)


def build_multilevel_static_hamiltonian(parameters: MultiLevelParameters) -> Qobj:
    _validate_multilevel_parameters(parameters)
    return Qobj(np.diag(np.asarray(parameters.energies, dtype=np.complex128)))


def build_multilevel_dipole_operator(parameters: MultiLevelParameters) -> Qobj:
    n_levels = _validate_multilevel_parameters(parameters)
    dipole = np.asarray(parameters.dipole_matrix, dtype=np.complex128)
    if dipole.shape != (n_levels, n_levels):
        raise ValueError("dipole_matrix 的形状必须是 N x N。")
    return Qobj(dipole)


def build_multilevel_lab_hamiltonian(parameters: MultiLevelParameters) -> list[Qobj | list[object]]:
    h0 = build_multilevel_static_hamiltonian(parameters)
    mu_operator = build_multilevel_dipole_operator(parameters)
    return [
        h0,
        [
            -mu_operator,
            lambda t, args: total_electric_field_value(float(t), args["fields"]),
        ],
    ]


def _collapse_operator_from_channel(parameters: MultiLevelParameters, channel: CollapseChannel) -> Qobj:
    n_levels = len(parameters.energies)
    if channel.rate < 0:
        raise ValueError(f"collapse channel {channel.name} 的 rate 不能为负。")

    if channel.kind == "relaxation":
        if channel.from_level is None or channel.to_level is None:
            raise ValueError(f"relaxation channel {channel.name} 需要 from_level 和 to_level。")
        if not (0 <= channel.from_level < n_levels and 0 <= channel.to_level < n_levels):
            raise ValueError(f"relaxation channel {channel.name} 的 level index 超界。")
        return np.sqrt(channel.rate) * (basis(n_levels, channel.to_level) * basis(n_levels, channel.from_level).dag())

    if channel.kind == "pure_dephasing":
        if channel.dephase_level is None:
            raise ValueError(f"pure_dephasing channel {channel.name} 需要 dephase_level。")
        if not 0 <= channel.dephase_level < n_levels:
            raise ValueError(f"pure_dephasing channel {channel.name} 的 level index 超界。")
        return np.sqrt(channel.rate) * (basis(n_levels, channel.dephase_level) * basis(n_levels, channel.dephase_level).dag())

    if channel.kind == "custom":
        if channel.operator is None:
            raise ValueError(f"custom channel {channel.name} 需要 operator。")
        operator = np.asarray(channel.operator, dtype=np.complex128)
        if operator.shape != (n_levels, n_levels):
            raise ValueError(f"custom channel {channel.name} 的 operator 必须是 N x N。")
        return np.sqrt(channel.rate) * Qobj(operator)

    raise ValueError(f"不支持的 collapse channel kind: {channel.kind}")


def build_multilevel_c_ops(parameters: MultiLevelParameters) -> list[Qobj]:
    _validate_multilevel_parameters(parameters)
    return [_collapse_operator_from_channel(parameters, channel) for channel in parameters.collapse_channels]


def optical_bloch_to_multilevel_parameters(parameters: OpticalBlochParameters) -> MultiLevelParameters:
    """把 two-level OpticalBlochParameters 映射成等价的 N=2 multi-level 参数。"""
    epsilon_2 = parameters.epsilon_1 + compute_energy_gap(
        detuning=parameters.detuning,
        omega_drive=parameters.omega_drive,
        hbar=parameters.hbar,
    )
    collapse_channels: list[CollapseChannel] = []
    if parameters.gamma_phi > 0:
        # 两个 projector 去相干通道共同作用时，
        # rho_12 的额外衰减是 (gamma_0 + gamma_1) / 2。
        # 这里对两个能级各放一个相同的 gamma_phi，可与 two-level sigma_z 约定匹配。
        collapse_channels.extend(
            [
                CollapseChannel(
                    name="dephase_level_0",
                    kind="pure_dephasing",
                    dephase_level=0,
                    rate=parameters.gamma_phi,
                ),
                CollapseChannel(
                    name="dephase_level_1",
                    kind="pure_dephasing",
                    dephase_level=1,
                    rate=parameters.gamma_phi,
                ),
            ]
        )
    if parameters.gamma1 > 0:
        collapse_channels.append(
            CollapseChannel(
                name="relaxation_1_to_0",
                kind="relaxation",
                from_level=1,
                to_level=0,
                rate=parameters.gamma1,
            )
        )
    return MultiLevelParameters(
        energies=(parameters.epsilon_1, epsilon_2),
        dipole_matrix=[
            [0.0, parameters.dipole],
            [parameters.dipole, 0.0],
        ],
        t_start=parameters.t_start,
        t_end=parameters.t_final if parameters.t_end is None else parameters.t_end,
        dt=parameters.dt,
        fields=parameter_fields(parameters),
        hbar=parameters.hbar,
        level_labels=("1", "2"),
        collapse_channels=tuple(collapse_channels),
        tlist=None if parameters.tlist is None else np.asarray(parameters.tlist, dtype=float),
        times_fs=None if parameters.times_fs is None else np.asarray(parameters.times_fs, dtype=float),
    )


def initial_multilevel_density_matrix(n_levels: int, occupied_level: int = 0) -> Qobj:
    """返回指定单能级占据的初态密度矩阵。"""
    if not 0 <= occupied_level < n_levels:
        raise ValueError("occupied_level 超界。")
    ket = basis(n_levels, occupied_level)
    return ket * ket.dag()


def coherent_multilevel_density_matrix(amplitudes: list[complex] | tuple[complex, ...] | np.ndarray) -> Qobj:
    """根据给定复振幅构造纯态密度矩阵。"""
    vector = np.asarray(amplitudes, dtype=np.complex128)
    norm = np.linalg.norm(vector)
    if norm == 0:
        raise ValueError("amplitudes 不能全为 0。")
    vector = vector / norm
    ket = Qobj(vector.reshape((-1, 1)))
    return ket * ket.dag()


def simulate_multilevel_lab_frame(
    parameters: MultiLevelParameters,
    rho0: Qobj | None = None,
) -> tuple[np.ndarray, list[Qobj]]:
    n_levels = _validate_multilevel_parameters(parameters)
    times = _default_tlist(parameters)
    result = mesolve(
        H=build_multilevel_lab_hamiltonian(parameters),
        rho0=initial_multilevel_density_matrix(n_levels) if rho0 is None else rho0,
        tlist=times,
        c_ops=build_multilevel_c_ops(parameters),
        e_ops=[],
        args={"fields": parameters.fields},
    )
    return times, list(result.states)


def _evaluate_multilevel_sanity_checks(result: MultiLevelResult) -> dict[str, Any]:
    trace_error = result.max_trace_error()
    hermiticity_error = result.max_hermiticity_error()
    populations = result.populations()
    return {
        "trace_error_small": {
            "value": trace_error,
            "threshold": 1e-8,
            "passed": bool(trace_error < 1e-8),
        },
        "hermiticity_error_small": {
            "value": hermiticity_error,
            "threshold": 1e-8,
            "passed": bool(hermiticity_error < 1e-8),
        },
        "population_sum_final": float(np.sum(populations[-1].real)),
    }


def run_multilevel_case(
    parameters: MultiLevelParameters,
    rho0: Qobj | None = None,
) -> MultiLevelResult:
    times, lab_states = simulate_multilevel_lab_frame(parameters, rho0=rho0)
    result = MultiLevelResult(
        parameters=parameters,
        times=times,
        times_fs=parameters.times_fs,
        lab_states=lab_states,
    )
    result.sanity_checks = _evaluate_multilevel_sanity_checks(result)
    return result


def two_level_multilevel_equivalence_check(parameters: OpticalBlochParameters) -> dict[str, float]:
    """比较 two-level exact lab-frame 与等价 N=2 multi-level 路径的差异。"""
    from .solvers import simulate_lab_frame

    times_two, states_two = simulate_lab_frame(parameters, rho0=initial_density_matrix())
    multi_parameters = optical_bloch_to_multilevel_parameters(parameters)
    times_multi, states_multi = simulate_multilevel_lab_frame(
        multi_parameters,
        rho0=initial_multilevel_density_matrix(2, occupied_level=0),
    )

    density_two = np.stack([state.full() for state in states_two], axis=0)
    density_multi = np.stack([state.full() for state in states_multi], axis=0)

    if density_two.shape != density_multi.shape or not np.allclose(times_two, times_multi):
        raise ValueError("two-level 与 multi-level equivalence check 的时间网格不一致。")

    rho11_diff = float(np.max(np.abs(density_two[:, 0, 0] - density_multi[:, 0, 0])))
    rho22_diff = float(np.max(np.abs(density_two[:, 1, 1] - density_multi[:, 1, 1])))
    rho12_diff = float(np.max(np.abs(density_two[:, 0, 1] - density_multi[:, 0, 1])))
    rho21_diff = float(np.max(np.abs(density_two[:, 1, 0] - density_multi[:, 1, 0])))
    return {
        "rho11_max_difference": rho11_diff,
        "rho22_max_difference": rho22_diff,
        "rho12_max_difference": rho12_diff,
        "rho21_max_difference": rho21_diff,
        "overall_max_difference": max(rho11_diff, rho22_diff, rho12_diff, rho21_diff),
    }


__all__ = [
    "CollapseChannel",
    "MultiLevelParameters",
    "build_multilevel_static_hamiltonian",
    "build_multilevel_dipole_operator",
    "build_multilevel_lab_hamiltonian",
    "build_multilevel_c_ops",
    "optical_bloch_to_multilevel_parameters",
    "initial_multilevel_density_matrix",
    "coherent_multilevel_density_matrix",
    "simulate_multilevel_lab_frame",
    "run_multilevel_case",
    "two_level_multilevel_equivalence_check",
]
