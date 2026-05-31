"""N-level Hamiltonian 和 Lindblad collapse operator 构造。"""

from __future__ import annotations

import numpy as np
from qutip import Qobj, basis

from .fields import FieldConfig, total_electric_field_value
from .fields.solver_inputs import CodeCarrierField, CodeGaussianCarrierField
from .parameters import NLevelSolverParams, as_complex_matrix


def electric_field(times: np.ndarray, amplitude: float, omega_drive: float) -> np.ndarray:
    field = CodeCarrierField(amplitude_code=amplitude, omega_code=omega_drive)
    return np.asarray(field(np.asarray(times, dtype=float)), dtype=float)


def compute_detuning(epsilon_1: float, epsilon_2: float, omega_drive: float, hbar: float) -> float:
    return (epsilon_2 - epsilon_1) - hbar * omega_drive


def compute_energy_gap(detuning: float, omega_drive: float, hbar: float) -> float:
    return hbar * omega_drive + detuning


def dimension(parameters: NLevelSolverParams) -> int:
    return len(parameters.energies)


def _basis_operators() -> tuple[Qobj, Qobj, Qobj]:
    ket_0 = basis(2, 0)
    ket_1 = basis(2, 1)
    projector_0 = ket_0 * ket_0.dag()
    projector_1 = ket_1 * ket_1.dag()
    dipole_operator = ket_0 * ket_1.dag() + ket_1 * ket_0.dag()
    return projector_0, projector_1, dipole_operator


def sigma_minus_operator() -> Qobj:
    return basis(2, 0) * basis(2, 1).dag()


def sigma_z_operator() -> Qobj:
    projector_0, projector_1, _dipole = _basis_operators()
    return projector_0 - projector_1


def initial_density_matrix(n_levels: int = 2, occupied_level: int = 0) -> Qobj:
    ket = basis(n_levels, occupied_level)
    return ket * ket.dag()


def excited_density_matrix() -> Qobj:
    return basis(2, 1) * basis(2, 1).dag()


def coherent_superposition_density_matrix() -> Qobj:
    psi = (basis(2, 0) + basis(2, 1)).unit()
    return psi * psi.dag()


def default_field_config(parameters: NLevelSolverParams):
    if parameters.pulse_sigma is None:
        return CodeCarrierField(
            amplitude_code=parameters.field_amplitude,
            omega_code=parameters.omega_drive,
        )
    return CodeGaussianCarrierField(
        amplitude_code=parameters.field_amplitude,
        omega_code=parameters.omega_drive,
        center_code=0.0 if parameters.pulse_center is None else parameters.pulse_center,
        sigma_code=parameters.pulse_sigma,
    )


def parameter_fields(parameters: NLevelSolverParams) -> tuple[FieldConfig, ...]:
    if parameters.fields is not None:
        return tuple(parameters.fields)
    return (default_field_config(parameters),)


def pulse_envelope(time: float, pulse_center: float | None, pulse_sigma: float | None) -> float:
    if pulse_sigma is None:
        return 1.0
    field = CodeGaussianCarrierField(
        amplitude_code=0.0,
        omega_code=0.0,
        center_code=0.0 if pulse_center is None else pulse_center,
        sigma_code=pulse_sigma,
    )
    return float(field.envelope(time))


def build_static_hamiltonian(parameters: NLevelSolverParams) -> Qobj:
    return Qobj(np.diag(np.asarray(parameters.energies, dtype=np.complex128)))


def build_lab_hamiltonian(parameters: NLevelSolverParams) -> list[Qobj | list[object]]:
    h0 = build_static_hamiltonian(parameters)
    dipole_operator = Qobj(as_complex_matrix(parameters.dipole_matrix))
    # H_int(t) = -E_code(t) * mu_code_matrix。这里所有量已经是 solver code unit。
    return [
        h0,
        [
            -dipole_operator,
            lambda t, args: total_electric_field_value(float(t), args["fields"]),
        ],
    ]


def build_rwa_hamiltonian(parameters: NLevelSolverParams) -> list[Qobj | list[object]]:
    energies = np.asarray(parameters.energies, dtype=float)
    n_levels = len(energies)
    shifted = energies - energies[0]
    if n_levels >= 2:
        shifted[1] = shifted[1] - parameters.omega_drive
    h_static = Qobj(np.diag(shifted.astype(np.complex128)))
    coupling = as_complex_matrix(parameters.coupling_matrix or parameters.dipole_matrix)
    # RWA path 使用慢变量 coupling matrix；光学 carrier 已经移除。
    h_coupling = Qobj(-coupling)
    return [h_static, [h_coupling, lambda t, args: float(args["drive"](t))]]


def _collapse_projector(n_levels: int, level: int) -> Qobj:
    ket = basis(n_levels, level)
    return ket * ket.dag()


def build_c_ops(parameters: NLevelSolverParams) -> list[Qobj]:
    n_levels = dimension(parameters)
    c_ops: list[Qobj] = []
    for channel in parameters.relaxation_channels:
        rate = float(channel.get("rate_code", channel.get("rate", 0.0)))
        if rate <= 0:
            continue
        from_level = int(channel["from_level"])
        to_level = int(channel["to_level"])
        c_ops.append(np.sqrt(rate) * (basis(n_levels, to_level) * basis(n_levels, from_level).dag()))
    for channel in parameters.pure_dephasing_channels:
        rate = float(channel.get("rate_code", channel.get("rate", 0.0)))
        if rate <= 0:
            continue
        level = int(channel["level"])
        c_ops.append(np.sqrt(rate) * _collapse_projector(n_levels, level))
    return c_ops


__all__ = [
    "electric_field",
    "compute_detuning",
    "compute_energy_gap",
    "_basis_operators",
    "sigma_minus_operator",
    "sigma_z_operator",
    "initial_density_matrix",
    "excited_density_matrix",
    "coherent_superposition_density_matrix",
    "default_field_config",
    "parameter_fields",
    "pulse_envelope",
    "build_lab_hamiltonian",
    "build_rwa_hamiltonian",
    "build_c_ops",
]
