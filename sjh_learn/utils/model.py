"""两能级模型相关的算符、哈密顿量与耗散构造。"""

from __future__ import annotations

import numpy as np
from qutip import Qobj, basis

from .fields import FieldConfig, electric_field_array, electric_field_value, total_electric_field_value
from .parameters import OpticalBlochParameters


def electric_field(times: np.ndarray, amplitude: float, omega_drive: float) -> np.ndarray:
    return electric_field_array(
        np.asarray(times, dtype=float),
        FieldConfig(amplitude=amplitude, omega=omega_drive),
    )


def compute_detuning(
    epsilon_1: float,
    epsilon_2: float,
    omega_drive: float,
    hbar: float,
) -> float:
    return (epsilon_2 - epsilon_1) - hbar * omega_drive


def compute_energy_gap(
    detuning: float,
    omega_drive: float,
    hbar: float,
) -> float:
    return hbar * omega_drive + detuning


def _basis_operators() -> tuple[Qobj, Qobj, Qobj]:
    ket_1 = basis(2, 0)
    ket_2 = basis(2, 1)
    projector_1 = ket_1 * ket_1.dag()
    projector_2 = ket_2 * ket_2.dag()
    dipole_operator = ket_1 * ket_2.dag() + ket_2 * ket_1.dag()
    return projector_1, projector_2, dipole_operator


def sigma_minus_operator() -> Qobj:
    ket_1 = basis(2, 0)
    ket_2 = basis(2, 1)
    return ket_1 * ket_2.dag()


def sigma_z_operator() -> Qobj:
    projector_1, projector_2, _dipole = _basis_operators()
    return projector_1 - projector_2


def initial_density_matrix() -> Qobj:
    ket_1 = basis(2, 0)
    return ket_1 * ket_1.dag()


def excited_density_matrix() -> Qobj:
    ket_2 = basis(2, 1)
    return ket_2 * ket_2.dag()


def coherent_superposition_density_matrix() -> Qobj:
    ket_1 = basis(2, 0)
    ket_2 = basis(2, 1)
    psi = (ket_1 + ket_2).unit()
    return psi * psi.dag()


def default_field_config(parameters: OpticalBlochParameters) -> FieldConfig:
    """把两能级旧参数转换成默认的单电场配置。"""
    if parameters.pulse_sigma is None:
        return FieldConfig(
            amplitude=parameters.field_amplitude,
            omega=parameters.omega_drive,
        )
    return FieldConfig(
        amplitude=parameters.field_amplitude,
        omega=parameters.omega_drive,
        envelope="gaussian",
        center=0.0 if parameters.pulse_center is None else parameters.pulse_center,
        sigma=parameters.pulse_sigma,
    )


def parameter_fields(parameters: OpticalBlochParameters) -> tuple[FieldConfig, ...]:
    """返回求解时实际使用的电场列表。"""
    if parameters.fields is not None:
        return tuple(parameters.fields)
    return (default_field_config(parameters),)


def pulse_envelope(time: float, pulse_center: float | None, pulse_sigma: float | None) -> float:
    """兼容保留的包络接口，内部委托给 fields.py。"""
    if pulse_sigma is None:
        field = FieldConfig(amplitude=0.0, omega=0.0, envelope="constant")
    else:
        field = FieldConfig(
            amplitude=0.0,
            omega=0.0,
            envelope="gaussian",
            center=0.0 if pulse_center is None else pulse_center,
            sigma=pulse_sigma,
        )
    from .fields import envelope_value

    return envelope_value(time, field)


def build_lab_hamiltonian(parameters: OpticalBlochParameters) -> list[Qobj | list[object]]:
    projector_1, projector_2, dipole_operator = _basis_operators()
    epsilon_2 = parameters.epsilon_1 + compute_energy_gap(
        detuning=parameters.detuning,
        omega_drive=parameters.omega_drive,
        hbar=parameters.hbar,
    )
    h0 = parameters.epsilon_1 * projector_1 + epsilon_2 * projector_2
    h_drive = -parameters.dipole * dipole_operator
    return [
        h0,
        [
            h_drive,
            lambda t, args: total_electric_field_value(float(t), args["fields"]),
        ],
    ]


def build_rwa_hamiltonian(parameters: OpticalBlochParameters) -> Qobj:
    g = parameters.dipole * parameters.field_amplitude
    return Qobj(
        np.array(
            [
                [0.0, -g],
                [-g, parameters.detuning],
            ],
            dtype=np.complex128,
        )
    )


def build_c_ops(parameters: OpticalBlochParameters) -> list[Qobj]:
    c_ops: list[Qobj] = []
    if parameters.gamma_phi > 0:
        c_ops.append(np.sqrt(parameters.gamma_phi / 2.0) * sigma_z_operator())
    if parameters.gamma1 > 0:
        c_ops.append(np.sqrt(parameters.gamma1) * sigma_minus_operator())
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
