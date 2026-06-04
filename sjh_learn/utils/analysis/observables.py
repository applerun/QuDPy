"""analysis 层的物理 observable 工具。

这些函数只从已经求解完成的 density matrix trajectory 和物理参数计算后处理量。
它们不属于 solver/core，也不应被 `DynamicsResult` 用来自动追加物理 observable。
"""

from __future__ import annotations

import numpy as np


DEBYE_TO_C_M = 3.33564e-30
FS_INV_TO_S_INV = 1.0e15
HBAR_J_S = 1.054571817e-34
EPSILON0_F_PER_M = 8.8541878128e-12


def _as_density_trajectory(rho_t) -> np.ndarray:
    rho = np.asarray(rho_t, dtype=np.complex128)
    if rho.ndim != 3 or rho.shape[1] != rho.shape[2]:
        raise ValueError("rho_t must have shape=(T, N, N).")
    if rho.shape[0] == 0:
        raise ValueError("rho_t must contain at least one time point.")
    return rho


def _as_dipole_matrix(dipole_matrix_D, dimension: int) -> np.ndarray:
    mu = np.asarray(dipole_matrix_D, dtype=np.complex128)
    if mu.shape != (dimension, dimension):
        raise ValueError("dipole_matrix_D must have shape=(N, N), matching rho_t.")
    if not np.allclose(mu, mu.conj().T, rtol=1e-10, atol=1e-12):
        raise ValueError("dipole_matrix_D must be Hermitian: mu[j, i] = conj(mu[i, j]).")
    diagonal = np.diag(mu)
    if np.max(np.abs(diagonal.imag)) > 1e-12:
        raise ValueError("diagonal elements of dipole_matrix_D must be real within numerical tolerance.")
    return mu


def dipole_expectation_D(rho_t, dipole_matrix_D) -> np.ndarray:
    """计算单个量子体系的偶极矩期望值，单位 Debye。

    物理约定为 `p(t) = Tr[rho(t) mu] = sum_ij rho_ij(t) mu_ji`。实现使用
    `np.einsum("tij,ji->t", rho_t, dipole_matrix_D)`，显式保留指标关系，
    并避免逐时间点构造 `rho @ mu` 的中间矩阵。

    `dipole_matrix_D` 必须是用户侧物理偶极矩矩阵，不能使用已经乘过场强
    或 solver code-unit 归一化因子的 coupling matrix。
    """

    rho = _as_density_trajectory(rho_t)
    mu = _as_dipole_matrix(dipole_matrix_D, rho.shape[1])
    return np.einsum("tij,ji->t", rho, mu)


def polarization_C_per_m2(
    rho_t,
    dipole_matrix_D,
    number_density_m3: float,
    *,
    imag_tolerance: float = 1e-10,
) -> np.ndarray:
    """计算宏观 polarization，单位 C/m^2。

    `number_density_m3` 的单位是 `m^-3`，必须显式给出。公式为
    `P(t) = number_density_m3 * Tr[rho(t) mu_D] * DEBYE_TO_C_M`。
    对 Hermitian density matrix 和 Hermitian dipole matrix，结果应为实数；
    如果虚部超过 `imag_tolerance`，直接报错，避免静默丢弃物理问题。
    """

    density = float(number_density_m3)
    if density < 0:
        raise ValueError("number_density_m3 must be non-negative.")
    polarization = density * dipole_expectation_D(rho_t, dipole_matrix_D) * DEBYE_TO_C_M
    max_imag = float(np.max(np.abs(polarization.imag)))
    if max_imag > imag_tolerance:
        raise ValueError(
            "polarization_C_per_m2 should be real for Hermitian rho and dipole_matrix_D; "
            f"max imaginary part is {max_imag:.3e}."
        )
    return polarization.real


def chi_two_level_linear(
    omega_fs_inv,
    omega_eg_fs_inv: float,
    mu_ge_D: complex,
    gamma2_fs_inv: float,
    number_density_m3: float,
    population_difference: float = 1.0,
) -> np.ndarray:
    """two-level analytic linear-response susceptibility 教学参考公式。

    这是 analysis 层的 analytic/teaching helper，不是 core two-level API。
    `mu_ge_D` 可以是 complex；线性响应强度使用 `mu_ge * mu_eg = |mu_ge|^2`，
    因此单一 two-level transition 的整体偶极相位不会改变吸收强度。
    """

    omega_s_inv = np.asarray(omega_fs_inv, dtype=float) * FS_INV_TO_S_INV
    omega_eg_s_inv = float(omega_eg_fs_inv) * FS_INV_TO_S_INV
    gamma2_s_inv = float(gamma2_fs_inv) * FS_INV_TO_S_INV
    if gamma2_s_inv < 0:
        raise ValueError("gamma2_fs_inv must be non-negative.")
    density = float(number_density_m3)
    if density < 0:
        raise ValueError("number_density_m3 must be non-negative.")
    mu_C_m = float(abs(mu_ge_D)) * DEBYE_TO_C_M
    prefactor = density * (mu_C_m**2) / (EPSILON0_F_PER_M * HBAR_J_S)
    denominator = omega_eg_s_inv - omega_s_inv - 1j * gamma2_s_inv
    return prefactor * float(population_difference) / denominator


__all__ = [
    "DEBYE_TO_C_M",
    "EPSILON0_F_PER_M",
    "FS_INV_TO_S_INV",
    "HBAR_J_S",
    "dipole_expectation_D",
    "polarization_C_per_m2",
    "chi_two_level_linear",
]
