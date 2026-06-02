"""谱学 observable 的物理单位工具。"""

from __future__ import annotations

import numpy as np


DEBYE_TO_C_M = 3.33564e-30
FS_INV_TO_S_INV = 1.0e15
HBAR_J_S = 1.054571817e-34
EPSILON0_F_PER_M = 8.8541878128e-12


def _as_density_trajectory(rho_t) -> np.ndarray:
    rho = np.asarray(rho_t, dtype=np.complex128)
    if rho.ndim != 3 or rho.shape[1] != rho.shape[2]:
        raise ValueError("rho_t 必须是 shape=(T, N, N) 的 density-matrix trajectory。")
    return rho


def _as_dipole_matrix(dipole_matrix_D, dimension: int) -> np.ndarray:
    mu = np.asarray(dipole_matrix_D, dtype=np.complex128)
    if mu.shape != (dimension, dimension):
        raise ValueError("dipole_matrix_D 必须是 shape=(N, N)，并与 rho_t 的 N 一致。")
    return mu


def dipole_expectation_D(rho_t, dipole_matrix_D) -> np.ndarray:
    """计算单个量子系统的偶极矩期望值，单位 Debye。

    物理约定：`p(t) = Tr[rho(t) mu] = sum_ij rho_ij(t) mu_ji`。
    这里必须使用用户侧物理偶极矩 `dipole_matrix_D`，不能使用已经乘过场强
    和归一化因子的 `coupling_matrix_code`。
    """

    rho = _as_density_trajectory(rho_t)
    mu = _as_dipole_matrix(dipole_matrix_D, rho.shape[1])
    return np.einsum("tij,ji->t", rho, mu)


def polarization_C_per_m2(rho_t, dipole_matrix_D, number_density_m3: float) -> np.ndarray:
    """计算宏观 polarization，单位 C/m^2。

    `number_density_m3` 的单位是 `m^-3`。单分子偶极矩先由 Debye 转为
    `C*m`，再乘 number density，得到 `P(t)` 的 `C/m^2`。
    """

    density = float(number_density_m3)
    if density < 0:
        raise ValueError("number_density_m3 不能为负。")
    return density * dipole_expectation_D(rho_t, dipole_matrix_D) * DEBYE_TO_C_M


def chi_two_level_linear(
    omega_fs_inv,
    omega_eg_fs_inv: float,
    mu_ge_D: float,
    gamma2_fs_inv: float,
    number_density_m3: float,
    population_difference: float = 1.0,
) -> np.ndarray:
    """二能级线性响应 susceptibility 参考公式。

    输入角频率使用 `fs^-1`，函数内部转换到 `s^-1`。`mu_ge_D` 使用 Debye，
    内部转换为 `C*m`。这里的公式是
    `chi = N |mu|^2 / (epsilon0 hbar) * population_difference
    / (omega_eg - omega - 1j gamma2)`。
    """

    omega_s_inv = np.asarray(omega_fs_inv, dtype=float) * FS_INV_TO_S_INV
    omega_eg_s_inv = float(omega_eg_fs_inv) * FS_INV_TO_S_INV
    gamma2_s_inv = float(gamma2_fs_inv) * FS_INV_TO_S_INV
    if gamma2_s_inv < 0:
        raise ValueError("gamma2_fs_inv 不能为负。")
    density = float(number_density_m3)
    if density < 0:
        raise ValueError("number_density_m3 不能为负。")
    mu_C_m = float(abs(mu_ge_D)) * DEBYE_TO_C_M
    prefactor = density * (mu_C_m**2) / (EPSILON0_F_PER_M * HBAR_J_S)
    denominator = omega_eg_s_inv - omega_s_inv - 1j * gamma2_s_inv
    return prefactor * float(population_difference) / denominator


__all__ = [
    "DEBYE_TO_C_M",
    "dipole_expectation_D",
    "polarization_C_per_m2",
    "chi_two_level_linear",
]
