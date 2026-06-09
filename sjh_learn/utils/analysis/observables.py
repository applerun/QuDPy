"""analysis 层的谱学 observable 工具。

这些函数只从已经求解完成的 density matrix trajectory 和物理参数计算后处理量；
它们不属于 solver/core，也不应被 `DynamicsResult` 用来自动追加物理 observable。
"""

from __future__ import annotations

from sjh_learn.utils.spectroscopy.observables import DEBYE_TO_C_M, dipole_expectation_D, polarization_C_per_m2
from sjh_learn.utils.spectroscopy.theory import (
    EPSILON0_F_PER_M,
    FS_INV_TO_S_INV,
    HBAR_J_S,
    chi_two_level_linear,
)


__all__ = [
    "DEBYE_TO_C_M",
    "EPSILON0_F_PER_M",
    "FS_INV_TO_S_INV",
    "HBAR_J_S",
    "dipole_expectation_D",
    "polarization_C_per_m2",
    "chi_two_level_linear",
]
