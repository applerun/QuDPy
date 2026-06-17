"""TA pump_dark_probe 模式下的 dark exact propagation 工具。

本模块只负责无场段的精确 Liouvillian 传播，不判断 delay mode，不写输出，
也不计算 absorption。runner 后续可在 pump segment 结束后调用本函数。
"""

from __future__ import annotations

from collections.abc import Sequence

from qutip import Qobj, liouvillian, operator_to_vector, vector_to_operator


def propagate_dark_exact(
    *,
    rho_before_dark: Qobj,
    hamiltonian: Qobj,
    c_ops: Sequence[Qobj],
    duration_fs: float,
) -> Qobj:
    """用 time-independent Liouvillian 指数传播 dark 段。

    参数约定：
    - ``rho_before_dark`` 是 dark 段起点的 density matrix；
    - ``hamiltonian`` 使用 fs^-1 单位，通常是不含外场的 H0；
    - ``c_ops`` 使用 fs^-1 速率构造；
    - ``duration_fs`` 是无场传播时间，单位 fs。

    传播公式：
        ``rho_after_dark = exp(L_dark * duration_fs) rho_before_dark``

    这里不重新引入 mesolve dark tlist；长暗段应由该精确指数传播处理。
    """

    duration = float(duration_fs)
    if duration < 0.0:
        raise ValueError("duration_fs must be non-negative.")

    if duration == 0.0:
        return rho_before_dark.copy()

    liouvillian_dark = liouvillian(hamiltonian, list(c_ops))
    rho_vec_final = (liouvillian_dark * duration).expm() * operator_to_vector(rho_before_dark)
    return vector_to_operator(rho_vec_final)


def propagate_dark_exact_trajectory(
    *,
    rho_before_dark: Qobj,
    hamiltonian: Qobj,
    c_ops: Sequence[Qobj],
    times: Sequence[float],
) -> list[Qobj]:
    """在 dark 段完整时间轴上生成 field-free exact trajectory。

    `times` 使用与 Hamiltonian / c_ops 相同的 solver time unit。第一个时间点
    是 dark window 起点；后续每个点使用 `exp(L_dark * (t - t0))` 从同一个
    起点密度矩阵传播得到。这样展示层可以看到完整 dark window，而不是只有
    起点和终点。
    """

    time_values = [float(item) for item in times]
    if not time_values:
        raise ValueError("times must contain at least one point.")
    t0 = time_values[0]
    if any(time < t0 for time in time_values):
        raise ValueError("times must be sorted in non-decreasing order.")

    liouvillian_dark = liouvillian(hamiltonian, list(c_ops))
    rho_vec_initial = operator_to_vector(rho_before_dark)
    states: list[Qobj] = []
    for time in time_values:
        duration = float(time) - t0
        if duration == 0.0:
            states.append(rho_before_dark.copy())
        else:
            states.append(vector_to_operator((liouvillian_dark * duration).expm() * rho_vec_initial))
    return states


__all__ = [
    "propagate_dark_exact",
    "propagate_dark_exact_trajectory",
]
