"""TA piecewise dark segment 的无场精确传播。"""

from __future__ import annotations

import numpy as np
from qutip import Qobj, basis, liouvillian, operator_to_vector, vector_to_operator

from sjh_learn.utils.core import ParaNormalizer
from sjh_learn.utils.core.results import DynamicsResult

from .ta_settings import TaDelayScanSettings
from .ta_specs import TaSegmentSpec


def build_dark_c_ops_physical(settings: TaDelayScanSettings) -> list[Qobj]:
    """按物理 fs^-1 单位构造 dark Liouvillian 使用的 collapse operators。"""

    n_levels = len(settings.energies_eV)
    c_ops: list[Qobj] = []

    relaxation_channels = (
        {"from_level": 2, "to_level": 1, "rate_fs_inv": 1.0 / float(settings.T1_2_to_1_fs)},
        {"from_level": 1, "to_level": 0, "rate_fs_inv": 1.0 / float(settings.T1_1_to_0_fs)},
    )
    for channel in relaxation_channels:
        rate = float(channel["rate_fs_inv"])
        if rate <= 0:
            continue
        from_level = int(channel["from_level"])
        to_level = int(channel["to_level"])
        c_ops.append(np.sqrt(rate) * (basis(n_levels, to_level) * basis(n_levels, from_level).dag()))

    pure_dephasing_channels = (
        {"level": 1, "rate_fs_inv": 1.0 / float(settings.Tphi_1_fs)},
        {"level": 2, "rate_fs_inv": 1.0 / float(settings.Tphi_2_fs)},
    )
    for channel in pure_dephasing_channels:
        rate = float(channel["rate_fs_inv"])
        if rate <= 0:
            continue
        level = int(channel["level"])
        ket = basis(n_levels, level)
        c_ops.append(np.sqrt(rate) * (ket * ket.dag()))

    return c_ops


def run_dark_segment_exact(
    segment: TaSegmentSpec,
    *,
    settings: TaDelayScanSettings,
    rho0: Qobj | None,
):
    """用 time-independent Liouvillian exponential 传播 dark segment。

    dark 段没有光场，但在 lab frame 中 coherence 仍会因 H0 旋转。直接用稀疏
    tlist 的 mesolve 积分长暗段可能触发积分器 nsteps 限制，因此这里使用
    `expm(L_dark * duration)` 做精确传播。
    """

    if rho0 is None:
        raise ValueError("Dark segment requires rho0 from the previous segment.")

    ckp = segment.checkpoint_path
    if ckp is not None and ckp.exists() and not settings.force_run:
        return DynamicsResult.from_ckp(ckp)

    duration_fs = float(segment.t_end_fs) - float(segment.t_start_fs)
    if duration_fs < 0:
        raise ValueError(f"Dark segment has negative duration: {duration_fs:g} fs.")

    energies_fs_inv = np.asarray(
        ParaNormalizer.energy_eV_to_fs_inv(np.asarray(settings.energies_eV, dtype=float)),
        dtype=float,
    )
    h0 = Qobj(np.diag(energies_fs_inv.astype(np.complex128)))
    liouvillian_dark = liouvillian(h0, build_dark_c_ops_physical(settings))
    rho_vec_final = (liouvillian_dark * duration_fs).expm() * operator_to_vector(rho0)
    rho_final = vector_to_operator(rho_vec_final)

    times_fs = np.asarray([segment.t_start_fs, segment.t_end_fs], dtype=float)
    result = DynamicsResult(
        mode="dark_exact",
        times=times_fs.copy(),
        times_fs=times_fs,
        states=[rho0, rho_final],
        parameters=None,
        physical_params=segment.params,
        solver_params=None,
        metadata={
            "segment_role": "dark",
            "propagation": "exact_liouvillian_expm",
            "duration_fs": duration_fs,
            "note": "无光场段使用 H0 与 Lindblad 耗散的 time-independent Liouvillian 精确传播。",
        },
    )
    result.sanity_checks = {
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

    if ckp is not None:
        ckp.parent.mkdir(parents=True, exist_ok=True)
        result.save_ckp(ckp)

    return result
