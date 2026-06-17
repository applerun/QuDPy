#!/usr/bin/env python3
"""QuDPy spectroscopy postprocessing 最小 smoke test。"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import sjh_learn.utils.spectroscopy as spectroscopy
from sjh_learn.utils.core import NLevelPhysicalParams, ParaNormalizer, run_case
from sjh_learn.utils.core.config import RWA_DISABLED_MESSAGE
from sjh_learn.utils.fields import make_default_carrier_field


def make_params(*, solver_mode: str = "lab_exact") -> NLevelPhysicalParams:
    return NLevelPhysicalParams(
        basis=("g", "e"),
        energies_eV=(0.0, 1.55),
        dipole_matrix_D=((0.0, 3.0), (3.0, 0.0)),
        t_start_fs=0.0,
        t_end_fs=0.2,
        dt_fs=0.1,
        field=make_default_carrier_field(E0_MV_per_cm=0.01, laser_energy_eV=1.55),
        solver_mode=solver_mode,
    )


def check_import() -> None:
    if not hasattr(spectroscopy, "lab_frame_fft_response"):
        raise AssertionError("spectroscopy import missing lab_frame_fft_response.")
    print("spectroscopy import ok")


def check_lab_exact() -> None:
    result = run_case(make_params(), normalizer=ParaNormalizer(time_scale_fs=1.0, auto_scale=False))
    if result.mode != "lab_exact":
        raise AssertionError(f"expected lab_exact, got {result.mode!r}")
    print("lab_exact run_case ok")


def check_rwa_disabled() -> None:
    try:
        run_case(make_params(solver_mode="rwa"), normalizer=ParaNormalizer(time_scale_fs=1.0, auto_scale=False))
    except RuntimeError as exc:
        if RWA_DISABLED_MESSAGE not in str(exc):
            raise AssertionError(f"unexpected RWA error message: {exc}") from exc
        print("rwa disabled guard ok")
        return
    raise AssertionError("solver_mode='rwa' should raise RuntimeError while FORCE_RWA=False.")


def check_lab_frame_fft_response() -> None:
    t_fs = np.linspace(0.0, 31.0, 32)
    omega = 0.4
    E = np.cos(omega * t_fs)
    P = 0.2 * E
    rho = 0.1 * E.astype(np.complex128)
    response = spectroscopy.lab_frame_fft_response_legacy(
        t_fs=t_fs,
        E_MV_per_cm=E,
        P_C_per_m2=P,
        rhoij=rho,
        window=None,
        subtract_mean=True,
        rel_threshold=1e-8,
        zero_padding_factor=1,
    )
    required = {"omega_fs_inv", "energy_eV", "E_fft", "P_fft", "rhoij_fft", "P_over_E", "neg_omega_im_P_over_E"}
    missing = required.difference(response)
    if missing:
        raise AssertionError(f"lab_frame_fft_response missing keys: {sorted(missing)}")
    if response["omega_fs_inv"].size == 0:
        raise AssertionError("lab_frame_fft_response returned empty positive-frequency response.")
    print("lab_frame_fft_response ok")


def main() -> None:
    check_import()
    check_lab_exact()
    check_rwa_disabled()
    check_lab_frame_fft_response()


if __name__ == "__main__":
    main()
