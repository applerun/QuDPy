#!/usr/bin/env python3
"""Analytic two-level linear susceptibility reference."""

from __future__ import annotations

import csv
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from sjh_learn.utils import (
    NLevelPhysicalParams,
    ParaNormalizer,
    PureDephasingChannel,
    chi_two_level_linear,
)


OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "absorption_01_chi_analytic"
RESULTS_CSV = OUTPUT_DIR / "results.csv"
FIGURE_PATH = OUTPUT_DIR / "chi_spectrum.png"


def make_physical_params() -> NLevelPhysicalParams:
    """构造 N=2 物理系统；这里不运行 mesolve，只借用 normalizer 做单位换算。"""

    return NLevelPhysicalParams(
        basis=("g", "e"),
        energies_eV=(0.0, 1.55),
        dipole_matrix_D=((0.0, 3.0), (3.0, 0.0)),
        laser_energy_eV=1.55,
        field_MV_per_cm=0.01,
        t_start_fs=0.0,
        t_end_fs=100.0,
        dt_fs=1.0,
        pure_dephasing_channels=(
            PureDephasingChannel(name="pure_dephasing_level_0", level=0, Tphi_fs=300.0),
            PureDephasingChannel(name="pure_dephasing_level_1", level=1, Tphi_fs=300.0),
        ),
    )


def write_results_csv(path: Path, rows: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "laser_energy_eV",
        "omega_fs_inv",
        "Re_chi",
        "Im_chi",
        "abs_chi",
        "absorption_like_s_inv",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    physical = make_physical_params()
    solver = ParaNormalizer(time_scale_fs=1.0, auto_scale=False).normalize(physical)

    number_density_m3 = 1.0e24
    energy_axis_eV = np.linspace(1.45, 1.65, 600)
    omega_fs_inv = np.asarray(ParaNormalizer.energy_eV_to_fs_inv(energy_axis_eV), dtype=float)
    chi = chi_two_level_linear(
        omega_fs_inv=omega_fs_inv,
        omega_eg_fs_inv=solver.omega_eg_fs_inv,
        mu_ge_D=physical.dipole_matrix_D[0][1],
        gamma2_fs_inv=solver.gamma2_fs_inv,
        number_density_m3=number_density_m3,
        population_difference=1.0,
    )
    # 按本例采用的 exp(-i omega t) 约定和给定分母，resonance 附近 Im[chi] 为正；
    # 因此 absorption-like 量取 omega * Im[chi]，其中 omega 使用 SI 的 s^-1。
    absorption_like_s_inv = omega_fs_inv * 1.0e15 * chi.imag

    rows = [
        {
            "laser_energy_eV": float(energy),
            "omega_fs_inv": float(omega),
            "Re_chi": float(value.real),
            "Im_chi": float(value.imag),
            "abs_chi": float(abs(value)),
            "absorption_like_s_inv": float(absorption),
        }
        for energy, omega, value, absorption in zip(energy_axis_eV, omega_fs_inv, chi, absorption_like_s_inv)
    ]
    write_results_csv(RESULTS_CSV, rows)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 7.2), sharex=True)
    axes[0].plot(energy_axis_eV, chi.real, color="tab:blue")
    axes[0].set_ylabel("Re[chi]")
    axes[1].plot(energy_axis_eV, chi.imag, color="tab:orange")
    axes[1].set_ylabel("Im[chi]")
    axes[2].plot(energy_axis_eV, absorption_like_s_inv, color="tab:green")
    axes[2].set_ylabel("omega * Im[chi] (s^-1)")
    axes[2].set_xlabel("Laser energy (eV)")
    fig.suptitle(
        "Analytic linear susceptibility reference\n"
        f"N={number_density_m3:.2e} m^-3, mu_ge={physical.dipole_matrix_D[0][1]:.3g} D, "
        f"gamma2={solver.gamma2_fs_inv:.3g} fs^-1"
    )
    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=160)
    plt.close(fig)

    print("Analytic two-level chi example")
    print(f"number_density_m3 : {number_density_m3:.6g}")
    print(f"omega_eg_fs_inv   : {solver.omega_eg_fs_inv:.6g}")
    print(f"gamma2_fs_inv     : {solver.gamma2_fs_inv:.6g}")
    print(f"results_csv       : {RESULTS_CSV}")
    print(f"figure            : {FIGURE_PATH}")
    print("TODO: add a lab-frame CW scan comparison against this analytic reference.")


if __name__ == "__main__":
    main()
