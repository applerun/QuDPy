#!/usr/bin/env python3
"""RWA-only example: pure dephasing damps two-level coherence.

Expected behavior:
- Shorter Tphi makes coherence decay faster.
- Rabi oscillations are damped by pure dephasing.
- Pure dephasing is not excited-to-ground population relaxation.
- With no T1, there should be no T1-driven population loss.
"""

from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sjh_learn.examples.rwa_common import make_base_physical_params, run_rwa_case_from_physical_params
from sjh_learn.utils import build_preview_figure, save_figure, save_result_case, save_results_components_long


OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "rwa_02_dephasing"


def _tphi_label(tphi_fs: float | None) -> str:
    return "no_dephasing" if tphi_fs is None else f"Tphi_{tphi_fs:g}_fs"


def _case_name(tphi_fs: float | None) -> str:
    return "rwa_dephasing_none" if tphi_fs is None else f"rwa_dephasing_Tphi_{tphi_fs:g}"


def _save_case_result(result, output_dir: Path) -> None:
    preview_fig, _axes = build_preview_figure(result)
    try:
        save_result_case(
            result,
            output_dir,
            output_data=True,
            output_preview=True,
            preview_fig=preview_fig,
            preview_dpi=120,
            case_name=_case_name(result.physical_params.Tphi_fs),
            append_results_csv=False,
        )
    finally:
        plt.close(preview_fig)


def _collect_summary(result) -> dict[str, float | str | None]:
    _rho11, rho22, rho12, _rho21 = result.components()
    physical = result.physical_params
    solver = result.solver_params
    return {
        "case_name": _case_name(physical.Tphi_fs),
        "mode": result.mode,
        "field_MV_per_cm": physical.field_MV_per_cm,
        "Tphi_fs": physical.Tphi_fs,
        "gamma_phi_fs_inv": solver.gamma_phi_fs_inv,
        "gamma2_fs_inv": solver.gamma2_fs_inv,
        "max_rho22": float(np.max(rho22.real)),
        "final_rho22": float(rho22[-1].real),
        "max_abs_rho12": float(np.max(np.abs(rho12))),
        "final_abs_rho12": float(abs(rho12[-1])),
    }


def _write_results_csv(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "case_name",
        "mode",
        "field_MV_per_cm",
        "Tphi_fs",
        "gamma_phi_fs_inv",
        "gamma2_fs_inv",
        "max_rho22",
        "final_rho22",
        "max_abs_rho12",
        "final_abs_rho12",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def _plot_dephasing_summary(results, labels, output_path: Path):
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    for result, label in zip(results, labels):
        times, _time_label = result.plot_times_and_label()
        _rho11, rho22, rho12, _rho21 = result.components()
        axes[0].plot(times, result.drive_values(), label=label)
        axes[1].plot(times, rho22.real, label=label)
        axes[2].plot(times, np.abs(rho12), label=label)

    axes[0].set_ylabel("Omega(t)")
    axes[1].set_ylabel(r"$\rho_{22}$")
    axes[2].set_ylabel(r"$|\rho_{12}|$")
    axes[2].set_xlabel("Time (fs)")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.suptitle("RWA pure-dephasing scan")
    fig.tight_layout()
    save_figure(fig, output_path, dpi=160)
    return fig, axes


def main() -> None:
    base = make_base_physical_params()
    tphi_values = [None, 1000.0, 300.0, 100.0]

    results = []
    labels = []
    rows = []
    for tphi_fs in tphi_values:
        physical = replace(base, field_MV_per_cm=0.5, T1_fs=None, T2_fs=None, Tphi_fs=tphi_fs)
        result = run_rwa_case_from_physical_params(physical)
        results.append(result)
        labels.append(_tphi_label(tphi_fs))
        rows.append(_collect_summary(result))
        _save_case_result(result, OUTPUT_DIR)

    fig, _axes = _plot_dephasing_summary(results, labels, OUTPUT_DIR / "comparison.png")
    plt.close(fig)
    save_results_components_long(results, OUTPUT_DIR / "comparison_components.csv")
    _write_results_csv(rows, OUTPUT_DIR / "results.csv")

    print("RWA pure-dephasing example")
    print(f"output dir: {OUTPUT_DIR}")
    for row in rows:
        print(
            f"{row['case_name']}: gamma_phi_fs_inv={row['gamma_phi_fs_inv']:.6g}, "
            f"max_abs_rho12={row['max_abs_rho12']:.6f}, "
            f"final_abs_rho12={row['final_abs_rho12']:.6f}"
        )


if __name__ == "__main__":
    main()
