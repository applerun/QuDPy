from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from sjh_learn.utils import (
    ParaNormalizer,
    PhysicalParams,
    build_preview_figure,
    optical_params_from_solver,
    run_rwa_case,
    save_figure,
    save_result_case,
)


def make_base_physical_params() -> PhysicalParams:
    return PhysicalParams(
        energy_gap_eV=1.55,
        laser_energy_eV=1.55,
        dipole_D=3.0,
        field_MV_per_cm=0.1,
        t_start_fs=0.0,
        t_end_fs=1200.0,
        dt_fs=0.5,
        T1_fs=None,
        T2_fs=None,
        Tphi_fs=None,
        pulse_center_fs=None,
        pulse_sigma_fs=None,
    )


def run_rwa_case_from_physical_params(
    physical_params: PhysicalParams,
    *,
    normalizer: ParaNormalizer | None = None,
):
    local_normalizer = ParaNormalizer(time_scale_fs=1.0, auto_scale=False) if normalizer is None else normalizer
    solver = local_normalizer.normalize(physical_params)
    parameters = optical_params_from_solver(solver=solver, physical=physical_params, normalizer=local_normalizer)
    result = run_rwa_case(parameters)
    result.physical_params = physical_params
    result.solver_params = solver
    return result


def build_case_name_from_T1_Tphi(
    *,
    prefix: str,
    field_MV_per_cm: float,
    T1_fs: float | None = None,
    Tphi_fs: float | None = None,
) -> str:
    parts = [prefix, f"field_{field_MV_per_cm:g}"]
    parts.append("T1_none" if T1_fs is None else f"T1_{T1_fs:g}")
    parts.append("Tphi_none" if Tphi_fs is None else f"Tphi_{Tphi_fs:g}")
    return "_".join(parts)


def save_case_result(
    result,
    output_dir: str | Path,
    *,
    preview: bool = True,
    case_name: str | None = None,
):
    preview_fig = None
    if preview:
        preview_fig, _axes = build_preview_figure(result)
    try:
        return save_result_case(
            result,
            output_dir,
            output_data=True,
            output_preview=preview,
            preview_fig=preview_fig,
            preview_dpi=120,
            case_name=case_name,
            append_results_csv=False,
        )
    finally:
        if preview_fig is not None:
            plt.close(preview_fig)


def collect_summary_metrics(result, *, case_name: str | None = None) -> dict[str, Any]:
    _rho11, rho22, rho12, _rho21 = result.components()
    physical = result.physical_params
    solver = result.solver_params
    return {
        "case_name": case_name or build_case_name_from_T1_Tphi(
            prefix="rwa",
            field_MV_per_cm=physical.field_MV_per_cm,
            T1_fs=physical.T1_fs,
            Tphi_fs=physical.Tphi_fs,
        ),
        "mode": result.mode,
        "field_MV_per_cm": physical.field_MV_per_cm,
        "T1_fs": physical.T1_fs,
        "Tphi_fs": physical.Tphi_fs,
        "rabi_fs_inv": solver.rabi_fs_inv,
        "rabi_code": solver.rabi,
        "gamma1_fs_inv": solver.gamma1_fs_inv,
        "gamma_phi_fs_inv": solver.gamma_phi_fs_inv,
        "gamma2_fs_inv": solver.gamma2_fs_inv,
        "max_rho22": float(np.max(rho22.real)),
        "final_rho22": float(rho22[-1].real),
        "max_abs_rho12": float(np.max(np.abs(rho12))),
        "final_abs_rho12": float(abs(rho12[-1])),
    }


def plot_rwa_comparison(
    results,
    labels,
    output_path: str | Path,
    *,
    rows: tuple[str, ...] = ("drive", "rho22", "abs_rho12"),
    title: str = "RWA comparison",
):
    fig, axes = plt.subplots(len(rows), 1, figsize=(10, 3 * len(rows)), sharex=True)
    axes_array = np.atleast_1d(axes)
    for result, label in zip(results, labels):
        times, _time_label = result.plot_times_and_label()
        _rho11, rho22, rho12, _rho21 = result.components()
        for ax, row_name in zip(axes_array, rows):
            if row_name == "drive":
                drive_fs_inv = result.drive_fs_inv_values()
                if drive_fs_inv is not None:
                    ax.plot(times, drive_fs_inv, label=label)
                    ax.set_ylabel("Omega(t) (fs^-1)")
                else:
                    ax.plot(times, result.drive_code_values(), label=label)
                    ax.set_ylabel("Omega(t) (code unit)")
            elif row_name == "rho22":
                ax.plot(times, rho22.real, label=label)
                ax.set_ylabel(r"$\rho_{22}$")
            elif row_name == "abs_rho12":
                ax.plot(times, np.abs(rho12), label=label)
                ax.set_ylabel(r"$|\rho_{12}|$")
            else:
                raise ValueError(f"Unsupported row name: {row_name}")

    axes_array[-1].set_xlabel("Time (fs)")
    for ax in axes_array:
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.suptitle(title)
    fig.tight_layout()
    save_figure(fig, output_path, dpi=160)
    return fig, axes_array


def save_results_csv(rows: list[dict], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("rows must not be empty.")
    header = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)
    return path


def physical_params_with_field(base: PhysicalParams, field_MV_per_cm: float) -> PhysicalParams:
    return replace(base, field_MV_per_cm=field_MV_per_cm)


__all__ = [
    "build_case_name_from_T1_Tphi",
    "collect_summary_metrics",
    "make_base_physical_params",
    "physical_params_with_field",
    "plot_rwa_comparison",
    "run_rwa_case_from_physical_params",
    "save_case_result",
    "save_results_csv",
]
