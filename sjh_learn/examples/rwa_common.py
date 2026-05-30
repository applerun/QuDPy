from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path

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


def collect_summary_metrics(result) -> dict[str, float | str]:
    _rho11, rho22, rho12, _rho21 = result.components()
    physical = result.physical_params
    solver = result.solver_params
    return {
        "case_name": f"rwa_field_{physical.field_MV_per_cm:g}",
        "mode": result.mode,
        "field_MV_per_cm": physical.field_MV_per_cm,
        "rabi_fs_inv": solver.rabi_fs_inv,
        "rabi_code": solver.rabi,
        "max_rho22": float(np.max(rho22.real)),
        "final_rho22": float(rho22[-1].real),
        "max_abs_rho12": float(np.max(np.abs(rho12))),
        "final_abs_rho12": float(abs(rho12[-1])),
    }


def save_case_result(result, output_dir: str | Path, preview: bool = True):
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
            case_name=f"rwa_field_{result.physical_params.field_MV_per_cm:g}",
            append_results_csv=False,
        )
    finally:
        if preview_fig is not None:
            plt.close(preview_fig)


def plot_rwa_field_strength_summary(results, labels, output_path: str | Path):
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    for result, label in zip(results, labels):
        times, _time_label = result.plot_times_and_label()
        _rho11, rho22, rho12, _rho21 = result.components()
        drive_values = result.drive_values()
        axes[0].plot(times, drive_values, label=label)
        axes[1].plot(times, rho22.real, label=label)
        axes[2].plot(times, np.abs(rho12), label=label)

    axes[0].set_ylabel("Omega(t)")
    axes[1].set_ylabel(r"$\rho_{22}$")
    axes[2].set_ylabel(r"$|\rho_{12}|$")
    axes[2].set_xlabel("Time (fs)")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.suptitle("RWA field-strength scan")
    fig.tight_layout()
    save_figure(fig, output_path, dpi=160)
    return fig, axes


def write_results_csv(rows: list[dict], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "case_name",
        "mode",
        "field_MV_per_cm",
        "rabi_fs_inv",
        "rabi_code",
        "max_rho22",
        "final_rho22",
        "max_abs_rho12",
        "final_abs_rho12",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)
    return path


def physical_params_with_field(base: PhysicalParams, field_MV_per_cm: float) -> PhysicalParams:
    return replace(base, field_MV_per_cm=field_MV_per_cm)
