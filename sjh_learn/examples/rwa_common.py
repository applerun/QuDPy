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
    make_rotating_view,
    optical_params_from_solver,
    run_rwa_case,
    run_lab_case,
    save_figure,
    save_result_case,
    save_results_components_long,
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


def make_condition_groups() -> dict[str, PhysicalParams]:
    base = make_base_physical_params()
    return {
        "resonant_strong": replace(base, field_MV_per_cm=0.5, laser_energy_eV=base.energy_gap_eV),
        "resonant_weak": replace(base, field_MV_per_cm=0.1, laser_energy_eV=base.energy_gap_eV),
        "detuned_weak": replace(base, field_MV_per_cm=0.1, energy_gap_eV=1.55, laser_energy_eV=1.57),
    }


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


def run_three_mode_cases_from_physical_params(
    physical_params: PhysicalParams,
    *,
    normalizer: ParaNormalizer | None = None,
):
    local_normalizer = ParaNormalizer(time_scale_fs=1.0, auto_scale=False) if normalizer is None else normalizer
    solver = local_normalizer.normalize(physical_params)
    parameters = optical_params_from_solver(solver=solver, physical=physical_params, normalizer=local_normalizer)

    lab = run_lab_case(parameters)
    lab.physical_params = physical_params
    lab.solver_params = solver

    rotating = make_rotating_view(lab)

    rwa = run_rwa_case(parameters)
    rwa.physical_params = physical_params
    rwa.solver_params = solver
    return lab, rotating, rwa


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
    example_name: str | None = None,
    condition_name: str | None = None,
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
            example_name=example_name,
            condition_name=condition_name,
            append_results_csv=False,
        )
    finally:
        if preview_fig is not None:
            plt.close(preview_fig)


def rho12_phase_series(result, *, unwrap: bool = True, mask_threshold: float = 1e-8) -> np.ndarray:
    _rho11, _rho22, rho12, _rho21 = result.components()
    phase = np.angle(rho12)
    if unwrap:
        phase = np.unwrap(phase)
    phase = np.asarray(phase, dtype=float)
    phase[np.abs(rho12) < mask_threshold] = np.nan
    return phase


def collect_summary_metrics(
    result,
    *,
    case_name: str | None = None,
    condition_name: str | None = None,
    example_name: str | None = None,
) -> dict[str, Any]:
    _rho11, rho22, rho12, _rho21 = result.components()
    physical = result.physical_params
    solver = result.solver_params
    drive_dict = result.drive_dict or {}
    envelope = "gaussian" if physical.pulse_sigma_fs is not None else "constant"
    return {
        "example_name": example_name or "",
        "condition_name": condition_name or "",
        "case_name": case_name or build_case_name_from_T1_Tphi(
            prefix="rwa",
            field_MV_per_cm=physical.field_MV_per_cm,
            T1_fs=physical.T1_fs,
            Tphi_fs=physical.Tphi_fs,
        ),
        "mode": result.mode,
        "field_MV_per_cm": physical.field_MV_per_cm,
        "peak_E_MV_per_cm": 2.0 * physical.field_MV_per_cm,
        "energy_gap_eV": physical.energy_gap_eV,
        "laser_energy_eV": physical.laser_energy_eV,
        "detuning_eV": physical.energy_gap_eV - physical.laser_energy_eV,
        "detuning_fs_inv": solver.detuning_fs_inv,
        "rabi_fs_inv": solver.rabi_fs_inv,
        "rabi_code": solver.rabi,
        "drive_class": drive_dict.get("class"),
        "drive_expr": result.drive_expr,
        "envelope": envelope,
        "T1_fs": physical.T1_fs,
        "Tphi_fs": physical.Tphi_fs,
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
    rows: tuple[str, ...] = ("drive", "rho22", "abs_rho12", "phase_rho12"),
    title: str = "RWA comparison",
    include_phase: bool = True,
    colormap: str = "plasma",
):
    active_rows = rows if include_phase else tuple(row for row in rows if row != "phase_rho12")
    fig, axes = plt.subplots(len(active_rows), 1, figsize=(7.0, 2.1 * len(active_rows)), sharex=True)
    axes_array = np.atleast_1d(axes)
    cmap = plt.get_cmap(colormap)
    colors = cmap(np.linspace(0.15, 0.9, len(results)))

    for result, label, color in zip(results, labels, colors):
        times, _time_label = result.plot_times_and_label()
        _rho11, rho22, rho12, _rho21 = result.components()
        abs_rho12 = np.abs(rho12)
        phase_rho12 = rho12_phase_series(result, unwrap=True)
        for ax, row_name in zip(axes_array, active_rows):
            if row_name == "drive":
                drive_fs_inv = result.drive_fs_inv_values()
                if drive_fs_inv is not None:
                    ax.plot(times, drive_fs_inv, label=label, color=color)
                    ax.set_ylabel("Omega(t) (fs^-1)")
                else:
                    ax.plot(times, result.drive_code_values(), label=label, color=color)
                    ax.set_ylabel("Omega(t) (code unit)")
            elif row_name == "rho22":
                ax.plot(times, rho22.real, label=label, color=color)
                ax.set_ylabel(r"$\rho_{22}$")
            elif row_name == "abs_rho12":
                ax.plot(times, abs_rho12, label=label, color=color)
                ax.set_ylabel(r"$|\rho_{12}|$")
            elif row_name == "phase_rho12":
                ax.plot(times, phase_rho12, label=label, color=color)
                ax.set_ylabel(r"phase($\rho_{12}$) (rad)")
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


def save_group_outputs(
    output_dir: str | Path,
    results,
    labels,
    rows: list[dict[str, Any]],
    *,
    comparison_title: str,
    colormap: str = "plasma",
) -> None:
    output_root = Path(output_dir)
    fig, _axes = plot_rwa_comparison(
        results,
        labels,
        output_root / "comparison.png",
        title=comparison_title,
        include_phase=True,
        colormap=colormap,
    )
    plt.close(fig)
    save_results_components_long(results, output_root / "comparison_components.csv")
    save_results_csv(rows, output_root / "results.csv")


def run_example_group(
    *,
    output_dir: str | Path,
    base_physical: PhysicalParams,
    case_specs: list[dict[str, Any]],
    case_name_prefix: str,
    comparison_title: str,
    label_builder,
    colormap: str = "plasma",
    preview: bool = True,
    condition_name: str | None = None,
    example_name: str | None = None,
):
    resolved_example_name = example_name or Path(output_dir).parent.name
    results = []
    labels = []
    rows = []
    for spec in case_specs:
        physical = replace(base_physical, T1_fs=spec.get("T1_fs"), T2_fs=None, Tphi_fs=spec.get("Tphi_fs"))
        result = run_rwa_case_from_physical_params(physical)
        case_name = build_case_name_from_T1_Tphi(
            prefix=case_name_prefix,
            field_MV_per_cm=physical.field_MV_per_cm,
            T1_fs=physical.T1_fs,
            Tphi_fs=physical.Tphi_fs,
        )
        results.append(result)
        labels.append(label_builder(spec))
        rows.append(
            collect_summary_metrics(
                result,
                case_name=case_name,
                condition_name=condition_name,
                example_name=resolved_example_name,
            )
        )
        save_case_result(
            result,
            output_dir,
            preview=preview,
            case_name=case_name,
            example_name=resolved_example_name,
            condition_name=condition_name,
        )

    save_group_outputs(
        output_dir,
        results,
        labels,
        rows,
        comparison_title=comparison_title,
        colormap=colormap,
    )
    return rows


def plot_three_mode_field_comparison(
    field_cases,
    output_path: str | Path,
    *,
    title: str = "Field-strength comparison across frames",
    colormap: str = "plasma",
):
    fig, axes = plt.subplots(4, 3, figsize=(12.6, 8.4), sharex="col")
    cmap = plt.get_cmap(colormap)
    colors = cmap(np.linspace(0.15, 0.9, len(field_cases)))
    column_specs = [
        ("Lab frame", 0),
        ("Rotating view", 1),
        ("RWA", 2),
    ]

    for col_title, col_index in column_specs:
        axes[0, col_index].set_title(col_title)
    axes[0, 1].text(0.5, 0.5, "derived from lab drive", ha="center", va="center", transform=axes[0, 1].transAxes)

    for (label, lab, rotating, rwa), color in zip(field_cases, colors):
        for result, col_index in ((lab, 0), (rotating, 1), (rwa, 2)):
            times, _time_label = result.plot_times_and_label()
            _rho11, rho22, rho12, _rho21 = result.components()

            if col_index != 1:
                drive_values = result.field_MV_per_cm_values() if result.mode == "lab_exact" else result.drive_fs_inv_values()
                if drive_values is not None:
                    axes[0, col_index].plot(times, drive_values, label=label, color=color)

            axes[1, col_index].plot(times, rho22.real, label=label, color=color)
            axes[2, col_index].plot(times, np.abs(rho12), label=label, color=color)
            axes[3, col_index].plot(times, rho12_phase_series(result, unwrap=True), label=label, color=color)

    axes[0, 0].set_ylabel("E(t) (MV/cm)")
    axes[0, 1].set_ylabel("input")
    axes[0, 2].set_ylabel("Omega(t) (fs^-1)")
    for col_index in range(3):
        axes[1, col_index].set_ylabel(r"$\rho_{22}$")
        axes[2, col_index].set_ylabel(r"$|\rho_{12}|$")
        axes[3, col_index].set_ylabel(r"phase($\rho_{12}$) (rad)")
        axes[3, col_index].set_xlabel("Time (fs)")

    for row in axes:
        for ax in row:
            ax.grid(True, alpha=0.3)
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend()

    fig.suptitle(title)
    fig.tight_layout()
    save_figure(fig, output_path, dpi=160)
    return fig, axes


__all__ = [
    "build_case_name_from_T1_Tphi",
    "collect_summary_metrics",
    "make_base_physical_params",
    "make_condition_groups",
    "physical_params_with_field",
    "plot_rwa_comparison",
    "rho12_phase_series",
    "run_example_group",
    "run_three_mode_cases_from_physical_params",
    "run_rwa_case_from_physical_params",
    "save_case_result",
    "save_group_outputs",
    "save_results_csv",
    "plot_three_mode_field_comparison",
]
