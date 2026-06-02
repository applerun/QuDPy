from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from sjh_learn.utils import (
    NLevelPhysicalParams,
    ParaNormalizer,
    PureDephasingChannel,
    RelaxationChannel,
    build_preview_figure,
    make_rotating_view,
    optical_params_from_solver,
    run_rwa_case,
    run_lab_case,
    save_figure,
    save_result_case,
    save_results_components_long,
)


def make_n2_physical_params(
    *,
    energy_gap_eV: float = 1.55,
    laser_energy_eV: float = 1.55,
    dipole_D: float = 3.0,
    field_MV_per_cm: float = 0.1,
    t_start_fs: float = 0.0,
    t_end_fs: float = 1200.0,
    dt_fs: float = 0.5,
    T1_fs: float | None = None,
    Tphi_fs: float | None = None,
    pulse_center_fs: float | None = None,
    pulse_sigma_fs: float | None = None,
) -> NLevelPhysicalParams:
    relaxation = ()
    if T1_fs is not None:
        relaxation = (RelaxationChannel(name="relaxation_1_to_0", from_level=1, to_level=0, T1_fs=T1_fs),)
    dephasing = ()
    if Tphi_fs is not None:
        dephasing = (
            PureDephasingChannel(name="pure_dephasing_level_0", level=0, Tphi_fs=Tphi_fs),
            PureDephasingChannel(name="pure_dephasing_level_1", level=1, Tphi_fs=Tphi_fs),
        )
    return NLevelPhysicalParams(
        basis=("g", "e"),
        energies_eV=(0.0, energy_gap_eV),
        dipole_matrix_D=((0.0, dipole_D), (dipole_D, 0.0)),
        laser_energy_eV=laser_energy_eV,
        field_MV_per_cm=field_MV_per_cm,
        t_start_fs=t_start_fs,
        t_end_fs=t_end_fs,
        dt_fs=dt_fs,
        relaxation_channels=relaxation,
        pure_dephasing_channels=dephasing,
        pulse_center_fs=pulse_center_fs,
        pulse_sigma_fs=pulse_sigma_fs,
    )


def with_dissipation(
    physical: NLevelPhysicalParams,
    *,
    T1_fs: float | None = None,
    Tphi_fs: float | None = None,
) -> NLevelPhysicalParams:
    return replace(
        physical,
        relaxation_channels=()
        if T1_fs is None
        else (RelaxationChannel(name="relaxation_1_to_0", from_level=1, to_level=0, T1_fs=T1_fs),),
        pure_dephasing_channels=()
        if Tphi_fs is None
        else (
            PureDephasingChannel(name="pure_dephasing_level_0", level=0, Tphi_fs=Tphi_fs),
            PureDephasingChannel(name="pure_dephasing_level_1", level=1, Tphi_fs=Tphi_fs),
        ),
    )


def T1_fs_of(physical: NLevelPhysicalParams) -> float | None:
    return physical.relaxation_channels[0].T1_fs if physical.relaxation_channels else None


def Tphi_fs_of(physical: NLevelPhysicalParams) -> float | None:
    return physical.pure_dephasing_channels[0].Tphi_fs if physical.pure_dephasing_channels else None


def make_base_physical_params() -> NLevelPhysicalParams:
    return make_n2_physical_params()


def make_condition_groups() -> dict[str, NLevelPhysicalParams]:
    base = make_base_physical_params()
    return {
        "resonant_strong": replace(base, field_MV_per_cm=0.5, laser_energy_eV=base.energy_gap_eV),
        "resonant_weak": replace(base, field_MV_per_cm=0.1, laser_energy_eV=base.energy_gap_eV),
        "detuned_weak": replace(base, field_MV_per_cm=0.1, energies_eV=(0.0, 1.55), laser_energy_eV=1.57),
    }


def run_rwa_case_from_physical_params(
    physical_params: NLevelPhysicalParams,
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
    physical_params: NLevelPhysicalParams,
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
    save_populations_csv: bool = True,
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
            save_populations_csv=save_populations_csv,
            append_results_csv=False,
        )
    finally:
        if preview_fig is not None:
            plt.close(preview_fig)


def extract_two_level_observables(result, *, mask_threshold: float = 1e-8) -> dict[str, np.ndarray]:
    if result.dimension() != 2:
        raise ValueError("RWA two-level examples require a two-dimensional DynamicsResult.")
    rho_00 = result.matrix_element(0, 0)
    rho_11 = result.matrix_element(1, 1)
    rho_01 = result.matrix_element(0, 1)
    phase_rho_01 = np.angle(rho_01).astype(float)
    phase_rho_01_unwrapped = np.unwrap(np.angle(rho_01)).astype(float)
    mask = np.abs(rho_01) < mask_threshold
    phase_rho_01[mask] = np.nan
    phase_rho_01_unwrapped[mask] = np.nan
    return {
        "rho_00": rho_00,
        "rho_11": rho_11,
        "rho_01": rho_01,
        "abs_rho_01": np.abs(rho_01),
        "phase_rho_01": phase_rho_01,
        "phase_rho_01_unwrapped": phase_rho_01_unwrapped,
    }


def rho_01_phase_series(result, *, unwrap: bool = True, mask_threshold: float = 1e-8) -> np.ndarray:
    observables = extract_two_level_observables(result, mask_threshold=mask_threshold)
    phase = observables["phase_rho_01_unwrapped" if unwrap else "phase_rho_01"]
    return np.asarray(phase, dtype=float)


def _two_level_metrics(result) -> dict[str, float]:
    observables = extract_two_level_observables(result)
    rho_11 = observables["rho_11"]
    abs_rho_01 = observables["abs_rho_01"]
    return {
        "max_rho_11": float(np.max(rho_11.real)),
        "final_rho_11": float(rho_11[-1].real),
        "max_abs_rho_01": float(np.max(abs_rho_01)),
        "final_abs_rho_01": float(abs_rho_01[-1]),
    }


def _plot_two_level_row(ax, row_name: str, observables: dict[str, np.ndarray], times, label, color) -> None:
    if row_name == "rho_11":
        ax.plot(times, observables["rho_11"].real, label=label, color=color)
        ax.set_ylabel(r"$\rho_{11}$")
    elif row_name == "abs_rho_01":
        ax.plot(times, observables["abs_rho_01"], label=label, color=color)
        ax.set_ylabel(r"$|\rho_{01}|$")
    elif row_name == "phase_rho_01":
        ax.plot(times, observables["phase_rho_01_unwrapped"], label=label, color=color)
        ax.set_ylabel(r"phase($\rho_{01}$) (rad)")
    else:
        raise ValueError(f"Unsupported row name: {row_name}")


def collect_summary_metrics(
    result,
    *,
    case_name: str | None = None,
    condition_name: str | None = None,
    example_name: str | None = None,
) -> dict[str, Any]:
    two_level_metrics = _two_level_metrics(result)
    physical = result.physical_params
    solver = result.solver_params
    envelope = "gaussian" if physical.pulse_sigma_fs is not None else "constant"
    drive_class = "GaussianRwaDrivePhysical" if envelope == "gaussian" else "ConstantRwaDrivePhysical"
    drive_expr = (
        f"g(t) = {solver.rabi_fs_inv:.6g} fs^-1"
        if envelope == "constant"
        else (
            f"g(t) = {solver.rabi_fs_inv:.6g} fs^-1 * "
            f"exp[-(t_fs - {solver.pulse_center_fs:.6g})^2 / (2 * {solver.pulse_sigma_fs:.6g}^2)]"
        )
    )
    return {
        "example_name": example_name or "",
        "condition_name": condition_name or "",
        "case_name": case_name or build_case_name_from_T1_Tphi(
            prefix="rwa",
            field_MV_per_cm=physical.field_MV_per_cm,
            T1_fs=T1_fs_of(physical),
            Tphi_fs=Tphi_fs_of(physical),
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
        "drive_class": drive_class,
        "drive_expr": drive_expr,
        "envelope": envelope,
        "T1_fs": T1_fs_of(physical),
        "Tphi_fs": Tphi_fs_of(physical),
        "gamma1_fs_inv": solver.gamma1_fs_inv,
        "gamma_phi_fs_inv": solver.gamma_phi_fs_inv,
        "gamma2_fs_inv": solver.gamma2_fs_inv,
        **two_level_metrics,
    }


def plot_rwa_comparison(
    results,
    labels,
    output_path: str | Path,
    *,
    rows: tuple[str, ...] = ("drive", "rho_11", "abs_rho_01", "phase_rho_01"),
    title: str = "RWA comparison",
    include_phase: bool = True,
    colormap: str = "plasma",
):
    active_rows = rows if include_phase else tuple(row for row in rows if row != "phase_rho_01")
    fig, axes = plt.subplots(len(active_rows), 1, figsize=(7.0, 2.1 * len(active_rows)), sharex=True)
    axes_array = np.atleast_1d(axes)
    cmap = plt.get_cmap(colormap)
    colors = cmap(np.linspace(0.15, 0.9, len(results)))

    for result, label, color in zip(results, labels, colors):
        times, _time_label = result.plot_times_and_label()
        observables = extract_two_level_observables(result)
        for ax, row_name in zip(axes_array, active_rows):
            if row_name == "drive":
                drive_fs_inv = result.drive_fs_inv_values()
                if drive_fs_inv is not None:
                    ax.plot(times, drive_fs_inv, label=label, color=color)
                    ax.set_ylabel("Omega(t) (fs^-1)")
                else:
                    ax.plot(times, result.drive_code_values(), label=label, color=color)
                    ax.set_ylabel("Omega(t) (code unit)")
            else:
                _plot_two_level_row(ax, row_name, observables, times, label, color)

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


def physical_params_with_field(base: NLevelPhysicalParams, field_MV_per_cm: float) -> NLevelPhysicalParams:
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
    base_physical: NLevelPhysicalParams,
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
        physical = with_dissipation(base_physical, T1_fs=spec.get("T1_fs"), Tphi_fs=spec.get("Tphi_fs"))
        result = run_rwa_case_from_physical_params(physical)
        case_name = build_case_name_from_T1_Tphi(
            prefix=case_name_prefix,
            field_MV_per_cm=physical.field_MV_per_cm,
            T1_fs=T1_fs_of(physical),
            Tphi_fs=Tphi_fs_of(physical),
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
            observables = extract_two_level_observables(result)

            if col_index != 1:
                drive_values = result.field_MV_per_cm_values() if result.mode == "lab_exact" else result.drive_fs_inv_values()
                if drive_values is not None:
                    axes[0, col_index].plot(times, drive_values, label=label, color=color)

            axes[1, col_index].plot(times, observables["rho_11"].real, label=label, color=color)
            axes[2, col_index].plot(times, observables["abs_rho_01"], label=label, color=color)
            axes[3, col_index].plot(times, observables["phase_rho_01_unwrapped"], label=label, color=color)

    axes[0, 0].set_ylabel("E(t) (MV/cm)")
    axes[0, 1].set_ylabel("input")
    axes[0, 2].set_ylabel("Omega(t) (fs^-1)")
    for col_index in range(3):
        axes[1, col_index].set_ylabel(r"$\rho_{11}$")
        axes[2, col_index].set_ylabel(r"$|\rho_{01}|$")
        axes[3, col_index].set_ylabel(r"phase($\rho_{01}$) (rad)")
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
    "extract_two_level_observables",
    "make_base_physical_params",
    "make_condition_groups",
    "physical_params_with_field",
    "plot_rwa_comparison",
    "rho_01_phase_series",
    "run_example_group",
    "run_three_mode_cases_from_physical_params",
    "run_rwa_case_from_physical_params",
    "save_case_result",
    "save_group_outputs",
    "save_results_csv",
    "plot_three_mode_field_comparison",
]
