from __future__ import annotations

from pathlib import Path
from typing import Any

from sjh_learn.examples.cw_input.rwa_common import (
    collect_summary_metrics,
    make_n2_physical_params,
    plot_rwa_comparison,
    run_rwa_case_from_physical_params,
    save_case_result,
    save_results_csv,
    T1_fs_of,
    Tphi_fs_of,
    with_dissipation,
)
from sjh_learn.utils import NLevelPhysicalParams, save_results_components_long


def make_base_gaussian_physical_params() -> NLevelPhysicalParams:
    return make_n2_physical_params(
        energy_gap_eV=1.55,
        laser_energy_eV=1.55,
        dipole_D=3.0,
        field_MV_per_cm=0.5,
        t_start_fs=-500.0,
        t_end_fs=1200.0,
        dt_fs=0.5,
        pulse_center_fs=000.0,
        pulse_sigma_fs=200/1.665,
    )


def dissipation_scenarios() -> dict[str, dict[str, float | None]]:
    return {
        "free": {"T1_fs": None, "Tphi_fs": None},
        "dephasing": {"T1_fs": None, "Tphi_fs": 300.0},
        "redistribution": {"T1_fs": 300.0, "Tphi_fs": None},
        "both": {"T1_fs": 300.0, "Tphi_fs": 300.0},
    }


def value_tag(value: float | None) -> str:
    if value is None:
        return "none"
    return f"{value:g}".replace("-", "m").replace(".", "p")


def case_name_from_pulse(
    *,
    prefix: str,
    physical: NLevelPhysicalParams,
    extra: str | None = None,
) -> str:
    parts = [
        prefix,
        f"field_{value_tag(physical.field_MV_per_cm)}",
        f"sigma_{value_tag(physical.pulse_sigma_fs)}",
        f"T1_{value_tag(T1_fs_of(physical))}",
        f"Tphi_{value_tag(Tphi_fs_of(physical))}",
    ]
    if extra:
        parts.append(extra)
    return "_".join(parts)


def pulse_summary_metrics(
    result,
    *,
    case_name: str,
    condition_name: str,
    example_name: str,
) -> dict[str, Any]:
    row = collect_summary_metrics(
        result,
        case_name=case_name,
        condition_name=condition_name,
        example_name=example_name,
    )
    physical = result.physical_params
    row.update(
        {
            "pulse_center_fs": physical.pulse_center_fs,
            "pulse_sigma_fs": physical.pulse_sigma_fs,
            "pulse_fwhm_fs": None if physical.pulse_sigma_fs is None else 2.354820045 * physical.pulse_sigma_fs,
        }
    )
    return row


def run_pulse_case(
    physical: NLevelPhysicalParams,
    *,
    case_name: str,
    condition_name: str,
    example_name: str,
    output_dir: str | Path,
    preview: bool = True,
):
    result = run_rwa_case_from_physical_params(physical)
    save_case_result(
        result,
        output_dir,
        preview=preview,
        case_name=case_name,
        example_name=example_name,
        condition_name=condition_name,
    )
    return result


def save_pulse_group_outputs(
    *,
    output_dir: str | Path,
    results,
    labels,
    rows: list[dict[str, Any]],
    title: str,
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    fig, _axes = plot_rwa_comparison(
        results,
        labels,
        output / "comparison.png",
        title=title,
        colormap="plasma",
    )
    import matplotlib.pyplot as plt

    plt.close(fig)
    save_results_components_long(results, output / "comparison_components.csv")
    save_results_csv(rows, output / "results.csv")


def apply_dissipation(physical: NLevelPhysicalParams, scenario: dict[str, float | None]) -> NLevelPhysicalParams:
    return with_dissipation(physical, T1_fs=scenario.get("T1_fs"), Tphi_fs=scenario.get("Tphi_fs"))
