#!/usr/bin/env python3
"""Gaussian-pulse lab-frame exact example: width scan at fixed pulse energy."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import math
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from sjh_learn.examples.gau_pulse_lab.pulse_lab_common import (
    apply_dissipation,
    case_name_from_pulse,
    dissipation_scenarios,
    lab_pulse_summary_metrics,
    make_base_gaussian_physical_params,
    run_lab_pulse_case,
    save_lab_group_outputs,
)


EXAMPLE_NAME = "pulse_lab_02_width_dependence_fixed_energy"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / EXAMPLE_NAME


def rescale_field_for_fixed_energy(*, reference_field_MV_per_cm: float, reference_sigma_fs: float, sigma_fs: float) -> float:
    """固定 Gaussian pulse fluence 时的 E0 缩放：E0 = E0_ref * sqrt(sigma_ref / sigma)。"""

    if sigma_fs <= 0:
        raise ValueError(f"sigma_fs must be positive, got {sigma_fs!r}")
    if reference_sigma_fs <= 0:
        raise ValueError(f"reference_sigma_fs must be positive, got {reference_sigma_fs!r}")
    return reference_field_MV_per_cm * math.sqrt(reference_sigma_fs / sigma_fs)


def main() -> None:
    base = make_base_gaussian_physical_params()
    sigma_values = [30.0, 60.0, 100.0, 160.0, 240.0]
    reference_sigma_fs = float(base.pulse_sigma_fs)
    reference_field_MV_per_cm = float(base.field_MV_per_cm)
    selected = {name: spec for name, spec in dissipation_scenarios().items() if name in {"free", "dephasing", "redistribution"}}

    for condition_name, scenario in selected.items():
        group_dir = OUTPUT_DIR / condition_name
        rows = []
        for sigma in sigma_values:
            field_MV_per_cm = rescale_field_for_fixed_energy(
                reference_field_MV_per_cm=reference_field_MV_per_cm,
                reference_sigma_fs=reference_sigma_fs,
                sigma_fs=sigma,
            )
            physical = replace(base, pulse_sigma_fs=sigma, field_MV_per_cm=field_MV_per_cm)
            physical = apply_dissipation(physical, scenario)
            case_name = case_name_from_pulse(prefix="width_fixed_energy_scan", physical=physical)
            result = run_lab_pulse_case(
                physical,
                case_name=case_name,
                condition_name=condition_name,
                example_name=EXAMPLE_NAME,
                output_dir=group_dir,
            )
            row = lab_pulse_summary_metrics(
                result,
                case_name=case_name,
                condition_name=condition_name,
                example_name=EXAMPLE_NAME,
            )
            row.update(
                {
                    "width_scan_normalization": "fixed_pulse_energy",
                    "reference_sigma_fs": reference_sigma_fs,
                    "reference_field_MV_per_cm": reference_field_MV_per_cm,
                    "relative_fluence": (field_MV_per_cm**2 * sigma) / (reference_field_MV_per_cm**2 * reference_sigma_fs),
                    "field_scaling_rule": "E0 = E0_ref * sqrt(sigma_ref / sigma)",
                }
            )
            rows.append(row)
        save_lab_group_outputs(output_dir=group_dir, rows=rows)
        print(f"{condition_name}: {group_dir}")

    print("Gaussian pulse lab-frame exact width dependence at fixed pulse energy")
    print(f"reference_sigma_fs: {reference_sigma_fs:g}")
    print(f"reference_field_MV_per_cm: {reference_field_MV_per_cm:g}")
    print(f"output root: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
