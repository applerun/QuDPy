#!/usr/bin/env python3
"""Gaussian-pulse lab-frame exact example: pulse-width dependent dynamics."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
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


EXAMPLE_NAME = "pulse_lab_02_width_dependence"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / EXAMPLE_NAME


def main() -> None:
    base = make_base_gaussian_physical_params()
    sigma_values = [30.0, 60.0, 100.0, 160.0, 240.0]
    selected = {name: spec for name, spec in dissipation_scenarios().items() if name in {"free", "dephasing", "redistribution"}}

    for condition_name, scenario in selected.items():
        group_dir = OUTPUT_DIR / condition_name
        rows = []
        for sigma in sigma_values:
            physical = replace(base, pulse_sigma_fs=sigma)
            physical = apply_dissipation(physical, scenario)
            case_name = case_name_from_pulse(prefix="width_scan", physical=physical)
            result = run_lab_pulse_case(
                physical,
                case_name=case_name,
                condition_name=condition_name,
                example_name=EXAMPLE_NAME,
                output_dir=group_dir,
            )
            rows.append(
                lab_pulse_summary_metrics(
                    result,
                    case_name=case_name,
                    condition_name=condition_name,
                    example_name=EXAMPLE_NAME,
                )
            )
        save_lab_group_outputs(output_dir=group_dir, rows=rows)
        print(f"{condition_name}: {group_dir}")

    print("Gaussian pulse lab-frame exact width dependence")
    print(f"output root: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
