#!/usr/bin/env python3
"""Gaussian-pulse RWA example: field-strength dependent dynamics.

The scan is repeated for free, pure-dephasing, and redistribution scenarios.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from sjh_learn.examples.gau_pulse.pulse_common import (
    apply_dissipation,
    case_name_from_pulse,
    dissipation_scenarios,
    make_base_gaussian_physical_params,
    pulse_summary_metrics,
    run_pulse_case,
    save_pulse_group_outputs,
)


EXAMPLE_NAME = "pulse_03_field_strength_dependence"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / EXAMPLE_NAME


def main() -> None:
    base = make_base_gaussian_physical_params()
    field_values = [0.1, 0.2, 0.5, 1.0]
    selected = {name: spec for name, spec in dissipation_scenarios().items() if name in {"free", "dephasing", "redistribution"}}

    for condition_name, scenario in selected.items():
        group_dir = OUTPUT_DIR / condition_name
        results = []
        labels = []
        rows = []
        for field in field_values:
            physical = replace(base, field_MV_per_cm=field)
            physical = apply_dissipation(physical, scenario)
            case_name = case_name_from_pulse(prefix="field_scan", physical=physical)
            result = run_pulse_case(
                physical,
                case_name=case_name,
                condition_name=condition_name,
                example_name=EXAMPLE_NAME,
                output_dir=group_dir,
            )
            results.append(result)
            labels.append(f"{field:g} MV/cm")
            rows.append(
                pulse_summary_metrics(
                    result,
                    case_name=case_name,
                    condition_name=condition_name,
                    example_name=EXAMPLE_NAME,
                )
            )
        save_pulse_group_outputs(
            output_dir=group_dir,
            results=results,
            labels=labels,
            rows=rows,
            title=f"Gaussian field-strength scan: {condition_name}",
        )
        print(f"{condition_name}: {group_dir}")

    print("Gaussian pulse field-strength dependence")
    print(f"output root: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

