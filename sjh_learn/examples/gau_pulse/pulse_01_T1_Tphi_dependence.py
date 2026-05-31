#!/usr/bin/env python3
"""Gaussian-pulse RWA example: Tphi/T1 dependent dynamics at fixed pulse.

The pulse is a slow RWA Gaussian drive envelope. This script keeps pulse center,
width, and field strength fixed, then scans pure dephasing and one-way
excited-to-ground T1 redistribution.
"""

from __future__ import annotations

from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from sjh_learn.examples.gau_pulse.pulse_common import (
    case_name_from_pulse,
    dissipation_scenarios,
    make_base_gaussian_physical_params,
    pulse_summary_metrics,
    run_pulse_case,
    save_pulse_group_outputs,
    apply_dissipation,
)


EXAMPLE_NAME = "pulse_01_T1_Tphi_dependence"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / EXAMPLE_NAME


def _run_group(condition_name: str, physical_cases, labels) -> list[dict]:
    group_dir = OUTPUT_DIR / condition_name
    results = []
    rows = []
    for physical, label in zip(physical_cases, labels):
        case_name = case_name_from_pulse(prefix=condition_name, physical=physical)
        result = run_pulse_case(
            physical,
            case_name=case_name,
            condition_name=condition_name,
            example_name=EXAMPLE_NAME,
            output_dir=group_dir,
        )
        results.append(result)
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
        title=f"Gaussian pulse fixed-parameter scan: {condition_name}",
    )
    return rows


def main() -> None:
    base = make_base_gaussian_physical_params()

    tphi_values = [None, 1000.0, 300.0, 100.0]
    dephasing_cases = [apply_dissipation(base, {"T1_fs": None, "Tphi_fs": tphi}) for tphi in tphi_values]
    _run_group(
        "Tphi_scan",
        dephasing_cases,
        ["no dephasing" if value is None else f"Tphi={value:g} fs" for value in tphi_values],
    )

    t1_values = [None, 1000.0, 300.0, 100.0]
    redistribution_cases = [apply_dissipation(base, {"T1_fs": t1, "Tphi_fs": None}) for t1 in t1_values]
    _run_group(
        "T1_scan",
        redistribution_cases,
        ["no redistribution" if value is None else f"T1={value:g} fs" for value in t1_values],
    )

    scenarios = dissipation_scenarios()
    combined_cases = [apply_dissipation(base, scenario) for scenario in scenarios.values()]
    _run_group("four_scenarios", combined_cases, list(scenarios.keys()))

    print("Gaussian pulse T1/Tphi dependence")
    print(f"output root: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
