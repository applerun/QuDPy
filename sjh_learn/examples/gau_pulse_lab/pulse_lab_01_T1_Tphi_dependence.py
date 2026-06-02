#!/usr/bin/env python3
"""Gaussian-pulse lab-frame exact example: Tphi/T1 dependent dynamics.

lab-frame exact pulse simulation 保留 optical carrier 的快速振荡。每个 case
会额外保存 FFT-based response-like spectrum；`rho_01_fft / E_fft` 不是
严格 macroscopic susceptibility chi。
"""

from __future__ import annotations

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


EXAMPLE_NAME = "pulse_lab_01_T1_Tphi_dependence"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / EXAMPLE_NAME


def _run_group(condition_name: str, physical_cases) -> list[dict]:
    group_dir = OUTPUT_DIR / condition_name
    rows = []
    for physical in physical_cases:
        case_name = case_name_from_pulse(prefix=condition_name, physical=physical)
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
    return rows


def main() -> None:
    base = make_base_gaussian_physical_params()

    tphi_values = [None, 1000.0, 300.0, 100.0]
    dephasing_cases = [apply_dissipation(base, {"T1_fs": None, "Tphi_fs": tphi}) for tphi in tphi_values]
    _run_group("Tphi_scan", dephasing_cases)

    t1_values = [None, 1000.0, 300.0, 100.0]
    redistribution_cases = [apply_dissipation(base, {"T1_fs": t1, "Tphi_fs": None}) for t1 in t1_values]
    _run_group("T1_scan", redistribution_cases)

    scenarios = dissipation_scenarios()
    combined_cases = [apply_dissipation(base, scenario) for scenario in scenarios.values()]
    _run_group("four_scenarios", combined_cases)

    print("Gaussian pulse lab-frame exact T1/Tphi dependence")
    print(f"output root: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
