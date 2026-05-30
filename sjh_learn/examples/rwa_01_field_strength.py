#!/usr/bin/env python3
"""RWA-only example: electric-field strength changes Rabi oscillation speed."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sjh_learn.examples.rwa_common import (
    collect_summary_metrics,
    make_base_physical_params,
    physical_params_with_field,
    plot_rwa_field_strength_summary,
    run_rwa_case_from_physical_params,
    save_case_result,
    write_results_csv,
)
from sjh_learn.utils import save_results_components_long


OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "rwa_01_field_strength"


def main() -> None:
    base = make_base_physical_params()
    field_values = [0.1, 0.2, 0.5, 1.0]

    results = []
    labels = []
    rows = []
    for field in field_values:
        physical = physical_params_with_field(base, field)
        result = run_rwa_case_from_physical_params(physical)
        results.append(result)
        labels.append(f"{field:g} MV/cm")
        rows.append(collect_summary_metrics(result))
        save_case_result(result, OUTPUT_DIR, preview=True)

    fig, _axes = plot_rwa_field_strength_summary(results, labels, OUTPUT_DIR / "comparison.png")
    plt.close(fig)
    save_results_components_long(results, OUTPUT_DIR / "comparison_components.csv")
    write_results_csv(rows, OUTPUT_DIR / "results.csv")

    print("RWA field-strength example")
    print(f"output dir: {OUTPUT_DIR}")
    for row in rows:
        print(
            f"{row['case_name']}: rabi_fs_inv={row['rabi_fs_inv']:.6g}, "
            f"max_rho22={row['max_rho22']:.6f}, final_rho22={row['final_rho22']:.6f}"
        )


if __name__ == "__main__":
    main()
