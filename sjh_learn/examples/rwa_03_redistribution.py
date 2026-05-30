#!/usr/bin/env python3
"""RWA-only example: T1 redistribution damps population and coherence.

In this round, redistribution is simplified to excited-to-ground T1 relaxation:
    C_down = sqrt(gamma1) |g><e|
This is a one-way downward channel. Bidirectional or thermal redistribution can
be added later; upward transitions are intentionally not included here.

Expected behavior:
- Shorter T1 makes excited-state population relax to the ground state more easily.
- Rabi oscillations are damped by population relaxation.
- Under continuous RWA drive, the system approaches a steady state set by drive and relaxation.
- Relaxation affects population and also contributes coherence decay through 1 / (2 T1).
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

import matplotlib.pyplot as plt

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sjh_learn.examples.rwa_common import (
    build_case_name_from_T1_Tphi,
    collect_summary_metrics,
    make_base_physical_params,
    plot_rwa_comparison,
    run_rwa_case_from_physical_params,
    save_case_result,
    save_results_csv,
)
from sjh_learn.utils import save_results_components_long


OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "rwa_03_redistribution"


def _t1_label(t1_fs: float | None) -> str:
    return "no_redistribution" if t1_fs is None else f"T1_{t1_fs:g}_fs"


def main() -> None:
    base = make_base_physical_params()
    t1_values = [None, 1000.0, 300.0, 100.0]

    results = []
    labels = []
    rows = []
    for t1_fs in t1_values:
        physical = replace(base, field_MV_per_cm=0.5, T1_fs=t1_fs, T2_fs=None, Tphi_fs=None)
        result = run_rwa_case_from_physical_params(physical)
        case_name = build_case_name_from_T1_Tphi(
            prefix="rwa_redistribution",
            field_MV_per_cm=physical.field_MV_per_cm,
            T1_fs=physical.T1_fs,
            Tphi_fs=physical.Tphi_fs,
        )
        results.append(result)
        labels.append(_t1_label(t1_fs))
        row = collect_summary_metrics(result, case_name=case_name)
        rows.append(
            {
                "case_name": row["case_name"],
                "mode": row["mode"],
                "field_MV_per_cm": row["field_MV_per_cm"],
                "T1_fs": row["T1_fs"],
                "gamma1_fs_inv": row["gamma1_fs_inv"],
                "gamma2_fs_inv": row["gamma2_fs_inv"],
                "max_rho22": row["max_rho22"],
                "final_rho22": row["final_rho22"],
                "max_abs_rho12": row["max_abs_rho12"],
                "final_abs_rho12": row["final_abs_rho12"],
            }
        )
        save_case_result(result, OUTPUT_DIR, preview=True, case_name=case_name)

    fig, _axes = plot_rwa_comparison(results, labels, OUTPUT_DIR / "comparison.png", title="RWA redistribution scan")
    plt.close(fig)
    save_results_components_long(results, OUTPUT_DIR / "comparison_components.csv")
    save_results_csv(rows, OUTPUT_DIR / "results.csv")

    print("RWA redistribution example")
    print(f"output dir: {OUTPUT_DIR}")
    for row in rows:
        print(
            f"{row['case_name']}: gamma1_fs_inv={row['gamma1_fs_inv']:.6g}, "
            f"final_rho22={row['final_rho22']:.6f}, "
            f"final_abs_rho12={row['final_abs_rho12']:.6f}"
        )


if __name__ == "__main__":
    main()
