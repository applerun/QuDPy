#!/usr/bin/env python3
"""RWA-only example: compare dephasing, redistribution, and both.

In this round, redistribution is simplified to excited-to-ground T1 relaxation:
    C_down = sqrt(gamma1) |g><e|
This is a one-way downward channel. Bidirectional or thermal redistribution can
be added later; upward transitions are intentionally not included here.

Expected behavior:
- no dissipation: Rabi oscillation does not decay.
- dephasing only: coherence decays and Rabi oscillation is damped.
- redistribution only: population is pulled toward a ground-state-dominated steady state.
- both: population and coherence are both damped and the system reaches steady state faster.
- With both T1 and Tphi present, coherence decay follows the direction
  1 / T2 = 1 / (2 T1) + 1 / Tphi.
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


OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "rwa_04_dephasing_and_redistribution"


def main() -> None:
    base = make_base_physical_params()
    case_specs = [
        ("no_dissipation", None, None),
        ("dephasing_only", None, 300.0),
        ("redistribution_only", 300.0, None),
        ("both", 300.0, 300.0),
    ]

    results = []
    labels = []
    rows = []
    for label, t1_fs, tphi_fs in case_specs:
        physical = replace(base, field_MV_per_cm=0.5, T1_fs=t1_fs, T2_fs=None, Tphi_fs=tphi_fs)
        result = run_rwa_case_from_physical_params(physical)
        case_name = build_case_name_from_T1_Tphi(
            prefix=f"rwa_{label}",
            field_MV_per_cm=physical.field_MV_per_cm,
            T1_fs=physical.T1_fs,
            Tphi_fs=physical.Tphi_fs,
        )
        results.append(result)
        labels.append(label)
        row = collect_summary_metrics(result, case_name=case_name)
        rows.append(
            {
                "case_name": row["case_name"],
                "mode": row["mode"],
                "field_MV_per_cm": row["field_MV_per_cm"],
                "T1_fs": row["T1_fs"],
                "Tphi_fs": row["Tphi_fs"],
                "gamma1_fs_inv": row["gamma1_fs_inv"],
                "gamma_phi_fs_inv": row["gamma_phi_fs_inv"],
                "gamma2_fs_inv": row["gamma2_fs_inv"],
                "max_rho22": row["max_rho22"],
                "final_rho22": row["final_rho22"],
                "max_abs_rho12": row["max_abs_rho12"],
                "final_abs_rho12": row["final_abs_rho12"],
            }
        )
        save_case_result(result, OUTPUT_DIR, preview=True, case_name=case_name)

    fig, _axes = plot_rwa_comparison(
        results,
        labels,
        OUTPUT_DIR / "comparison.png",
        title="RWA dephasing and redistribution",
    )
    plt.close(fig)
    save_results_components_long(results, OUTPUT_DIR / "comparison_components.csv")
    save_results_csv(rows, OUTPUT_DIR / "results.csv")

    print("RWA dephasing and redistribution example")
    print(f"output dir: {OUTPUT_DIR}")
    for row in rows:
        print(
            f"{row['case_name']}: gamma1_fs_inv={row['gamma1_fs_inv']:.6g}, "
            f"gamma_phi_fs_inv={row['gamma_phi_fs_inv']:.6g}, "
            f"final_rho22={row['final_rho22']:.6f}, "
            f"final_abs_rho12={row['final_abs_rho12']:.6f}"
        )


if __name__ == "__main__":
    main()
