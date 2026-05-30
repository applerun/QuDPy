#!/usr/bin/env python3
"""RWA-only example: pure dephasing damps two-level coherence.

Expected behavior:
- Shorter Tphi makes coherence decay faster.
- Rabi oscillations are damped by pure dephasing.
- Pure dephasing is not excited-to-ground population relaxation.
- With no T1, there should be no T1-driven population loss.
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


OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "rwa_02_dephasing"


def _tphi_label(tphi_fs: float | None) -> str:
    return "no_dephasing" if tphi_fs is None else f"Tphi_{tphi_fs:g}_fs"


def main() -> None:
    base = make_base_physical_params()
    tphi_values = [None, 1000.0, 300.0, 100.0, 50, 20]
    results = []
    labels = []
    rows = []
    for tphi_fs in tphi_values:
        physical = replace(base, field_MV_per_cm=0.5, T1_fs=None, T2_fs=None, Tphi_fs=tphi_fs)
        result = run_rwa_case_from_physical_params(physical)
        case_name = build_case_name_from_T1_Tphi(
            prefix="rwa_dephasing",
            field_MV_per_cm=physical.field_MV_per_cm,
            T1_fs=physical.T1_fs,
            Tphi_fs=physical.Tphi_fs,
        )
        results.append(result)
        labels.append(_tphi_label(tphi_fs))
        rows.append(collect_summary_metrics(result, case_name=case_name))
        save_case_result(result, OUTPUT_DIR, preview=True, case_name=case_name)

    fig, _axes = plot_rwa_comparison(results, labels, OUTPUT_DIR / "comparison.png", title="RWA pure-dephasing scan")
    plt.close(fig)
    save_results_components_long(results, OUTPUT_DIR / "comparison_components.csv")
    save_results_csv(rows, OUTPUT_DIR / "results.csv")

    print("RWA pure-dephasing example")
    print(f"output dir: {OUTPUT_DIR}")
    for row in rows:
        print(
            f"{row['case_name']}: gamma_phi_fs_inv={row['gamma_phi_fs_inv']:.6g}, "
            f"max_abs_rho12={row['max_abs_rho12']:.6f}, "
            f"final_abs_rho12={row['final_abs_rho12']:.6f}"
        )


if __name__ == "__main__":
    main()
