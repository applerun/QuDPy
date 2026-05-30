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

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sjh_learn.examples.rwa_common import (
    make_condition_groups,
    run_example_group,
)


OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "rwa_02_dephasing"


def _tphi_label(tphi_fs: float | None) -> str:
    return "no_dephasing" if tphi_fs is None else f"Tphi_{tphi_fs:g}_fs"


def main() -> None:
    tphi_values = [None, 1000.0, 300.0, 100.0]
    case_specs = [{"T1_fs": None, "Tphi_fs": tphi_fs} for tphi_fs in tphi_values]

    for condition_name, base in make_condition_groups().items():
        group_dir = OUTPUT_DIR / condition_name
        rows = run_example_group(
            output_dir=group_dir,
            base_physical=replace(base, T1_fs=None, T2_fs=None, Tphi_fs=None),
            case_specs=case_specs,
            case_name_prefix="rwa_dephasing",
            comparison_title=f"RWA pure-dephasing scan: {condition_name}",
            label_builder=lambda spec: _tphi_label(spec["Tphi_fs"]),
            colormap="plasma",
            condition_name=condition_name,
        )

        print(f"condition: {condition_name}")
        print(f"output dir: {group_dir}")
        for row in rows:
            print(
                f"{row['case_name']}: gamma_phi_fs_inv={row['gamma_phi_fs_inv']:.6g}, "
                f"max_abs_rho12={row['max_abs_rho12']:.6f}, "
                f"final_abs_rho12={row['final_abs_rho12']:.6f}"
            )
    print("RWA pure-dephasing example")
    print(f"output root: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
