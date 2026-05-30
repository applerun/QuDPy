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

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sjh_learn.examples.rwa_common import (
    make_condition_groups,
    run_example_group,
)


OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "rwa_04_dephasing_and_redistribution"


def main() -> None:
    case_specs = [
        {"label": "no_dissipation", "T1_fs": None, "Tphi_fs": None},
        {"label": "dephasing_only", "T1_fs": None, "Tphi_fs": 300.0},
        {"label": "redistribution_only", "T1_fs": 300.0, "Tphi_fs": None},
        {"label": "both", "T1_fs": 300.0, "Tphi_fs": 300.0},
    ]

    for condition_name, base in make_condition_groups().items():
        group_dir = OUTPUT_DIR / condition_name
        rows = run_example_group(
            output_dir=group_dir,
            base_physical=replace(base, T1_fs=None, T2_fs=None, Tphi_fs=None),
            case_specs=case_specs,
            case_name_prefix="rwa_combined",
            comparison_title=f"RWA dephasing and redistribution: {condition_name}",
            label_builder=lambda spec: spec["label"],
            colormap="plasma",
            condition_name=condition_name,
        )
        print(f"condition: {condition_name}")
        print(f"output dir: {group_dir}")
        for row in rows:
            print(
                f"{row['case_name']}: gamma1_fs_inv={row['gamma1_fs_inv']:.6g}, "
                f"gamma_phi_fs_inv={row['gamma_phi_fs_inv']:.6g}, "
                f"final_rho22={row['final_rho22']:.6f}, "
                f"final_abs_rho12={row['final_abs_rho12']:.6f}"
            )

    print("RWA dephasing and redistribution example")
    print(f"output root: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
