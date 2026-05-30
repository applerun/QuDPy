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

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sjh_learn.examples.rwa_common import (
    make_condition_groups,
    run_example_group,
)


OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "rwa_03_redistribution"


def _t1_label(t1_fs: float | None) -> str:
    return "no_redistribution" if t1_fs is None else f"T1_{t1_fs:g}_fs"


def main() -> None:
    t1_values = [None, 1000.0, 300.0, 100.0]
    case_specs = [{"T1_fs": t1_fs, "Tphi_fs": None} for t1_fs in t1_values]

    for condition_name, base in make_condition_groups().items():
        group_dir = OUTPUT_DIR / condition_name
        rows = run_example_group(
            output_dir=group_dir,
            base_physical=replace(base, T1_fs=None, T2_fs=None, Tphi_fs=None),
            case_specs=case_specs,
            case_name_prefix="rwa_redistribution",
            comparison_title=f"RWA redistribution scan: {condition_name}",
            label_builder=lambda spec: _t1_label(spec["T1_fs"]),
            colormap="plasma",
            condition_name=condition_name,
        )
        print(f"condition: {condition_name}")
        print(f"output dir: {group_dir}")
        for row in rows:
            print(
                f"{row['case_name']}: gamma1_fs_inv={row['gamma1_fs_inv']:.6g}, "
                f"final_rho22={row['final_rho22']:.6f}, "
                f"final_abs_rho12={row['final_abs_rho12']:.6f}"
            )

    print("RWA redistribution example")
    print(f"output root: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
