#!/usr/bin/env python3
"""RWA-only 示例：比较 dephasing、redistribution 以及二者同时存在。

这里把 redistribution 简化为 excited-to-ground 的单向 relaxation channel：
    C_{0 <- 1} = sqrt(rate) |0><1|
这不是 thermal redistribution；向上跃迁和双向平衡过程以后再单独加入。

预期行为：
- no dissipation：Rabi oscillation 不衰减。
- dephasing only：coherence 衰减，并阻尼 Rabi oscillation。
- redistribution only：population 被拉向 ground-state-dominated steady state。
- both：population 和 coherence 都被阻尼，系统更快到达 steady state。
- T1 和 Tphi 同时存在时，coherence decay 的方向满足
  1 / T2 = 1 / (2 T1) + 1 / Tphi。
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sjh_learn.examples.cw_input.rwa_common import (
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
            base_physical=base,
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
                f"final_rho_11={row['final_rho_11']:.6f}, "
                f"final_abs_rho_01={row['final_abs_rho_01']:.6f}"
            )

    print("RWA dephasing and redistribution example")
    print(f"output root: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
