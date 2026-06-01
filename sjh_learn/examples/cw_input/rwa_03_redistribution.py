#!/usr/bin/env python3
"""RWA-only 示例：population relaxation 会阻尼 population 和 coherence。

这里把 redistribution 简化为 excited-to-ground 的单向 relaxation channel：
    C_{0 <- 1} = sqrt(rate) |0><1|
这不是 thermal redistribution；向上跃迁和双向平衡过程以后再单独加入。

预期行为：
- T1 越短，excited-state population 越容易回到 ground state。
- population relaxation 会阻尼 Rabi oscillation。
- 连续 RWA drive 下，系统会趋向由 drive 和 relaxation 共同决定的 steady state。
- relaxation 影响 population，也会通过 1 / (2 T1) 贡献 coherence decay。
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


OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "rwa_03_redistribution"


def _t1_label(t1_fs: float | None) -> str:
    return "no_redistribution" if t1_fs is None else f"T1_{t1_fs:g}_fs"


def main() -> None:
    t1_values = [None, 1000.0, 300.0, 100.0, 50, 20]
    case_specs = [{"T1_fs": t1_fs, "Tphi_fs": None} for t1_fs in t1_values]

    for condition_name, base in make_condition_groups().items():
        group_dir = OUTPUT_DIR / condition_name
        rows = run_example_group(
            output_dir=group_dir,
            base_physical=base,
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
                f"final_rho_11={row['final_rho_11']:.6f}, "
                f"final_abs_rho_01={row['final_abs_rho_01']:.6f}"
            )

    print("RWA redistribution example")
    print(f"output root: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
