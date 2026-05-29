#!/usr/bin/env python3
"""最小多能级实验室系 demo。"""

from __future__ import annotations

from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sjh_learn.utils import (
    CollapseChannel,
    FieldConfig,
    MultiLevelParameters,
    run_multilevel_case,
    save_multilevel_plot,
)


OUTPUT_PATH = Path(__file__).resolve().parent / "optical_bloch_plots" / "multilevel_demo.png"


def main() -> None:
    """运行三能级实验室系示例。"""
    parameters = MultiLevelParameters(
        energies=(0.0, 1.0, 1.5),
        dipole_matrix=[
            [0.0, 0.05, 0.0],
            [0.05, 0.0, 0.03],
            [0.0, 0.03, 0.0],
        ],
        t_start=0.0,
        t_end=120.0,
        dt=0.05,
        fields=(
            FieldConfig(
                amplitude=0.5,
                omega=1.0,
                phase=0.0,
                envelope="constant",
                name="cw_drive",
            ),
        ),
        collapse_channels=(
            CollapseChannel(name="2_to_1", kind="relaxation", from_level=2, to_level=1, rate=0.02),
            CollapseChannel(name="1_to_0", kind="relaxation", from_level=1, to_level=0, rate=0.01),
        ),
    )

    result = run_multilevel_case(parameters)
    save_multilevel_plot(result, OUTPUT_PATH, populations=None, coherences=[(0, 1), (1, 2)])

    print("多能级实验室系示例")
    print(f"dimension           : {result.dimension()}")
    print(f"final populations   : {[float(value.real) for value in result.populations()[-1]]}")
    print(f"trace error         : {result.max_trace_error():.3e}")
    print(f"Hermiticity error   : {result.max_hermiticity_error():.3e}")
    print(f"output plot         : {OUTPUT_PATH}")
    print(f"sanity checks       : {result.sanity_checks}")


if __name__ == "__main__":
    main()
