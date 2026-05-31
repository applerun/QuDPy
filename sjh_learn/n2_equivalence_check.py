#!/usr/bin/env python3
"""比较 two-level exact lab-frame 与等价 N=2 multi-level 路径。"""

from __future__ import annotations

from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sjh_learn.utils import OpticalBlochParameters, two_level_multilevel_equivalence_check


def main() -> None:
    parameters = OpticalBlochParameters(
        t_start=0.0,
        t_final=20.0,
        dt=0.01,
        hbar=1.0,
        energies=(0.0, 1.25),
        dipole_matrix=((0.0, 0.08), (0.08, 0.0)),
        field_amplitude=0.7,
        omega_drive=1.0,
        relaxation_channels=(
            {"name": "relaxation_1_to_0", "from_level": 1, "to_level": 0, "rate_code": 0.03},
        ),
        pure_dephasing_channels=(
            {"name": "pure_dephasing_level_0", "level": 0, "rate_code": 0.02},
            {"name": "pure_dephasing_level_1", "level": 1, "rate_code": 0.02},
        ),
        gamma1=0.03,
        gamma_phi=0.02,
    )
    differences = two_level_multilevel_equivalence_check(parameters)
    print("Two-level vs N=2 multi-level equivalence check")
    for key, value in differences.items():
        print(f"{key:<24}: {value:.6e}")


if __name__ == "__main__":
    main()
