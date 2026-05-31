#!/usr/bin/env python3
"""Gaussian-pulse RWA example: pulse-width dependent dynamics at fixed pulse energy.

This example scans pulse_sigma_fs while rescaling the peak field amplitude so that
the approximate pulse fluence / pulse energy is kept constant.

For a Gaussian field envelope

    f(t) = exp[-(t - t0)^2 / (2 sigma^2)]

the cycle-averaged pulse fluence scales as

    fluence ∝ E0^2 * integral f(t)^2 dt ∝ E0^2 * sigma

when the beam area is fixed. Therefore, to keep pulse energy constant,

    E0(sigma) = E0_ref * sqrt(sigma_ref / sigma)

This differs from pulse_02_width_dependence.py, which keeps peak field fixed.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import math
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from sjh_learn.examples.gau_pulse.pulse_common import (
    apply_dissipation,
    case_name_from_pulse,
    dissipation_scenarios,
    make_base_gaussian_physical_params,
    pulse_summary_metrics,
    run_pulse_case,
    save_pulse_group_outputs,
)


EXAMPLE_NAME = "pulse_02_width_dependence_fixed_energy"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / EXAMPLE_NAME


def rescale_field_for_fixed_energy(
    *,
    reference_field_MV_per_cm: float,
    reference_sigma_fs: float,
    sigma_fs: float,
) -> float:
    """Return E0 for fixed Gaussian-pulse fluence / energy.

    For the envelope f(t)=exp[-(t-t0)^2/(2*sigma^2)], the intensity envelope
    scales as f(t)^2, so the time-integrated intensity is proportional to

        E0^2 * sigma

    Therefore, fixed fluence requires

        E0(sigma) = E0_ref * sqrt(sigma_ref / sigma)

    Parameters
    ----------
    reference_field_MV_per_cm:
        Reference field amplitude E0_ref in MV/cm.

    reference_sigma_fs:
        Reference Gaussian sigma in fs.

    sigma_fs:
        Current Gaussian sigma in fs.

    Returns
    -------
    float
        Rescaled field amplitude E0 in MV/cm.
    """
    if sigma_fs <= 0:
        raise ValueError(f"sigma_fs must be positive, got {sigma_fs!r}")
    if reference_sigma_fs <= 0:
        raise ValueError(f"reference_sigma_fs must be positive, got {reference_sigma_fs!r}")

    return reference_field_MV_per_cm * math.sqrt(reference_sigma_fs / sigma_fs)


def main() -> None:
    base = make_base_gaussian_physical_params()

    sigma_values = [30.0, 60.0, 100.0, 160.0, 240.0]

    # Use the base pulse as the reference pulse energy / fluence.
    reference_sigma_fs = float(base.pulse_sigma_fs)
    reference_field_MV_per_cm = float(base.field_MV_per_cm)

    selected = {
        name: spec
        for name, spec in dissipation_scenarios().items()
        if name in {"free", "dephasing", "redistribution"}
    }

    for condition_name, scenario in selected.items():
        group_dir = OUTPUT_DIR / condition_name
        results = []
        labels = []
        rows = []

        for sigma in sigma_values:
            field_MV_per_cm = rescale_field_for_fixed_energy(
                reference_field_MV_per_cm=reference_field_MV_per_cm,
                reference_sigma_fs=reference_sigma_fs,
                sigma_fs=sigma,
            )

            physical = replace(
                base,
                pulse_sigma_fs=sigma,
                field_MV_per_cm=field_MV_per_cm,
            )
            physical = apply_dissipation(physical, scenario)

            case_name = case_name_from_pulse(
                prefix="width_fixed_energy_scan",
                physical=physical,
            )

            result = run_pulse_case(
                physical,
                case_name=case_name,
                condition_name=condition_name,
                example_name=EXAMPLE_NAME,
                output_dir=group_dir,
            )

            results.append(result)
            labels.append(f"sigma={sigma:g} fs, E0={field_MV_per_cm:.3g} MV/cm")

            row = pulse_summary_metrics(
                result,
                case_name=case_name,
                condition_name=condition_name,
                example_name=EXAMPLE_NAME,
            )

            # Add explicit scan metadata to results.csv.
            row.update(
                {
                    "width_scan_normalization": "fixed_pulse_energy",
                    "reference_sigma_fs": reference_sigma_fs,
                    "reference_field_MV_per_cm": reference_field_MV_per_cm,
                    "pulse_sigma_fs": sigma,
                    "field_MV_per_cm": field_MV_per_cm,
                    "relative_fluence": (field_MV_per_cm**2 * sigma)
                    / (reference_field_MV_per_cm**2 * reference_sigma_fs),
                    "field_scaling_rule": "E0 = E0_ref * sqrt(sigma_ref / sigma)",
                }
            )
            rows.append(row)

        save_pulse_group_outputs(
            output_dir=group_dir,
            results=results,
            labels=labels,
            rows=rows,
            title=f"Gaussian pulse-width scan at fixed pulse energy: {condition_name}",
        )
        print(f"{condition_name}: {group_dir}")

    print("Gaussian pulse-width dependence at fixed pulse energy")
    print(f"reference_sigma_fs: {reference_sigma_fs:g}")
    print(f"reference_field_MV_per_cm: {reference_field_MV_per_cm:g}")
    print(f"output root: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()