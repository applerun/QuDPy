#!/usr/bin/env python3
"""Two-level optical Bloch demo.

The top-level script decides which simulation cases to run and how to compose
the comparison figure. Solver functions each return one trajectory.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
from time import perf_counter

import matplotlib.pyplot as plt

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sjh_learn.utils import (
    ParaNormalizer,
    PhysicalParameterSweep,
    PhysicalParams,
    QuantumResultIO,
    default_output_path,
    make_rotating_view,
    optical_params_from_solver,
    plot_density_components,
    run_lab_case,
    run_rwa_case,
    save_figure,
    save_parameter_summary,
)


BASE_PHYSICAL_PARAMS = PhysicalParams(
    energy_gap_eV=1.55,
    laser_energy_eV=1.55,
    dipole_D=3.0,
    field_MV_per_cm=0.3,
    t_start_fs=0.0,
    t_end_fs=1000.0,
    dt_fs=0.5,
    T1_fs=500.0,
    T2_fs=None,
    Tphi_fs=300.0,
    pulse_center_fs=None,
    pulse_sigma_fs=None,
)

PHYSICAL_SWEEP = PhysicalParameterSweep(
    base_params=BASE_PHYSICAL_PARAMS,
    field_MV_per_cm_values=(0.5, 1, 2),
    laser_energy_eV_values=(1.55, 1.57),
)

NORMALIZER = ParaNormalizer(time_scale_fs=None, auto_scale=True)

OUTPUT_DIR = Path(__file__).resolve().parent / "optical_bloch_plots"
SUMMARY_PATH = OUTPUT_DIR / "parameter_summary.json"
RESULT_IO = QuantumResultIO(str(OUTPUT_DIR / "quantum_results_single"))


def run_one_physical_point(physical_params: PhysicalParams):
    solver = NORMALIZER.normalize(physical_params)
    parameters = optical_params_from_solver(solver=solver, physical=physical_params, normalizer=NORMALIZER)

    lab = run_lab_case(parameters)
    lab.physical_params = physical_params
    lab.solver_params = solver

    rotating = make_rotating_view(lab)

    rwa = run_rwa_case(parameters)
    rwa.physical_params = physical_params
    rwa.solver_params = solver

    return lab, rotating, rwa


def save_comparison_figure(lab, rotating, rwa, output_path: Path) -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(18, 7), sharex="col")
    plot_density_components(lab, axes=axes[:, 0])
    plot_density_components(rotating, axes=axes[:, 1])
    plot_density_components(rwa, axes=axes[:, 2])

    physical = lab.physical_params
    solver = lab.solver_params
    if physical is not None and solver is not None:
        fig.suptitle(
            "Two-Level Optical Bloch Comparison\n"
            f"Eg={physical.energy_gap_eV:.4f} eV, EL={physical.laser_energy_eV:.4f} eV, "
            f"Field={physical.field_MV_per_cm:.4f} MV/cm, Dipole={physical.dipole_D:.4f} D\n"
            f"T1={physical.T1_fs}, Tphi={physical.Tphi_fs}, "
            f"Delta={solver.detuning_fs_inv:.6g} fs^-1, Rabi={solver.rabi_fs_inv:.6g} fs^-1"
        )
    else:
        fig.suptitle("Two-Level Optical Bloch Comparison")

    fig.tight_layout()
    save_figure(fig, output_path, dpi=160)
    plt.close(fig)
    return output_path


def main() -> None:
    print("Two-level optical Bloch demo with explicit single-mode cases")
    print(f"energy_gap_eV     : {BASE_PHYSICAL_PARAMS.energy_gap_eV}")
    print(f"laser_energy_eV   : {BASE_PHYSICAL_PARAMS.laser_energy_eV}")
    print(f"dipole_D          : {BASE_PHYSICAL_PARAMS.dipole_D}")
    print(f"field_MV_per_cm   : {list(PHYSICAL_SWEEP.field_MV_per_cm_values)}")
    print(f"laser scan eV     : {list(PHYSICAL_SWEEP.laser_energy_eV_values)}")
    print(f"time range fs     : {BASE_PHYSICAL_PARAMS.t_start_fs} -> {BASE_PHYSICAL_PARAMS.t_end_fs}")
    print(f"dt_fs             : {BASE_PHYSICAL_PARAMS.dt_fs}")
    print(f"T1_fs             : {BASE_PHYSICAL_PARAMS.T1_fs}")
    print(f"Tphi_fs           : {BASE_PHYSICAL_PARAMS.Tphi_fs}")

    field_values = PHYSICAL_SWEEP.field_MV_per_cm_values or (PHYSICAL_SWEEP.base_params.field_MV_per_cm,)
    laser_values = PHYSICAL_SWEEP.laser_energy_eV_values or (PHYSICAL_SWEEP.base_params.laser_energy_eV,)

    print("\nSolve timing:")
    saved_paths: list[Path] = []
    all_results = []
    case_index = 0
    total_cases = len(field_values) * len(laser_values)
    sweep_start = perf_counter()

    for laser_energy_eV in laser_values:
        for field_MV_per_cm in field_values:
            case_index += 1
            physical_params = replace(
                PHYSICAL_SWEEP.base_params,
                laser_energy_eV=laser_energy_eV,
                field_MV_per_cm=field_MV_per_cm,
            )

            case_start = perf_counter()
            lab, rotating, rwa = run_one_physical_point(physical_params)
            solve_elapsed_s = perf_counter() - case_start

            output_path = default_output_path(OUTPUT_DIR, lab)
            save_comparison_figure(lab, rotating, rwa, output_path)
            saved_paths.append(output_path)

            for result in (lab, rotating, rwa):
                RESULT_IO.save_case(
                    result,
                    output_data=True,
                    output_preview=False,
                    save_npz=True,
                    save_csv=True,
                    save_json=True,
                )
                all_results.append(result)

            print(
                f"case {case_index}/{total_cases} "
                f"field={field_MV_per_cm:g} MV/cm, "
                f"laser={laser_energy_eV:g} eV -> "
                f"{solve_elapsed_s:.3f} s"
            )
            print(f"output figure      : {output_path}")
            print(f"lab final summary  : {lab.summary_dict()}")
            print(f"rwa final summary  : {rwa.summary_dict()}")

    total_elapsed_s = perf_counter() - sweep_start
    print(f"\ntotal solve time  : {total_elapsed_s:.3f} s")

    print("\nGenerated comparison figures:")
    for output_path in saved_paths:
        print(output_path)

    summary_path = save_parameter_summary(all_results, SUMMARY_PATH)
    print(f"\nparameter summary  : {summary_path}")
    print(f"result data root   : {RESULT_IO.outdir}")


if __name__ == "__main__":
    main()
