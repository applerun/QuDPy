from __future__ import annotations

from pathlib import Path
import json
import sys

import matplotlib.pyplot as plt
import numpy as np


EXAMPLE_NAME = "compare_two_vs_three_level_absorption_lab_exact_fast_2to1_dissi"

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from sjh_learn.utils.core import (
    NLevelPhysicalParams,
    RelaxationChannel,
    PureDephasingChannel,
    run_case,
)
from sjh_learn.utils.analysis import DynamicsAnalysis
from sjh_learn.utils.fields import make_default_gaussian_carrier_field
from sjh_learn.utils.io import save_result_case
from sjh_learn.utils.spectroscopy.spectra import lab_frame_fft_response
from sjh_learn.utils.plotting import (
    plot_normalized_curve,
    set_axis_ylim_from_curves,
    set_energy_axis,
    add_top_omega_axis,
)


FORCE_RERUN = False


def make_analysis_from_result(result):
    if hasattr(DynamicsAnalysis, "from_result"):
        return DynamicsAnalysis.from_result(result)
    return DynamicsAnalysis.from_dynamics_res(result)


def get_coherence(analysis: DynamicsAnalysis, pair=(0, 1)) -> np.ndarray:
    if hasattr(analysis, "coherence"):
        return analysis.coherence(pair=pair)
    return analysis.rho12(pair=pair)


def result_ckp_path(output_dir: Path, case_name: str) -> Path:
    """Use an explicit checkpoint cache independent of save_result_case layout."""

    return output_dir / "checkpoints" / f"{case_name}.ckp"


def reset_results_csv(output_dir: Path) -> None:
    """Remove stale root-level results.csv before a multi-case debug run.

    This makes the first save_result_case call behave like an overwrite, while
    later calls can append cleanly in the same run.
    """
    results_csv = output_dir / "results.csv"
    if results_csv.exists():
        results_csv.unlink()
        print(f"Removed stale results.csv: {results_csv}")


def json_safe(value):
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return json_safe(value.to_dict())
    if hasattr(value, "__dataclass_fields__"):
        return json_safe(
            {
                name: getattr(value, name)
                for name in value.__dataclass_fields__
            }
        )
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_safe(payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def stringify_written_paths(written) -> dict[str, str]:
    if isinstance(written, dict):
        return {str(key): str(value) for key, value in written.items()}
    return {"output": str(written)}


def make_two_level_params() -> NLevelPhysicalParams:
    return NLevelPhysicalParams(
        energies_eV=(0.0, 1.50),
        dipole_matrix_D=(
            (0.0, 3.0),
            (3.0, 0.0),
        ),
        t_start_fs=-1000.0,
        t_end_fs=1000.0,
        dt_fs=0.05,
        field=make_default_gaussian_carrier_field(
            E0_MV_per_cm=0.1,
            laser_energy_eV=1.625,
            pulse_center_fs=0.0,
            pulse_sigma_fs=5.0,
        ),
        basis=("0", "1"),
        relaxation_channels=(
            RelaxationChannel(
                name="relaxation_1_to_0",
                from_level=1,
                to_level=0,
                T1_fs=800.0,
            ),
        ),
        pure_dephasing_channels=(
            PureDephasingChannel(
                name="pure_dephasing_level_1",
                level=1,
                Tphi_fs=250.0,
            ),
        ),
        solver_mode="lab_exact",
        input_description="Two-level lab-exact absorption validation case.",
        input_metadata={
            "example": EXAMPLE_NAME,
            "system": "two_level",
            "transitions_eV": {"0_to_1": 1.50},
            "dipoles_D": {"mu_01": 3.0},
        },
    )


def make_three_level_params() -> NLevelPhysicalParams:
    return NLevelPhysicalParams(
        energies_eV=(0.0, 1.50, 1.75),
        dipole_matrix_D=(
            (0.0, 3.0, 2.0),
            (3.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
        ),
        t_start_fs=-1000.0,
        t_end_fs=1000.0,
        dt_fs=0.05,
        field=make_default_gaussian_carrier_field(
            E0_MV_per_cm=0.1,
            laser_energy_eV=1.625,
            pulse_center_fs=0.0,
            pulse_sigma_fs=5.0,
        ),
        basis=("0", "1", "2"),
        relaxation_channels=(
            RelaxationChannel(
                name="relaxation_1_to_0",
                from_level=1,
                to_level=0,
                T1_fs=800.0,
            ),
            RelaxationChannel(
                name="relaxation_2_to_0",
                from_level=2,
                to_level=0,
                T1_fs=800.0,
            ),
        ),
        pure_dephasing_channels=(
            PureDephasingChannel(
                name="pure_dephasing_level_1",
                level=1,
                Tphi_fs=250.0,
            ),
            PureDephasingChannel(
                name="pure_dephasing_level_2",
                level=2,
                Tphi_fs=250.0,
            ),
        ),
        solver_mode="lab_exact",
        input_description="Three-level lab-exact absorption validation case.",
        input_metadata={
            "example": EXAMPLE_NAME,
            "system": "three_level",
            "transitions_eV": {
                "0_to_1": 1.50,
                "0_to_2": 1.75,
            },
            "dipoles_D": {
                "mu_01": 3.0,
                "mu_02": 2.0,
                "mu_12": 0.0,
            },
        },
    )


def run_case_and_extract(
    *,
    params: NLevelPhysicalParams,
    case_name: str,
    output_dir: Path,
    number_density_m3: float,
    coherence_pairs: tuple[tuple[int, int], ...],
    append_results_csv: bool,
) -> dict[str, object]:
    print(f"Running {case_name}")

    ckp_path = result_ckp_path(output_dir, case_name)
    result = run_case(
        params,
        load_ckp=ckp_path,
        save_ckp=ckp_path,
        force_run=FORCE_RERUN,
    )

    analysis = make_analysis_from_result(result)

    t_fs = analysis.time_fs()
    E_t, _, _ = analysis.input_signal(kind="field")
    P_t = analysis.full_polarization_C_per_m2(
        number_density_m3=number_density_m3,
    )

    coherences = {}
    responses = {}

    for pair in coherence_pairs:
        rhoij_t = get_coherence(analysis, pair=pair)
        coherences[pair] = rhoij_t

        responses[pair] = lab_frame_fft_response(
            t_fs=t_fs,
            E_MV_per_cm=E_t,
            P_C_per_m2=P_t,
            rho12=rhoij_t,
            window="hann",
            subtract_mean=True,
            rel_threshold=1e-5,
            zero_padding_factor=4,
        )

    written = save_result_case(
        result,
        output_dir,
        output_data=True,
        output_preview=True,
        case_name=case_name,
        example_name=EXAMPLE_NAME,
        condition_name="lab_exact_absorption_validation",
        append_results_csv=append_results_csv,
        save_populations_csv=True,
    )

    return {
        "params": params,
        "result": result,
        "checkpoint_path": ckp_path,
        "case_outputs": stringify_written_paths(written),
        "time_fs": t_fs,
        "E_t": E_t,
        "P_t": P_t,
        "coherences": coherences,
        "responses": responses,
    }


def save_debug_csv(
    *,
    output_dir: Path,
    two_record: dict[str, object],
    three_record: dict[str, object],
) -> None:
    csv_dir = output_dir / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)

    t_two = two_record["time_fs"]
    t_three = three_record["time_fs"]

    if np.allclose(t_two, t_three, rtol=1e-8, atol=1e-10):
        t_fs = t_two
        two_rho01 = two_record["coherences"][(0, 1)]
        three_rho01 = three_record["coherences"][(0, 1)]
        three_rho02 = three_record["coherences"][(0, 2)]

        time_table = np.column_stack(
            [
                t_fs,
                two_record["E_t"],
                np.real(two_record["P_t"]),
                np.real(three_record["P_t"]),
                np.real(two_rho01),
                np.imag(two_rho01),
                np.real(three_rho01),
                np.imag(three_rho01),
                np.real(three_rho02),
                np.imag(three_rho02),
            ]
        )

        np.savetxt(
            csv_dir / "two_vs_three_time_domain.csv",
            time_table,
            delimiter=",",
            header=(
                "time_fs,E_MV_per_cm,"
                "Re_P_two_C_per_m2,Re_P_three_C_per_m2,"
                "Re_rho01_two,Im_rho01_two,"
                "Re_rho01_three,Im_rho01_three,"
                "Re_rho02_three,Im_rho02_three"
            ),
            comments="",
        )
    else:
        print("Warning: two-level and three-level time axes differ; skip combined time CSV.")

    two_resp = two_record["responses"][(0, 1)]
    three_resp_01 = three_record["responses"][(0, 1)]
    three_resp_02 = three_record["responses"][(0, 2)]

    spec_table = np.column_stack(
        [
            two_resp["energy_eV"],
            two_resp["abs_E_fft"],
            two_resp["neg_omega_im_P_over_E"],
            three_resp_01["neg_omega_im_P_over_E"],
            two_resp["im_rho12_over_E"],
            three_resp_01["im_rho12_over_E"],
            three_resp_02["im_rho12_over_E"],
        ]
    )

    np.savetxt(
        csv_dir / "two_vs_three_spectrum.csv",
        spec_table,
        delimiter=",",
        header=(
            "energy_eV,abs_E_fft,"
            "two_neg_omega_im_P_over_E,"
            "three_neg_omega_im_P_over_E,"
            "two_im_rho01_over_E,"
            "three_im_rho01_over_E,"
            "three_im_rho02_over_E"
        ),
        comments="",
    )


def build_compare_figure(
    *,
    output_dir: Path,
    two_record: dict[str, object],
    three_record: dict[str, object],
    e_min: float = 1.35,
    e_max: float = 1.90,
) -> Path:
    style = {
        "font.size": 15,
        "axes.titlesize": 17,
        "axes.labelsize": 15,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 10,
        "figure.titlesize": 20,
    }

    t_fs = two_record["time_fs"]
    E_t = two_record["E_t"]

    two_resp = two_record["responses"][(0, 1)]
    three_resp_01 = three_record["responses"][(0, 1)]
    three_resp_02 = three_record["responses"][(0, 2)]

    energy = two_resp["energy_eV"]
    band = (energy >= e_min) & (energy <= e_max)
    time_window = (t_fs >= -80.0) & (t_fs <= 80.0)

    with plt.rc_context(style):
        fig = plt.figure(figsize=(18, 10))
        gs = fig.add_gridspec(2, 3, wspace=0.34, hspace=0.42)

        ax_t = fig.add_subplot(gs[0, 0])
        ax_w = fig.add_subplot(gs[1, 0])
        ax_abs = fig.add_subplot(gs[:, 1])
        ax_coh = fig.add_subplot(gs[:, 2])

        ax_t.plot(t_fs[time_window], E_t[time_window], linewidth=2.2)
        ax_t.set_title("Input pulse")
        ax_t.set_xlabel("Time (fs)")
        ax_t.set_ylabel("E(t) [MV/cm]")

        E_max = float(np.nanmax(np.abs(E_t[time_window])))
        if E_max == 0.0:
            E_max = 1.0
        ax_t.set_ylim(-1.05 * E_max, 1.05 * E_max)

        curves_w = []
        curves_w.append(
            plot_normalized_curve(
                ax_w,
                energy[band],
                two_resp["abs_E_fft"][band],
                label="norm |E(w)|",
                linewidth=2.3,
            )
        )
        ax_w.set_title("Input spectrum")
        ax_w.set_xlabel("Energy (eV)")
        ax_w.set_ylabel("Normalized amplitude")
        set_energy_axis(ax_w, e_min, e_max)
        add_top_omega_axis(ax_w, y_pos=0.82)
        set_axis_ylim_from_curves(ax_w, curves_w, min_headroom=0.16)
        ax_w.legend(loc="best")

        curves_abs = []
        curves_abs.append(
            plot_normalized_curve(
                ax_abs,
                energy[band],
                two_resp["neg_omega_im_P_over_E"][band],
                label="2-level total -w Im[P/E]",
                linewidth=2.6,
            )
        )
        curves_abs.append(
            plot_normalized_curve(
                ax_abs,
                three_resp_01["energy_eV"][band],
                three_resp_01["neg_omega_im_P_over_E"][band],
                label="3-level total -w Im[P/E]",
                linewidth=2.6,
                linestyle="--",
            )
        )

        ax_abs.axvline(1.50, linestyle=":", linewidth=1.5, label="E10 = 1.50 eV")
        ax_abs.axvline(1.75, linestyle=":", linewidth=1.5, label="E20 = 1.75 eV")
        ax_abs.set_title("Total absorption-like response")
        ax_abs.set_xlabel("Energy (eV)")
        ax_abs.set_ylabel("Normalized signal")
        set_energy_axis(ax_abs, e_min, e_max)
        add_top_omega_axis(ax_abs)
        set_axis_ylim_from_curves(ax_abs, curves_abs, min_headroom=0.22)
        ax_abs.legend(loc="best")

        curves_coh = []
        curves_coh.append(
            plot_normalized_curve(
                ax_coh,
                energy[band],
                -two_resp["im_rho12_over_E"][band],
                label="2-level -Im[rho01/E]",
                linewidth=2.2,
            )
        )
        curves_coh.append(
            plot_normalized_curve(
                ax_coh,
                three_resp_01["energy_eV"][band],
                -three_resp_01["im_rho12_over_E"][band],
                label="3-level -Im[rho01/E]",
                linewidth=2.2,
                linestyle="--",
            )
        )
        curves_coh.append(
            plot_normalized_curve(
                ax_coh,
                three_resp_02["energy_eV"][band],
                -three_resp_02["im_rho12_over_E"][band],
                label="3-level -Im[rho02/E]",
                linewidth=2.2,
                linestyle="-.",
            )
        )

        ax_coh.axvline(1.50, linestyle=":", linewidth=1.5)
        ax_coh.axvline(1.75, linestyle=":", linewidth=1.5)
        ax_coh.set_title("Coherence-channel response")
        ax_coh.set_xlabel("Energy (eV)")
        ax_coh.set_ylabel("Normalized signal")
        set_energy_axis(ax_coh, e_min, e_max)
        add_top_omega_axis(ax_coh)
        set_axis_ylim_from_curves(ax_coh, curves_coh, min_headroom=0.22)
        ax_coh.legend(loc="best")

        fig.suptitle(
            "Two-level vs three-level lab-exact absorption validation\n"
            "No RWA; default preview enabled",
            y=0.97,
        )

        fig.subplots_adjust(top=0.84, left=0.07, right=0.985, bottom=0.08)

        path = output_dir / "figures" / "two_vs_three_level_absorption_compare.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=180)
        plt.close(fig)

    return path


def main():
    output_dir = Path(__file__).resolve().parent / "outputs" / "three_level_absorption_lab_exact"
    simulation_root = output_dir / "simulation"
    final_output_root = output_dir / "real" / "final_output"

    output_dir.mkdir(parents=True, exist_ok=True)
    simulation_root.mkdir(parents=True, exist_ok=True)
    final_output_root.mkdir(parents=True, exist_ok=True)

    reset_results_csv(simulation_root)

    number_density_m3 = 1.0e24

    two_record = run_case_and_extract(
        params=make_two_level_params(),
        case_name="two_level_lab_exact_absorption",
        output_dir=simulation_root,
        number_density_m3=number_density_m3,
        coherence_pairs=((0, 1),),
        append_results_csv=True,
    )

    three_record = run_case_and_extract(
        params=make_three_level_params(),
        case_name="three_level_lab_exact_absorption",
        output_dir=simulation_root,
        number_density_m3=number_density_m3,
        coherence_pairs=((0, 1), (0, 2)),
        append_results_csv=True,
    )

    save_debug_csv(
        output_dir=final_output_root,
        two_record=two_record,
        three_record=three_record,
    )

    fig_path = build_compare_figure(
        output_dir=final_output_root,
        two_record=two_record,
        three_record=three_record,
        e_min=1.35,
        e_max=1.90,
    )

    workflow_metadata = {
        "example_name": EXAMPLE_NAME,
        "metadata_role": "workflow_summary",
        "description": (
            "Two-level vs three-level lab-exact absorption validation. "
            "Per-case physical parameters, field metadata, relaxation channels, "
            "pure dephasing channels, solver metadata, and sanity checks are saved "
            "by save_result_case under simulation/."
        ),
        "case_metadata_policy": {
            "workflow_metadata": "Only workflow-level settings, output paths, and compact sanity summary.",
            "case_wise_metadata": (
                "Complete simulation case metadata is stored under simulation/<case_name>/ "
                "or simulation/res_per_case/<case_name>/ depending on the active save_result_case layout."
            ),
            "checkpoint_policy": "Checkpoints are stored under simulation/checkpoints/ and are independent of the case export layout.",
        },
        "number_density_m3": number_density_m3,
        "spectroscopy": {
            "window": "hann",
            "subtract_mean": True,
            "rel_threshold": 1e-5,
            "zero_padding_factor": 4,
            "energy_range_for_figure_eV": [1.35, 1.90],
        },
        "outputs": {
            "root": str(output_dir),
            "simulation_root": str(simulation_root),
            "simulation_results_csv": str(simulation_root / "results.csv"),
            "checkpoints": str(simulation_root / "checkpoints"),
            "final_output_root": str(final_output_root),
            "debug_csv_root": str(final_output_root / "csv"),
            "compare_figure": str(fig_path),
            "workflow_metadata": str(output_dir / "workflow_metadata.json"),
            "case_outputs": {
                "two_level": two_record["case_outputs"],
                "three_level": three_record["case_outputs"],
            },
            "checkpoint_paths": {
                "two_level": str(two_record["checkpoint_path"]),
                "three_level": str(three_record["checkpoint_path"]),
            },
        },
        "sanity_summary": {
            "two_level": {
                "dimension": int(two_record["result"].dimension()),
                "max_trace_error": float(two_record["result"].max_trace_error()),
                "max_hermiticity_error": float(two_record["result"].max_hermiticity_error()),
                "n_time_points": int(np.asarray(two_record["time_fs"]).size),
            },
            "three_level": {
                "dimension": int(three_record["result"].dimension()),
                "max_trace_error": float(three_record["result"].max_trace_error()),
                "max_hermiticity_error": float(three_record["result"].max_hermiticity_error()),
                "n_time_points": int(np.asarray(three_record["time_fs"]).size),
            },
        },
    }
    write_json(output_dir / "workflow_metadata.json", workflow_metadata)

    print("Saved compare figure:")
    print(f"  - {fig_path}")
    print("Saved CSV root:")
    print(f"  - {final_output_root / 'csv'}")
    print("Saved result cases root:")
    print(f"  - {simulation_root}")
    print("Saved summary CSV:")
    print(f"  - {simulation_root / 'results.csv'}")
    print("Saved workflow metadata:")
    print(f"  - {output_dir / 'workflow_metadata.json'}")
    print("Done.")


if __name__ == "__main__":
    main()
