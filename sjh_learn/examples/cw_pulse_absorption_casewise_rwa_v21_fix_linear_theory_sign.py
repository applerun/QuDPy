from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

from sjh_learn.utils.core import (
    NLevelPhysicalParams,
    RelaxationChannel,
    PureDephasingChannel,
    run_case,
)
from sjh_learn.utils.core import config as core_config
from sjh_learn.utils.core.normalization import ParaNormalizer
from sjh_learn.utils.analysis import DynamicsAnalysis
from sjh_learn.utils.plotting import (
    add_top_omega_axis,
    normalize_for_shape,
    plot_normalized_curve,
    real_if_close_or_abs_for_plot,
    set_axis_ylim_from_curves,
    set_energy_axis,
    sorted_xy_for_plot,
)
from sjh_learn.utils.spectroscopy.rwa import choose_rwa_reconstructed_p_over_e_response
from sjh_learn.utils.spectroscopy.spectra import lab_frame_fft_response, rwa_fft_response
from sjh_learn.utils.spectroscopy.theory import EPSILON0_F_PER_M, chi_two_level_linear, gamma2_fs_inv_from_T1_Tphi
from sjh_learn.utils.io import save_result_case


MV_PER_CM_TO_V_PER_M = 1.0e8
DEBYE_TO_C_M = 3.33564e-30
FORCE_RERUN = False
RUN_ONLY = "few_cycle_ultrastrong_with_perm"




@dataclass(frozen=True)
class VariantConfig:
    name: str
    label: str
    field_MV_per_cm: float
    pulse_sigma_fs: float
    T1_fs: Optional[float]
    Tphi_fs: Optional[float]
    dipole_matrix_D: tuple[tuple[complex, complex], tuple[complex, complex]]
    baseline_delta_text: str
    note: str
    plot_e_min: float = 1.50
    plot_e_max: float = 1.60
    compute_rwa: bool = False


def make_analysis_from_result(result):
    if hasattr(DynamicsAnalysis, "from_result"):
        return DynamicsAnalysis.from_result(result)
    return DynamicsAnalysis.from_dynamics_res(result)


def get_coherence(analysis: DynamicsAnalysis, pair=(0, 1)) -> np.ndarray:
    if hasattr(analysis, "coherence"):
        return analysis.coherence(pair=pair)
    return analysis.rho12(pair=pair)


def result_ckp_path(output_dir: Path, case_name: str) -> Path:
    return output_dir / "res_per_case" / case_name / "data" / "result.ckp"


def make_params(
    *,
    variant: VariantConfig,
    energy_gap_eV: float,
    laser_energy_eV: float,
    t_start_fs: float,
    t_end_fs: float,
    dt_fs: float,
    solver_mode: str,
) -> NLevelPhysicalParams:
    relaxation_channels = ()
    if variant.T1_fs is not None:
        relaxation_channels = (
            RelaxationChannel(
                name="relaxation_1_to_0",
                from_level=1,
                to_level=0,
                T1_fs=variant.T1_fs,
            ),
        )
    pure_dephasing_channels = ()
    if variant.Tphi_fs is not None:
        pure_dephasing_channels = (
            PureDephasingChannel(
                name="pure_dephasing_level_1",
                level=1,
                Tphi_fs=variant.Tphi_fs,
            ),
        )

    return NLevelPhysicalParams(
        energies_eV=(0.0, energy_gap_eV),
        dipole_matrix_D=variant.dipole_matrix_D,
        field_MV_per_cm=variant.field_MV_per_cm,
        laser_energy_eV=laser_energy_eV,
        t_start_fs=t_start_fs,
        t_end_fs=t_end_fs,
        dt_fs=dt_fs,
        basis=("g", "e"),
        relaxation_channels=relaxation_channels,
        pure_dephasing_channels=pure_dephasing_channels,
        pulse_center_fs=0.0,
        pulse_sigma_fs=variant.pulse_sigma_fs,
        solver_mode=solver_mode,
        input_description="Weak Gaussian carrier pulse for lab-vs-RWA absorption-like comparisons.",
        input_metadata={
            "variant": variant.name,
            "variant_label": variant.label,
            "variant_note": variant.note,
            "baseline_delta": variant.baseline_delta_text,
            "purpose": "case-wise lab and RWA comparison",
            "target_energy_window_eV": [variant.plot_e_min, variant.plot_e_max],
        },
    )


def run_variant(
    *,
    variant: VariantConfig,
    output_dir: Path,
    energy_gap_eV: float,
    laser_energy_eV: float,
    t_start_fs: float,
    t_end_fs: float,
    dt_fs: float,
    mu_ge_D: float,
    number_density_m3: float,
) -> dict[str, object]:
    print(f"Running variant: {variant.name} ({variant.label})")
    if variant.compute_rwa and not core_config.FORCE_RWA:
        raise RuntimeError(core_config.RWA_DISABLED_MESSAGE)
    lab_params = make_params(
        variant=variant,
        energy_gap_eV=energy_gap_eV,
        laser_energy_eV=laser_energy_eV,
        t_start_fs=t_start_fs,
        t_end_fs=t_end_fs,
        dt_fs=dt_fs,
        solver_mode="lab_exact",
    )

    lab_case_name = f"{variant.name}_lab_exact"
    lab_ckp = result_ckp_path(output_dir, lab_case_name)

    lab_result = run_case(
        lab_params,
        load_ckp=lab_ckp,
        force_run=FORCE_RERUN,
    )
    lab_analysis = make_analysis_from_result(lab_result)

    t_fs = lab_analysis.time_fs()
    P_t = lab_analysis.full_polarization_C_per_m2(number_density_m3=number_density_m3)
    E_t, _, _ = lab_analysis.input_signal(kind="field")
    rho12_lab_t = get_coherence(lab_analysis, pair=(0, 1))

    lab_response = lab_frame_fft_response(
        t_fs=t_fs,
        E_MV_per_cm=E_t,
        P_C_per_m2=P_t,
        rhoij=rho12_lab_t,
        window="hann",
        subtract_mean=True,
        rel_threshold=1e-5,
        zero_padding_factor=4,
    )

    omega_fs_inv = lab_response["omega_fs_inv"]
    gamma2_fs_inv = gamma2_fs_inv_from_T1_Tphi(variant.T1_fs, variant.Tphi_fs)
    chi = chi_two_level_linear(
        omega_fs_inv=omega_fs_inv,
        omega_eg_fs_inv=energy_gap_eV * ParaNormalizer.EV_TO_FS_INV,
        mu_ge_D=np.asarray(variant.dipole_matrix_D, dtype=np.complex128)[0, 1],
        gamma2_fs_inv=gamma2_fs_inv,
        number_density_m3=number_density_m3,
        population_difference=1.0,
    )
    theory_P_over_E_MVcm = EPSILON0_F_PER_M * chi * MV_PER_CM_TO_V_PER_M
    # chi_two_level_linear() follows the analytic/physics Fourier convention, where
    # the absorptive part is +omega * Im[P/E]. The numerical FFT-based lab
    # spectrum still uses neg_omega_im_P_over_E because np.fft.fft has the
    # opposite sign convention for positive frequencies.
    theory_abs_like = omega_fs_inv * np.imag(theory_P_over_E_MVcm)

    save_result_case(
        lab_result,
        output_dir,
        output_preview=True,
        case_name=lab_case_name,
        example_name="cw_pulse_absorption_casewise_rwa",
        append_results_csv=True,
        save_populations_csv=False,
    )

    if variant.compute_rwa:
        rwa_params = replace(lab_params, solver_mode="rwa")
        rwa_case_name = f"{variant.name}_rwa"
        rwa_ckp = result_ckp_path(output_dir, rwa_case_name)

        rwa_result = run_case(
            rwa_params,
            load_ckp=rwa_ckp,
            force_run=FORCE_RERUN,
        )
        rwa_analysis = make_analysis_from_result(rwa_result)

        t_rwa_fs = rwa_analysis.time_fs()
        if not np.allclose(t_fs, t_rwa_fs, rtol=1e-8, atol=1e-10):
            raise ValueError("lab_exact and RWA time axes do not match.")
        g_t, _, _ = rwa_analysis.input_signal(kind="drive")
        rho12_rwa_t = get_coherence(rwa_analysis, pair=(0, 1))

        rwa_response = rwa_fft_response(
            t_fs=t_fs,
            g_fs_inv=g_t,
            rho12_rwa=rho12_rwa_t,
            laser_energy_eV=laser_energy_eV,
            window="hann",
            subtract_mean=False,
            rel_threshold=1e-5,
            zero_padding_factor=4,
        )

        P_rwa_recon_t, rwa_recon_p_response, rwa_recon_carrier_label = choose_rwa_reconstructed_p_over_e_response(
            t_fs=t_fs,
            E_t=E_t,
            rho12_rwa_t=rho12_rwa_t,
            rho12_lab_t=rho12_lab_t,
            dipole_matrix_D=variant.dipole_matrix_D,
            number_density_m3=number_density_m3,
            laser_energy_eV=laser_energy_eV,
            lab_reference_response=lab_response,
            energy_window_eV=(variant.plot_e_min, variant.plot_e_max),
        )

        save_result_case(
            rwa_result,
            output_dir,
            output_preview=True,
            case_name=rwa_case_name,
            example_name="cw_pulse_absorption_casewise_rwa",
            append_results_csv=True,
            save_populations_csv=False,
        )
    else:
        print(f"Skipping RWA for variant: {variant.name}")
        g_t = np.full_like(t_fs, np.nan + 0j, dtype=np.complex128)
        rho12_rwa_t = np.full_like(rho12_lab_t, np.nan + 0j, dtype=np.complex128)
        P_rwa_recon_t = np.full_like(t_fs, np.nan, dtype=float)
        rwa_response = None
        rwa_recon_p_response = None
        rwa_recon_carrier_label = "RWA skipped for this case"

    time_dir = output_dir / "time_domain_csv"
    spec_dir = output_dir / "spectrum_csv"
    time_dir.mkdir(parents=True, exist_ok=True)
    spec_dir.mkdir(parents=True, exist_ok=True)

    time_table = np.column_stack([
        t_fs,
        E_t,
        np.real(g_t),
        np.imag(g_t),
        np.real(rho12_lab_t),
        np.imag(rho12_lab_t),
        np.real(rho12_rwa_t),
        np.imag(rho12_rwa_t),
        np.real(P_t),
        np.imag(P_t),
        P_rwa_recon_t,
    ])
    np.savetxt(
        time_dir / f"{variant.name}_time_domain_signals.csv",
        time_table,
        delimiter=",",
        header="time_fs,E_MV_per_cm,Re_g_rwa_fs_inv,Im_g_rwa_fs_inv,Re_rho12_lab,Im_rho12_lab,Re_rho12_rwa,Im_rho12_rwa,Re_P_C_per_m2,Im_P_C_per_m2,P_rwa_reconstructed_C_per_m2",
        comments="",
    )

    lab_spec = np.column_stack([
        lab_response["energy_eV"],
        lab_response["abs_E_fft"],
        lab_response["abs_rho12_over_E"],
        lab_response["im_rho12_over_E"],
        lab_response["omega_im_rho12_over_E"],
        lab_response["neg_omega_im_P_over_E"],
        theory_abs_like,
    ])
    np.savetxt(
        spec_dir / f"{variant.name}_lab_spectrum.csv",
        lab_spec,
        delimiter=",",
        header="energy_eV,abs_E_fft,abs_rho12_over_E,im_rho12_over_E,omega_im_rho12_over_E,neg_omega_im_P_over_E,linear_theory_abs_like",
        comments="",
    )

    if variant.compute_rwa:
        rwa_spec = np.column_stack([
            rwa_response["energy_eV"],
            rwa_response["abs_g_fft"],
            rwa_response["abs_rho12_rwa_over_g"],
            rwa_response["im_rho12_rwa_over_g"],
            rwa_response["omega_im_rho12_rwa_over_g"],
        ])
        np.savetxt(
            spec_dir / f"{variant.name}_rwa_spectrum.csv",
            rwa_spec,
            delimiter=",",
            header="energy_eV,abs_g_fft,abs_rho12_rwa_over_g,im_rho12_rwa_over_g,omega_im_rho12_rwa_over_g",
            comments="",
        )

        rwa_recon_spec = np.column_stack([
            rwa_recon_p_response["energy_eV"],
            rwa_recon_p_response["neg_omega_im_P_over_E"],
        ])
        np.savetxt(
            spec_dir / f"{variant.name}_rwa_reconstructed_P_over_E_spectrum.csv",
            rwa_recon_spec,
            delimiter=",",
            header="energy_eV,rwa_reconstructed_neg_omega_im_P_over_E",
            comments="",
        )

    return {
        "variant": variant,
        "time_fs": t_fs,
        "E_t": E_t,
        "g_t": g_t,
        "lab_response": lab_response,
        "rwa_response": rwa_response,
        "rwa_recon_p_response": rwa_recon_p_response,
        "rwa_recon_carrier_label": rwa_recon_carrier_label,
        "theory_abs_like": theory_abs_like,
    }

def build_lab_only_figure(record: dict[str, object], *, output_dir: Path, e_min: float, e_max: float) -> Path:
    style = {
        "font.size": 17,
        "axes.titlesize": 19,
        "axes.labelsize": 17,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "legend.fontsize": 12,
        "figure.titlesize": 21,
    }
    variant = record["variant"]
    t_fs = record["time_fs"]
    E_t = record["E_t"]
    lab_response = record["lab_response"]
    theory_abs_like = record["theory_abs_like"]

    band_input = (lab_response["energy_eV"] >= e_min) & (lab_response["energy_eV"] <= e_max)
    panel3_min = e_min
    panel3_max = e_max
    band_resp = (lab_response["energy_eV"] >= panel3_min) & (lab_response["energy_eV"] <= panel3_max)
    time_window = (t_fs >= -8.0 * variant.pulse_sigma_fs) & (t_fs <= 8.0 * variant.pulse_sigma_fs)
    neg_omega_im_rho12 = -lab_response["omega_im_rho12_over_E"]

    with plt.rc_context(style):
        fig = plt.figure(figsize=(16, 9))
        gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.15], wspace=0.32, hspace=0.46)
        ax_t = fig.add_subplot(gs[0, 0])
        ax_w = fig.add_subplot(gs[1, 0])
        ax_resp = fig.add_subplot(gs[:, 1])

        ax_t.plot(t_fs[time_window], E_t[time_window], linewidth=2.3)
        ax_t.set_title("Input t", pad=8)
        ax_t.set_xlabel("Time (fs)", labelpad=4)
        ax_t.set_ylabel("E(t) [MV/cm]")
        E_max = float(np.nanmax(np.abs(E_t[time_window]))) if np.any(time_window) else float(np.nanmax(np.abs(E_t)))
        if E_max == 0.0:
            E_max = 1.0
        ax_t.set_ylim(-1.05 * E_max, 1.05 * E_max)

        y1 = plot_normalized_curve(ax_w, lab_response["energy_eV"][band_input], lab_response["abs_E_fft"][band_input], label="norm |E(w)|", linewidth=2.5, color="C0")
        ax_w.set_title("Input w", pad=10)
        ax_w.set_xlabel("Energy (eV)")
        ax_w.set_ylabel("Normalized amplitude")
        set_energy_axis(ax_w, e_min, e_max)
        set_axis_ylim_from_curves(ax_w, [y1], min_headroom=0.16)
        ax_w.legend(loc="best")

        curves = []
        curves.append(plot_normalized_curve(ax_resp, lab_response["energy_eV"][band_resp], lab_response["abs_rho12_over_E"][band_resp], label="norm |rho12/E|", linewidth=2.2, color="C0"))
        curves.append(plot_normalized_curve(ax_resp, lab_response["energy_eV"][band_resp], -lab_response["im_rho12_over_E"][band_resp], label="norm -Im[rho12/E]", linewidth=2.2, color="C1", linestyle="--"))
        curves.append(plot_normalized_curve(ax_resp, lab_response["energy_eV"][band_resp], neg_omega_im_rho12[band_resp], label="norm -w Im[rho12/E]", linewidth=2.2, color="C2", linestyle="-."))
        curves.append(plot_normalized_curve(ax_resp, lab_response["energy_eV"][band_resp], lab_response["neg_omega_im_P_over_E"][band_resp], label="norm -w Im[P/E]", linewidth=2.4, color="C3", linestyle=":"))
        curves.append(plot_normalized_curve(ax_resp, lab_response["energy_eV"][band_resp], theory_abs_like[band_resp], label="linear theory", linewidth=2.4, color="black", linestyle="-"))
        ax_resp.set_title("Lab-frame response", pad=8)
        ax_resp.set_xlabel("Energy (eV)")
        ax_resp.set_ylabel("Normalized signal")
        set_energy_axis(ax_resp, panel3_min, panel3_max)
        set_axis_ylim_from_curves(ax_resp, curves, min_headroom=0.20)
        ax_resp.legend(loc="best", ncol=1)

        fig.suptitle(f"Lab-only case: {variant.label}\nΔ from baseline: {variant.baseline_delta_text}", y=0.965)
        fig.subplots_adjust(top=0.80, left=0.10, right=0.985, bottom=0.09)
        path = output_dir / "casewise_figs" / f"{variant.name}_lab_only.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=180)
        plt.close(fig)
    return path


def build_lab_vs_rwa_figure(record: dict[str, object], *, output_dir: Path, e_min: float, e_max: float) -> Path:
    style = {
        "font.size": 16,
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 10,
        "figure.titlesize": 21,
    }
    variant = record["variant"]
    t_fs = record["time_fs"]
    E_t = record["E_t"]
    g_t = record["g_t"]
    lab_response = record["lab_response"]
    rwa_response = record["rwa_response"]
    rwa_recon_p_response = record["rwa_recon_p_response"]
    rwa_recon_carrier_label = record["rwa_recon_carrier_label"]

    lab_band = (lab_response["energy_eV"] >= e_min) & (lab_response["energy_eV"] <= e_max)
    rwa_band = (rwa_response["energy_eV"] >= e_min) & (rwa_response["energy_eV"] <= e_max)
    rwa_recon_band = (rwa_recon_p_response["energy_eV"] >= e_min) & (rwa_recon_p_response["energy_eV"] <= e_max)
    time_window = (t_fs >= -8.0 * variant.pulse_sigma_fs) & (t_fs <= 8.0 * variant.pulse_sigma_fs)

    neg_omega_im_rho12_lab = -lab_response["omega_im_rho12_over_E"]
    neg_omega_im_rho12_rwa = -(rwa_response["energy_eV"] * ParaNormalizer.EV_TO_FS_INV) * rwa_response["im_rho12_rwa_over_g"]

    with plt.rc_context(style):
        fig = plt.figure(figsize=(18.8, 11.8))
        outer = fig.add_gridspec(2, 3, width_ratios=[0.82, 1.0, 1.0], wspace=0.46, hspace=0.42)
        left_top = outer[0, 0].subgridspec(2, 1, hspace=1.35)

        ax_input_t = fig.add_subplot(left_top[0, 0])
        ax_input_w = fig.add_subplot(left_top[1, 0])
        ax_abs = fig.add_subplot(outer[0, 1])
        ax_im = fig.add_subplot(outer[0, 2])
        ax_wim = fig.add_subplot(outer[1, 0])
        ax_pe = fig.add_subplot(outer[1, 1])
        ax_theory = fig.add_subplot(outer[1, 2])

        # Panel 1: input t
        ax_right = ax_input_t.twinx()
        g_t_plot, g_t_label = real_if_close_or_abs_for_plot(g_t)
        ax_input_t.plot(t_fs[time_window], E_t[time_window], linewidth=2.0, label="lab E(t)", color="C0")
        ax_right.plot(t_fs[time_window], g_t_plot[time_window], linestyle="--", linewidth=2.0, label=f"RWA {g_t_label}", color="C1")
        ax_input_t.set_title("Input t", pad=8)
        ax_input_t.set_xlabel("Time (fs)", labelpad=2)
        ax_input_t.set_ylabel("E(t) [MV/cm]", labelpad=2)
        ax_right.set_ylabel("g(t) [fs$^{-1}$]", labelpad=2)
        E_max = float(np.nanmax(np.abs(E_t[time_window]))) if np.any(time_window) else float(np.nanmax(np.abs(E_t)))
        g_max = float(np.nanmax(np.abs(g_t_plot[time_window]))) if np.any(time_window) else float(np.nanmax(np.abs(g_t_plot)))
        if E_max == 0.0:
            E_max = 1.0
        if g_max == 0.0:
            g_max = 1.0
        ax_input_t.set_ylim(-1.05 * E_max, 1.05 * E_max)
        ax_right.set_ylim(-1.05 * g_max, 1.05 * g_max)
        lines1, labels1 = ax_input_t.get_legend_handles_labels()
        lines2, labels2 = ax_right.get_legend_handles_labels()
        ax_input_t.legend(lines1 + lines2, labels1 + labels2, loc="best")

        # Panel 2: input w
        curves = []
        curves.append(plot_normalized_curve(ax_input_w, lab_response["energy_eV"][lab_band], lab_response["abs_E_fft"][lab_band], label="lab norm |E(w)|", linewidth=2.2, color="C0"))
        curves.append(plot_normalized_curve(ax_input_w, rwa_response["energy_eV"][rwa_band], rwa_response["abs_g_fft"][rwa_band], label="RWA norm |g(W)|", linewidth=2.2, color="C1", linestyle="--"))
        ax_input_w.set_title("Input w", pad=8)
        ax_input_w.set_xlabel("Energy (eV)")
        ax_input_w.set_ylabel("Normalized amplitude")
        set_energy_axis(ax_input_w, e_min, e_max)
        add_top_omega_axis(ax_input_w,y_pos = 0.8)
        set_axis_ylim_from_curves(ax_input_w, curves, min_headroom=0.28)
        ax_input_w.legend(loc="best")

        # Panel 3: |rho12/E|
        curves = []
        curves.append(plot_normalized_curve(ax_abs, lab_response["energy_eV"][lab_band], lab_response["abs_rho12_over_E"][lab_band], label="lab", linewidth=2.2, color="C0"))
        curves.append(plot_normalized_curve(ax_abs, rwa_response["energy_eV"][rwa_band], rwa_response["abs_rho12_rwa_over_g"][rwa_band], label="RWA", linewidth=2.2, color="C1", linestyle="--"))
        ax_abs.set_title("|rho12 / input|", pad=10)
        ax_abs.set_xlabel("Energy (eV)")
        ax_abs.set_ylabel("Normalized signal", labelpad=2)
        set_energy_axis(ax_abs, e_min, e_max)
        add_top_omega_axis(ax_abs)
        set_axis_ylim_from_curves(ax_abs, curves, min_headroom=0.28)
        ax_abs.legend(loc="best")

        # Panel 4: -Im[rho12/input]
        curves = []
        curves.append(plot_normalized_curve(ax_im, lab_response["energy_eV"][lab_band], -lab_response["im_rho12_over_E"][lab_band], label="lab", linewidth=2.2, color="C0"))
        curves.append(plot_normalized_curve(ax_im, rwa_response["energy_eV"][rwa_band], -rwa_response["im_rho12_rwa_over_g"][rwa_band], label="RWA", linewidth=2.2, color="C1", linestyle="--"))
        ax_im.set_title("-Im[rho12 / input]", pad=10)
        ax_im.set_xlabel("Energy (eV)")
        ax_im.set_ylabel("Normalized signal")
        set_energy_axis(ax_im, e_min, e_max)
        add_top_omega_axis(ax_im)
        set_axis_ylim_from_curves(ax_im, curves, min_headroom=0.28)
        ax_im.legend(loc="best")

        # Panel 5: -w Im[rho12/input]
        curves = []
        curves.append(plot_normalized_curve(ax_wim, lab_response["energy_eV"][lab_band], neg_omega_im_rho12_lab[lab_band], label="lab", linewidth=2.2, color="C0"))
        curves.append(plot_normalized_curve(ax_wim, rwa_response["energy_eV"][rwa_band], neg_omega_im_rho12_rwa[rwa_band], label="RWA", linewidth=2.2, color="C1", linestyle="--"))
        ax_wim.set_title("-w Im[rho12 / input]", pad=10)
        ax_wim.set_xlabel("Energy (eV)")
        ax_wim.set_ylabel("Normalized signal")
        set_energy_axis(ax_wim, e_min, e_max)
        add_top_omega_axis(ax_wim)
        set_axis_ylim_from_curves(ax_wim, curves, min_headroom=0.28)
        ax_wim.legend(loc="best")

        # Panel 6: -w Im[P/E]
        curves = []
        curves.append(plot_normalized_curve(ax_pe, lab_response["energy_eV"][lab_band], lab_response["neg_omega_im_P_over_E"][lab_band], label="lab", linewidth=2.3, color="C3"))
        curves.append(plot_normalized_curve(ax_pe, rwa_recon_p_response["energy_eV"][rwa_recon_band], rwa_recon_p_response["neg_omega_im_P_over_E"][rwa_recon_band], label="RWA recon", linewidth=2.3, color="C4", linestyle="--"))
        ax_pe.set_title("-w Im[P / E]", pad=10)
        ax_pe.set_xlabel("Energy (eV)")
        ax_pe.set_ylabel("Normalized signal")
        set_energy_axis(ax_pe, e_min, e_max)
        add_top_omega_axis(ax_pe)
        set_axis_ylim_from_curves(ax_pe, curves, min_headroom=0.28)
        ax_pe.legend(loc="best")
        # ax_pe.text(0.02, 0.05, rwa_recon_carrier_label, transform=ax_pe.transAxes, fontsize=9, va="bottom", ha="left")

        # Panel 7: lab absorption vs theory
        curves = []
        curves.append(plot_normalized_curve(ax_theory, lab_response["energy_eV"][lab_band], lab_response["neg_omega_im_P_over_E"][lab_band], label="lab -w Im[P/E]", linewidth=2.3, color="C3"))
        curves.append(plot_normalized_curve(ax_theory, lab_response["energy_eV"][lab_band], record["theory_abs_like"][lab_band], label="linear theory", linewidth=2.3, color="black", linestyle="--"))
        ax_theory.set_title("Lab absorption vs theory", pad=10)
        ax_theory.set_xlabel("Energy (eV)")
        ax_theory.set_ylabel("Normalized signal")
        set_energy_axis(ax_theory, e_min, e_max)
        add_top_omega_axis(ax_theory)
        set_axis_ylim_from_curves(ax_theory, curves, min_headroom=0.28)
        ax_theory.legend(loc="best")

        fig.suptitle(f"Lab vs RWA case: {variant.label}\nΔ from baseline: {variant.baseline_delta_text}", y=0.985)
        fig.subplots_adjust(top=0.84, left=0.065, right=0.985, bottom=0.08)
        path = output_dir / "casewise_figs" / f"{variant.name}_lab_vs_rwa.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=180)
        plt.close(fig)
    return path


def main():
    output_dir = Path("../scratch/outputs/cw_pulse_absorption_compare")
    output_dir.mkdir(parents=True, exist_ok=True)

    energy_gap_eV = 1.55
    laser_energy_eV = 1.55
    mu_ge_D = 3.0
    number_density_m3 = 1.0e24

    base_field_MV_per_cm = 1
    base_pulse_sigma_fs = 30/1.6
    base_T1_fs = 500.0
    base_Tphi_fs = 300.0

    t_start_fs = -1000.0
    t_end_fs = 1000.0
    dt_fs = 0.05

    permanent_mu_gg_D = 0.0
    permanent_mu_ee_D = 10.0
    # 自由设置跃迁偶极矩相位
    phase_deg = 45.0
    phase_rad = np.deg2rad(phase_deg)

    mu_ge_complex = mu_ge_D * np.exp(1j * phase_rad)
    mu_eg_complex = np.conjugate(mu_ge_complex)

    variants = [
        VariantConfig(
            name="baseline",
            label="baseline",
            field_MV_per_cm=base_field_MV_per_cm,
            pulse_sigma_fs=base_pulse_sigma_fs,
            T1_fs=base_T1_fs,
            Tphi_fs=base_Tphi_fs,
            dipole_matrix_D=((0.0, mu_ge_D), (mu_ge_D, 0.0)),
            baseline_delta_text=(f"baseline: E0 = {base_field_MV_per_cm:.4g} MV/cm, sigma = {base_pulse_sigma_fs:.2f} fs, T1 = {base_T1_fs:.1f} fs, Tphi = {base_Tphi_fs:.1f} fs"),
            note="No permanent dipole: mu_gg = 0 D, mu_ee = 0 D.",
        ),
        VariantConfig(
            name = f"complex_transition_mu_phase_{int(phase_deg)}deg",
            label = f"complex transition dipole, phase = {phase_deg:.0f} deg",
            field_MV_per_cm = base_field_MV_per_cm,
            pulse_sigma_fs = base_pulse_sigma_fs,
            T1_fs = base_T1_fs,
            Tphi_fs = base_Tphi_fs,
            dipole_matrix_D = (
                (0.0, mu_ge_complex),
                (mu_eg_complex, 0.0),
            ),
            baseline_delta_text = (
                f"transition dipole phase: mu_ge = {mu_ge_D:.1f} * exp(i {phase_deg:.0f} deg) D; "
            ),
            note = (
                "Two-level gauge test with a complex transition dipole. "
                "For a single isolated two-level system, changing this phase should not change physical observables, "
                "although rho12 itself may acquire a different phase convention."
            ),
            plot_e_min = 1.50,
            plot_e_max = 1.60,
            compute_rwa = False,
        ),
        VariantConfig(
            name="imag_transition_mu",
            label="imaginary transition dipole",
            field_MV_per_cm=base_field_MV_per_cm,
            pulse_sigma_fs=base_pulse_sigma_fs,
            T1_fs=base_T1_fs,
            Tphi_fs=base_Tphi_fs,
            dipole_matrix_D=((0.0, 1j * mu_ge_D), (-1j * mu_ge_D, 0.0)),
            baseline_delta_text="transition dipole phase: mu_ge = 3 D -> 3i D; mu_eg = 3 D -> -3i D",
            note="Hermitian complex transition dipole test. For a single two-level system this should be gauge-equivalent to the real-mu baseline in physical observables. RWA is intentionally skipped for this case because the current drive export path can still return an all-zero complex drive.",
            compute_rwa=False,
        ),
        VariantConfig(
            name="field_x5",
            label="field ×5",
            field_MV_per_cm=5 * base_field_MV_per_cm,
            pulse_sigma_fs=base_pulse_sigma_fs,
            T1_fs=base_T1_fs,
            Tphi_fs=base_Tphi_fs,
            dipole_matrix_D=((0.0, mu_ge_D), (mu_ge_D, 0.0)),
            baseline_delta_text=f"E0: {base_field_MV_per_cm:.4g} → {5 * base_field_MV_per_cm:.4g} MV/cm (+400%)",
            note="Only the field amplitude is changed relative to baseline.",
            compute_rwa = False,
        ),
        VariantConfig(
            name="pulse_x1p5",
            label="pulse width ×1.5",
            field_MV_per_cm=base_field_MV_per_cm,
            pulse_sigma_fs=1.5 * base_pulse_sigma_fs,
            T1_fs=base_T1_fs,
            Tphi_fs=base_Tphi_fs,
            dipole_matrix_D=((0.0, mu_ge_D), (mu_ge_D, 0.0)),
            baseline_delta_text=f"sigma: {base_pulse_sigma_fs:.2f} → {1.5 * base_pulse_sigma_fs:.2f} fs (×1.5, +50%)",
            note="Pulse is broader in time, so the input spectrum is narrower.",
            compute_rwa = False,
        ),
        VariantConfig(
            name="with_permanent_dipole",
            label="add permanent dipole",
            field_MV_per_cm=base_field_MV_per_cm,
            pulse_sigma_fs=base_pulse_sigma_fs,
            T1_fs=base_T1_fs,
            Tphi_fs=base_Tphi_fs,
            dipole_matrix_D=((permanent_mu_gg_D, mu_ge_D), (mu_ge_D, permanent_mu_ee_D)),
            baseline_delta_text=f"add diagonal dipoles: mu_gg = {permanent_mu_gg_D:.1f} D, mu_ee = {permanent_mu_ee_D:.1f} D, Δmu_diag = {permanent_mu_ee_D - permanent_mu_gg_D:.1f} D",
            note="Chosen permanent dipole test case: only the excited-state diagonal dipole is nonzero.",
            compute_rwa = False,
        ),
        VariantConfig(
            name="short_T1",
            label="short T1",
            field_MV_per_cm=base_field_MV_per_cm,
            pulse_sigma_fs=base_pulse_sigma_fs,
            T1_fs=100.0,
            Tphi_fs=base_Tphi_fs,
            dipole_matrix_D=((0.0, mu_ge_D), (mu_ge_D, 0.0)),
            baseline_delta_text=f"T1: {base_T1_fs:.1f} → 100.0 fs (-80%)",
            note="Population relaxation is made much faster than baseline.",
            compute_rwa = False,
        ),
        VariantConfig(
            name="remove_T1",
            label="remove T1",
            field_MV_per_cm=base_field_MV_per_cm,
            pulse_sigma_fs=base_pulse_sigma_fs,
            T1_fs=None,
            Tphi_fs=base_Tphi_fs,
            dipole_matrix_D=((0.0, mu_ge_D), (mu_ge_D, 0.0)),
            baseline_delta_text=f"remove T1: {base_T1_fs:.1f} fs → None",
            note="No population relaxation channel; only pure dephasing remains.",
            compute_rwa = False,
        ),
        VariantConfig(
            name="remove_Tphi",
            label="remove Tphi",
            field_MV_per_cm=base_field_MV_per_cm,
            pulse_sigma_fs=base_pulse_sigma_fs,
            T1_fs=base_T1_fs,
            Tphi_fs=None,
            dipole_matrix_D=((0.0, mu_ge_D), (mu_ge_D, 0.0)),
            baseline_delta_text=f"remove Tphi: {base_Tphi_fs:.1f} fs → None",
            note="No pure dephasing channel; only T1 contributes to gamma2.",
            compute_rwa = False,
        ),
        VariantConfig(
            name="extreme_perm_strong_broad",
            label="extreme Δmu + strong broad pulse",
            field_MV_per_cm=50.0,
            pulse_sigma_fs=2.5,
            T1_fs=base_T1_fs,
            Tphi_fs=base_Tphi_fs,
            dipole_matrix_D=((0.0, mu_ge_D), (mu_ge_D, 30.0)),
            baseline_delta_text="E0: 1 → 50 MV/cm; sigma: 18.75 → 2.50 fs; mu_ee: 0 → 30 D",
            note="Extreme two-level test: large diagonal dipole difference plus a strong, short pulse. Plot window is kept inside the main input bandwidth.",
            plot_e_min=1.15,
            plot_e_max=1.95,
            compute_rwa = False,
        ),
        VariantConfig(
            name="few_cycle_ultrastrong_no_perm",
            label="few-cycle ultra-strong, no Δmu",
            field_MV_per_cm=150.0,
            pulse_sigma_fs=1.0,
            T1_fs=base_T1_fs,
            Tphi_fs=base_Tphi_fs,
            dipole_matrix_D=((0.0, mu_ge_D), (mu_ge_D, 0.0)),
            baseline_delta_text="E0: 1 → 150 MV/cm; sigma: 18.75 → 1.00 fs; no diagonal dipole",
            note="Extreme two-level test: few-cycle, ultra-strong pulse to stress the positive-frequency/coherence approximation.",
            plot_e_min=0.85,
            plot_e_max=2.25,
            compute_rwa = False,
        ),
        VariantConfig(
            name="few_cycle_ultrastrong_with_perm",
            label="few-cycle ultra-strong + Δmu",
            field_MV_per_cm=100.0,
            pulse_sigma_fs=1.0,
            T1_fs=base_T1_fs,
            Tphi_fs=base_Tphi_fs,
            dipole_matrix_D=((0.0, mu_ge_D), (mu_ge_D, 20.0)),
            baseline_delta_text="E0: 1 → 100 MV/cm; sigma: 18.75 → 1.00 fs; mu_ee: 0 → 20 D",
            note="Extreme two-level test: combines a few-cycle strong pulse with a large diagonal dipole difference.",
            plot_e_min=1.45,
            plot_e_max=1.65,
        ),
        VariantConfig(
            name = "remove_T1_weak_pump",
            label = "remove T1_weak_pump",
            field_MV_per_cm = base_field_MV_per_cm / 5,
            pulse_sigma_fs = base_pulse_sigma_fs,
            T1_fs = None,
            Tphi_fs = base_Tphi_fs,
            dipole_matrix_D = ((0.0, mu_ge_D), (mu_ge_D, 0.0)),
            baseline_delta_text = f"remove T1: {base_T1_fs:.1f} fs → None, 1/5 $E_0$",
            note = "No population relaxation channel; only pure dephasing remains.",
            compute_rwa = False,
        ),

    ]

    lab_only_paths = []
    lab_vs_rwa_paths = []

    selected_variants = variants if RUN_ONLY is None else [variant for variant in variants if variant.name == RUN_ONLY]
    if not selected_variants:
        raise ValueError(f"RUN_ONLY={RUN_ONLY!r} did not match any VariantConfig.name.")

    for variant in selected_variants:
        record = run_variant(
            variant=variant,
            output_dir=output_dir,
            energy_gap_eV=energy_gap_eV,
            laser_energy_eV=laser_energy_eV,
            t_start_fs=t_start_fs,
            t_end_fs=t_end_fs,
            dt_fs=dt_fs,
            mu_ge_D=mu_ge_D,
            number_density_m3=number_density_m3,
        )
        lab_only_paths.append(
            build_lab_only_figure(
                record,
                output_dir=output_dir,
                e_min=variant.plot_e_min,
                e_max=variant.plot_e_max,
            )
        )
        if variant.compute_rwa:
            lab_vs_rwa_paths.append(
                build_lab_vs_rwa_figure(
                    record,
                    output_dir=output_dir,
                    e_min=variant.plot_e_min,
                    e_max=variant.plot_e_max,
                )
            )
        else:
            print(f"Skipped lab-vs-RWA figure for {variant.name} because compute_rwa=False")

    print("Saved lab-only figures:")
    for path in lab_only_paths:
        print(f"  - {path}")
    print("Saved lab-vs-rwa figures:")
    for path in lab_vs_rwa_paths:
        print(f"  - {path}")
    print(f"Saved result cases root: {output_dir / 'res_per_case'}")
    print("Done.")


if __name__ == "__main__":
    main()
