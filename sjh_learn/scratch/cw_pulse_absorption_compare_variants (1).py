from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from sjh_learn.utils.core import (
    NLevelPhysicalParams,
    RelaxationChannel,
    PureDephasingChannel,
    run_case,
)
from sjh_learn.utils.core.normalization import ParaNormalizer
from sjh_learn.utils.analysis import DynamicsAnalysis
from sjh_learn.utils.analysis.observables import (
    EPSILON0_F_PER_M,
    chi_two_level_linear,
)
from sjh_learn.utils.io import save_result_case


MV_PER_CM_TO_V_PER_M = 1.0e8


@dataclass(frozen=True)
class VariantConfig:
    name: str
    label: str
    field_MV_per_cm: float
    pulse_sigma_fs: float
    T1_fs: float
    dipole_matrix_D: tuple[tuple[float, float], tuple[float, float]]
    note: str


def make_analysis_from_result(result):
    """兼容新旧 DynamicsAnalysis 构造函数。"""
    if hasattr(DynamicsAnalysis, "from_result"):
        return DynamicsAnalysis.from_result(result)
    return DynamicsAnalysis.from_dynamics_res(result)


def get_coherence(analysis: DynamicsAnalysis, pair=(0, 1)) -> np.ndarray:
    """兼容 coherence() / rho12() 两种命名。"""
    if hasattr(analysis, "coherence"):
        return analysis.coherence(pair=pair)
    return analysis.rho12(pair=pair)


def apply_window(values: np.ndarray, window: str | None) -> np.ndarray:
    if window is None or window == "none":
        return values
    if window == "hann":
        return values * np.hanning(values.size)
    raise ValueError("window must be None, 'none', or 'hann'.")


def safe_complex_divide(
    numerator: np.ndarray,
    denominator: np.ndarray,
    *,
    rel_threshold: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Return numerator/denominator and a mask of reliable frequencies."""
    denominator_abs = np.abs(denominator)
    max_denominator = float(np.max(denominator_abs))

    if max_denominator == 0.0:
        raise ValueError("The input spectrum is identically zero.")

    valid = denominator_abs > rel_threshold * max_denominator
    ratio = np.full_like(numerator, np.nan + 1j * np.nan, dtype=np.complex128)
    ratio[valid] = numerator[valid] / denominator[valid]
    return ratio, valid


def normalize_for_shape(y: np.ndarray) -> np.ndarray:
    """Normalize a real or complex vector by its finite max absolute value."""
    y = np.asarray(y)
    if np.iscomplexobj(y):
        y = np.abs(y)
    y = y.astype(float)

    finite = np.isfinite(y)
    if not np.any(finite):
        return y

    scale = np.nanmax(np.abs(y[finite]))
    if scale == 0.0:
        return y
    return y / scale


def interp_to_grid(x: np.ndarray, y: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Interpolate finite y(x) to a common plotting/output grid."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    grid = np.asarray(grid, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(finite) < 2:
        return np.full_like(grid, np.nan, dtype=float)
    x_valid = x[finite]
    y_valid = y[finite]
    order = np.argsort(x_valid)
    return np.interp(grid, x_valid[order], y_valid[order], left=np.nan, right=np.nan)


def fft_pulse_response(
    *,
    t_fs: np.ndarray,
    E_MV_per_cm: np.ndarray,
    P_C_per_m2: np.ndarray,
    rho12: np.ndarray,
    window: str | None = "hann",
    subtract_mean: bool = True,
    rel_threshold: float = 1e-6,
    zero_padding_factor: int = 4,
) -> dict[str, np.ndarray]:
    """Compute pulse-derived response spectra.

    E is kept in MV/cm, so P_over_E has units:
        (C/m^2) / (MV/cm)

    For comparison with SI chi:
        P/E_MVcm = epsilon0 * chi * 1e8

    zero_padding_factor only interpolates the plotted spectrum. The true spectral
    resolution is still controlled by the total time window.
    """
    t_fs = np.asarray(t_fs, dtype=float)
    E_MV_per_cm = np.asarray(E_MV_per_cm, dtype=float)
    P_C_per_m2 = np.asarray(P_C_per_m2, dtype=np.complex128)
    rho12 = np.asarray(rho12, dtype=np.complex128)

    if t_fs.ndim != 1:
        raise ValueError("t_fs must be 1D.")
    if not (E_MV_per_cm.shape == P_C_per_m2.shape == rho12.shape == t_fs.shape):
        raise ValueError("t_fs, E, P, and rho12 must have the same shape.")

    dt = np.diff(t_fs)
    if dt.size == 0:
        raise ValueError("Need at least two time points.")
    if not np.allclose(dt, dt[0], rtol=1e-5, atol=1e-10):
        raise ValueError("FFT requires a uniformly sampled time axis.")

    E_signal = E_MV_per_cm.astype(np.complex128)
    P_signal = P_C_per_m2.astype(np.complex128)
    rho_signal = rho12.astype(np.complex128)

    if subtract_mean:
        E_signal = E_signal - np.mean(E_signal)
        P_signal = P_signal - np.mean(P_signal)
        rho_signal = rho_signal - np.mean(rho_signal)

    if zero_padding_factor < 1:
        raise ValueError("zero_padding_factor must be >= 1.")

    n_samples = t_fs.size
    n_fft_target = int(n_samples * zero_padding_factor)
    n_fft = 1 << int(np.ceil(np.log2(max(n_fft_target, n_samples))))

    E_fft = np.fft.fft(apply_window(E_signal, window), n=n_fft)
    P_fft = np.fft.fft(apply_window(P_signal, window), n=n_fft)
    rho_fft = np.fft.fft(apply_window(rho_signal, window), n=n_fft)

    freq_fs_inv = np.fft.fftfreq(n_fft, d=float(dt[0]))
    omega_fs_inv = 2.0 * np.pi * freq_fs_inv
    energy_eV = omega_fs_inv / ParaNormalizer.EV_TO_FS_INV

    P_over_E, valid_E = safe_complex_divide(P_fft, E_fft, rel_threshold=rel_threshold)
    rho_over_E, _ = safe_complex_divide(rho_fft, E_fft, rel_threshold=rel_threshold)

    # Positive optical frequencies only.
    pos = freq_fs_inv > 0
    mask = pos & valid_E

    return {
        "frequency_fs_inv": freq_fs_inv[mask],
        "omega_fs_inv": omega_fs_inv[mask],
        "energy_eV": energy_eV[mask],
        "E_fft": E_fft[mask],
        "P_fft": P_fft[mask],
        "rho12_fft": rho_fft[mask],
        "P_over_E": P_over_E[mask],
        "rho12_over_E": rho_over_E[mask],
        "abs_E_fft": np.abs(E_fft[mask]),
        "abs_rho12_fft": np.abs(rho_fft[mask]),
        "abs_rho12_over_E": np.abs(rho_over_E[mask]),
        "omega_Im_P_over_E": omega_fs_inv[mask] * np.imag(P_over_E[mask]),
    }


def make_params(
    *,
    variant: VariantConfig,
    energy_gap_eV: float,
    laser_energy_eV: float,
    t_start_fs: float,
    t_end_fs: float,
    dt_fs: float,
    Tphi_fs: float,
) -> NLevelPhysicalParams:
    return NLevelPhysicalParams(
        energies_eV=(0.0, energy_gap_eV),
        dipole_matrix_D=variant.dipole_matrix_D,
        field_MV_per_cm=variant.field_MV_per_cm,
        laser_energy_eV=laser_energy_eV,
        t_start_fs=t_start_fs,
        t_end_fs=t_end_fs,
        dt_fs=dt_fs,
        basis=("g", "e"),
        relaxation_channels=(
            RelaxationChannel(
                name="relaxation_1_to_0",
                from_level=1,
                to_level=0,
                T1_fs=variant.T1_fs,
            ),
        ),
        pure_dephasing_channels=(
            PureDephasingChannel(
                name="pure_dephasing_level_1",
                level=1,
                Tphi_fs=Tphi_fs,
            ),
        ),
        pulse_center_fs=0.0,
        pulse_sigma_fs=variant.pulse_sigma_fs,
        solver_mode="lab_exact",
        input_description=(
            "Weak narrow Gaussian carrier pulse for comparing time-domain "
            "P(w)/E(w) response with analytic two-level linear response."
        ),
        input_metadata={
            "variant": variant.name,
            "variant_label": variant.label,
            "variant_note": variant.note,
            "purpose": "weak-pulse absorption spectrum sensitivity check",
            "target_energy_window_eV": [1.4, 1.7],
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
    Tphi_fs: float,
    mu_ge_D: float,
    number_density_m3: float,
) -> dict[str, object]:
    print(f"Running variant: {variant.name}  ({variant.label})")

    params = make_params(
        variant=variant,
        energy_gap_eV=energy_gap_eV,
        laser_energy_eV=laser_energy_eV,
        t_start_fs=t_start_fs,
        t_end_fs=t_end_fs,
        dt_fs=dt_fs,
        Tphi_fs=Tphi_fs,
    )
    result = run_case(params)
    analysis = make_analysis_from_result(result)

    t_fs = analysis.time_fs()
    P_t = analysis.full_polarization_C_per_m2(number_density_m3=number_density_m3)
    E_t, input_name, input_unit = analysis.input_signal(kind="field")
    rho12_t = get_coherence(analysis, pair=(0, 1))

    response = fft_pulse_response(
        t_fs=t_fs,
        E_MV_per_cm=E_t,
        P_C_per_m2=P_t,
        rho12=rho12_t,
        window="hann",
        subtract_mean=True,
        rel_threshold=1e-5,
        zero_padding_factor=4,
    )

    energy_eV = response["energy_eV"]
    omega_fs_inv = response["omega_fs_inv"]

    gamma2_fs_inv = 0.5 / variant.T1_fs + 1.0 / Tphi_fs
    chi = chi_two_level_linear(
        omega_fs_inv=omega_fs_inv,
        omega_eg_fs_inv=energy_gap_eV * ParaNormalizer.EV_TO_FS_INV,
        mu_ge_D=mu_ge_D,
        gamma2_fs_inv=gamma2_fs_inv,
        number_density_m3=number_density_m3,
        population_difference=1.0,
    )
    theory_P_over_E_MVcm = EPSILON0_F_PER_M * chi * MV_PER_CM_TO_V_PER_M
    theory_abs_like = omega_fs_inv * np.imag(theory_P_over_E_MVcm)

    # With the numpy FFT convention used here and H_int = -mu E,
    # positive absorption-like signal is -omega * Im[P(w)/E(w)].
    # The raw omega * Im[P/E] is saved for sign diagnostics.
    time_domain_abs_like = -response["omega_Im_P_over_E"]

    time_dir = output_dir / "time_domain_csv"
    spec_dir = output_dir / "spectrum_csv"
    time_dir.mkdir(parents=True, exist_ok=True)
    spec_dir.mkdir(parents=True, exist_ok=True)

    time_table = np.column_stack(
        [
            t_fs,
            E_t,
            np.real(rho12_t),
            np.imag(rho12_t),
            np.abs(rho12_t),
            np.real(P_t),
            np.imag(P_t),
            np.abs(P_t),
        ]
    )
    time_header = (
        "time_fs,"
        "E_MV_per_cm,"
        "Re_rho12,"
        "Im_rho12,"
        "abs_rho12,"
        "Re_P_C_per_m2,"
        "Im_P_C_per_m2,"
        "abs_P_C_per_m2"
    )
    time_csv_path = time_dir / f"{variant.name}_time_domain_signals.csv"
    np.savetxt(time_csv_path, time_table, delimiter=",", header=time_header, comments="")

    spec_table = np.column_stack(
        [
            energy_eV,
            response["abs_E_fft"],
            response["abs_rho12_fft"],
            response["abs_rho12_over_E"],
            np.real(response["P_over_E"]),
            np.imag(response["P_over_E"]),
            response["omega_Im_P_over_E"],
            time_domain_abs_like,
            np.real(theory_P_over_E_MVcm),
            np.imag(theory_P_over_E_MVcm),
            theory_abs_like,
            normalize_for_shape(time_domain_abs_like),
            normalize_for_shape(theory_abs_like),
        ]
    )
    spec_header = (
        "energy_eV,"
        "abs_E_fft,"
        "abs_rho12_fft,"
        "abs_rho12_over_E,"
        "Re_P_over_E,"
        "Im_P_over_E,"
        "raw_omega_Im_P_over_E_time_domain,"
        "abs_like_minus_omega_Im_P_over_E_time_domain,"
        "Re_theory_P_over_E,"
        "Im_theory_P_over_E,"
        "omega_Im_P_over_E_linear_response,"
        "norm_time_domain_abs_like,"
        "norm_linear_response_abs_like"
    )
    spectrum_csv_path = spec_dir / f"{variant.name}_spectrum.csv"
    np.savetxt(spectrum_csv_path, spec_table, delimiter=",", header=spec_header, comments="")

    save_result_case(
        result,
        output_dir,
        output_preview=True,
        case_name=variant.name,
        example_name="cw_pulse_absorption_compare_variants",
        append_results_csv=True,
        save_populations_csv=False,
    )

    return {
        "variant": variant,
        "result": result,
        "time_fs": t_fs,
        "E_t": E_t,
        "P_t": P_t,
        "rho12_t": rho12_t,
        "response": response,
        "time_domain_abs_like": time_domain_abs_like,
        "theory_abs_like": theory_abs_like,
        "input_name": input_name,
        "input_unit": input_unit,
        "time_csv_path": time_csv_path,
        "spectrum_csv_path": spectrum_csv_path,
    }


def main():
    custom_plot_style = {
        "font.size": 20,
        "axes.titlesize": 22,
        "axes.labelsize": 20,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 14,
        "figure.titlesize": 24,
    }

    output_dir = Path("outputs/cw_pulse_absorption_compare")
    output_dir.mkdir(parents=True, exist_ok=True)

    energy_gap_eV = 1.55
    laser_energy_eV = 1.55
    mu_ge_D = 3.0
    number_density_m3 = 1.0e24

    base_field_MV_per_cm = 0.001
    base_pulse_sigma_fs = 6.0
    base_T1_fs = 500.0
    Tphi_fs = 300.0

    # Symmetric time window. A narrow pulse is used so that one time-domain run
    # covers the 1.4-1.7 eV response region.
    t_start_fs = -1000.0
    t_end_fs = 1000.0
    dt_fs = 0.05

    # Permanent dipoles are diagonal dipole elements. Only the difference matters
    # for the field coupling; here mu_gg=0 and mu_ee=1 D is a visible test case.
    permanent_mu_ee_D = 1.0

    variants = [
        VariantConfig(
            name="baseline",
            label="baseline",
            field_MV_per_cm=base_field_MV_per_cm,
            pulse_sigma_fs=base_pulse_sigma_fs,
            T1_fs=base_T1_fs,
            dipole_matrix_D=((0.0, mu_ge_D), (mu_ge_D, 0.0)),
            note="weak field, no permanent dipole, baseline pulse width and T1",
        ),
        VariantConfig(
            name="field_x10",
            label="E0 x10",
            field_MV_per_cm=10.0 * base_field_MV_per_cm,
            pulse_sigma_fs=base_pulse_sigma_fs,
            T1_fs=base_T1_fs,
            dipole_matrix_D=((0.0, mu_ge_D), (mu_ge_D, 0.0)),
            note="electric field amplitude multiplied by 10",
        ),
        VariantConfig(
            name="with_permanent_dipole",
            label="permanent dipole",
            field_MV_per_cm=base_field_MV_per_cm,
            pulse_sigma_fs=base_pulse_sigma_fs,
            T1_fs=base_T1_fs,
            dipole_matrix_D=((0.0, mu_ge_D), (mu_ge_D, permanent_mu_ee_D)),
            note="include diagonal permanent dipole: mu_gg=0 D, mu_ee=1 D",
        ),
        VariantConfig(
            name="wider_pulse",
            label="wider pulse",
            field_MV_per_cm=base_field_MV_per_cm,
            pulse_sigma_fs=2.0 * base_pulse_sigma_fs,
            T1_fs=base_T1_fs,
            dipole_matrix_D=((0.0, mu_ge_D), (mu_ge_D, 0.0)),
            note="pulse width sigma doubled; spectrum becomes narrower",
        ),
        VariantConfig(
            name="short_T1",
            label="short T1",
            field_MV_per_cm=base_field_MV_per_cm,
            pulse_sigma_fs=base_pulse_sigma_fs,
            T1_fs=100.0,
            dipole_matrix_D=((0.0, mu_ge_D), (mu_ge_D, 0.0)),
            note="population relaxation time shortened from 500 fs to 100 fs",
        ),
    ]

    records = []
    for variant in variants:
        record = run_variant(
            variant=variant,
            output_dir=output_dir,
            energy_gap_eV=energy_gap_eV,
            laser_energy_eV=laser_energy_eV,
            t_start_fs=t_start_fs,
            t_end_fs=t_end_fs,
            dt_fs=dt_fs,
            Tphi_fs=Tphi_fs,
            mu_ge_D=mu_ge_D,
            number_density_m3=number_density_m3,
        )
        records.append(record)

    e_min = 1.4
    e_max = 1.7
    common_energy = np.linspace(e_min, e_max, 1201)

    combined_columns = [common_energy]
    combined_header = ["energy_eV"]

    for record in records:
        variant = record["variant"]
        response = record["response"]
        energy_eV = response["energy_eV"]
        td_abs_norm = normalize_for_shape(record["time_domain_abs_like"])
        coherence_norm = normalize_for_shape(response["abs_rho12_over_E"])
        input_norm = normalize_for_shape(response["abs_E_fft"])
        theory_norm = normalize_for_shape(record["theory_abs_like"])

        combined_columns.extend(
            [
                interp_to_grid(energy_eV, td_abs_norm, common_energy),
                interp_to_grid(energy_eV, coherence_norm, common_energy),
                interp_to_grid(energy_eV, input_norm, common_energy),
                interp_to_grid(energy_eV, theory_norm, common_energy),
            ]
        )
        combined_header.extend(
            [
                f"{variant.name}_norm_abs_like",
                f"{variant.name}_norm_abs_rho12_over_E",
                f"{variant.name}_norm_abs_E_fft",
                f"{variant.name}_norm_theory_abs_like",
            ]
        )

    combined_csv_path = output_dir / "combined_spectra_1p4_1p7_eV.csv"
    np.savetxt(
        combined_csv_path,
        np.column_stack(combined_columns),
        delimiter=",",
        header=",".join(combined_header),
        comments="",
    )

    # Use the baseline record for the time-domain pulse/coherence panel.
    baseline_record = records[0]
    t_fs = baseline_record["time_fs"]
    base_sigma = baseline_record["variant"].pulse_sigma_fs
    time_window = (t_fs >= -8.0 * base_sigma) & (t_fs <= 8.0 * base_sigma)

    with plt.rc_context(custom_plot_style):
        fig, axes = plt.subplots(4, 1, figsize=(15.0, 22.0), sharex=False)

        # (1) Baseline time-domain pulse and coherence.
        axes[0].plot(
            t_fs[time_window],
            normalize_for_shape(baseline_record["E_t"][time_window]),
            label="baseline norm E(t)",
            linewidth=2.5,
        )
        axes[0].plot(
            t_fs[time_window],
            normalize_for_shape(np.real(baseline_record["rho12_t"][time_window])),
            label="baseline norm Re[rho12(t)]",
            linewidth=2.0,
        )
        axes[0].plot(
            t_fs[time_window],
            normalize_for_shape(np.imag(baseline_record["rho12_t"][time_window])),
            label="baseline norm Im[rho12(t)]",
            linewidth=2.0,
        )
        axes[0].plot(
            t_fs[time_window],
            normalize_for_shape(np.abs(baseline_record["rho12_t"][time_window])),
            label="baseline norm |rho12(t)|",
            linewidth=2.0,
        )
        axes[0].set_title("Time domain pulse")
        axes[0].set_xlabel("Time (fs)")
        axes[0].set_ylabel("Normalized amplitude")
        axes[0].legend(loc="best")

        # (2) Input spectra together.
        for record in records:
            variant = record["variant"]
            response = record["response"]
            band = (response["energy_eV"] >= e_min) & (response["energy_eV"] <= e_max)
            axes[1].plot(
                response["energy_eV"][band],
                normalize_for_shape(response["abs_E_fft"][band]),
                label=variant.label,
                linewidth=2.3,
            )
        axes[1].set_title("Input spectra together")
        axes[1].set_xlabel("Energy (eV)")
        axes[1].set_ylabel("norm |E(w)|")
        axes[1].legend(loc="best")

        # (3) Coherence response spectra together.
        for record in records:
            variant = record["variant"]
            response = record["response"]
            band = (response["energy_eV"] >= e_min) & (response["energy_eV"] <= e_max)
            axes[2].plot(
                response["energy_eV"][band],
                normalize_for_shape(response["abs_rho12_over_E"][band]),
                label=variant.label,
                linewidth=2.3,
            )
        axes[2].set_title("Coherence response spectra together")
        axes[2].set_xlabel("Energy (eV)")
        axes[2].set_ylabel("norm |rho12(w)| / |E(w)|")
        axes[2].legend(loc="best")

        # (4) Absorption-like spectra together.
        for record in records:
            variant = record["variant"]
            response = record["response"]
            band = (response["energy_eV"] >= e_min) & (response["energy_eV"] <= e_max)
            axes[3].plot(
                response["energy_eV"][band],
                normalize_for_shape(record["time_domain_abs_like"][band]),
                label=variant.label,
                linewidth=2.3,
            )

        baseline = records[0]
        baseline_response = baseline["response"]
        band = (baseline_response["energy_eV"] >= e_min) & (baseline_response["energy_eV"] <= e_max)
        axes[3].plot(
            baseline_response["energy_eV"][band],
            normalize_for_shape(baseline["theory_abs_like"][band]),
            "--",
            label="linear theory baseline",
            linewidth=2.5,
        )
        short_t1 = next(record for record in records if record["variant"].name == "short_T1")
        short_response = short_t1["response"]
        band = (short_response["energy_eV"] >= e_min) & (short_response["energy_eV"] <= e_max)
        axes[3].plot(
            short_response["energy_eV"][band],
            normalize_for_shape(short_t1["theory_abs_like"][band]),
            "--",
            label="linear theory short T1",
            linewidth=2.5,
        )
        axes[3].set_title("Absorption-like spectra together")
        axes[3].set_xlabel("Energy (eV)")
        axes[3].set_ylabel("norm -w Im[P/E]")
        axes[3].legend(loc="best")

        fig.tight_layout()
        spectra_png_path = output_dir / "cw_pulse_absorption_all_spectra.png"
        fig.savefig(spectra_png_path, dpi=180)
        plt.close(fig)

    print(f"Saved combined spectra CSV : {combined_csv_path}")
    print(f"Saved all-spectra figure   : {spectra_png_path}")
    print(f"Saved result cases root    : {output_dir / 'res_per_case'}")
    print("Done.")


if __name__ == "__main__":
    main()
