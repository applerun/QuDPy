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
from sjh_learn.utils.core.normalization import ParaNormalizer
from sjh_learn.utils.analysis import DynamicsAnalysis
from sjh_learn.utils.analysis.observables import EPSILON0_F_PER_M, chi_two_level_linear
from sjh_learn.utils.io import save_result_case


MV_PER_CM_TO_V_PER_M = 1.0e8
DEBYE_TO_C_M = 3.33564e-30
FORCE_RERUN = True



@dataclass(frozen=True)
class VariantConfig:
    name: str
    label: str
    field_MV_per_cm: float
    pulse_sigma_fs: float
    T1_fs: Optional[float]
    Tphi_fs: Optional[float]
    dipole_matrix_D: tuple[tuple[float, float], tuple[float, float]]
    baseline_delta_text: str
    note: str
    plot_e_min: float = 1.50
    plot_e_max: float = 1.60


def make_analysis_from_result(result):
    if hasattr(DynamicsAnalysis, "from_result"):
        return DynamicsAnalysis.from_result(result)
    return DynamicsAnalysis.from_dynamics_res(result)


def get_coherence(analysis: DynamicsAnalysis, pair=(0, 1)) -> np.ndarray:
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
    denominator_abs = np.abs(denominator)
    max_denominator = float(np.max(denominator_abs))
    if max_denominator == 0.0:
        raise ValueError("The input spectrum is identically zero.")
    valid = denominator_abs > rel_threshold * max_denominator
    ratio = np.full_like(numerator, np.nan + 1j * np.nan, dtype=np.complex128)
    ratio[valid] = numerator[valid] / denominator[valid]
    return ratio, valid


def normalize_for_shape(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y)
    if np.iscomplexobj(y):
        y = np.real(y)
    y = y.astype(float)
    finite = np.isfinite(y)
    if not np.any(finite):
        return y
    scale = np.nanmax(np.abs(y[finite]))
    if scale == 0.0:
        return y
    return y / scale


def sorted_xy_for_plot(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if x.size == 0:
        return x, y
    order = np.argsort(x)
    return x[order], y[order]


def plot_normalized_curve(ax, x, y, *, label: str, **plot_kwargs) -> np.ndarray:
    x, y = sorted_xy_for_plot(x, y)
    if x.size == 0:
        return np.array([], dtype=float)
    y_norm = normalize_for_shape(y)
    ax.plot(x, y_norm, label=label, **plot_kwargs)
    return y_norm


def add_top_omega_axis(ax,x_pos = 0.13 ,y_pos = 0.92):
    secax = ax.secondary_xaxis(
        "top",
        functions=(
            lambda energy_eV: energy_eV * ParaNormalizer.EV_TO_FS_INV,
            lambda omega_fs_inv: omega_fs_inv / ParaNormalizer.EV_TO_FS_INV,
        ),
    )
    secax.set_xlabel("")
    secax.tick_params(axis="x", pad=1)
    # Keep this label attached to the top x-axis, but move it away from the left y-axis.
    ax.text(
        x_pos,
        y_pos,
        "Ang. Freq. (fs$^{-1}$)",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=12,
        clip_on=False,
    )
    return secax

def set_energy_axis(ax, e_min: float, e_max: float, *, n_ticks: int = 3):
    ax.set_xlim(e_min, e_max)
    ax.set_xticks(np.linspace(e_min, e_max, n_ticks))


def result_ckp_path(output_dir: Path, case_name: str) -> Path:
    return output_dir / "res_per_case" / case_name / "data" / "result.ckp"


def set_axis_ylim_from_curves(ax, curves: list[np.ndarray], *, min_headroom: float = 0.12):
    finite_values = []
    for curve in curves:
        curve = np.asarray(curve, dtype=float)
        finite = curve[np.isfinite(curve)]
        if finite.size > 0:
            finite_values.append(finite)
    if not finite_values:
        return
    all_values = np.concatenate(finite_values)
    y_min = float(np.min(all_values))
    y_max = float(np.max(all_values))
    if y_min == y_max:
        if y_max == 0.0:
            y_min, y_max = -1.0, 1.0
        else:
            pad = 0.1 * abs(y_max)
            y_min -= pad
            y_max += pad
    span = y_max - y_min
    pad = max(min_headroom * span, 0.05)
    ax.set_ylim(y_min - pad, y_max + pad)


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
    t_fs = np.asarray(t_fs, dtype=float)
    E_MV_per_cm = np.asarray(E_MV_per_cm, dtype=float)
    P_C_per_m2 = np.asarray(P_C_per_m2, dtype=np.complex128)
    rho12 = np.asarray(rho12, dtype=np.complex128)

    dt = np.diff(t_fs)
    if dt.size == 0 or not np.allclose(dt, dt[0], rtol=1e-5, atol=1e-10):
        raise ValueError("FFT requires a uniformly sampled time axis with at least two points.")

    E_signal = E_MV_per_cm.astype(np.complex128)
    P_signal = P_C_per_m2.astype(np.complex128)
    rho_signal = rho12.astype(np.complex128)
    if subtract_mean:
        E_signal = E_signal - np.mean(E_signal)
        P_signal = P_signal - np.mean(P_signal)
        rho_signal = rho_signal - np.mean(rho_signal)

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

    pos = freq_fs_inv > 0
    mask = pos & valid_E

    return {
        "omega_fs_inv": omega_fs_inv[mask],
        "energy_eV": energy_eV[mask],
        "E_fft": E_fft[mask],
        "P_fft": P_fft[mask],
        "rho12_fft": rho_fft[mask],
        "P_over_E": P_over_E[mask],
        "rho12_over_E": rho_over_E[mask],
        "abs_E_fft": np.abs(E_fft[mask]),
        "abs_rho12_over_E": np.abs(rho_over_E[mask]),
        "im_rho12_over_E": np.imag(rho_over_E[mask]),
        "omega_im_rho12_over_E": omega_fs_inv[mask] * np.imag(rho_over_E[mask]),
        "neg_omega_im_P_over_E": -omega_fs_inv[mask] * np.imag(P_over_E[mask]),
    }


def fft_rwa_response(
    *,
    t_fs: np.ndarray,
    g_fs_inv: np.ndarray,
    rho12_rwa: np.ndarray,
    laser_energy_eV: float,
    window: str | None = "hann",
    subtract_mean: bool = False,
    rel_threshold: float = 1e-6,
    zero_padding_factor: int = 4,
) -> dict[str, np.ndarray]:
    t_fs = np.asarray(t_fs, dtype=float)
    g_fs_inv = np.asarray(g_fs_inv, dtype=float)
    rho12_rwa = np.asarray(rho12_rwa, dtype=np.complex128)

    dt = np.diff(t_fs)
    if dt.size == 0 or not np.allclose(dt, dt[0], rtol=1e-5, atol=1e-10):
        raise ValueError("FFT requires a uniformly sampled time axis with at least two points.")

    g_signal = g_fs_inv.astype(np.complex128)
    rho_signal = rho12_rwa.astype(np.complex128)
    if subtract_mean:
        g_signal = g_signal - np.mean(g_signal)
        rho_signal = rho_signal - np.mean(rho_signal)

    n_samples = t_fs.size
    n_fft_target = int(n_samples * zero_padding_factor)
    n_fft = 1 << int(np.ceil(np.log2(max(n_fft_target, n_samples))))

    g_fft = np.fft.fft(apply_window(g_signal, window), n=n_fft)
    rho_fft = np.fft.fft(apply_window(rho_signal, window), n=n_fft)

    freq_offset_fs_inv = np.fft.fftfreq(n_fft, d=float(dt[0]))
    omega_offset_fs_inv = 2.0 * np.pi * freq_offset_fs_inv
    energy_eV = laser_energy_eV + omega_offset_fs_inv / ParaNormalizer.EV_TO_FS_INV

    rho_over_g, valid_g = safe_complex_divide(rho_fft, g_fft, rel_threshold=rel_threshold)
    idx = np.where(valid_g)[0]

    return {
        "omega_offset_fs_inv": omega_offset_fs_inv[idx],
        "energy_eV": energy_eV[idx],
        "g_fft": g_fft[idx],
        "rho12_rwa_over_g": rho_over_g[idx],
        "abs_g_fft": np.abs(g_fft[idx]),
        "abs_rho12_rwa_over_g": np.abs(rho_over_g[idx]),
        "im_rho12_rwa_over_g": np.imag(rho_over_g[idx]),
        "omega_im_rho12_rwa_over_g": omega_offset_fs_inv[idx] * np.imag(rho_over_g[idx]),
    }



def reconstruct_rwa_lab_polarization_C_per_m2(
    *,
    t_fs: np.ndarray,
    rho12_rwa: np.ndarray,
    variant: VariantConfig,
    number_density_m3: float,
    laser_energy_eV: float,
    carrier_sign: int,
) -> np.ndarray:
    """Reconstruct a lab-frame polarization from the RWA coherence.

    This is a diagnostic quantity. RWA stores a slow coherence, so to compare it
    with lab-frame P(w)/E(w), we multiply the optical carrier back and then use
    the off-diagonal dipole contribution:

        P(t) ~= N * mu_ge * [rho_ge^lab(t) + conj(rho_ge^lab(t))].

    Diagonal permanent-dipole contributions are not reconstructed here because
    they require population terms and mainly live near zero/baseband frequency.
    """
    if carrier_sign not in (-1, 1):
        raise ValueError("carrier_sign must be -1 or +1.")

    t_fs = np.asarray(t_fs, dtype=float)
    rho12_rwa = np.asarray(rho12_rwa, dtype=np.complex128)
    mu_ge_D = float(variant.dipole_matrix_D[0][1])
    omega_L_fs_inv = laser_energy_eV * ParaNormalizer.EV_TO_FS_INV

    rho12_lab_like = rho12_rwa * np.exp(1j * carrier_sign * omega_L_fs_inv * t_fs)
    polarization = (
        float(number_density_m3)
        * DEBYE_TO_C_M
        * mu_ge_D
        * (rho12_lab_like + np.conjugate(rho12_lab_like))
    )
    return np.asarray(polarization.real, dtype=float)


def normalized_correlation(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    finite = np.isfinite(a) & np.isfinite(b)
    if np.count_nonzero(finite) < 3:
        return -np.inf
    aa = normalize_for_shape(a[finite])
    bb = normalize_for_shape(b[finite])
    aa = aa - np.mean(aa)
    bb = bb - np.mean(bb)
    denom = float(np.sqrt(np.sum(aa * aa) * np.sum(bb * bb)))
    if denom == 0.0:
        return -np.inf
    return float(np.sum(aa * bb) / denom)


def choose_rwa_reconstructed_p_over_e_response(
    *,
    t_fs: np.ndarray,
    E_t: np.ndarray,
    rho12_rwa_t: np.ndarray,
    rho12_lab_t: np.ndarray,
    variant: VariantConfig,
    number_density_m3: float,
    laser_energy_eV: float,
    lab_reference_response: dict[str, np.ndarray],
    e_min: float = 1.4,
    e_max: float = 1.7,
) -> tuple[np.ndarray, dict[str, np.ndarray], str]:
    """Try both carrier signs and keep the one closer to lab -w Im[P/E]."""
    best_score = -np.inf
    best_P = None
    best_response = None
    best_label = "unknown carrier sign"

    reference = lab_reference_response["neg_omega_im_P_over_E"]
    for sign in (-1, 1):
        P_recon = reconstruct_rwa_lab_polarization_C_per_m2(
            t_fs=t_fs,
            rho12_rwa=rho12_rwa_t,
            variant=variant,
            number_density_m3=number_density_m3,
            laser_energy_eV=laser_energy_eV,
            carrier_sign=sign,
        )
        response = fft_pulse_response(
            t_fs=t_fs,
            E_MV_per_cm=E_t,
            P_C_per_m2=P_recon,
            rho12=rho12_lab_t,
            window="hann",
            subtract_mean=True,
            rel_threshold=1e-5,
            zero_padding_factor=4,
        )
        band = (response["energy_eV"] >= e_min) & (response["energy_eV"] <= e_max)
        score = normalized_correlation(
            response["neg_omega_im_P_over_E"][band],
            reference[band],
        )
        if score > best_score:
            best_score = score
            best_P = P_recon
            best_response = response
            best_label = f"rho12_RWA * exp({sign:+d} i omega_L t), corr={score:.3f}"

    if best_P is None or best_response is None:
        raise RuntimeError("Failed to reconstruct RWA lab-frame polarization.")
    return best_P, best_response, best_label

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


def gamma2_fs_inv_from_variant(variant: VariantConfig) -> float:
    gamma2 = 0.0
    if variant.T1_fs is not None:
        gamma2 += 0.5 / variant.T1_fs
    if variant.Tphi_fs is not None:
        gamma2 += 1.0 / variant.Tphi_fs
    return gamma2


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
    lab_params = make_params(
        variant=variant,
        energy_gap_eV=energy_gap_eV,
        laser_energy_eV=laser_energy_eV,
        t_start_fs=t_start_fs,
        t_end_fs=t_end_fs,
        dt_fs=dt_fs,
        solver_mode="lab_exact",
    )
    rwa_params = replace(lab_params, solver_mode="rwa")

    lab_case_name = f"{variant.name}_lab_exact"
    rwa_case_name = f"{variant.name}_rwa"
    lab_ckp = result_ckp_path(output_dir, lab_case_name)
    rwa_ckp = result_ckp_path(output_dir, rwa_case_name)

    lab_result = run_case(
        lab_params,
        load_ckp=lab_ckp,
        force_run=FORCE_RERUN,
    )
    rwa_result = run_case(
        rwa_params,
        load_ckp=rwa_ckp,
        force_run=FORCE_RERUN,
    )
    lab_analysis = make_analysis_from_result(lab_result)
    rwa_analysis = make_analysis_from_result(rwa_result)

    t_fs = lab_analysis.time_fs()
    P_t = lab_analysis.full_polarization_C_per_m2(number_density_m3=number_density_m3)
    E_t, _, _ = lab_analysis.input_signal(kind="field")
    rho12_lab_t = get_coherence(lab_analysis, pair=(0, 1))

    t_rwa_fs = rwa_analysis.time_fs()
    if not np.allclose(t_fs, t_rwa_fs, rtol=1e-8, atol=1e-10):
        raise ValueError("lab_exact and RWA time axes do not match.")
    g_t, _, _ = rwa_analysis.input_signal(kind="drive")
    rho12_rwa_t = get_coherence(rwa_analysis, pair=(0, 1))

    lab_response = fft_pulse_response(
        t_fs=t_fs,
        E_MV_per_cm=E_t,
        P_C_per_m2=P_t,
        rho12=rho12_lab_t,
        window="hann",
        subtract_mean=True,
        rel_threshold=1e-5,
        zero_padding_factor=4,
    )
    rwa_response = fft_rwa_response(
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
        variant=variant,
        number_density_m3=number_density_m3,
        laser_energy_eV=laser_energy_eV,
        lab_reference_response=lab_response,
        e_min=variant.plot_e_min,
        e_max=variant.plot_e_max,
    )

    omega_fs_inv = lab_response["omega_fs_inv"]
    gamma2_fs_inv = gamma2_fs_inv_from_variant(variant)
    chi = chi_two_level_linear(
        omega_fs_inv=omega_fs_inv,
        omega_eg_fs_inv=energy_gap_eV * ParaNormalizer.EV_TO_FS_INV,
        mu_ge_D=mu_ge_D,
        gamma2_fs_inv=gamma2_fs_inv,
        number_density_m3=number_density_m3,
        population_difference=1.0,
    )
    theory_P_over_E_MVcm = EPSILON0_F_PER_M * chi * MV_PER_CM_TO_V_PER_M
    theory_abs_like = -omega_fs_inv * np.imag(theory_P_over_E_MVcm)

    save_result_case(
        lab_result,
        output_dir,
        output_preview=True,
        case_name=lab_case_name,
        example_name="cw_pulse_absorption_casewise_rwa",
        append_results_csv=True,
        save_populations_csv=False,
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

    time_dir = output_dir / "time_domain_csv"
    spec_dir = output_dir / "spectrum_csv"
    time_dir.mkdir(parents=True, exist_ok=True)
    spec_dir.mkdir(parents=True, exist_ok=True)

    time_table = np.column_stack([
        t_fs,
        E_t,
        g_t,
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
        header="time_fs,E_MV_per_cm,g_rwa_fs_inv,Re_rho12_lab,Im_rho12_lab,Re_rho12_rwa,Im_rho12_rwa,Re_P_C_per_m2,Im_P_C_per_m2,P_rwa_reconstructed_C_per_m2",
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
        header="energy_eV,abs_E_fft,abs_rho12_over_E,im_rho12_over_E,omega_im_rho12_over_E,neg_omega_im_P_over_E,linear_theory_neg_omega_im_P_over_E",
        comments="",
    )
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
        curves.append(plot_normalized_curve(ax_resp, lab_response["energy_eV"][band_resp], -theory_abs_like[band_resp], label="linear theory", linewidth=2.4, color="black", linestyle="-"))
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
        ax_input_t.plot(t_fs[time_window], E_t[time_window], linewidth=2.0, label="lab E(t)", color="C0")
        ax_right.plot(t_fs[time_window], g_t[time_window], linestyle="--", linewidth=2.0, label="RWA g(t)", color="C1")
        ax_input_t.set_title("Input t", pad=8)
        ax_input_t.set_xlabel("Time (fs)", labelpad=2)
        ax_input_t.set_ylabel("E(t) [MV/cm]", labelpad=2)
        ax_right.set_ylabel("g(t) [fs$^{-1}$]", labelpad=2)
        E_max = float(np.nanmax(np.abs(E_t[time_window]))) if np.any(time_window) else float(np.nanmax(np.abs(E_t)))
        g_max = float(np.nanmax(np.abs(g_t[time_window]))) if np.any(time_window) else float(np.nanmax(np.abs(g_t)))
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
        curves.append(plot_normalized_curve(ax_theory, lab_response["energy_eV"][lab_band], -record["theory_abs_like"][lab_band], label="linear theory", linewidth=2.3, color="black", linestyle="--"))
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
    output_dir = Path("outputs/cw_pulse_absorption_compare")
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
            name="field_x10",
            label="field ×10",
            field_MV_per_cm=10.0 * base_field_MV_per_cm,
            pulse_sigma_fs=base_pulse_sigma_fs,
            T1_fs=base_T1_fs,
            Tphi_fs=base_Tphi_fs,
            dipole_matrix_D=((0.0, mu_ge_D), (mu_ge_D, 0.0)),
            baseline_delta_text=f"E0: {base_field_MV_per_cm:.4g} → {10.0 * base_field_MV_per_cm:.4g} MV/cm (×10, +900%)",
            note="Only the field amplitude is changed relative to baseline.",
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
            plot_e_min=0.85,
            plot_e_max=2.25,
        ),
    ]

    lab_only_paths = []
    lab_vs_rwa_paths = []
    for variant in variants:
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
        lab_vs_rwa_paths.append(
            build_lab_vs_rwa_figure(
                record,
                output_dir=output_dir,
                e_min=variant.plot_e_min,
                e_max=variant.plot_e_max,
            )
        )

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
