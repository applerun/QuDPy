from __future__ import annotations

import csv
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from sjh_learn.examples.cw_input.rwa_common import (
    T1_fs_of,
    Tphi_fs_of,
    collect_summary_metrics,
    save_case_result,
    save_results_csv,
)
from sjh_learn.examples.gau_pulse.pulse_common import (
    apply_dissipation,
    case_name_from_pulse,
    dissipation_scenarios,
    make_base_gaussian_physical_params,
    pulse_summary_metrics,
)
from sjh_learn.utils import GaussianCarrierFieldPhysical, NLevelPhysicalParams, ParaNormalizer, run_physical_case


HBAR_EV_FS = 0.6582119569
HC_EV_NM = 1239.841984


def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def make_lab_gaussian_field(physical: NLevelPhysicalParams) -> GaussianCarrierFieldPhysical:
    """从同一个 physical 参数生成 lab-frame Gaussian carrier field。

    这里的公式与 lab-frame exact solver 的物理约定一致：
    `E(t) = 2 E0 exp[-(t-t0)^2/(2 sigma^2)] cos(omega_L t + phase)`。
    FFT 分析使用这个 physical field 采样，避免另写一套不同的光场。
    """

    if physical.pulse_center_fs is None or physical.pulse_sigma_fs is None:
        raise ValueError("lab Gaussian pulse example requires pulse_center_fs and pulse_sigma_fs.")
    return GaussianCarrierFieldPhysical(
        E0_MV_per_cm=physical.field_MV_per_cm,
        pulse_center_fs=physical.pulse_center_fs,
        pulse_sigma_fs=physical.pulse_sigma_fs,
        laser_energy_eV=physical.laser_energy_eV,
        phase_rad=0.0,
    )


def compute_fft_spectrum(t_fs, signal, apply_window: bool = True) -> dict[str, np.ndarray | bool | str | float | int]:
    """计算正频率 FFT 谱，并给出 Hz、fs^-1、eV、nm 坐标。

    `t_fs` 单位是 fs，必须等间隔。默认 Hann window 用于降低有限时间窗导致的
    spectral leakage；metadata 中会记录 window 设置。
    """

    times = np.asarray(t_fs, dtype=float)
    values = np.asarray(signal, dtype=np.complex128)
    if times.ndim != 1:
        raise ValueError("t_fs must be a one-dimensional array.")
    if values.ndim != 1 or len(values) != len(times):
        raise ValueError("signal must be one-dimensional and have the same length as t_fs.")
    if len(times) < 2:
        raise ValueError("FFT requires at least two time points.")

    dt_values = np.diff(times)
    dt_fs = float(dt_values[0])
    if dt_fs <= 0:
        raise ValueError("t_fs must be strictly increasing.")
    if not np.allclose(dt_values, dt_fs, rtol=1e-9, atol=1e-12):
        raise ValueError("t_fs must be evenly spaced for FFT.")

    window = np.hanning(len(values)) if apply_window else np.ones(len(values), dtype=float)
    fft_values = np.fft.fft(values * window)
    frequency_Hz = np.fft.fftfreq(len(values), d=dt_fs * 1.0e-15)
    positive = frequency_Hz > 0
    frequency_Hz = frequency_Hz[positive]
    fft_values = fft_values[positive]
    omega_fs_inv = 2.0 * np.pi * frequency_Hz * 1.0e-15
    energy_eV = HBAR_EV_FS * omega_fs_inv
    wavelength_nm = np.full_like(energy_eV, np.nan, dtype=float)
    energy_positive = energy_eV > 0
    wavelength_nm[energy_positive] = HC_EV_NM / energy_eV[energy_positive]

    return {
        "frequency_Hz": frequency_Hz,
        "omega_fs_inv": omega_fs_inv,
        "energy_eV": energy_eV,
        "wavelength_nm": wavelength_nm,
        "fft": fft_values,
        "dt_fs": dt_fs,
        "n_time_points": len(times),
        "total_time_window_fs": float(times[-1] - times[0]),
        "apply_window": bool(apply_window),
        "window": "hann" if apply_window else "none",
    }


def _phase(values: np.ndarray) -> np.ndarray:
    phase = np.angle(values).astype(float)
    phase[np.isnan(values.real) | np.isnan(values.imag)] = np.nan
    return phase


def compute_response_spectrum(
    *,
    t_fs,
    electric_field_MV_per_cm,
    rho_01,
    response_threshold: float = 1e-6,
    apply_window: bool = True,
) -> dict[str, Any]:
    if response_threshold <= 0:
        raise ValueError("response_threshold must be positive.")

    field_spec = compute_fft_spectrum(t_fs, electric_field_MV_per_cm, apply_window=apply_window)
    rho_spec = compute_fft_spectrum(t_fs, rho_01, apply_window=apply_window)
    E_fft = np.asarray(field_spec["fft"], dtype=np.complex128)
    rho_fft = np.asarray(rho_spec["fft"], dtype=np.complex128)
    max_field = float(np.max(np.abs(E_fft)))
    if max_field == 0:
        raise ValueError("max(abs(E_fft)) is zero; cannot compute response-like spectrum.")
    valid = np.abs(E_fft) > response_threshold * max_field
    response = np.full(E_fft.shape, np.nan + 1j * np.nan, dtype=np.complex128)
    response[valid] = rho_fft[valid] / E_fft[valid]
    return {
        "frequency_Hz": field_spec["frequency_Hz"],
        "omega_fs_inv": field_spec["omega_fs_inv"],
        "energy_eV": field_spec["energy_eV"],
        "wavelength_nm": field_spec["wavelength_nm"],
        "E_fft": E_fft,
        "rho_01_fft": rho_fft,
        "rho_01_over_E": response,
        "response_valid_mask": valid,
        "metadata": {
            "apply_window": field_spec["apply_window"],
            "window": field_spec["window"],
            "dt_fs": field_spec["dt_fs"],
            "n_time_points": field_spec["n_time_points"],
            "total_time_window_fs": field_spec["total_time_window_fs"],
            "fft_response_threshold": response_threshold,
            "positive_frequency_only": True,
            "rho_element_used": "rho_01",
            "field_source": "lab-frame Gaussian carrier field",
            "warning": (
                "rho_01_fft / E_fft is a response-like quantity, not a calibrated susceptibility "
                "unless polarization normalization, number density, epsilon0, and Fourier convention are handled."
            ),
        },
    }


def save_fft_spectrum_csv(spectrum: dict[str, Any], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    E_fft = spectrum["E_fft"]
    rho_fft = spectrum["rho_01_fft"]
    response = spectrum["rho_01_over_E"]
    rows = []
    for index in range(len(spectrum["frequency_Hz"])):
        rows.append(
            {
                "frequency_Hz": float(spectrum["frequency_Hz"][index]),
                "omega_fs_inv": float(spectrum["omega_fs_inv"][index]),
                "energy_eV": float(spectrum["energy_eV"][index]),
                "wavelength_nm": float(spectrum["wavelength_nm"][index]),
                "Re_E_fft": float(E_fft[index].real),
                "Im_E_fft": float(E_fft[index].imag),
                "abs_E_fft": float(abs(E_fft[index])),
                "phase_E_fft": float(np.angle(E_fft[index])),
                "Re_rho_01_fft": float(rho_fft[index].real),
                "Im_rho_01_fft": float(rho_fft[index].imag),
                "abs_rho_01_fft": float(abs(rho_fft[index])),
                "phase_rho_01_fft": float(np.angle(rho_fft[index])),
                "Re_rho_01_over_E": float(response[index].real),
                "Im_rho_01_over_E": float(response[index].imag),
                "abs_rho_01_over_E": float(abs(response[index])),
                "phase_rho_01_over_E": float(_phase(response)[index]),
                "response_valid_mask": bool(spectrum["response_valid_mask"][index]),
            }
        )

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def save_fft_spectrum_plot(spectrum: dict[str, Any], output_path: str | Path, *, title: str) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    energy = np.asarray(spectrum["energy_eV"], dtype=float)
    positive_energy = energy > 0
    valid_response = positive_energy & np.asarray(spectrum["response_valid_mask"], dtype=bool)

    fig, axes = plt.subplots(3, 1, figsize=(7.2, 7.2), sharex=True)
    axes[0].plot(energy[positive_energy], np.abs(spectrum["E_fft"])[positive_energy])
    axes[0].set_ylabel("|FFT[E(t)]|")
    axes[1].plot(energy[positive_energy], np.abs(spectrum["rho_01_fft"])[positive_energy])
    axes[1].set_ylabel("|FFT[rho_01(t)]|")
    axes[2].plot(energy[valid_response], np.abs(spectrum["rho_01_over_E"])[valid_response])
    axes[2].set_ylabel("|FFT[rho_01(t)] / FFT[E(t)]|")
    axes[2].set_xlabel("Energy (eV)")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def save_fft_meta(spectrum: dict[str, Any], output_path: str | Path, *, case_name: str, physical) -> Path:
    path = Path(output_path)
    payload = {
        "case_name": case_name,
        **spectrum["metadata"],
        "field_MV_per_cm": physical.field_MV_per_cm,
        "laser_energy_eV": physical.laser_energy_eV,
        "pulse_center_fs": physical.pulse_center_fs,
        "pulse_sigma_fs": physical.pulse_sigma_fs,
    }
    path.write_text(json.dumps(_json_safe(payload), indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def save_lab_fft_outputs(
    result,
    case_dir: str | Path,
    *,
    case_name: str,
    response_threshold: float = 1e-6,
    apply_window: bool = True,
) -> dict[str, Path]:
    physical = result.physical_params
    if physical is None:
        raise ValueError("lab FFT outputs require result.physical_params.")
    if result.dimension() != 2:
        raise ValueError("lab Gaussian pulse FFT example currently expects N=2.")
    if result.times_fs is None:
        raise ValueError("lab FFT outputs require result.times_fs.")

    t_fs = np.asarray(result.times_fs, dtype=float)
    field = make_lab_gaussian_field(physical)
    electric_field = np.asarray(field(t_fs), dtype=float)
    rho_01 = result.matrix_element(0, 1)
    spectrum = compute_response_spectrum(
        t_fs=t_fs,
        electric_field_MV_per_cm=electric_field,
        rho_01=rho_01,
        response_threshold=response_threshold,
        apply_window=apply_window,
    )

    root = Path(case_dir)
    written = {
        "fft_spectrum_csv": save_fft_spectrum_csv(spectrum, root / "fft_spectrum.csv"),
        "fft_spectrum_png": save_fft_spectrum_plot(
            spectrum,
            root / "fft_spectrum.png",
            title=f"FFT-based response-like spectrum: {case_name}",
        ),
        "fft_meta": save_fft_meta(spectrum, root / "fft_meta.json", case_name=case_name, physical=physical),
    }
    return written


def run_lab_pulse_case(
    physical: NLevelPhysicalParams,
    *,
    case_name: str,
    condition_name: str,
    example_name: str,
    output_dir: str | Path,
    preview: bool = True,
    response_threshold: float = 1e-6,
    apply_window: bool = True,
):
    normalizer = ParaNormalizer(time_scale_fs=1.0, auto_scale=False)
    result = run_physical_case(physical, normalizer=normalizer)
    saved = save_case_result(
        result,
        output_dir,
        preview=preview,
        case_name=case_name,
        example_name=example_name,
        condition_name=condition_name,
        save_populations_csv=False,
    )
    save_lab_fft_outputs(
        result,
        saved["case_dir"],
        case_name=case_name,
        response_threshold=response_threshold,
        apply_window=apply_window,
    )
    return result


def lab_pulse_summary_metrics(
    result,
    *,
    case_name: str,
    condition_name: str,
    example_name: str,
) -> dict[str, Any]:
    row = pulse_summary_metrics(
        result,
        case_name=case_name,
        condition_name=condition_name,
        example_name=example_name,
    )
    row["solver_frame"] = "lab_exact"
    row["field_source"] = "lab-frame Gaussian carrier field"
    return row


def save_lab_group_outputs(*, output_dir: str | Path, rows: list[dict[str, Any]]) -> None:
    save_results_csv(rows, Path(output_dir) / "results.csv")
