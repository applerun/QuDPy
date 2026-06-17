"""手动检查 piecewise stitch 后的三能级/二能级吸收谱输出。

该脚本的物理参数与 examples/absorption/three_level_absorption_lab_exact_casewise_meta.py
保持一致，用于排查 piecewise/stitch 与 lab-frame absorption 后处理是否一致。

主检查点：
1. two-level 和 three-level 均分别运行 full-window 与 piecewise 两条路径；
2. run_case(...) 返回 PieceDynamicsResultSeries 后，显式 stitch() 成 DynamicsResult；
3. 使用 DynamicsAnalysis + lab_frame_fft_response_legacy 复现原 example 的谱学计算方式；
4. 额外并排保存 lab_frame_absorption_response 的新 helper 输出，方便和 legacy 对照；
5. full/piecewise 均通过 save_result_case(...) 导出 QuDPy 标准 case output。

注意：
- 该脚本不是 unittest；用于人工检查 rho、population、coherence、preview 和 absorption。
- absorption / difference 仍是 spectroscopy / experiment 后处理，不放入 core。
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


if __package__ is None or __package__ == "":
    # 允许放在 scratch/ 下直接运行。
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sjh_learn.utils.core import (
    DynamicsResult,
    NLevelPhysicalParams,
    PureDephasingChannel,
    RelaxationChannel,
    run_case,
)
from sjh_learn.utils.analysis import DynamicsAnalysis
from sjh_learn.utils.fields import make_default_gaussian_carrier_field
from sjh_learn.utils.fields.field_windows import FieldActiveWindowSettings
from sjh_learn.utils.io import save_result_case
from sjh_learn.utils.json_utils import write_json
from sjh_learn.utils.plotting import (
    add_top_omega_axis,
    plot_multilevel_components,
    plot_normalized_curve,
    set_axis_ylim_from_curves,
    set_energy_axis,
)
from sjh_learn.utils.spectroscopy import lab_frame_absorption_response, polarization_C_per_m2
from sjh_learn.utils.spectroscopy.absorption_spectra import (
    lab_frame_fft_response_legacy as lab_frame_fft_response,
)


EXAMPLE_NAME = "debug_piecewise_three_level_absorption_param_matched"
REFERENCE_EXAMPLE_NAME = "compare_two_vs_three_level_absorption_lab_exact_fast_2to1_dissi"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "piecewise_three_level_absorption_param_matched"

# 与正常 example 保持一致的核心参数。
T_START_FS = -1000.0
T_END_FS = 1000.0
DT_FS = 0.05
E0_MV_PER_CM = 0.1
LASER_EV = 1.625
PULSE_CENTER_FS = 0.0
PULSE_SIGMA_FS = 5.0
NUMBER_DENSITY_M3 = 1.0e24
LEGACY_REL_THRESHOLD = 1.0e-5
NEW_REL_THRESHOLD = 1.0e-5
ZERO_PADDING_FACTOR = 4
PLOT_ENERGY_MIN_EV = 1.35
PLOT_ENERGY_MAX_EV = 1.90

# piecewise 仅用于检查和加速；物理谱学计算仍使用 stitched DynamicsResult。
PIECEWISE_REL_THRESHOLD = 1.0e-3
PIECEWISE_PADDING_FS = 80.0
PIECEWISE_MERGE_GAP_FS = 3.0


def as_dynamics_result(result: Any) -> DynamicsResult:
    """把 run_case 返回值转成 DynamicsResult。"""

    if isinstance(result, DynamicsResult):
        return result
    stitch = getattr(type(result), "stitch", None)
    if callable(stitch):
        stitched = result.stitch()
        if not isinstance(stitched, DynamicsResult):
            raise TypeError(
                "result.stitch() did not return a DynamicsResult: "
                f"{type(stitched).__name__}"
            )
        return stitched
    raise TypeError(
        "run_case result must be DynamicsResult or have stitch() -> DynamicsResult; "
        f"got {type(result).__name__}"
    )


def make_two_level_params() -> NLevelPhysicalParams:
    return NLevelPhysicalParams(
        energies_eV=(0.0, 1.50),
        dipole_matrix_D=((0.0, 3.0), (3.0, 0.0)),
        t_start_fs=T_START_FS,
        t_end_fs=T_END_FS,
        dt_fs=DT_FS,
        field=make_default_gaussian_carrier_field(
            E0_MV_per_cm=E0_MV_PER_CM,
            laser_energy_eV=LASER_EV,
            pulse_center_fs=PULSE_CENTER_FS,
            pulse_sigma_fs=PULSE_SIGMA_FS,
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
            "reference_example": REFERENCE_EXAMPLE_NAME,
            "system": "two_level",
            "transitions_eV": {"0_to_1": 1.50},
            "dipoles_D": {"mu_01": 3.0},
        },
    )


def make_three_level_params() -> NLevelPhysicalParams:
    return NLevelPhysicalParams(
        energies_eV=(0.0, 1.50, 1.75),
        dipole_matrix_D=((0.0, 3.0, 2.0), (3.0, 0.0, 0.0), (2.0, 0.0, 0.0)),
        t_start_fs=T_START_FS,
        t_end_fs=T_END_FS,
        dt_fs=DT_FS,
        field=make_default_gaussian_carrier_field(
            E0_MV_per_cm=E0_MV_PER_CM,
            laser_energy_eV=LASER_EV,
            pulse_center_fs=PULSE_CENTER_FS,
            pulse_sigma_fs=PULSE_SIGMA_FS,
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
            "reference_example": REFERENCE_EXAMPLE_NAME,
            "system": "three_level",
            "transitions_eV": {"0_to_1": 1.50, "0_to_2": 1.75},
            "dipoles_D": {"mu_01": 3.0, "mu_02": 2.0, "mu_12": 0.0},
        },
    )


def make_analysis_from_result(result: DynamicsResult) -> DynamicsAnalysis:
    if hasattr(DynamicsAnalysis, "from_result"):
        return DynamicsAnalysis.from_result(result)
    return DynamicsAnalysis.from_dynamics_res(result)


def get_coherence(analysis: DynamicsAnalysis, pair: tuple[int, int]) -> np.ndarray:
    if hasattr(analysis, "coherence"):
        return analysis.coherence(pair=pair)
    return analysis.rho12(pair=pair)


def make_piecewise_settings(params: NLevelPhysicalParams) -> FieldActiveWindowSettings:
    return FieldActiveWindowSettings(
        rel_threshold=PIECEWISE_REL_THRESHOLD,
        padding_fs=PIECEWISE_PADDING_FS,
        dt_fs=params.dt_fs,
        t_start_fs=params.t_start_fs,
        t_end_fs=params.t_end_fs,
        merge_gap_fs=PIECEWISE_MERGE_GAP_FS,
        force_single_window=True,
    )


def run_one_path(
    *,
    params: NLevelPhysicalParams,
    case_name: str,
    output_dir: Path,
    piecewise: bool,
    coherence_pairs: tuple[tuple[int, int], ...],
    append_results_csv: bool,
) -> dict[str, Any]:
    """运行 full 或 piecewise 路径，并按原 example 的分析方式提取谱。"""

    print(f"Running {case_name} [piecewise={piecewise}]")
    series = run_case(
        params,
        piecewise=piecewise,
        piecewise_settings=make_piecewise_settings(params) if piecewise else None,
    )
    result = as_dynamics_result(series)
    analysis = make_analysis_from_result(result)

    t_fs = analysis.time_fs()
    E_t, _, _ = analysis.input_signal(kind="field")
    P_t = analysis.full_polarization_C_per_m2(number_density_m3=NUMBER_DENSITY_M3)

    coherences: dict[tuple[int, int], np.ndarray] = {}
    legacy_responses: dict[tuple[int, int], dict[str, np.ndarray]] = {}
    new_responses: dict[tuple[int, int], dict[str, np.ndarray]] = {}

    for pair in coherence_pairs:
        rhoij_t = get_coherence(analysis, pair=pair)
        coherences[pair] = rhoij_t
        legacy_responses[pair] = lab_frame_fft_response(
            t_fs=t_fs,
            E_MV_per_cm=E_t,
            P_C_per_m2=P_t,
            rho12=rhoij_t,
            window="hann",
            subtract_mean=True,
            rel_threshold=LEGACY_REL_THRESHOLD,
            zero_padding_factor=ZERO_PADDING_FACTOR,
        )
        new_responses[pair] = lab_frame_absorption_response(
            t_fs=t_fs,
            E_MV_per_cm=E_t,
            P_C_per_m2=P_t,
            window="hann",
            subtract_mean=True,
            rel_threshold=NEW_REL_THRESHOLD,
            zero_padding_factor=ZERO_PADDING_FACTOR,
        )

    written = save_result_case(
        series,
        output_dir,
        output_data=True,
        output_preview=True,
        save_npz=True,
        save_csv=True,
        save_populations_csv=True,
        save_json=True,
        save_human_meta=True,
        save_debug_meta=True,
        case_name=case_name,
        example_name=EXAMPLE_NAME,
        condition_name="piecewise_absorption_debug",
        append_results_csv=append_results_csv,
        preview_component_pairs=coherence_pairs,
        preview_dpi=200,
    )

    return {
        "params": params,
        "series": series,
        "result": result,
        "case_outputs": {str(k): str(v) for k, v in written.items()} if isinstance(written, dict) else {"output": str(written)},
        "time_fs": t_fs,
        "E_t": E_t,
        "P_t": P_t,
        "coherences": coherences,
        "legacy_responses": legacy_responses,
        "new_responses": new_responses,
    }


def total_time_points(series: Any) -> int:
    return int(sum(len(piece_result.result.times) for piece_result in series.piece_results))


def active_solver_time_points(series: Any) -> int:
    return int(
        sum(
            len(piece_result.result.times)
            for piece_result in series.piece_results
            if piece_result.piece.kind == "active"
        )
    )


def _as_array(payload: dict[str, Any], key: str) -> np.ndarray:
    if key not in payload:
        raise KeyError(f"Missing response key: {key!r}; available={sorted(payload.keys())}")
    return np.asarray(payload[key])


def _legacy_xy(response: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    return _as_array(response, "energy_eV"), _as_array(response, "neg_omega_im_P_over_E")


def _first_existing_key(payload: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        if key in payload:
            return key
    return None


def _new_xy(response: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, str, str] | None:
    x_key = _first_existing_key(response, ("energy_eV", "frequency_THz", "frequency", "freq", "omega_fs_inv"))
    y_key = _first_existing_key(response, ("neg_omega_im_P_over_E", "absorption", "response", "signal", "omega_im_P_over_E"))
    if x_key is None or y_key is None:
        print(f"Cannot infer new response axes from keys: {sorted(response.keys())}")
        return None
    return _as_array(response, x_key), _as_array(response, y_key), x_key, y_key


def _energy_band(x: np.ndarray) -> np.ndarray:
    return (x >= PLOT_ENERGY_MIN_EV) & (x <= PLOT_ENERGY_MAX_EV)


def _save_figure(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_components(record: dict[str, Any], path: Path, title: str) -> Path:
    dim = record["result"].dimension()
    populations = tuple(range(min(dim, 3)))
    coherences = tuple(record["coherences"].keys())
    fig, _axes = plot_multilevel_components(
        record["result"],
        populations=populations,
        coherences=coherences,
        title=title,
    )
    return _save_figure(fig, path)


def plot_full_vs_piecewise_time(
    *,
    full_record: dict[str, Any],
    piecewise_record: dict[str, Any],
    pair: tuple[int, int],
    path: Path,
    title: str,
) -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 6.5), sharex=True)
    full_t = np.asarray(full_record["time_fs"])
    piece_t = np.asarray(piecewise_record["time_fs"])
    full_density = full_record["result"].density_array()
    piece_density = piecewise_record["result"].density_array()

    axes[0].plot(full_t, full_density[:, 0, 0].real, label="full rho_00")
    axes[0].plot(full_t, full_density[:, 1, 1].real, label="full rho_11")
    axes[0].plot(piece_t, piece_density[:, 0, 0].real, "--", label="piecewise rho_00")
    axes[0].plot(piece_t, piece_density[:, 1, 1].real, "--", label="piecewise rho_11")
    axes[0].set_ylabel("Population")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best")

    axes[1].plot(full_t, np.abs(full_record["coherences"][pair]), label=f"full |rho_{pair[0]}{pair[1]}|")
    axes[1].plot(piece_t, np.abs(piecewise_record["coherences"][pair]), "--", label=f"piecewise |rho_{pair[0]}{pair[1]}|")
    axes[1].set_xlabel("Time (fs)")
    axes[1].set_ylabel("Coherence magnitude")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best")

    fig.suptitle(title)
    return _save_figure(fig, path)


def plot_legacy_full_vs_piecewise(
    *,
    full_record: dict[str, Any],
    piecewise_record: dict[str, Any],
    pair: tuple[int, int],
    path: Path,
    title: str,
) -> Path:
    full_resp = full_record["legacy_responses"][pair]
    piece_resp = piecewise_record["legacy_responses"][pair]
    x_full, y_full = _legacy_xy(full_resp)
    x_piece, y_piece = _legacy_xy(piece_resp)
    band_full = _energy_band(x_full)
    band_piece = _energy_band(x_piece)

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.plot(x_full[band_full], y_full[band_full], label="full legacy")
    ax.plot(x_piece[band_piece], y_piece[band_piece], "--", label="piecewise legacy")
    for transition in (1.50, 1.75):
        ax.axvline(transition, color="0.4", linestyle=":", linewidth=1.0)
    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("-ω Im[P/E]")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    return _save_figure(fig, path)


def plot_new_vs_legacy(
    *,
    record: dict[str, Any],
    pair: tuple[int, int],
    path: Path,
    title: str,
) -> Path | None:
    new_xy = _new_xy(record["new_responses"][pair])
    if new_xy is None:
        return None
    x_new, y_new, x_key, y_key = new_xy
    x_legacy, y_legacy = _legacy_xy(record["legacy_responses"][pair])
    new_band = _energy_band(x_new)
    legacy_band = _energy_band(x_legacy)

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.plot(x_new[new_band], y_new[new_band], label=f"new {y_key}")
    ax.plot(x_legacy[legacy_band], y_legacy[legacy_band], "--", label="legacy -ω Im[P/E]")
    for transition in (1.50, 1.75):
        ax.axvline(transition, color="0.4", linestyle=":", linewidth=1.0)
    ax.set_xlabel(str(x_key))
    ax.set_ylabel("response")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    return _save_figure(fig, path)


def build_reference_compare_figure(
    *,
    two_full: dict[str, Any],
    three_full: dict[str, Any],
    output_dir: Path,
) -> Path:
    """复现原 two-vs-three compare figure 的核心图。"""

    style = {
        "font.size": 15,
        "axes.titlesize": 17,
        "axes.labelsize": 15,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 10,
        "figure.titlesize": 20,
    }
    t_fs = np.asarray(two_full["time_fs"])
    E_t = np.asarray(two_full["E_t"])
    two_resp = two_full["legacy_responses"][(0, 1)]
    three_resp_01 = three_full["legacy_responses"][(0, 1)]
    three_resp_02 = three_full["legacy_responses"][(0, 2)]
    energy = two_resp["energy_eV"]
    band = _energy_band(energy)
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

        curves_w = [
            plot_normalized_curve(
                ax_w,
                energy[band],
                two_resp["abs_E_fft"][band],
                label="norm |E(w)|",
                linewidth=2.3,
            )
        ]
        ax_w.set_title("Input spectrum")
        ax_w.set_xlabel("Energy (eV)")
        ax_w.set_ylabel("Normalized amplitude")
        set_energy_axis(ax_w, PLOT_ENERGY_MIN_EV, PLOT_ENERGY_MAX_EV)
        add_top_omega_axis(ax_w, y_pos=0.82)
        set_axis_ylim_from_curves(ax_w, curves_w, min_headroom=0.16)
        ax_w.legend(loc="best")

        curves_abs = [
            plot_normalized_curve(
                ax_abs,
                energy[band],
                two_resp["neg_omega_im_P_over_E"][band],
                label="2-level total -ω Im[P/E]",
                linewidth=2.6,
            ),
            plot_normalized_curve(
                ax_abs,
                three_resp_01["energy_eV"][band],
                three_resp_01["neg_omega_im_P_over_E"][band],
                label="3-level total -ω Im[P/E]",
                linewidth=2.6,
                linestyle="--",
            ),
        ]
        ax_abs.axvline(1.50, linestyle=":", linewidth=1.5, label="E10 = 1.50 eV")
        ax_abs.axvline(1.75, linestyle=":", linewidth=1.5, label="E20 = 1.75 eV")
        ax_abs.set_title("Total absorption-like response")
        ax_abs.set_xlabel("Energy (eV)")
        ax_abs.set_ylabel("Normalized signal")
        set_energy_axis(ax_abs, PLOT_ENERGY_MIN_EV, PLOT_ENERGY_MAX_EV)
        add_top_omega_axis(ax_abs)
        set_axis_ylim_from_curves(ax_abs, curves_abs, min_headroom=0.22)
        ax_abs.legend(loc="best")

        curves_coh = [
            plot_normalized_curve(
                ax_coh,
                energy[band],
                -two_resp["im_rho12_over_E"][band],
                label="2-level -Im[rho01/E]",
                linewidth=2.2,
            ),
            plot_normalized_curve(
                ax_coh,
                three_resp_01["energy_eV"][band],
                -three_resp_01["im_rho12_over_E"][band],
                label="3-level -Im[rho01/E]",
                linewidth=2.2,
                linestyle="--",
            ),
            plot_normalized_curve(
                ax_coh,
                three_resp_02["energy_eV"][band],
                -three_resp_02["im_rho12_over_E"][band],
                label="3-level -Im[rho02/E]",
                linewidth=2.2,
                linestyle="-.",
            ),
        ]
        ax_coh.axvline(1.50, linestyle=":", linewidth=1.5)
        ax_coh.axvline(1.75, linestyle=":", linewidth=1.5)
        ax_coh.set_title("Coherence-channel response")
        ax_coh.set_xlabel("Energy (eV)")
        ax_coh.set_ylabel("Normalized signal")
        set_energy_axis(ax_coh, PLOT_ENERGY_MIN_EV, PLOT_ENERGY_MAX_EV)
        add_top_omega_axis(ax_coh)
        set_axis_ylim_from_curves(ax_coh, curves_coh, min_headroom=0.22)
        ax_coh.legend(loc="best")

        fig.suptitle(
            "Two-level vs three-level lab-exact absorption validation\n"
            "No RWA; full-window stitched result",
            y=0.97,
        )
        fig.subplots_adjust(top=0.84, left=0.07, right=0.985, bottom=0.08)

        path = output_dir / "figures" / "two_vs_three_level_absorption_compare_full_window.png"
        return _save_figure(fig, path)


def save_record_payload(record: dict[str, Any], output_dir: Path, stem: str) -> dict[str, Path]:
    written: dict[str, Path] = {}
    for family in ("legacy_responses", "new_responses"):
        for pair, response in record[family].items():
            pair_name = f"rho_{pair[0]}{pair[1]}"
            base = output_dir / f"{stem}_{family}_{pair_name}"
            np.savez_compressed(base.with_suffix(".npz"), **response)
            write_json(base.with_suffix(".json"), response)
            written[f"{stem}_{family}_{pair_name}_npz"] = base.with_suffix(".npz")
            written[f"{stem}_{family}_{pair_name}_json"] = base.with_suffix(".json")
    return written


def print_piece_summary(label: str, series: Any) -> None:
    print(f"\n[{label} pieces]")
    for idx, piece_result in enumerate(series.piece_results):
        piece = piece_result.piece
        result = piece_result.result
        print(
            f"{idx:02d} {piece.piece_name:30} "
            f"kind={piece.kind:6} "
            f"window=[{piece.window.start_fs:.3f}, {piece.window.end_fs:.3f}] fs "
            f"n={len(result.times)}"
        )


def print_record_summary(label: str, record: dict[str, Any]) -> None:
    result = record["result"]
    series = record["series"]
    print(f"\n[{label}]")
    print(f"piece count              : {len(series.piece_results)}")
    print(f"display raw time points  : {total_time_points(series)}")
    print(f"active solver points     : {active_solver_time_points(series)}")
    print(f"stitched time points     : {len(result.times)}")
    print(f"max_trace_error          : {result.max_trace_error():.3e}")
    print(f"max_hermiticity_error    : {result.max_hermiticity_error():.3e}")
    print_piece_summary(label, series)


def print_standard_output_files(label: str, case_outputs: dict[str, str]) -> None:
    case_dir = case_outputs.get("case_dir")
    if not case_dir:
        print(f"\n[{label} meta.json output_files] missing case_dir")
        return
    meta_path = Path(case_dir) / "meta.json"
    if not meta_path.exists():
        print(f"\n[{label} meta.json output_files] missing meta.json: {meta_path}")
        return
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    print(f"\n[{label} meta.json output_files]")
    print(json.dumps(payload.get("output_files", {}), indent=2, ensure_ascii=False))


def print_written_paths(written: dict[str, Path]) -> None:
    print("\n[files]")
    for key in sorted(written):
        print(f"{key:48}: {written[key]}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "figures").mkdir(parents=True, exist_ok=True)

    cases = {
        "two_level": {
            "params": make_two_level_params(),
            "pairs": ((0, 1),),
        },
        "three_level": {
            "params": make_three_level_params(),
            "pairs": ((0, 1), (0, 2)),
        },
    }

    records: dict[str, dict[str, Any]] = {}
    written: dict[str, Path] = {}

    append_results_csv = False
    for system_name, cfg in cases.items():
        params = cfg["params"]
        pairs = cfg["pairs"]

        full_key = f"{system_name}_full"
        piecewise_key = f"{system_name}_piecewise"

        records[full_key] = run_one_path(
            params=params,
            case_name=f"{system_name}_full_window",
            output_dir=OUTPUT_DIR / "simulation",
            piecewise=False,
            coherence_pairs=pairs,
            append_results_csv=append_results_csv,
        )
        append_results_csv = True
        records[piecewise_key] = run_one_path(
            params=params,
            case_name=f"{system_name}_piecewise_window",
            output_dir=OUTPUT_DIR / "simulation",
            piecewise=True,
            coherence_pairs=pairs,
            append_results_csv=True,
        )

        written.update(save_record_payload(records[full_key], OUTPUT_DIR, full_key))
        written.update(save_record_payload(records[piecewise_key], OUTPUT_DIR, piecewise_key))

        written[f"{system_name}_full_components_png"] = plot_components(
            records[full_key],
            OUTPUT_DIR / "figures" / f"{system_name}_full_components.png",
            title=f"{system_name} full stitched components",
        )
        written[f"{system_name}_piecewise_components_png"] = plot_components(
            records[piecewise_key],
            OUTPUT_DIR / "figures" / f"{system_name}_piecewise_components.png",
            title=f"{system_name} piecewise stitched components",
        )
        written[f"{system_name}_full_vs_piecewise_time_png"] = plot_full_vs_piecewise_time(
            full_record=records[full_key],
            piecewise_record=records[piecewise_key],
            pair=pairs[0],
            path=OUTPUT_DIR / "figures" / f"{system_name}_full_vs_piecewise_time.png",
            title=f"{system_name} full vs piecewise time-domain check",
        )
        written[f"{system_name}_legacy_full_vs_piecewise_png"] = plot_legacy_full_vs_piecewise(
            full_record=records[full_key],
            piecewise_record=records[piecewise_key],
            pair=pairs[0],
            path=OUTPUT_DIR / "figures" / f"{system_name}_legacy_full_vs_piecewise.png",
            title=f"{system_name} legacy FFT full vs piecewise",
        )
        maybe_new_vs_legacy = plot_new_vs_legacy(
            record=records[full_key],
            pair=pairs[0],
            path=OUTPUT_DIR / "figures" / f"{system_name}_new_vs_legacy_full.png",
            title=f"{system_name} new vs legacy absorption, full",
        )
        if maybe_new_vs_legacy is not None:
            written[f"{system_name}_new_vs_legacy_full_png"] = maybe_new_vs_legacy

        series_json = OUTPUT_DIR / f"{system_name}_piecewise_series.json"
        write_json(series_json, records[piecewise_key]["series"].to_dict())
        written[f"{system_name}_piecewise_series_json"] = series_json

    written["reference_two_vs_three_full_compare_png"] = build_reference_compare_figure(
        two_full=records["two_level_full"],
        three_full=records["three_level_full"],
        output_dir=OUTPUT_DIR,
    )

    workflow_metadata = {
        "example_name": EXAMPLE_NAME,
        "reference_example_name": REFERENCE_EXAMPLE_NAME,
        "description": "Piecewise debug script using the same physical parameters as the validated two-vs-three lab-exact absorption example.",
        "parameters": {
            "t_start_fs": T_START_FS,
            "t_end_fs": T_END_FS,
            "dt_fs": DT_FS,
            "E0_MV_per_cm": E0_MV_PER_CM,
            "laser_energy_eV": LASER_EV,
            "pulse_sigma_fs": PULSE_SIGMA_FS,
            "number_density_m3": NUMBER_DENSITY_M3,
            "legacy_rel_threshold": LEGACY_REL_THRESHOLD,
            "new_rel_threshold": NEW_REL_THRESHOLD,
            "zero_padding_factor": ZERO_PADDING_FACTOR,
            "plot_energy_range_eV": [PLOT_ENERGY_MIN_EV, PLOT_ENERGY_MAX_EV],
            "piecewise": {
                "rel_threshold": PIECEWISE_REL_THRESHOLD,
                "padding_fs": PIECEWISE_PADDING_FS,
                "merge_gap_fs": PIECEWISE_MERGE_GAP_FS,
            },
        },
        "case_outputs": {key: record["case_outputs"] for key, record in records.items()},
        "summary": {
            key: {
                "piece_count": int(len(record["series"].piece_results)),
                "raw_time_points": int(total_time_points(record["series"])),
                "active_solver_points": int(active_solver_time_points(record["series"])),
                "stitched_time_points": int(len(record["result"].times)),
                "max_trace_error": float(record["result"].max_trace_error()),
                "max_hermiticity_error": float(record["result"].max_hermiticity_error()),
            }
            for key, record in records.items()
        },
    }
    workflow_metadata_path = OUTPUT_DIR / "workflow_metadata.json"
    write_json(workflow_metadata_path, workflow_metadata)
    written["workflow_metadata_json"] = workflow_metadata_path

    print(f"output directory         : {OUTPUT_DIR}")
    for key in ("two_level_full", "two_level_piecewise", "three_level_full", "three_level_piecewise"):
        print_record_summary(key, records[key])
        print_standard_output_files(key, records[key]["case_outputs"])
    print_written_paths(written)


if __name__ == "__main__":
    main()
