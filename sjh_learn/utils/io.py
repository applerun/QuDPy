"""Result output and case-level data management helpers.

User-facing CSV and figure outputs should prefer physical units when available.
Internal code-unit diagnostics are still saved, but only under explicit names.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .results import DynamicsResult


ResultLike = DynamicsResult

HC_EV_NM = 1239.8419843320026


def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def format_value_tag(value: float) -> str:
    return f"{value:.4f}".rstrip("0").rstrip(".").replace("-", "m").replace(".", "p")


def default_output_path(output_dir: Path, result: ResultLike) -> Path:
    physical = getattr(result, "physical_params", None)
    solver = getattr(result, "solver_params", None)
    parameters = getattr(result, "parameters", None)
    if physical is not None and solver is not None:
        amplitude_tag = format_value_tag(physical.field_MV_per_cm)
        detuning_tag = format_value_tag(solver.detuning_fs_inv)
    else:
        amplitude_tag = format_value_tag(getattr(parameters, "field_amplitude", 0.0))
        detuning_tag = format_value_tag(getattr(parameters, "detuning", 0.0))
    return output_dir / f"comparison_E0_{amplitude_tag}_Delta_{detuning_tag}.png"


def save_parameter_summary(results: list[ResultLike], output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    data = [result.parameter_summary_dict() for result in results]
    output.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return output


def _append_results_csv_row(path: Path, header: list[str], row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerow([row.get(column, "") for column in header])
        return

    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        existing_header = next(reader, None)
    if existing_header != header:
        raise ValueError(
            "results.csv header mismatch. Delete the old file or use another output directory.\n"
            f"existing: {existing_header}\n"
            f"expected: {header}"
        )

    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([row.get(column, "") for column in header])


def _safe_case_name(value: str) -> str:
    cleaned = []
    for char in str(value):
        cleaned.append(char if (char.isalnum() or char in ("-", "_")) else "_")
    return "".join(cleaned).strip("_") or "case"


def _case_name(result: ResultLike) -> str:
    physical = result.physical_params
    if physical is not None:
        return _safe_case_name(
            f"{result.mode}_field_{format_value_tag(physical.field_MV_per_cm)}_"
            f"laser_{format_value_tag(physical.laser_energy_eV)}"
        )
    return _safe_case_name(f"{result.mode}_N{result.dimension()}")


def _summary_row(case_name: str, result: ResultLike) -> dict[str, Any]:
    times = result.times_fs if result.times_fs is not None else result.times
    dimension = result.dimension()
    max_trace_error = result.max_trace_error()
    max_hermiticity_error = result.max_hermiticity_error()
    final_populations = ";".join(f"{float(value.real):.12g}" for value in result.populations()[-1])
    row: dict[str, Any] = {
        "case_name": case_name,
        "result_type": type(result).__name__,
        "mode": getattr(result, "mode", ""),
        "source_mode": getattr(result, "source_mode", ""),
        "time_start_fs": float(times[0]) if len(times) else "",
        "time_end_fs": float(times[-1]) if len(times) else "",
        "n_time_points": len(result.times),
        "dimension": dimension,
        "max_trace_error": max_trace_error,
        "max_hermiticity_error": max_hermiticity_error,
        "final_populations": final_populations,
    }
    physical = getattr(result, "physical_params", None)
    solver = getattr(result, "solver_params", None)
    row["field_MV_per_cm"] = "" if physical is None else physical.field_MV_per_cm
    row["laser_energy_eV"] = "" if physical is None else physical.laser_energy_eV
    row["detuning_fs_inv"] = "" if solver is None else solver.detuning_fs_inv
    return row


def _trajectory_summary(result: ResultLike) -> dict[str, Any]:
    times = result.times_fs if result.times_fs is not None else result.times
    summary: dict[str, Any] = {
        "n_time_points": int(len(result.times)),
        "time_range_fs": [float(times[0]), float(times[-1])] if len(times) else [],
        "max_trace_error": result.max_trace_error(),
        "max_hermiticity_error": result.max_hermiticity_error(),
    }
    if result.dimension() == 2:
        rho11, rho22, rho12, _rho21 = result.components()
        phase_rho12 = np.angle(rho12)
        phase_rho12_unwrapped = np.unwrap(phase_rho12)
        if abs(rho12[-1]) < 1e-8:
            final_phase_rho12 = None
            final_phase_rho12_unwrapped = None
        else:
            final_phase_rho12 = float(phase_rho12[-1])
            final_phase_rho12_unwrapped = float(phase_rho12_unwrapped[-1])
        summary.update(
            {
                "final_rho11": float(rho11[-1].real),
                "final_rho22": float(rho22[-1].real),
                "final_abs_rho12": float(abs(rho12[-1])),
                "final_phase_rho12": final_phase_rho12,
                "final_phase_rho12_unwrapped": final_phase_rho12_unwrapped,
                "max_rho22": float(np.max(rho22.real)),
                "max_abs_rho12": float(np.max(np.abs(rho12))),
            }
        )
    else:
        final_populations = result.populations()[-1]
        summary["final_populations"] = [float(value.real) for value in final_populations]
    return summary


def _field_envelope(physical: Any) -> str:
    return "gaussian" if getattr(physical, "pulse_sigma_fs", None) is not None else "constant"


def _input_field_metadata(physical: Any, solver: Any) -> dict[str, Any] | None:
    if physical is None:
        return None
    envelope = _field_envelope(physical)
    data: dict[str, Any] = {
        "description": "Physical lab-frame optical field.",
        "class": "GaussianCarrierField" if envelope == "gaussian" else "CarrierField",
        "expression": "E(t) = 2 E0 f(t) cos(omega_L t + phase)",
        "E0_MV_per_cm": physical.field_MV_per_cm,
        "peak_E_MV_per_cm": 2.0 * physical.field_MV_per_cm,
        "omega_L_fs_inv": None if solver is None else solver.omega_L_fs_inv,
        "laser_energy_eV": physical.laser_energy_eV,
        "phase_rad": 0.0,
        "envelope": envelope,
        "field_unit": "MV/cm",
        "time_unit": "fs",
        "amplitude_convention": "field_MV_per_cm is E0 in E(t) = 2 E0 f(t) cos(omega_L t + phase).",
    }
    if envelope == "gaussian":
        data["pulse_center_fs"] = physical.pulse_center_fs
        data["pulse_sigma_fs"] = physical.pulse_sigma_fs
    return data


def _solver_representation_metadata(result: ResultLike) -> dict[str, Any]:
    mode = getattr(result, "mode", None)
    if mode == "lab_exact":
        return {
            "mode": "lab_exact",
            "description": "Direct lab-frame propagation with the full time-dependent optical carrier.",
            "hamiltonian_type": "time-dependent lab-frame Hamiltonian",
            "uses_lab_field_directly": True,
            "uses_rwa_drive": False,
            "uses_rotating_transform": False,
            "carrier_retained": True,
            "counter_rotating_terms_retained": True,
            "rwa_approximation_applied": False,
            "independently_solved_by_mesolve": True,
        }
    if mode == "rotating_view":
        return {
            "mode": "rotating_view",
            "description": "Derived view obtained by transforming lab-frame density matrices into a rotating frame.",
            "hamiltonian_type": "not independently solved",
            "uses_lab_field_directly": False,
            "uses_rwa_drive": False,
            "uses_rotating_transform": True,
            "source_mode": getattr(result, "source_mode", None) or "lab_exact",
            "carrier_retained": "not_applicable",
            "counter_rotating_terms_retained": "inherited_from_lab_exact",
            "rwa_approximation_applied": False,
            "independently_solved_by_mesolve": False,
        }
    if mode == "rwa":
        return {
            "mode": "rwa",
            "description": "Propagation with the effective RWA Hamiltonian.",
            "hamiltonian_type": "RWA effective Hamiltonian",
            "uses_lab_field_directly": False,
            "uses_rwa_drive": True,
            "uses_rotating_transform": False,
            "carrier_retained": False,
            "counter_rotating_terms_retained": False,
            "rwa_approximation_applied": True,
            "independently_solved_by_mesolve": True,
        }
    return {
        "mode": mode,
        "description": "Simulation representation is not specialized for this mode.",
        "independently_solved_by_mesolve": None,
    }


def _lab_frame_solver_metadata(result: ResultLike, physical: Any, solver: Any) -> dict[str, Any] | None:
    if getattr(result, "mode", None) != "lab_exact":
        return None
    envelope = _field_envelope(physical) if physical is not None else "unknown"
    return {
        "description": "Direct lab-frame solver using the physical optical carrier.",
        "field_class": "GaussianCarrierField" if envelope == "gaussian" else "CarrierField",
        "field_expression": "E(t) = 2 E0 f(t) cos(omega_L t + phase)",
        "interaction": "H_int(t) = -mu E(t)",
        "carrier_retained": True,
        "counter_rotating_terms_retained": True,
        "hamiltonian_note": "The full lab-frame optical carrier is retained in the time-dependent Hamiltonian.",
    }


def _rotating_transform_metadata(result: ResultLike, solver: Any) -> dict[str, Any] | None:
    if getattr(result, "mode", None) != "rotating_view":
        return None
    return {
        "description": "Post-processing transformation applied to lab-frame density matrix trajectory.",
        "source_result_mode": getattr(result, "source_mode", None) or "lab_exact",
        "reference_frequency_fs_inv": None if solver is None else solver.omega_L_fs_inv,
        "reference_frequency_source": "laser_energy_eV",
        "transform_type": "density matrix unitary frame transformation",
        "independently_solved_by_mesolve": False,
        "rwa_approximation_applied": False,
    }


def _input_drive_metadata(result: ResultLike, physical: Any, solver: Any, parameters: Any) -> dict[str, Any] | None:
    mode = getattr(result, "mode", None)
    if mode != "rwa":
        return {
            "description": f"No RWA drive is used in {mode} mode.",
            "uses_rwa_drive": False,
        }
    drive_dict = getattr(result, "drive_dict", None) or {}
    envelope = _field_envelope(physical) if physical is not None else "constant"
    return {
        "description": "Effective slow drive entering the RWA Hamiltonian.",
        "class": drive_dict.get("class"),
        "drive_name": getattr(result, "drive_name", None),
        "drive_symbol": "g(t)",
        "expression": "g(t) = mu E0 f(t) / hbar",
        "amplitude_fs_inv": None if solver is None else solver.rabi_fs_inv,
        "amplitude_code": None if solver is None else solver.rabi,
        "source": "derived from dipole_D and field_MV_per_cm",
        "domain": "RWA",
        "drive_unit_physical": "fs^-1",
        "drive_unit_code": "code",
        "envelope": envelope,
        "amplitude_convention": "input_drive is the slow RWA coupling after removing the optical carrier.",
        "drive_expr": getattr(result, "drive_expr", None),
        "parameters_field_amplitude_code": getattr(parameters, "field_amplitude", None),
    }


def _human_metadata(
    result: ResultLike,
    *,
    example_name: str | None = None,
    condition_name: str | None = None,
    case_name: str | None = None,
    output_files: dict[str, str] | None = None,
) -> dict[str, Any]:
    physical = getattr(result, "physical_params", None)
    solver = getattr(result, "solver_params", None)
    parameters = getattr(result, "parameters", None)
    meta: dict[str, Any] = {
        "result_type": type(result).__name__,
        "example_name": example_name,
        "condition_name": condition_name,
        "case_name": case_name,
        "mode": getattr(result, "mode", None),
        "source_mode": getattr(result, "source_mode", None),
    }

    if physical is not None:
        detuning_eV = physical.energy_gap_eV - physical.laser_energy_eV
        meta["inputs_physical"] = {
            "energy_gap_eV": physical.energy_gap_eV,
            "laser_energy_eV": physical.laser_energy_eV,
            "detuning_eV": detuning_eV,
            "wavelength_nm": HC_EV_NM / physical.laser_energy_eV if physical.laser_energy_eV else None,
            "dipole_D": physical.dipole_D,
            "field_MV_per_cm": physical.field_MV_per_cm,
            "T1_fs": physical.T1_fs,
            "Tphi_fs": physical.Tphi_fs,
            "T2_fs": physical.T2_fs,
            "t_start_fs": physical.t_start_fs,
            "t_end_fs": physical.t_end_fs,
            "dt_fs": physical.dt_fs,
            "pulse_center_fs": physical.pulse_center_fs,
            "pulse_sigma_fs": physical.pulse_sigma_fs,
        }
    else:
        meta["inputs_physical"] = None

    meta["input_field"] = _input_field_metadata(physical, solver)
    meta["solver_representation"] = _solver_representation_metadata(result)
    meta["lab_frame_solver"] = _lab_frame_solver_metadata(result, physical, solver)
    meta["rotating_transform"] = _rotating_transform_metadata(result, solver)
    meta["input_drive"] = _input_drive_metadata(result, physical, solver, parameters)

    if solver is not None:
        meta["derived_physical"] = {
            "omega_eg_fs_inv": solver.omega_eg_fs_inv,
            "omega_L_fs_inv": solver.omega_L_fs_inv,
            "detuning_fs_inv": solver.detuning_fs_inv,
            "rabi_fs_inv": solver.rabi_fs_inv,
            "gamma1_fs_inv": solver.gamma1_fs_inv,
            "gamma_phi_fs_inv": solver.gamma_phi_fs_inv,
            "gamma2_fs_inv": solver.gamma2_fs_inv,
            "T2_effective_fs": (1.0 / solver.gamma2_fs_inv) if solver.gamma2_fs_inv > 0 else None,
            "rabi_period_fs": (2.0 * np.pi / abs(solver.rabi_fs_inv)) if solver.rabi_fs_inv != 0 else None,
        }
        meta["solver_code_summary"] = {
            "time_scale_fs": solver.time_scale_fs,
            "detuning_code": solver.detuning,
            "field_amplitude_code": getattr(parameters, "field_amplitude", None),
            "omega_drive_code": getattr(parameters, "omega_drive", None),
            "gamma1_code": solver.gamma1,
            "gamma_phi_code": solver.gamma_phi,
            "gamma2_code": solver.gamma2,
            "t_start_code": solver.t_start,
            "t_end_code": solver.t_end,
            "dt_code": solver.dt,
        }
    else:
        meta["derived_physical"] = None
        meta["solver_code_summary"] = None

    meta["trajectory_summary"] = _trajectory_summary(result)
    meta["output_files"] = output_files or {}
    return _json_safe(meta)


def _debug_metadata(
    result: ResultLike,
    *,
    example_name: str | None = None,
    condition_name: str | None = None,
    case_name: str | None = None,
) -> dict[str, Any]:
    metadata = result.metadata_dict()
    metadata["example_name"] = example_name
    metadata["condition_name"] = condition_name
    metadata["case_name"] = case_name
    return metadata


def _relative_output_files(case_dir: Path, written: dict[str, Path]) -> dict[str, str]:
    names = {
        "density_npz": "density",
        "components_csv": "components",
        "populations_csv": "populations",
        "selected_elements_csv": "selected_elements",
        "preview": "preview",
        "full": "full",
        "debug_meta": "debug_metadata",
        "meta": "metadata",
    }
    output_files: dict[str, str] = {}
    for key, label in names.items():
        path = written.get(key)
        if path is not None:
            output_files[label] = path.relative_to(case_dir).as_posix()
    return output_files


def _write_metadata_files(
    result: ResultLike,
    case_dir: Path,
    written: dict[str, Path],
    *,
    example_name: str | None = None,
    condition_name: str | None = None,
    case_name: str | None = None,
    save_human_meta: bool = True,
    save_debug_meta: bool = True,
) -> dict[str, Path]:
    metadata_paths: dict[str, Path] = {}
    debug_path = case_dir / "debug_meta.json"
    meta_path = case_dir / "meta.json"
    metadata_outputs = {**written}
    if save_debug_meta:
        metadata_outputs["debug_meta"] = debug_path
    output_files = _relative_output_files(case_dir, metadata_outputs)

    if save_debug_meta:
        debug_path.write_text(
            json.dumps(
                _debug_metadata(
                    result,
                    example_name=example_name,
                    condition_name=condition_name,
                    case_name=case_name,
                ),
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        metadata_paths["debug_meta"] = debug_path

    if save_human_meta:
        meta_path.write_text(
            json.dumps(
                _human_metadata(
                    result,
                    example_name=example_name,
                    condition_name=condition_name,
                    case_name=case_name,
                    output_files=output_files,
                ),
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        metadata_paths["meta"] = meta_path

    return metadata_paths


def save_figure(fig, output_path: str | Path, dpi: int = 120) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=int(dpi))
    return output


def save_result_data(
    result: ResultLike,
    output_dir: str | Path,
    *,
    save_npz: bool = True,
    save_csv: bool = True,
    save_json: bool = True,
    save_human_meta: bool = True,
    save_debug_meta: bool = True,
    example_name: str | None = None,
    condition_name: str | None = None,
    case_name: str | None = None,
    selected_elements: dict[str, tuple[int, int]] | None = None,
) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    if save_npz:
        npz_path = output / "density.npz"
        np.savez_compressed(npz_path, **result.to_npz_dict())
        written["density_npz"] = npz_path

    if save_csv:
        if result.dimension() == 2:
            components_path = output / "components.csv"
            result.components_dataframe().to_csv(components_path, index=False)
            written["components_csv"] = components_path

        populations_path = output / "populations.csv"
        result.populations_dataframe().to_csv(populations_path, index=False)
        written["populations_csv"] = populations_path

        if selected_elements:
            selected_path = output / "selected_elements.csv"
            result.selected_elements_dataframe(selected_elements).to_csv(selected_path, index=False)
            written["selected_elements_csv"] = selected_path

    if save_json:
        written.update(
            _write_metadata_files(
                result,
                output.parent,
                written,
                example_name=example_name,
                condition_name=condition_name,
                case_name=case_name,
                save_human_meta=save_human_meta,
                save_debug_meta=save_debug_meta,
            )
        )

    return written


def save_results_components_long(results: list[ResultLike], output_path: str | Path) -> Path:
    if not results:
        raise ValueError("results must not be empty.")
    frames = []
    for result in results:
        if result.dimension() != 2:
            raise ValueError("save_results_components_long() is only defined for two-level results.")
        frame = result.components_dataframe().copy()
        frame.insert(0, "mode", result.mode)
        frames.append(frame)

    pd = __import__("pandas")
    combined = pd.concat(frames, ignore_index=True)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(path, index=False)
    return path


def save_result_case(
    result: ResultLike,
    output_dir: str | Path,
    *,
    output_data: bool = True,
    output_preview: bool = False,
    fig=None,
    preview_fig=None,
    preview_dpi: int = 120,
    full_fig=None,
    full_dpi: int = 300,
    save_npz: bool = True,
    save_csv: bool = True,
    save_json: bool = True,
    save_human_meta: bool = True,
    save_debug_meta: bool = True,
    example_name: str | None = None,
    condition_name: str | None = None,
    case_name: str | None = None,
    selected_elements: dict[str, tuple[int, int]] | None = None,
    append_results_csv: bool = True,
) -> dict[str, Path]:
    root = Path(output_dir)
    name = _safe_case_name(case_name) if case_name is not None else _case_name(result)
    case_dir = root / "res_per_case" / name
    data_dir = case_dir / "data"
    figs_dir = case_dir / "figs"
    case_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {"case_dir": case_dir}

    if output_data:
        written.update(
            save_result_data(
                result,
                data_dir,
                save_npz=save_npz,
                save_csv=save_csv,
                save_json=False,
                selected_elements=selected_elements,
            )
        )

    generated_preview = False
    local_preview_fig = fig if preview_fig is None else preview_fig
    if output_preview and local_preview_fig is None:
        from .plotting import build_preview_figure

        local_preview_fig, _axes = build_preview_figure(result)
        generated_preview = True

    if output_preview and local_preview_fig is not None:
        preview_path = figs_dir / "preview.png"
        written["preview"] = save_figure(local_preview_fig, preview_path, dpi=preview_dpi)
        if generated_preview:
            import matplotlib.pyplot as plt

            plt.close(local_preview_fig)

    if full_fig is not None:
        full_path = figs_dir / "full.png"
        written["full"] = save_figure(full_fig, full_path, dpi=full_dpi)

    if save_json:
        written.update(
            _write_metadata_files(
                result,
                case_dir,
                written,
                example_name=example_name,
                condition_name=condition_name,
                case_name=name,
                save_human_meta=save_human_meta,
                save_debug_meta=save_debug_meta,
            )
        )

    if append_results_csv:
        header = [
            "case_name",
            "result_type",
            "mode",
            "source_mode",
            "time_start_fs",
            "time_end_fs",
            "n_time_points",
            "dimension",
            "field_MV_per_cm",
            "laser_energy_eV",
            "detuning_fs_inv",
            "max_trace_error",
            "max_hermiticity_error",
            "final_populations",
        ]
        results_csv = root / "results.csv"
        _append_results_csv_row(results_csv, header, _summary_row(name, result))
        written["results_csv"] = results_csv
    return written


@dataclass
class QuantumResultIO:
    outdir: str
    results_csv_name: str = "results.csv"

    def save_case(self, result: ResultLike, **kwargs) -> dict[str, Path]:
        return save_result_case(result, self.outdir, **kwargs)


ResultManager = QuantumResultIO


__all__ = [
    "QuantumResultIO",
    "ResultManager",
    "default_output_path",
    "save_parameter_summary",
    "save_result_data",
    "save_results_components_long",
    "save_figure",
    "save_result_case",
]
