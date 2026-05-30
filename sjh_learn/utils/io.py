"""Result output and case-level data management helpers."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .results import DynamicsResult


ResultLike = DynamicsResult


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
        meta_path = output.parent / "meta.json"
        meta_path.write_text(json.dumps(result.metadata_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        written["meta"] = meta_path

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
                save_json=save_json,
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
