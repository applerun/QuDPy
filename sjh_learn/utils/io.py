"""结果输出相关工具。"""

from __future__ import annotations

import json
from pathlib import Path

from .results import OpticalBlochResult


def format_value_tag(value: float) -> str:
    return f"{value:.4f}".rstrip("0").rstrip(".").replace("-", "m").replace(".", "p")


def default_output_path(output_dir: Path, result: OpticalBlochResult) -> Path:
    if result.physical_params is not None and result.solver_params is not None:
        amplitude_tag = format_value_tag(result.physical_params.field_MV_per_cm)
        detuning_tag = format_value_tag(result.solver_params.detuning_fs_inv)
    else:
        amplitude_tag = format_value_tag(result.parameters.field_amplitude)
        detuning_tag = format_value_tag(result.parameters.detuning)
    return output_dir / f"comparison_E0_{amplitude_tag}_Delta_{detuning_tag}.png"


def save_parameter_summary(results: list[OpticalBlochResult], output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    data = [result.parameter_summary_dict() for result in results]
    output.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return output


__all__ = ["default_output_path", "save_parameter_summary"]
