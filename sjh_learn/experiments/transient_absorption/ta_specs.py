"""TA workflow 使用的 specification 数据结构。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


@dataclass(frozen=True)
class TaSegmentSpec:
    """单个可运行 segment 的描述。

    一个 delay case 可以包含一个 full segment，也可以包含 pump、dark、probe
    三个 segment。此对象只描述任务，不负责运行。
    """

    segment_key: str
    role: Literal["full", "pump", "dark", "probe", "probe_only"]
    field: Any
    t_start_fs: float
    t_end_fs: float
    dt_fs: float
    params: Any
    checkpoint_path: Path | None = None


@dataclass(frozen=True)
class TaDelayCaseSpec:
    """某个 delay 对应的完整计算任务。"""

    case_key: str
    delay_fs: float
    mode: Literal["full_overlap", "piecewise"]
    pump_center_fs: float
    probe_center_fs: float
    segments: tuple[TaSegmentSpec, ...]


@dataclass(frozen=True)
class TaDelayScanOutputs:
    """delay-scan runner 的主要输出路径。"""

    output_dir: Path
    simulation_dir: Path
    res_per_delay_dir: Path
    checkpoints_dir: Path
    final_output_dir: Path
    case_specs_csv: Path
    segment_summary_json: Path
    all_difference_spectra_csv: Path
    workflow_metadata_json: Path


@dataclass
class TaDelayResultRecord:
    """单个 delay 的运行结果索引。"""

    case_key: str
    delay_fs: float
    mode: str
    case_dir: Path
    difference_spectrum_csv: Path
