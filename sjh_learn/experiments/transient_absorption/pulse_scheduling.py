"""TA delay-scan 的时间窗和命名工具。"""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class FullWindow:
    t_start_fs: float
    t_end_fs: float


@dataclass(frozen=True)
class PiecewiseWindows:
    pump_start_fs: float
    pump_end_fs: float
    dark_start_fs: float
    dark_end_fs: float
    probe_start_fs: float
    probe_end_fs: float


def safe_delay_tag(value: float) -> str:
    """把 delay 数值转换成适合文件名的短标签。"""

    value = float(value)
    if abs(value) < 1e-12:
        value = 0.0
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    return text.replace("-", "m").replace(".", "p")


def align_floor(value: float, dt_fs: float) -> float:
    return math.floor(float(value) / float(dt_fs)) * float(dt_fs)


def align_ceil(value: float, dt_fs: float) -> float:
    return math.ceil(float(value) / float(dt_fs)) * float(dt_fs)


def count_time_points(t_start_fs: float, t_end_fs: float, dt_fs: float) -> int:
    return int(round((float(t_end_fs) - float(t_start_fs)) / float(dt_fs))) + 1


def compute_pulse_centers(*, delay_fs: float, probe_center_fs: float) -> tuple[float, float]:
    """返回 pump/probe 中心时间。

    delay 约定为 pump-probe delay：delay_fs > 0 表示 pump 早于 probe。
    """

    probe_center = float(probe_center_fs)
    pump_center = probe_center - float(delay_fs)
    return pump_center, probe_center


def classify_delay_mode(*, delay_fs: float, piecewise_min_positive_delay_fs: float) -> str:
    """判断当前 delay 使用 full overlap 还是 piecewise。"""

    if float(delay_fs) > float(piecewise_min_positive_delay_fs):
        return "piecewise"
    return "full_overlap"


def compute_full_overlap_window(
    *,
    pump_center_fs: float,
    probe_center_fs: float,
    pump_half_width_fs: float,
    probe_half_width_fs: float,
    extra_padding_fs: float,
    dt_fs: float,
) -> FullWindow:
    left = min(
        float(pump_center_fs) - float(pump_half_width_fs),
        float(probe_center_fs) - float(probe_half_width_fs),
    )
    right = max(
        float(pump_center_fs) + float(pump_half_width_fs),
        float(probe_center_fs) + float(probe_half_width_fs),
    )
    return FullWindow(
        t_start_fs=align_floor(left - float(extra_padding_fs), dt_fs),
        t_end_fs=align_ceil(right + float(extra_padding_fs), dt_fs),
    )


def compute_piecewise_windows(
    *,
    pump_center_fs: float,
    probe_center_fs: float,
    pump_half_width_fs: float,
    probe_half_width_fs: float,
    dt_fs: float,
) -> PiecewiseWindows:
    pump_start = align_floor(float(pump_center_fs) - float(pump_half_width_fs), dt_fs)
    pump_end = align_ceil(float(pump_center_fs) + float(pump_half_width_fs), dt_fs)
    probe_start = align_floor(float(probe_center_fs) - float(probe_half_width_fs), dt_fs)
    probe_end = align_ceil(float(probe_center_fs) + float(probe_half_width_fs), dt_fs)

    if pump_end >= probe_start:
        raise ValueError(
            f"Piecewise window invalid: pump_end={pump_end:g} fs is not earlier than "
            f"probe_start={probe_start:g} fs."
        )

    return PiecewiseWindows(
        pump_start_fs=pump_start,
        pump_end_fs=pump_end,
        dark_start_fs=pump_end,
        dark_end_fs=probe_start,
        probe_start_fs=probe_start,
        probe_end_fs=probe_end,
    )


def check_time_points(
    *,
    key: str,
    t_start_fs: float,
    t_end_fs: float,
    dt_fs: float,
    max_points: int,
) -> None:
    n_points = count_time_points(t_start_fs, t_end_fs, dt_fs)
    if n_points > int(max_points):
        raise ValueError(
            f"{key} requires {n_points} time points, exceeding max_points={max_points}. "
            "Use piecewise propagation or shorten the saved time window."
        )
