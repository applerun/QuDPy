"""物理电场 active-window 检测工具。

本模块只做电场时间域分析，不知道 TA / 2DES / solver 的具体 workflow。
它可以用于多脉冲实验中检测每个子场的亮区，也可以用于其它 field diagnostic。

共同阈值约定：
    对一组 field，先计算每个 field 在检测时间轴上的最大绝对幅值，取其中最小值
    作为共同参考幅值。active window 的绝对阈值为：

        ``rel_threshold * min_i(max(abs(E_i)))``

    因此强 pump 和弱 probe 使用同一个绝对暗区标准，避免弱 probe 因按自身峰值
    归一化而得到过宽的亮区。

窗口语义：
    ``ActiveWindow`` 表示一个连续 active 时间窗。一个 field 可以产生多个
    ``ActiveWindow``。多个 field 的窗口如果重叠，或间隔小于等于
    ``merge_gap_fs``，会在本模块中合并为一个窗口。合并后的 ``name`` 只由
    原始 field 名称通过 ``"_and_"`` 拼接得到，例如 ``pump_and_probe``，不会
    自动添加 ``active_`` 或 ``window_`` 等实验层前缀。
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np


from .lab_fields import FieldPhyRoot
from .field_series import FieldPhySeries




_NAME_JOINER = "_and_"


@dataclass(frozen=True)
class ActiveWindow:
    """物理 field 的连续 active 时间窗。

    ``name`` 只表示原始 field 名称，或多个原始 field 名称的组合。例如
    ``pump``、``probe``、``pump_and_probe``。实验层如果需要 active piece 名称，
    应在外部自行添加 ``active_`` 前缀。
    """

    start_fs: float
    end_fs: float
    name: str | None = None

    def __post_init__(self) -> None:
        start = float(self.start_fs)
        end = float(self.end_fs)
        if end < start:
            raise ValueError("ActiveWindow.end_fs must be >= start_fs.")
        object.__setattr__(self, "start_fs", start)
        object.__setattr__(self, "end_fs", end)
        if self.name is not None:
            object.__setattr__(self, "name", str(self.name))

    @property
    def duration_fs(self) -> float:
        """窗口持续时间。"""

        return float(self.end_fs) - float(self.start_fs)

    def padded(self, padding_fs: float) -> "ActiveWindow":
        """返回两端扩展后的窗口，并保留 ``name``。"""

        padding = float(padding_fs)
        return ActiveWindow(
            start_fs=float(self.start_fs) - padding,
            end_fs=float(self.end_fs) + padding,
            name=self.name,
        )

    def overlaps(self, other: "ActiveWindow") -> bool:
        """判断两个窗口是否重叠或相接。"""

        return not (self.end_fs < other.start_fs or other.end_fs < self.start_fs)

    def gap_to(self, other: "ActiveWindow") -> float:
        """返回当前窗口到后一个窗口之间的 dark gap。

        如果两个窗口重叠或相接，返回 0。
        """

        if self.end_fs >= other.start_fs:
            return 0.0
        return float(other.start_fs) - float(self.end_fs)

    def is_before(self, other: "ActiveWindow") -> bool:
        """判断当前窗口是否完全早于另一个窗口。"""

        return self.end_fs < other.start_fs

    def to_dict(self) -> dict[str, float | str | None]:
        return {
            "start_fs": float(self.start_fs),
            "end_fs": float(self.end_fs),
            "name": self.name,
        }


@dataclass(frozen=True)
class FieldActiveWindowSettings:
    """物理电场 active-window 检测设置。

    ``rel_threshold`` 只用于 field active window 判定。对于一组 field，共同参考
    幅值取 ``min(max(abs(E_i)))``。因此暗区判据为：

        ``abs(E_i(t)) < rel_threshold * min_i(max(abs(E_i)))``

    ``merge_gap_fs`` 用于把被短 dark gap 分开的 active segments 合并，常用于
    避免 lab-frame carrier 零点把一个 pulse envelope 切成许多碎窗口。

    ``min_window_fs`` 用于过滤过短的毛刺窗口。

    ``force_single_window`` 会在所有 field window 检测、合并后，将最终结果强制
    合并为一个总包络窗口。它适合 probe readout window 这类只需要一个总包络的
    场景；默认关闭，以免吞掉真实的 dark interval。
    """

    rel_threshold: float = 1e-3
    padding_fs: float = 0.0
    dt_fs: float = 0.2
    t_start_fs: float = -1000.0
    t_end_fs: float = 1000.0
    merge_gap_fs: float = 0.0
    force_single_window: bool = False

    def __post_init__(self) -> None:
        if self.rel_threshold <= 0:
            raise ValueError("rel_threshold must be positive.")
        if self.dt_fs <= 0:
            raise ValueError("dt_fs must be positive.")
        if self.padding_fs < 0:
            raise ValueError("padding_fs must be non-negative.")
        if self.merge_gap_fs < 0:
            raise ValueError("merge_gap_fs must be non-negative.")

        t_start = float(self.t_start_fs)
        t_end = float(self.t_end_fs)
        if t_start >= t_end:
            raise ValueError(
                "t_start_fs must be smaller than t_end_fs, "
                f"got {self.t_start_fs} and {self.t_end_fs}."
            )

        object.__setattr__(self, "t_start_fs", t_start)
        object.__setattr__(self, "t_end_fs", t_end)
        object.__setattr__(self, "dt_fs", float(self.dt_fs))
        object.__setattr__(self, "rel_threshold", float(self.rel_threshold))
        object.__setattr__(self, "padding_fs", float(self.padding_fs))
        object.__setattr__(self, "merge_gap_fs", float(self.merge_gap_fs))
        object.__setattr__(self, "force_single_window", bool(self.force_single_window))


def _as_field_tuple(field_or_fields: FieldPhyRoot | Iterable[FieldPhyRoot]) -> tuple[FieldPhyRoot, ...]:
    """把任意物理 field 输入标准化为 field tuple。"""

    if isinstance(field_or_fields, FieldPhySeries):
        return tuple(field_or_fields.fields)

    if isinstance(field_or_fields, FieldPhyRoot):
        return (field_or_fields,)

    fields = tuple(field_or_fields)
    if not fields:
        raise ValueError("field_or_fields must contain at least one field.")

    for field in fields:
        if not isinstance(field, FieldPhyRoot):
            raise TypeError("all items in field_or_fields must be FieldPhyRoot instances.")

    return fields


def _field_names_for_input(
    field_or_fields: FieldPhyRoot | Iterable[FieldPhyRoot],
    *,
    field_names: Iterable[str] | None,
) -> tuple[str, ...]:
    """为输入 fields 生成与 field tuple 对齐的名称。"""

    fields = _as_field_tuple(field_or_fields)

    if field_names is not None:
        names = tuple(str(name) for name in field_names)
    elif isinstance(field_or_fields, FieldPhySeries):
        if field_or_fields.sub_field_names is None:
            names = tuple(f"field_{idx}" for idx in range(len(fields)))
        else:
            names = tuple(str(name) for name in field_or_fields.sub_field_names)
    elif isinstance(field_or_fields, FieldPhyRoot):
        names = (str(getattr(field_or_fields, "name", None) or "field_0"),)
    else:
        names = tuple(
            str(getattr(field, "name", None) or f"field_{idx}")
            for idx, field in enumerate(fields)
        )

    if len(names) != len(fields):
        raise ValueError("field_names length must match number of fields.")
    if len(set(names)) != len(names):
        raise ValueError("field_names must be unique.")
    return names


def make_window_time_axis(
    *,
    t_start_fs: float,
    t_end_fs: float,
    dt_fs: float,
) -> np.ndarray:
    """生成用于 active-window 检测的采样时间轴。"""

    if t_end_fs <= t_start_fs:
        raise ValueError("t_end_fs must be greater than t_start_fs.")
    if dt_fs <= 0:
        raise ValueError("dt_fs must be positive.")

    n = int(np.floor((float(t_end_fs) - float(t_start_fs)) / float(dt_fs))) + 1
    return float(t_start_fs) + float(dt_fs) * np.arange(n, dtype=float)


def common_active_threshold(
    field_or_fields: FieldPhyRoot | Iterable[FieldPhyRoot],
    t_fs: np.ndarray,
    *,
    rel_threshold: float,
) -> float:
    """计算一组 field 共用的 active-window 绝对幅值阈值。"""

    fields = _as_field_tuple(field_or_fields)

    maxima: list[float] = []
    for field in fields:
        values = np.asarray(field(t_fs), dtype=float)
        maxima.append(float(np.max(np.abs(values))))

    reference = min(maxima)
    if reference <= 0.0:
        raise ValueError("All fields must have non-zero amplitude for active-window detection.")
    return float(rel_threshold) * reference


def _split_active_index_segments(active: np.ndarray) -> tuple[np.ndarray, ...]:
    """把 active mask 中的 True index 拆成连续 index 段。"""

    idx = np.where(active)[0]
    if idx.size == 0:
        return ()

    breaks = np.where(np.diff(idx) > 1)[0]
    return tuple(np.split(idx, breaks + 1))


def _name_parts(name: str | None) -> tuple[str, ...]:
    """把 window name 拆成原始 field 名称序列。"""

    if name is None or name == "":
        return ()
    return tuple(part for part in str(name).split(_NAME_JOINER) if part)


def _join_unique_names(windows: Iterable[ActiveWindow]) -> str | None:
    """按首次出现顺序合并 window 原始名称。"""

    names: list[str] = []
    for window in windows:
        for part in _name_parts(window.name):
            if part not in names:
                names.append(part)
    return _NAME_JOINER.join(names) if names else None


def _merge_window_group(windows: list[ActiveWindow]) -> ActiveWindow:
    """把一组窗口合并成一个窗口。"""

    if not windows:
        raise ValueError("windows must not be empty.")
    return ActiveWindow(
        start_fs=min(window.start_fs for window in windows),
        end_fs=max(window.end_fs for window in windows),
        name=_join_unique_names(windows),
    )





def merge_active_windows(
    windows: Iterable[ActiveWindow],
    *,
    merge_gap_fs: float = 0.0,
    force_single_window: bool = False,
) -> tuple[ActiveWindow, ...]:
    """合并重叠或被短 gap 分开的 active windows。

    合并后的 ``name`` 只由原始 field names 通过 ``"_and_"`` 拼接得到。
    不会添加 ``active_``、``window_`` 等实验层前缀。
    """

    gap_limit = float(merge_gap_fs)
    if gap_limit < 0:
        raise ValueError("merge_gap_fs must be non-negative.")

    ordered = sorted(tuple(windows), key=lambda item: item.start_fs)
    if not ordered:
        return ()

    if force_single_window:
        return (_merge_window_group(list(ordered)),)

    groups: list[list[ActiveWindow]] = []
    current: list[ActiveWindow] = [ordered[0]]
    current_end = float(ordered[0].end_fs)

    for window in ordered[1:]:
        gap = max(0.0, float(window.start_fs) - current_end)
        if gap <= gap_limit:
            current.append(window)
            current_end = max(current_end, float(window.end_fs))
        else:
            groups.append(current)
            current = [window]
            current_end = float(window.end_fs)
    groups.append(current)

    return tuple(_merge_window_group(group) for group in groups)


def detect_field_active_windows(
    field: FieldPhyRoot,
    t_fs: np.ndarray,
    *,
    abs_threshold: float,
    padding_fs: float = 0.0,
    merge_gap_fs: float = 0.0,
    force_single_window: bool = False,
    name: str | None = None,
) -> tuple[ActiveWindow, ...]:
    """用绝对幅值阈值检测单个 field 的一个或多个 active windows。"""

    if not isinstance(field, FieldPhyRoot):
        raise TypeError("field must be a FieldPhyRoot instance.")

    values = np.asarray(field(t_fs), dtype=float)
    active = np.abs(values) >= float(abs_threshold)
    segments = _split_active_index_segments(active)
    if not segments:
        return ()

    raw_windows = tuple(
        ActiveWindow(
            start_fs=float(t_fs[segment[0]]),
            end_fs=float(t_fs[segment[-1]]),
            name=name,
        )
        for segment in segments
    )

    merged = merge_active_windows(raw_windows, merge_gap_fs=float(merge_gap_fs))
    padded = tuple(window.padded(float(padding_fs)) for window in merged)

    # padding 可能让同一 field 的相邻窗口重新重叠，因此再合并一次。
    return merge_active_windows(
        padded,
        merge_gap_fs=0.0,
        force_single_window=bool(force_single_window),
    )


def detect_active_windows(
    field_or_fields: FieldPhyRoot | Iterable[FieldPhyRoot],
    *,
    settings: FieldActiveWindowSettings,
    field_names: Iterable[str] | None = None,
) -> tuple[tuple[ActiveWindow, ...], float]:
    """检测一个或多个 field 的 active windows。

    返回 ``(windows, threshold)``。``windows`` 是一层 flat tuple，元素可以是
    单场窗口，也可以是多个 field 重叠后自动合并得到的窗口。

    如果输入为 ``FieldPhySeries``，本函数只展开其子场并使用
    ``series.sub_field_names`` 作为默认名称；输出结构不会因为输入是
    ``FieldPhySeries`` 而特殊化。
    """

    fields = _as_field_tuple(field_or_fields)
    names = _field_names_for_input(field_or_fields, field_names=field_names)

    t_fs = make_window_time_axis(
        t_start_fs=settings.t_start_fs,
        t_end_fs=settings.t_end_fs,
        dt_fs=settings.dt_fs,
    )
    threshold = common_active_threshold(
        fields,
        t_fs,
        rel_threshold=float(settings.rel_threshold),
    )

    detected: list[ActiveWindow] = []
    for key, field in zip(names, fields):
        detected.extend(
            detect_field_active_windows(
                field,
                t_fs,
                abs_threshold=threshold,
                padding_fs=float(settings.padding_fs),
                merge_gap_fs=float(settings.merge_gap_fs),
                force_single_window=False,
                name=key,
            )
        )

    merged = merge_active_windows(
        detected,
        merge_gap_fs=float(settings.merge_gap_fs),
        force_single_window=bool(settings.force_single_window),
    )
    return merged, float(threshold)




__all__ = [
    "ActiveWindow",
    "FieldActiveWindowSettings",
    "common_active_threshold",
    "detect_active_windows",
    "detect_field_active_windows",
    "make_window_time_axis",
    "merge_active_windows",
]


if __name__ == "__main__":
    import json

    class DebugGaussianPulse(FieldPhyRoot):
        """用于调试 active-window detection 的简单物理 field。"""

        def __init__(
            self,
            *,
            name: str,
            centers_fs: tuple[float, ...],
            sigma_fs: float = 20.0,
            carrier_period_fs: float = 4.0,
            amplitude: float = 1.0,
        ) -> None:
            self.name = name
            self.centers_fs = tuple(float(item) for item in centers_fs)
            self.sigma_fs = float(sigma_fs)
            self.carrier_period_fs = float(carrier_period_fs)
            self.amplitude = float(amplitude)

        def physical_E_MV_per_cm(self, t_fs: np.ndarray) -> np.ndarray:
            t = np.asarray(t_fs, dtype=float)
            field = np.zeros_like(t)

            for center in self.centers_fs:
                envelope = np.exp(-0.5 * ((t - center) / self.sigma_fs) ** 2)
                carrier = np.cos(2.0 * np.pi * (t - center) / self.carrier_period_fs)
                field += self.amplitude * envelope * carrier

            return field

        @property
        def reference_MV_per_cm(self) -> float | None:
            return self.amplitude

        @property
        def normalization_rate_candidates_fs_inv(self) -> tuple[float, ...]:
            return ()

        def __repr__(self) -> str:
            return (
                f"DebugGaussianPulse(name={self.name!r}, "
                f"centers_fs={self.centers_fs!r}, sigma_fs={self.sigma_fs!r})"
            )

    def _print_windows(title: str, windows: tuple[ActiveWindow, ...], threshold: float) -> None:
        print(f"\n{title}")
        print(f"threshold = {threshold:.6g}")
        print(json.dumps([window.to_dict() for window in windows], indent=2, ensure_ascii=False))

    # 示例 1：单个 field 内有两个分离 pulse。
    # merge_gap_fs 小时，应得到两个 windows。
    two_pulse_field = DebugGaussianPulse(
        name="probe",
        centers_fs=(-100.0, 100.0),
        sigma_fs=12.0,
        carrier_period_fs=4.0,
        amplitude=1.0,
    )

    settings_separate = FieldActiveWindowSettings(
        rel_threshold=1e-2,
        padding_fs=0.0,
        dt_fs=0.2,
        t_start_fs=-200.0,
        t_end_fs=200.0,
        merge_gap_fs=5.0,
        force_single_window=False,
    )

    windows, threshold = detect_active_windows(
        two_pulse_field,
        settings=settings_separate,
        field_names=("probe",),
    )
    _print_windows("Single field with two separated pulses", windows, threshold)

    # 示例 2：force_single_window=True 时，强行合并为一个总包络 window。
    settings_force_single = FieldActiveWindowSettings(
        rel_threshold=1e-2,
        padding_fs=0.0,
        dt_fs=0.2,
        t_start_fs=-200.0,
        t_end_fs=200.0,
        merge_gap_fs=5.0,
        force_single_window=True,
    )

    windows, threshold = detect_active_windows(
        two_pulse_field,
        settings=settings_force_single,
        field_names=("probe",),
    )
    _print_windows("Single field with force_single_window=True", windows, threshold)

    # 示例 3：pump / probe 两个 field 有重叠，应该合并成 pump_and_probe。
    pump = DebugGaussianPulse(
        name="pump",
        centers_fs=(-10.0,),
        sigma_fs=20.0,
        carrier_period_fs=4.0,
        amplitude=1.0,
    )
    probe = DebugGaussianPulse(
        name="probe",
        centers_fs=(10.0,),
        sigma_fs=20.0,
        carrier_period_fs=4.0,
        amplitude=0.5,
    )

    series = FieldPhySeries(
        fields=(pump, probe),
        sub_field_names=("pump", "probe"),
    )

    settings_overlap = FieldActiveWindowSettings(
        rel_threshold=1e-2,
        padding_fs=0.0,
        dt_fs=0.2,
        t_start_fs=-100.0,
        t_end_fs=100.0,
        merge_gap_fs=5.0,
        force_single_window=False,
    )

    windows, threshold = detect_active_windows(series, settings=settings_overlap)
    _print_windows("Overlapped pump/probe FieldPhySeries", windows, threshold)

    # 示例 4：pump / probe 分离，应该得到两个 windows。
    pump_far = DebugGaussianPulse(
        name="pump",
        centers_fs=(-100.0,),
        sigma_fs=15.0,
        carrier_period_fs=4.0,
        amplitude=1.0,
    )
    probe_far = DebugGaussianPulse(
        name="probe",
        centers_fs=(100.0,),
        sigma_fs=15.0,
        carrier_period_fs=4.0,
        amplitude=0.5,
    )

    series_far = FieldPhySeries(
        fields=(pump_far, probe_far),
        sub_field_names=("pump", "probe"),
    )

    windows, threshold = detect_active_windows(series_far, settings=settings_separate)
    _print_windows("Separated pump/probe FieldPhySeries", windows, threshold)