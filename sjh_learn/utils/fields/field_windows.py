"""物理电场 active window 检测工具。

本模块只做电场时间域分析，不知道 TA / 2DES / solver 的具体 workflow。
它可以用于多脉冲实验中检测每个子场的亮区，也可以用于其它 field diagnostic。

共同阈值约定：
    对一组 field，先计算每个 field 在检测时间轴上的最大绝对幅值，取其中最小值
    作为共同参考幅值。active window 的绝对阈值为：

        ``rel_threshold * min_i(max(abs(E_i)))``

    因此强 pump 和弱 probe 使用同一个绝对暗区标准，避免弱 probe 因按自身峰值
    归一化而得到过宽的亮区。
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from .lab_fields import FieldPhyRoot
from .field_series import FieldPhySeries


@dataclass(frozen = True)
class ActiveWindow:
	"""物理 field 的有效亮区。"""

	start_fs: float
	end_fs: float

	def __post_init__(self) -> None:
		if self.end_fs < self.start_fs:
			raise ValueError("ActiveWindow.end_fs must be >= start_fs.")

	def padded(self, padding_fs: float) -> "ActiveWindow":
		padding = float(padding_fs)
		return ActiveWindow(
			start_fs = float(self.start_fs) - padding,
			end_fs = float(self.end_fs) + padding,
		)

	def overlaps(self, other: "ActiveWindow") -> bool:
		return not (self.end_fs < other.start_fs or other.end_fs < self.start_fs)

	def is_before(self, other: "ActiveWindow") -> bool:
		return self.end_fs < other.start_fs

	def to_dict(self) -> dict[str, float]:
		return {"start_fs": float(self.start_fs), "end_fs": float(self.end_fs)}


@dataclass(frozen = True)
class FieldActiveWindowSettings:
	"""物理电场 active window 检测设置。

	``rel_threshold`` 只用于 field active window 判定。对于一组 field，共同参考
	幅值取 ``min(max(abs(E_i)))``。因此暗区判据为：

		``abs(E_i(t)) < rel_threshold * min_i(max(abs(E_i)))``

	``padding_fs`` 会在检测到的亮区两端额外扩展，用于抵消采样或阈值边界误差。
	"""

	rel_threshold: float = 1e-3
	padding_fs: float = 0.0
	dt_fs: float = 0.2
	t_start_fs: float = -1000.0
	t_end_fs: float = 1000.0

	def __post_init__(self) -> None:
		if self.rel_threshold <= 0:
			raise ValueError("rel_threshold must be positive.")
		if self.dt_fs <= 0:
			raise ValueError("dt_fs must be positive.")

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


def as_field_tuple(field_or_fields: FieldPhyRoot | Iterable[FieldPhyRoot]) -> tuple[FieldPhyRoot, ...]:
	"""把任意物理 field 输入标准化为 field tuple。

	支持三种输入：
		- 单个 ``FieldPhyRoot``；
		- ``FieldPhySeries``，会自动展开为其子场；
		- 任意由 ``FieldPhyRoot`` 组成的 iterable。

	这样窗口检测工具可以支持所有 physical field，而不只支持 ``FieldPhySeries``。
	"""

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


def make_window_time_axis(
		*,
		t_start_fs: float,
		t_end_fs: float,
		dt_fs: float,
) -> np.ndarray:
	"""生成用于 active window 检测的采样时间轴。"""

	if t_end_fs <= t_start_fs:
		raise ValueError("t_end_fs must be greater than t_start_fs.")
	if dt_fs <= 0:
		raise ValueError("dt_fs must be positive.")

	n = int(np.floor((float(t_end_fs) - float(t_start_fs)) / float(dt_fs))) + 1
	return float(t_start_fs) + float(dt_fs) * np.arange(n, dtype = float)


def common_active_threshold(
		field_or_fields: FieldPhyRoot | Iterable[FieldPhyRoot],
		t_fs: np.ndarray,
		*,
		rel_threshold: float,
) -> float:
	"""计算一组 field 共用的 active-window 绝对幅值阈值。"""

	fields = as_field_tuple(field_or_fields)

	maxima: list[float] = []
	for field in fields:
		values = np.asarray(field(t_fs), dtype = float)
		maxima.append(float(np.max(np.abs(values))))

	reference = min(maxima)
	if reference <= 0.0:
		raise ValueError("All fields must have non-zero amplitude for active-window detection.")
	return float(rel_threshold) * reference


def detect_field_active_window(
		field: FieldPhyRoot,
		t_fs: np.ndarray,
		*,
		abs_threshold: float,
		padding_fs: float = 0.0,
) -> ActiveWindow | None:
	"""用绝对幅值阈值检测单个 field 的 active window。"""

	if not isinstance(field, FieldPhyRoot):
		raise TypeError("field must be a FieldPhyRoot instance.")

	values = np.asarray(field(t_fs), dtype = float)
	active = np.abs(values) >= float(abs_threshold)
	if not np.any(active):
		return None

	idx = np.where(active)[0]
	window = ActiveWindow(
		start_fs = float(t_fs[idx[0]]),
		end_fs = float(t_fs[idx[-1]]),
	)
	return window.padded(float(padding_fs))


def detect_active_windows(
		field_or_fields: FieldPhyRoot | Iterable[FieldPhyRoot],
		*,
		settings: FieldActiveWindowSettings,
		field_names: Iterable[str] | None = None,
) -> tuple[dict[str, ActiveWindow | None], float]:
	"""检测一个或多个 field 的 active window。

	返回 ``(windows, threshold)``。如果输入为 ``FieldPhySeries`` 且未显式传入
	``names``，则使用 ``series.sub_field_names`` 作为 key；如果输入为单个 field，
	默认 key 为 ``field_0``；如果输入为普通 iterable，则默认 key 为
	``field_0, field_1, ...``。
	"""

	if isinstance(field_or_fields, FieldPhySeries):
		fields = tuple(field_or_fields.fields)
		if field_names is None:
			assert field_or_fields.sub_field_names is not None
			keys = tuple(str(name) for name in field_or_fields.sub_field_names)
		else:
			keys = tuple(str(name) for name in field_names)
	else:
		fields = as_field_tuple(field_or_fields)
		if field_names is None:
			keys = tuple(f"field_{idx}" for idx in range(len(fields)))
		else:
			keys = tuple(str(name) for name in field_names)

	if len(keys) != len(fields):
		raise ValueError("names length must match number of fields.")

	t_fs = make_window_time_axis(t_start_fs = settings.t_start_fs, t_end_fs = settings.t_end_fs, dt_fs = settings.dt_fs)
	threshold = common_active_threshold(
		fields,
		t_fs,
		rel_threshold = float(settings.rel_threshold),
	)

	windows: dict[str, ActiveWindow | None] = {}
	for key, field in zip(keys, fields):
		windows[key] = detect_field_active_window(
			field,
			t_fs,
			abs_threshold = threshold,
			padding_fs = float(settings.padding_fs),
		)
	return windows, threshold


def detect_series_active_windows(
		series: FieldPhySeries,
		*,
		settings: FieldActiveWindowSettings,
) -> tuple[dict[str, ActiveWindow | None], float]:
	"""兼容旧调用名：检测 ``FieldPhySeries`` 中每个子场的 active window。"""

	return detect_active_windows(series, settings = settings)


__all__ = [
	"ActiveWindow",
	"FieldActiveWindowSettings",
	"as_field_tuple",
	"common_active_threshold",
	"detect_active_windows",
	"detect_field_active_window",
	"detect_series_active_windows",
	"make_window_time_axis",
]
