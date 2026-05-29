"""外加电场定义与求值工具。"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, exp
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class FieldConfig:
    """单个外加电场的配置。"""

    amplitude: float
    omega: float
    phase: float = 0.0
    envelope: str = "constant"
    center: float | None = None
    sigma: float | None = None
    name: str = "field"


def envelope_value(t: float, field: FieldConfig) -> float:
    """返回给定时刻的包络函数值。"""
    if field.envelope == "constant":
        return 1.0
    if field.envelope == "gaussian":
        if field.center is None or field.sigma is None:
            raise ValueError("gaussian envelope 需要同时提供 center 和 sigma。")
        if field.sigma <= 0:
            raise ValueError("gaussian envelope 的 sigma 必须为正。")
        return float(exp(-((t - field.center) ** 2) / (2.0 * field.sigma**2)))
    raise ValueError(f"不支持的 envelope 类型: {field.envelope}")


def electric_field_value(t: float, field: FieldConfig) -> float:
    """返回单个电场在时刻 t 的值。"""
    return 2.0 * field.amplitude * envelope_value(t, field) * cos(field.omega * t + field.phase)


def electric_field_array(times: np.ndarray, field: FieldConfig) -> np.ndarray:
    """返回单个电场在整个时间数组上的值。"""
    return np.asarray([electric_field_value(float(time), field) for time in np.asarray(times, dtype=float)])


def _iter_fields(fields: FieldConfig | Iterable[FieldConfig]) -> tuple[FieldConfig, ...]:
    if isinstance(fields, FieldConfig):
        return (fields,)
    return tuple(fields)


def total_electric_field_value(t: float, fields: FieldConfig | Iterable[FieldConfig]) -> float:
    """返回多个电场叠加后的总场。"""
    field_list = _iter_fields(fields)
    return float(sum(electric_field_value(t, field) for field in field_list))


def total_electric_field_array(times: np.ndarray, fields: FieldConfig | Iterable[FieldConfig]) -> np.ndarray:
    """返回多个电场在整个时间数组上的总场。"""
    field_list = _iter_fields(fields)
    return np.asarray(
        [total_electric_field_value(float(time), field_list) for time in np.asarray(times, dtype=float)]
    )


__all__ = [
    "FieldConfig",
    "envelope_value",
    "electric_field_value",
    "electric_field_array",
    "total_electric_field_value",
    "total_electric_field_array",
]
