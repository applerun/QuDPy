"""物理 field 线性叠加容器。

本模块只定义通用的物理场叠加对象，不包含 TA / 2DES 等具体实验语义。
``FieldPhySeries`` 仍然是 ``FieldPhyRoot``，因此可直接传入
``NLevelPhysicalParams(..., field=...)``。

边界说明：
    - ``FieldPhySeries`` 表示同一时刻多个物理 field 的线性叠加。
    - 它不表示分段传播，也不表示 delay scan 或 parameter sweep。
    - TA / 2DES 等 specific field 应放在 ``fields/specific/`` 下。
    - active window 检测属于电场分析工具，应放在 ``field_windows.py``。
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np

from .lab_fields import FieldPhyRoot


def _metadata_copy(metadata: dict[str, Any] | None) -> dict[str, Any]:
    """复制 metadata，避免外部 dict 被内部对象共享修改。"""

    return dict(metadata or {})


@dataclass(frozen=True)
class FieldPhySeries(FieldPhyRoot):
    """多个物理 field 的线性叠加。

    ``FieldPhySeries`` 是 physical field 层的组合对象，不是 solver code-unit
    field，也不表示 piecewise schedule。它只表示：

        ``E_total(t_fs) = sum_k E_k(t_fs)``

    支持按 index 或 subfield name 提取子场，便于外部 workflow 对 pump、probe
    或其它 pulse 分别做 active window 检测。
    """

    fields: tuple[FieldPhyRoot, ...]
    sub_field_names: tuple[str, ...] | None = None
    name: str = "field_phy_series"
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        fields = tuple(self.fields)
        if not fields:
            raise ValueError("FieldPhySeries requires at least one subfield.")

        for field in fields:
            if not isinstance(field, FieldPhyRoot):
                raise TypeError("FieldPhySeries.fields must contain FieldPhyRoot instances.")
        object.__setattr__(self, "fields", fields)

        if self.sub_field_names is None:
            names: list[str] = []
            for idx, field in enumerate(fields):
                payload = field.to_dict()
                names.append(str(payload.get("name") or getattr(field, "name", f"field_{idx}")))
            object.__setattr__(self, "sub_field_names", tuple(names))
            return

        names = tuple(str(name) for name in self.sub_field_names)
        if len(names) != len(fields):
            raise ValueError("sub_field_names length must match fields length.")
        if len(set(names)) != len(names):
            raise ValueError("sub_field_names must be unique.")
        object.__setattr__(self, "sub_field_names", names)

    @property
    def reference_MV_per_cm(self) -> float | None:
        """返回叠加场的参考幅度。

        对线性叠加场，采用各子场参考幅度绝对值之和作为保守归一化尺度。
        只要任一子场没有 reference，则返回 None。
        """

        references: list[float] = []
        for field in self.fields:
            reference = field.reference_MV_per_cm
            if reference is None:
                return None
            references.append(abs(float(reference)))
        total = sum(references)
        return None if total == 0.0 else float(total)

    @property
    def normalization_rate_candidates_fs_inv(self) -> tuple[float, ...]:
        """合并所有子场给出的 auto-scale 速率候选。"""

        candidates: list[float] = []
        for field in self.fields:
            candidates.extend(field.normalization_rate_candidates_fs_inv)
        return tuple(candidates)

    def physical_E_MV_per_cm(self, t_fs: np.ndarray) -> np.ndarray:
        total = np.zeros_like(t_fs, dtype=float)
        for field in self.fields:
            total = total + field(t_fs)
        return total

    def get_field(self, key: int | str) -> FieldPhyRoot:
        """按 index 或 subfield name 获取子场。"""

        if isinstance(key, int):
            return self.fields[key]
        if isinstance(key, str):
            assert self.sub_field_names is not None
            try:
                idx = self.sub_field_names.index(key)
            except ValueError as exc:
                raise KeyError(f"Unknown sub_field_name: {key!r}") from exc
            return self.fields[idx]
        raise TypeError("key must be int or str.")

    def __getitem__(self, key: int | str) -> FieldPhyRoot:
        return self.get_field(key)

    def __iter__(self) -> Iterator[FieldPhyRoot]:
        return iter(self.fields)

    def __len__(self) -> int:
        return len(self.fields)

    def __repr__(self) -> str:
        assert self.sub_field_names is not None
        items = ", ".join(
            f"{name}={field!r}"
            for name, field in zip(self.sub_field_names, self.fields)
        )
        return f"{self.__class__.__name__}({items})"

    def to_dict(self) -> dict[str, Any]:
        metadata = _metadata_copy(self.metadata)
        rebuildable = all(bool(field.to_dict().get("rebuildable", False)) for field in self.fields)
        return {
            "class": self.__class__.__name__,
            "repr": repr(self),
            "name": self.name,
            "time_unit": self.time_unit,
            "field_unit": self.field_unit,
            "rebuildable": rebuildable,
            "sub_field_names": list(self.sub_field_names or ()),
            "fields": [field.to_dict() for field in self.fields],
            "expression": "E_total(t_fs) = sum_k E_k(t_fs)",
            "description": metadata.get("description"),
            "metadata": metadata,
        }


__all__ = [
    "FieldPhySeries",
    "_metadata_copy",
]
