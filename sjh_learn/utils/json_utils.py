"""JSON 安全序列化辅助函数。"""

from __future__ import annotations

from dataclasses import fields as dataclass_fields, is_dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np


def _complex_matrix_to_json(value: Any) -> list:
    array = np.asarray(value, dtype=np.complex128)
    return [[{"real": float(item.real), "imag": float(item.imag)} for item in row] for row in array]


def make_json_safe(value: Any) -> Any:
    """递归转换为 `json.dumps` 可处理的对象。

    该函数只处理序列化边界，不改变 metadata key 或物理语义。complex number
    使用结构化对象保存，避免丢失 imaginary part。
    """

    if type(value).__name__ == "ParaNormalizer":
        return {"class": "ParaNormalizer", "note": "runtime object omitted from JSON metadata"}
    if type(value).__name__ == "Qobj" and hasattr(value, "full") and hasattr(value, "shape"):
        return {"qobj_shape": list(value.shape), "data": _complex_matrix_to_json(value.full())}
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return make_json_safe(value.to_dict())
    if is_dataclass(value) and not isinstance(value, type):
        return make_json_safe({item.name: getattr(value, item.name) for item in dataclass_fields(value)})
    if isinstance(value, dict):
        return {str(key): make_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return make_json_safe(value.tolist())
    if isinstance(value, np.generic):
        return make_json_safe(value.item())
    if isinstance(value, complex):
        return {"real": float(np.real(value)), "imag": float(np.imag(value))}
    if isinstance(value, Path):
        return str(value)
    if callable(value):
        return {"callable_serialized": False, "repr": repr(value)}
    return value


def write_json(path: str | Path, payload: Any, *, indent: int = 2) -> None:
    """把 payload 写成 UTF-8 JSON 文件。"""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(make_json_safe(payload), indent=indent, ensure_ascii=False),
        encoding="utf-8",
    )


__all__ = ["make_json_safe", "write_json"]
