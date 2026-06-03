"""Internal solver-unit input callables.

These classes are for Hamiltonian construction after ``ParaNormalizer`` has
converted physical inputs to solver code units. User-facing examples should
use normalized physical parameters rather than constructing these directly.
"""

from __future__ import annotations

from typing import Any

from .lab_fields import (
    CodeCarrierField,
    CodeCompositeField,
    CodeGaussianCarrierField,
    FieldConfig,
    electric_field_array,
    electric_field_value,
    envelope_value,
    total_electric_field_array,
    total_electric_field_value,
)
from .rwa_drives import CodeConstantDrive, CodeGaussianDrive


_SOLVER_INPUT_REGISTRY = {
    "CodeConstantDrive": CodeConstantDrive,
    "CodeGaussianDrive": CodeGaussianDrive,
    "CodeCarrierField": CodeCarrierField,
    "CodeGaussianCarrierField": CodeGaussianCarrierField,
    "CodeCompositeField": CodeCompositeField,
    # Legacy debug metadata names.
    "ConstantDrive": CodeConstantDrive,
    "GaussianDrive": CodeGaussianDrive,
    "CarrierField": CodeCarrierField,
    "GaussianCarrierField": CodeGaussianCarrierField,
    "CompositeField": CodeCompositeField,
}


def solver_input_from_dict(data: dict[str, Any]):
    class_name = data.get("class")
    cls = _SOLVER_INPUT_REGISTRY.get(class_name)
    if cls is None:
        raise ValueError(f"Unknown solver input class: {class_name!r}")
    return cls.from_dict(data)


__all__ = [
    "CodeConstantDrive",
    "CodeGaussianDrive",
    "CodeCarrierField",
    "CodeGaussianCarrierField",
    "CodeCompositeField",
    "FieldConfig",
    "solver_input_from_dict",
    "envelope_value",
    "electric_field_value",
    "electric_field_array",
    "total_electric_field_value",
    "total_electric_field_array",
]