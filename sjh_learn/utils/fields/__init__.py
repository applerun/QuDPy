"""Input field and RWA drive helpers.

User-facing classes use physical units. Solver-internal code-unit callables
live in :mod:`solver_inputs` and are imported explicitly by solver/model code.
"""

from __future__ import annotations

from typing import Any

from .lab_fields import (
    CarrierFieldPhysical,
    CompositeLabFieldPhysical,
    GaussianCarrierFieldPhysical,
    lab_field_from_dict,
    lab_total_field_array,
    lab_total_field_value,
)
from .rwa_drives import (
    ConstantRwaDrivePhysical,
    GaussianRwaDrivePhysical,
    drive_from_dict,
    make_rwa_drive_from_physical_field,
)
from .solver_inputs import (
    CodeCarrierField,
    CodeCompositeField,
    CodeConstantDrive,
    CodeGaussianCarrierField,
    CodeGaussianDrive,
    FieldConfig,
    electric_field_array,
    electric_field_value,
    envelope_value,
    solver_input_from_dict,
    total_electric_field_array,
    total_electric_field_value,
)

# Backward-compatible public names now point to physical user-facing classes.
CarrierField = CarrierFieldPhysical
GaussianCarrierField = GaussianCarrierFieldPhysical
CompositeField = CompositeLabFieldPhysical
ConstantDrive = ConstantRwaDrivePhysical
GaussianDrive = GaussianRwaDrivePhysical


def field_from_dict(data: dict[str, Any]):
    class_name = data.get("class")
    if class_name in {"CarrierFieldPhysical", "GaussianCarrierFieldPhysical", "CompositeLabFieldPhysical"}:
        return lab_field_from_dict(data)
    if class_name in {"ConstantRwaDrivePhysical", "GaussianRwaDrivePhysical"}:
        return drive_from_dict(data)
    return solver_input_from_dict(data)


__all__ = [
    "CarrierFieldPhysical",
    "GaussianCarrierFieldPhysical",
    "CompositeLabFieldPhysical",
    "ConstantRwaDrivePhysical",
    "GaussianRwaDrivePhysical",
    "make_rwa_drive_from_physical_field",
    "CodeConstantDrive",
    "CodeGaussianDrive",
    "CodeCarrierField",
    "CodeGaussianCarrierField",
    "CodeCompositeField",
    "CarrierField",
    "GaussianCarrierField",
    "CompositeField",
    "ConstantDrive",
    "GaussianDrive",
    "FieldConfig",
    "field_from_dict",
    "drive_from_dict",
    "lab_field_from_dict",
    "solver_input_from_dict",
    "envelope_value",
    "electric_field_value",
    "electric_field_array",
    "total_electric_field_value",
    "total_electric_field_array",
    "lab_total_field_value",
    "lab_total_field_array",
]
