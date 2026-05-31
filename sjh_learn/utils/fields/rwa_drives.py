"""User-facing physical RWA slow-drive helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..normalization import ParaNormalizer
from .lab_fields import CarrierFieldPhysical, GaussianCarrierFieldPhysical


def _asarray_or_float(values):
    array = np.asarray(values, dtype=float)
    if array.ndim == 0:
        return float(array)
    return array


@dataclass(frozen=True)
class ConstantRwaDrivePhysical:
    """Physical CW RWA slow coupling ``g(t)`` in fs^-1."""

    amplitude_fs_inv: float
    name: str = "physical_constant_rwa_drive"
    source: str = "derived from dipole_matrix_D and field_MV_per_cm"
    envelope: str = "constant"

    def __call__(self, t_fs):
        values = np.full_like(np.asarray(t_fs, dtype=float), self.amplitude_fs_inv, dtype=float)
        return _asarray_or_float(values)

    def to_dict(self) -> dict[str, Any]:
        return {
            "class": self.__class__.__name__,
            "name": self.name,
            "amplitude_fs_inv": self.amplitude_fs_inv,
            "source": self.source,
            "envelope": self.envelope,
            "time_unit": "fs",
            "drive_unit_physical": "fs^-1",
            "amplitude_convention": "input_drive is the slow RWA coupling g(t) after removing the optical carrier.",
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConstantRwaDrivePhysical":
        return cls(
            name=data.get("name", "physical_constant_rwa_drive"),
            amplitude_fs_inv=float(data["amplitude_fs_inv"]),
            source=data.get("source", "derived from dipole_matrix_D and field_MV_per_cm"),
        )

    def to_expr(self) -> str:
        return f"{self.name}: g(t) = {self.amplitude_fs_inv:.6g} fs^-1"

    def __repr__(self) -> str:
        return self.to_expr()


@dataclass(frozen=True)
class GaussianRwaDrivePhysical:
    """Physical Gaussian-envelope RWA slow coupling ``g(t)`` in fs^-1."""

    amplitude_fs_inv: float
    pulse_center_fs: float
    pulse_sigma_fs: float
    name: str = "physical_gaussian_rwa_drive"
    source: str = "derived from dipole_matrix_D and field_MV_per_cm"
    envelope: str = "gaussian"

    def __call__(self, t_fs):
        time = np.asarray(t_fs, dtype=float)
        values = self.amplitude_fs_inv * np.exp(-((time - self.pulse_center_fs) ** 2) / (2.0 * self.pulse_sigma_fs**2))
        return _asarray_or_float(values)

    def to_dict(self) -> dict[str, Any]:
        return {
            "class": self.__class__.__name__,
            "name": self.name,
            "amplitude_fs_inv": self.amplitude_fs_inv,
            "pulse_center_fs": self.pulse_center_fs,
            "pulse_sigma_fs": self.pulse_sigma_fs,
            "source": self.source,
            "envelope": self.envelope,
            "time_unit": "fs",
            "drive_unit_physical": "fs^-1",
            "amplitude_convention": "input_drive is the slow RWA coupling g(t) after removing the optical carrier.",
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GaussianRwaDrivePhysical":
        return cls(
            name=data.get("name", "physical_gaussian_rwa_drive"),
            amplitude_fs_inv=float(data["amplitude_fs_inv"]),
            pulse_center_fs=float(data["pulse_center_fs"]),
            pulse_sigma_fs=float(data["pulse_sigma_fs"]),
            source=data.get("source", "derived from dipole_matrix_D and field_MV_per_cm"),
        )

    def to_expr(self) -> str:
        return (
            f"{self.name}: g(t) = {self.amplitude_fs_inv:.6g} fs^-1 "
            f"* exp[-(t_fs - {self.pulse_center_fs:.6g})^2 / (2 * {self.pulse_sigma_fs:.6g}^2)]"
        )

    def __repr__(self) -> str:
        return self.to_expr()


def make_rwa_drive_from_physical_field(
    field: CarrierFieldPhysical | GaussianCarrierFieldPhysical,
    projected_dipole_D: float,
):
    """Build the physical RWA slow coupling from a physical lab-frame field."""

    amplitude_fs_inv = ParaNormalizer.rabi_fs_inv_from_mu_and_field(projected_dipole_D, field.E0_MV_per_cm)
    if isinstance(field, GaussianCarrierFieldPhysical):
        return GaussianRwaDrivePhysical(
            name="physical_gaussian_rwa_drive",
            amplitude_fs_inv=amplitude_fs_inv,
            pulse_center_fs=field.pulse_center_fs,
            pulse_sigma_fs=field.pulse_sigma_fs,
        )
    return ConstantRwaDrivePhysical(
        name="physical_constant_rwa_drive",
        amplitude_fs_inv=amplitude_fs_inv,
    )


_RWA_DRIVE_REGISTRY = {
    "ConstantRwaDrivePhysical": ConstantRwaDrivePhysical,
    "GaussianRwaDrivePhysical": GaussianRwaDrivePhysical,
}


def drive_from_dict(data: dict[str, Any]):
    class_name = data.get("class")
    cls = _RWA_DRIVE_REGISTRY.get(class_name)
    if cls is None:
        raise ValueError(f"Unknown physical RWA drive class: {class_name!r}")
    return cls.from_dict(data)


__all__ = [
    "ConstantRwaDrivePhysical",
    "GaussianRwaDrivePhysical",
    "make_rwa_drive_from_physical_field",
    "drive_from_dict",
]
