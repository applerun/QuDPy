"""User-facing lab-frame optical fields in physical units."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from ..normalization import ParaNormalizer


def _asarray_or_float(values):
    array = np.asarray(values, dtype=float)
    if array.ndim == 0:
        return float(array)
    return array


@dataclass(frozen=True)
class CarrierFieldPhysical:
    """Physical CW lab-frame optical field.

    ``__call__(t_fs)`` returns ``E(t)`` in MV/cm. The amplitude convention is
    ``E(t) = 2 E0 cos(omega_L t + phase)``.
    """

    E0_MV_per_cm: float
    omega_L_fs_inv: float | None = None
    name: str = "physical_carrier_field"
    laser_energy_eV: float | None = None
    phase_rad: float = 0.0
    envelope: str = "constant"

    def __post_init__(self) -> None:
        if self.omega_L_fs_inv is None:
            if self.laser_energy_eV is None:
                raise ValueError("Either omega_L_fs_inv or laser_energy_eV must be provided.")
            object.__setattr__(
                self,
                "omega_L_fs_inv",
                ParaNormalizer.energy_eV_to_fs_inv(float(self.laser_energy_eV)),
            )

    def __call__(self, t_fs):
        t = np.asarray(t_fs, dtype=float)
        values = 2.0 * self.E0_MV_per_cm * np.cos(self.omega_L_fs_inv * t + self.phase_rad)
        return _asarray_or_float(values)

    def to_dict(self) -> dict[str, Any]:
        return {
            "class": self.__class__.__name__,
            "name": self.name,
            "E0_MV_per_cm": self.E0_MV_per_cm,
            "peak_E_MV_per_cm": 2.0 * self.E0_MV_per_cm,
            "omega_L_fs_inv": self.omega_L_fs_inv,
            "laser_energy_eV": self.laser_energy_eV,
            "phase_rad": self.phase_rad,
            "envelope": self.envelope,
            "time_unit": "fs",
            "field_unit": "MV/cm",
            "amplitude_convention": "field_MV_per_cm is E0 in E(t) = 2 E0 f(t) cos(omega_L t + phase).",
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CarrierFieldPhysical":
        return cls(
            name=data.get("name", "physical_carrier_field"),
            E0_MV_per_cm=float(data["E0_MV_per_cm"]),
            omega_L_fs_inv=float(data["omega_L_fs_inv"]),
            laser_energy_eV=data.get("laser_energy_eV"),
            phase_rad=float(data.get("phase_rad", 0.0)),
        )

    def to_expr(self) -> str:
        return (
            f"{self.name}: E(t) = 2 * {self.E0_MV_per_cm:.6g} MV/cm "
            f"* cos({self.omega_L_fs_inv:.6g} fs^-1 * t_fs + {self.phase_rad:.6g})"
        )

    def __repr__(self) -> str:
        return self.to_expr()


@dataclass(frozen=True)
class GaussianCarrierFieldPhysical:
    """Physical Gaussian-envelope lab-frame optical field."""

    E0_MV_per_cm: float
    pulse_center_fs: float
    pulse_sigma_fs: float
    omega_L_fs_inv: float | None = None
    name: str = "physical_gaussian_carrier_field"
    laser_energy_eV: float | None = None
    phase_rad: float = 0.0
    envelope: str = "gaussian"

    def __post_init__(self) -> None:
        if self.omega_L_fs_inv is None:
            if self.laser_energy_eV is None:
                raise ValueError("Either omega_L_fs_inv or laser_energy_eV must be provided.")
            object.__setattr__(
                self,
                "omega_L_fs_inv",
                ParaNormalizer.energy_eV_to_fs_inv(float(self.laser_energy_eV)),
            )

    def envelope_values(self, t_fs):
        t = np.asarray(t_fs, dtype=float)
        values = np.exp(-((t - self.pulse_center_fs) ** 2) / (2.0 * self.pulse_sigma_fs**2))
        return _asarray_or_float(values)

    def __call__(self, t_fs):
        t = np.asarray(t_fs, dtype=float)
        values = (
            2.0
            * self.E0_MV_per_cm
            * np.exp(-((t - self.pulse_center_fs) ** 2) / (2.0 * self.pulse_sigma_fs**2))
            * np.cos(self.omega_L_fs_inv * t + self.phase_rad)
        )
        return _asarray_or_float(values)

    def to_dict(self) -> dict[str, Any]:
        return {
            "class": self.__class__.__name__,
            "name": self.name,
            "E0_MV_per_cm": self.E0_MV_per_cm,
            "peak_E_MV_per_cm": 2.0 * self.E0_MV_per_cm,
            "omega_L_fs_inv": self.omega_L_fs_inv,
            "laser_energy_eV": self.laser_energy_eV,
            "phase_rad": self.phase_rad,
            "envelope": self.envelope,
            "pulse_center_fs": self.pulse_center_fs,
            "pulse_sigma_fs": self.pulse_sigma_fs,
            "time_unit": "fs",
            "field_unit": "MV/cm",
            "amplitude_convention": "field_MV_per_cm is E0 in E(t) = 2 E0 f(t) cos(omega_L t + phase).",
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GaussianCarrierFieldPhysical":
        return cls(
            name=data.get("name", "physical_gaussian_carrier_field"),
            E0_MV_per_cm=float(data["E0_MV_per_cm"]),
            omega_L_fs_inv=float(data["omega_L_fs_inv"]),
            laser_energy_eV=data.get("laser_energy_eV"),
            phase_rad=float(data.get("phase_rad", 0.0)),
            pulse_center_fs=float(data["pulse_center_fs"]),
            pulse_sigma_fs=float(data["pulse_sigma_fs"]),
        )

    def to_expr(self) -> str:
        return (
            f"{self.name}: E(t) = 2 * {self.E0_MV_per_cm:.6g} MV/cm "
            f"* exp[-(t_fs - {self.pulse_center_fs:.6g})^2 / (2 * {self.pulse_sigma_fs:.6g}^2)] "
            f"* cos({self.omega_L_fs_inv:.6g} fs^-1 * t_fs + {self.phase_rad:.6g})"
        )

    def __repr__(self) -> str:
        return self.to_expr()


@dataclass(frozen=True)
class CompositeLabFieldPhysical:
    """Sum of physical lab-frame fields."""

    fields: tuple[CarrierFieldPhysical | GaussianCarrierFieldPhysical, ...]
    name: str = "physical_composite_lab_field"

    def __call__(self, t_fs):
        total = sum(field(t_fs) for field in self.fields)
        return _asarray_or_float(total)

    def to_dict(self) -> dict[str, Any]:
        return {
            "class": self.__class__.__name__,
            "name": self.name,
            "fields": [field.to_dict() for field in self.fields],
            "time_unit": "fs",
            "field_unit": "MV/cm",
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CompositeLabFieldPhysical":
        return cls(
            name=data.get("name", "physical_composite_lab_field"),
            fields=tuple(lab_field_from_dict(item) for item in data["fields"]),
        )

    def to_expr(self) -> str:
        return f"{self.name}: " + " + ".join(field.to_expr() for field in self.fields)

    def __repr__(self) -> str:
        return self.to_expr()


_LAB_FIELD_REGISTRY = {
    "CarrierFieldPhysical": CarrierFieldPhysical,
    "GaussianCarrierFieldPhysical": GaussianCarrierFieldPhysical,
    "CompositeLabFieldPhysical": CompositeLabFieldPhysical,
}


def lab_field_from_dict(data: dict[str, Any]):
    class_name = data.get("class")
    cls = _LAB_FIELD_REGISTRY.get(class_name)
    if cls is None:
        raise ValueError(f"Unknown physical lab field class: {class_name!r}")
    return cls.from_dict(data)


def lab_total_field_value(t_fs: float, fields: Iterable[CarrierFieldPhysical | GaussianCarrierFieldPhysical]) -> float:
    return float(sum(field(t_fs) for field in fields))


def lab_total_field_array(t_fs, fields: Iterable[CarrierFieldPhysical | GaussianCarrierFieldPhysical]) -> np.ndarray:
    times = np.asarray(t_fs, dtype=float)
    total = np.zeros_like(times, dtype=float)
    for field in fields:
        total = total + np.asarray(field(times), dtype=float)
    return total


__all__ = [
    "CarrierFieldPhysical",
    "GaussianCarrierFieldPhysical",
    "CompositeLabFieldPhysical",
    "lab_field_from_dict",
    "lab_total_field_value",
    "lab_total_field_array",
]
