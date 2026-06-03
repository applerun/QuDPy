"""Internal solver-unit lab-frame field callables."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


def _asarray_or_float(values):
    array = np.asarray(values, dtype = float)
    if array.ndim == 0:
        return float(array)
    return array


@dataclass(frozen = True)
class CodeCarrierField:
    """Internal solver-unit lab-frame carrier."""

    name: str = "code_carrier_field"
    amplitude_code: float = 1.0
    omega_code: float = 1.0
    phase: float = 0.0
    domain: str = "solver_code"
    time_unit: str = "code"
    amplitude_unit: str = "code"
    amplitude: float | None = None
    omega: float | None = None

    def __post_init__(self) -> None:
        if self.amplitude is not None:
            object.__setattr__(self, "amplitude_code", float(self.amplitude))
        if self.omega is not None:
            object.__setattr__(self, "omega_code", float(self.omega))

    def __call__(self, t):
        time = np.asarray(t, dtype = float)
        values = 2.0 * self.amplitude_code * np.cos(self.omega_code * time + self.phase)
        return _asarray_or_float(values)

    def to_dict(self) -> dict[str, Any]:
        return {
            "class"         : self.__class__.__name__,
            "name"          : self.name,
            "amplitude_code": self.amplitude_code,
            "omega_code"    : self.omega_code,
            "phase"         : self.phase,
            "domain"        : self.domain,
            "time_unit"     : self.time_unit,
            "amplitude_unit": self.amplitude_unit,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CodeCarrierField":
        return cls(
            name = data.get("name", "code_carrier_field"),
            amplitude_code = float(data.get("amplitude_code", data.get("amplitude", 1.0))),
            omega_code = float(data.get("omega_code", data.get("omega", 1.0))),
            phase = float(data.get("phase", 0.0)),
            domain = data.get("domain", "solver_code"),
            time_unit = data.get("time_unit", "code"),
            amplitude_unit = data.get("amplitude_unit", "code"),
        )

    def to_expr(self) -> str:
        return f"{self.name}: E_code(t) = 2 * {self.amplitude_code:.6g} * cos({self.omega_code:.6g} * t + {self.phase:.6g})"

    def __repr__(self) -> str:
        return self.to_expr()


@dataclass(frozen = True)
class CodeGaussianCarrierField:
    """Internal solver-unit Gaussian lab-frame carrier."""

    name: str = "code_gaussian_carrier_field"
    amplitude_code: float = 1.0
    omega_code: float = 1.0
    phase: float = 0.0
    center_code: float = 0.0
    sigma_code: float = 1.0
    domain: str = "solver_code"
    time_unit: str = "code"
    amplitude_unit: str = "code"
    amplitude: float | None = None
    omega: float | None = None
    center: float | None = None
    sigma: float | None = None

    def __post_init__(self) -> None:
        if self.amplitude is not None:
            object.__setattr__(self, "amplitude_code", float(self.amplitude))
        if self.omega is not None:
            object.__setattr__(self, "omega_code", float(self.omega))
        if self.center is not None:
            object.__setattr__(self, "center_code", float(self.center))
        if self.sigma is not None:
            object.__setattr__(self, "sigma_code", float(self.sigma))

    def envelope(self, t):
        time = np.asarray(t, dtype = float)
        values = np.exp(-((time - self.center_code) ** 2) / (2.0 * self.sigma_code ** 2))
        return _asarray_or_float(values)

    def __call__(self, t):
        time = np.asarray(t, dtype = float)
        values = 2.0 * self.amplitude_code * np.asarray(self.envelope(time)) * np.cos(
            self.omega_code * time + self.phase)
        return _asarray_or_float(values)

    def to_dict(self) -> dict[str, Any]:
        return {
            "class"         : self.__class__.__name__,
            "name"          : self.name,
            "amplitude_code": self.amplitude_code,
            "omega_code"    : self.omega_code,
            "phase"         : self.phase,
            "center_code"   : self.center_code,
            "sigma_code"    : self.sigma_code,
            "domain"        : self.domain,
            "time_unit"     : self.time_unit,
            "amplitude_unit": self.amplitude_unit,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CodeGaussianCarrierField":
        return cls(
            name = data.get("name", "code_gaussian_carrier_field"),
            amplitude_code = float(data.get("amplitude_code", data.get("amplitude", 1.0))),
            omega_code = float(data.get("omega_code", data.get("omega", 1.0))),
            phase = float(data.get("phase", 0.0)),
            center_code = float(data.get("center_code", data.get("center", 0.0))),
            sigma_code = float(data.get("sigma_code", data.get("sigma", 1.0))),
            domain = data.get("domain", "solver_code"),
            time_unit = data.get("time_unit", "code"),
            amplitude_unit = data.get("amplitude_unit", "code"),
        )

    def to_expr(self) -> str:
        return (
            f"{self.name}: E_code(t) = 2 * {self.amplitude_code:.6g} "
            f"* exp[-(t - {self.center_code:.6g})^2 / (2 * {self.sigma_code:.6g}^2)] "
            f"* cos({self.omega_code:.6g} * t + {self.phase:.6g})"
        )

    def __repr__(self) -> str:
        return self.to_expr()


@dataclass(frozen = True)
class CodeCompositeField:
    """Internal solver-unit sum of lab-frame fields."""

    fields: tuple[CodeCarrierField | CodeGaussianCarrierField, ...]
    name: str = "code_composite_field"

    def __call__(self, t):
        total = sum(field(t) for field in self.fields)
        return _asarray_or_float(total)

    def to_dict(self) -> dict[str, Any]:
        return {
            "class"         : self.__class__.__name__,
            "name"          : self.name,
            "fields"        : [field.to_dict() for field in self.fields],
            "domain"        : "solver_code",
            "time_unit"     : "code",
            "amplitude_unit": "code",
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CodeCompositeField":
        from . import solver_input_from_dict

        return cls(
            name = data.get("name", "code_composite_field"),
            fields = tuple(solver_input_from_dict(item) for item in data["fields"]),
        )

    def to_expr(self) -> str:
        return f"{self.name}: " + " + ".join(field.to_expr() for field in self.fields)

    def __repr__(self) -> str:
        return self.to_expr()


def envelope_value(time: float, field: CodeCarrierField | CodeGaussianCarrierField) -> float:
    if isinstance(field, CodeGaussianCarrierField):
        return float(field.envelope(time))
    return 1.0


def electric_field_value(time: float, field: CodeCarrierField | CodeGaussianCarrierField) -> float:
    return float(field(time))


def electric_field_array(times, field: CodeCarrierField | CodeGaussianCarrierField) -> np.ndarray:
    return np.asarray(field(np.asarray(times, dtype = float)), dtype = float)


def _iter_fields(fields):
    if isinstance(fields, (CodeCarrierField, CodeGaussianCarrierField, CodeCompositeField)):
        return (fields,)
    return tuple(fields)


def total_electric_field_value(time: float, fields: Iterable[CodeCarrierField | CodeGaussianCarrierField]) -> float:
    return float(sum(field(time) for field in _iter_fields(fields)))


def total_electric_field_array(times, fields: Iterable[CodeCarrierField | CodeGaussianCarrierField]) -> np.ndarray:
    time_array = np.asarray(times, dtype = float)
    total = np.zeros_like(time_array, dtype = float)
    for field in _iter_fields(fields):
        total = total + np.asarray(field(time_array), dtype = float)
    return total

__all__ = [
    "CodeCarrierField",
    "CodeGaussianCarrierField",
    "CodeCompositeField",
    "envelope_value",
    "electric_field_value",
    "electric_field_array",
    "total_electric_field_value",
    "total_electric_field_array",
]
