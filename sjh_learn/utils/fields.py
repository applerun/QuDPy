"""Callable input drives and laboratory-frame fields."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

import numpy as np


def _as_input(t):
    arr = np.asarray(t, dtype=float)
    return arr, np.ndim(t) == 0


def _maybe_scalar(values, scalar: bool):
    if scalar:
        return float(np.asarray(values).reshape(-1)[0])
    return np.asarray(values, dtype=float)


@dataclass(frozen=True)
class ConstantDrive:
    name: str = "constant_drive"
    amplitude: float = 1.0
    domain: str = "solver_code"
    time_unit: str = "code"
    amplitude_unit: str = "code"

    def __call__(self, t):
        arr, scalar = _as_input(t)
        values = np.full_like(arr, float(self.amplitude), dtype=float)
        return _maybe_scalar(values, scalar)

    def to_dict(self) -> dict[str, Any]:
        return {"class": type(self).__name__, **asdict(self)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConstantDrive":
        payload = {key: value for key, value in data.items() if key != "class"}
        return cls(**payload)

    def to_expr(self) -> str:
        return f"{self.name}: Omega(t) = {self.amplitude:g}"

    def __repr__(self) -> str:
        return self.to_expr()


@dataclass(frozen=True)
class GaussianDrive:
    name: str = "gaussian_drive"
    amplitude: float = 1.0
    center: float = 0.0
    sigma: float = 1.0
    domain: str = "solver_code"
    time_unit: str = "code"
    amplitude_unit: str = "code"

    def __call__(self, t):
        if self.sigma <= 0:
            raise ValueError("sigma must be positive.")
        arr, scalar = _as_input(t)
        values = self.amplitude * np.exp(-((arr - self.center) ** 2) / (2.0 * self.sigma**2))
        return _maybe_scalar(values, scalar)

    def to_dict(self) -> dict[str, Any]:
        return {"class": type(self).__name__, **asdict(self)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GaussianDrive":
        payload = {key: value for key, value in data.items() if key != "class"}
        return cls(**payload)

    def to_expr(self) -> str:
        return (
            f"{self.name}: Omega(t) = {self.amplitude:g} * "
            f"exp[-(t - {self.center:g})^2 / (2 * {self.sigma:g}^2)]"
        )

    def __repr__(self) -> str:
        return self.to_expr()


@dataclass(frozen=True)
class CarrierField:
    name: str = "carrier_field"
    amplitude: float = 1.0
    omega: float = 1.0
    phase: float = 0.0
    domain: str = "solver_code"
    time_unit: str = "code"
    amplitude_unit: str = "code"

    def __call__(self, t):
        arr, scalar = _as_input(t)
        values = 2.0 * self.amplitude * np.cos(self.omega * arr + self.phase)
        return _maybe_scalar(values, scalar)

    def to_dict(self) -> dict[str, Any]:
        return {"class": type(self).__name__, **asdict(self)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CarrierField":
        payload = {key: value for key, value in data.items() if key != "class"}
        return cls(**payload)

    def to_expr(self) -> str:
        return f"{self.name}: E(t) = 2 * {self.amplitude:g} * cos({self.omega:g} t + {self.phase:g})"

    def __repr__(self) -> str:
        return self.to_expr()


@dataclass(frozen=True)
class GaussianCarrierField:
    name: str = "gaussian_carrier_field"
    amplitude: float = 1.0
    omega: float = 1.0
    phase: float = 0.0
    center: float = 0.0
    sigma: float = 1.0
    domain: str = "solver_code"
    time_unit: str = "code"
    amplitude_unit: str = "code"

    def __call__(self, t):
        if self.sigma <= 0:
            raise ValueError("sigma must be positive.")
        arr, scalar = _as_input(t)
        envelope = np.exp(-((arr - self.center) ** 2) / (2.0 * self.sigma**2))
        values = 2.0 * self.amplitude * envelope * np.cos(self.omega * arr + self.phase)
        return _maybe_scalar(values, scalar)

    def envelope(self, t):
        arr, scalar = _as_input(t)
        values = np.exp(-((arr - self.center) ** 2) / (2.0 * self.sigma**2))
        return _maybe_scalar(values, scalar)

    def to_dict(self) -> dict[str, Any]:
        return {"class": type(self).__name__, **asdict(self)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GaussianCarrierField":
        payload = {key: value for key, value in data.items() if key != "class"}
        return cls(**payload)

    def to_expr(self) -> str:
        return (
            f"{self.name}: E(t) = 2 * {self.amplitude:g} * "
            f"exp[-(t - {self.center:g})^2 / (2 * {self.sigma:g}^2)] * "
            f"cos({self.omega:g} t + {self.phase:g})"
        )

    def __repr__(self) -> str:
        return self.to_expr()


@dataclass(frozen=True)
class CompositeField:
    fields: tuple[Any, ...]
    name: str = "composite_field"

    def __call__(self, t):
        arr, scalar = _as_input(t)
        values = np.zeros_like(arr, dtype=float)
        for field in self.fields:
            values = values + np.asarray(field(arr), dtype=float)
        return _maybe_scalar(values, scalar)

    def to_dict(self) -> dict[str, Any]:
        return {
            "class": type(self).__name__,
            "name": self.name,
            "fields": [field.to_dict() for field in self.fields],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CompositeField":
        return cls(
            fields=tuple(field_from_dict(item) for item in data["fields"]),
            name=data.get("name", "composite_field"),
        )

    def to_expr(self) -> str:
        return f"{self.name}: " + " + ".join(field.to_expr() for field in self.fields)

    def __repr__(self) -> str:
        return self.to_expr()


FIELD_CLASSES = {
    "ConstantDrive": ConstantDrive,
    "GaussianDrive": GaussianDrive,
    "CarrierField": CarrierField,
    "GaussianCarrierField": GaussianCarrierField,
    "CompositeField": CompositeField,
}


def field_from_dict(data: dict[str, Any]):
    class_name = data.get("class")
    if class_name not in FIELD_CLASSES:
        raise ValueError(f"Unsupported field/drive class: {class_name}")
    return FIELD_CLASSES[class_name].from_dict(data)


def _iter_fields(fields: Any | Iterable[Any]) -> tuple[Any, ...]:
    if isinstance(fields, (ConstantDrive, GaussianDrive, CarrierField, GaussianCarrierField, CompositeField)):
        return (fields,)
    return tuple(fields)


def envelope_value(t: float, field: Any) -> float:
    if isinstance(field, GaussianCarrierField):
        return float(field.envelope(t))
    return 1.0


def electric_field_value(t: float, field: Any) -> float:
    return float(field(t))


def electric_field_array(times: np.ndarray, field: Any) -> np.ndarray:
    return np.asarray(field(np.asarray(times, dtype=float)), dtype=float)


def total_electric_field_value(t: float, fields: Any | Iterable[Any]) -> float:
    field_list = _iter_fields(fields)
    return float(sum(float(field(t)) for field in field_list))


def total_electric_field_array(times: np.ndarray, fields: Any | Iterable[Any]) -> np.ndarray:
    field_list = _iter_fields(fields)
    arr = np.asarray(times, dtype=float)
    values = np.zeros_like(arr, dtype=float)
    for field in field_list:
        values += np.asarray(field(arr), dtype=float)
    return values


FieldConfig = CarrierField


__all__ = [
    "ConstantDrive",
    "GaussianDrive",
    "CarrierField",
    "GaussianCarrierField",
    "CompositeField",
    "FieldConfig",
    "field_from_dict",
    "envelope_value",
    "electric_field_value",
    "electric_field_array",
    "total_electric_field_value",
    "total_electric_field_array",
]
