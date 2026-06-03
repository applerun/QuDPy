"""Internal solver-unit RWA drive callables."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .lab_fields import _asarray_or_float


@dataclass(frozen = True)
class CodeConstantDrive:
    """Internal solver-unit RWA drive."""

    name: str = "code_constant_drive"
    amplitude_code: float = 1.0
    domain: str = "solver_code"
    time_unit: str = "code"
    amplitude_unit: str = "code"
    amplitude: float | None = None

    def __post_init__(self) -> None:
        if self.amplitude is not None:
            object.__setattr__(self, "amplitude_code", float(self.amplitude))

    def __call__(self, t):
        values = np.full_like(np.asarray(t, dtype = float), self.amplitude_code, dtype = float)
        return _asarray_or_float(values)

    def to_dict(self) -> dict[str, Any]:
        return {
            "class"         : self.__class__.__name__,
            "name"          : self.name,
            "amplitude_code": self.amplitude_code,
            "domain"        : self.domain,
            "time_unit"     : self.time_unit,
            "amplitude_unit": self.amplitude_unit,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CodeConstantDrive":
        return cls(
            name = data.get("name", "code_constant_drive"),
            amplitude_code = float(data.get("amplitude_code", data.get("amplitude", 1.0))),
            domain = data.get("domain", "solver_code"),
            time_unit = data.get("time_unit", "code"),
            amplitude_unit = data.get("amplitude_unit", "code"),
        )

    def to_expr(self) -> str:
        return f"{self.name}: g_code(t) = {self.amplitude_code:.6g}"

    def __repr__(self) -> str:
        return self.to_expr()


@dataclass(frozen = True)
class CodeGaussianDrive:
    """Internal solver-unit Gaussian RWA drive."""

    name: str = "code_gaussian_drive"
    amplitude_code: float = 1.0
    center_code: float = 0.0
    sigma_code: float = 1.0
    domain: str = "solver_code"
    time_unit: str = "code"
    amplitude_unit: str = "code"
    amplitude: float | None = None
    center: float | None = None
    sigma: float | None = None

    def __post_init__(self) -> None:
        if self.amplitude is not None:
            object.__setattr__(self, "amplitude_code", float(self.amplitude))
        if self.center is not None:
            object.__setattr__(self, "center_code", float(self.center))
        if self.sigma is not None:
            object.__setattr__(self, "sigma_code", float(self.sigma))

    def __call__(self, t):
        time = np.asarray(t, dtype = float)
        values = self.amplitude_code * np.exp(-((time - self.center_code) ** 2) / (2.0 * self.sigma_code ** 2))
        return _asarray_or_float(values)

    def to_dict(self) -> dict[str, Any]:
        return {
            "class"         : self.__class__.__name__,
            "name"          : self.name,
            "amplitude_code": self.amplitude_code,
            "center_code"   : self.center_code,
            "sigma_code"    : self.sigma_code,
            "domain"        : self.domain,
            "time_unit"     : self.time_unit,
            "amplitude_unit": self.amplitude_unit,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CodeGaussianDrive":
        return cls(
            name = data.get("name", "code_gaussian_drive"),
            amplitude_code = float(data.get("amplitude_code", data.get("amplitude", 1.0))),
            center_code = float(data.get("center_code", data.get("center", 0.0))),
            sigma_code = float(data.get("sigma_code", data.get("sigma", 1.0))),
            domain = data.get("domain", "solver_code"),
            time_unit = data.get("time_unit", "code"),
            amplitude_unit = data.get("amplitude_unit", "code"),
        )

    def to_expr(self) -> str:
        return (
            f"{self.name}: g_code(t) = {self.amplitude_code:.6g} * "
            f"exp[-(t - {self.center_code:.6g})^2 / (2 * {self.sigma_code:.6g}^2)]"
        )

    def __repr__(self) -> str:
        return self.to_expr()


__all__ = [
    "CodeConstantDrive",
    "CodeGaussianDrive",
]
