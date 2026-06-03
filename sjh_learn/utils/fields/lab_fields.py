"""用户侧 lab-frame 物理电场输入。

本模块只定义真实物理单位下的输入场：时间单位固定为 fs，电场单位固定为
MV/cm。solver 需要 code-unit callable 时，应由 `ParaNormalizer.make_code_field()`
生成内部 adapter；普通 examples 不应直接构造 code-unit field。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np


def _metadata_copy(metadata: dict[str, Any] | None) -> dict[str, Any]:
    return dict(metadata or {})


class FieldPhyRoot(ABC):
    """物理电场基类。

    子类的 `physical_E_MV_per_cm(t_fs)` 必须支持 numpy array 输入，并返回
    与 `t_fs` 相同 shape 的真实 lab-frame 电场，单位为 MV/cm。这里不做
    scalar-only callable 的自动兼容；shape 不匹配时直接 fail-fast。
    """

    time_unit = "fs"
    field_unit = "MV/cm"

    def __call__(self, t_fs):
        t_array = np.asarray(t_fs, dtype=float)
        values = np.asarray(self.physical_E_MV_per_cm(t_array), dtype=float)
        if values.shape != t_array.shape:
            raise ValueError(
                "physical_E_MV_per_cm(t_fs) must return an array with the same shape as t_fs. "
                f"got {values.shape}, expected {t_array.shape}."
            )
        if np.ndim(t_fs) == 0:
            return float(values)
        return values

    @abstractmethod
    def physical_E_MV_per_cm(self, t_fs: np.ndarray) -> np.ndarray:
        """返回真实 lab-frame 电场 `E(t_fs)`，单位 MV/cm。"""

    @abstractmethod
    def __repr__(self) -> str:
        """返回人可读、尽量可恢复的输入场描述。"""

    def to_dict(self) -> dict[str, Any]:
        return {
            "class": self.__class__.__name__,
            "repr": repr(self),
            "time_unit": self.time_unit,
            "field_unit": self.field_unit,
            "rebuildable": False,
        }

    @classmethod
    def rebuild(cls, payload):
        raise NotImplementedError(f"{cls.__name__}.rebuild() is not implemented.")


class FieldPhyCustomed(FieldPhyRoot):
    """用户自定义物理电场的推荐基类。

    用户应继承本类并实现 `physical_E_MV_per_cm(t_fs)`、`__repr__()`，
    如需从 metadata/checkpoint 恢复，还应实现 `@classmethod rebuild(...)`。
    """

    @classmethod
    def rebuild(cls, payload):
        raise NotImplementedError(
            "Custom physical fields are not rebuildable unless the subclass implements rebuild()."
        )


@dataclass(frozen=True)
class CarrierFieldPhysical(FieldPhyRoot):
    """CW lab-frame carrier field。

    物理约定：`E(t_fs) = 2 E0 cos(omega_L t_fs + phase)`，其中
    `E0_MV_per_cm` 是输入场强参数，单位 MV/cm。
    """

    E0_MV_per_cm: float
    omega_L_fs_inv: float
    phase_rad: float = 0.0
    name: str = "carrier_field_physical"
    metadata: dict[str, Any] | None = None

    def physical_E_MV_per_cm(self, t_fs: np.ndarray) -> np.ndarray:
        return 2.0 * float(self.E0_MV_per_cm) * np.cos(
            float(self.omega_L_fs_inv) * t_fs + float(self.phase_rad)
        )

    def __repr__(self) -> str:
        return (
            "CarrierFieldPhysical("
            f"E0_MV_per_cm={self.E0_MV_per_cm!r}, "
            f"omega_L_fs_inv={self.omega_L_fs_inv!r}, "
            f"phase_rad={self.phase_rad!r})"
        )

    def to_dict(self) -> dict[str, Any]:
        metadata = _metadata_copy(self.metadata)
        return {
            "class": self.__class__.__name__,
            "repr": repr(self),
            "name": self.name,
            "time_unit": self.time_unit,
            "field_unit": self.field_unit,
            "rebuildable": True,
            "E0_MV_per_cm": float(self.E0_MV_per_cm),
            "peak_E_MV_per_cm": 2.0 * float(self.E0_MV_per_cm),
            "omega_L_fs_inv": float(self.omega_L_fs_inv),
            "phase_rad": float(self.phase_rad),
            "envelope": "constant",
            "expression": "E(t_fs) = 2 E0 cos(omega_L t_fs + phase)",
            "amplitude_convention": "E0_MV_per_cm is E0 in E(t)=2E0 cos(...).",
            "description": metadata.get("description"),
            "metadata": metadata,
        }

    @classmethod
    def rebuild(cls, payload):
        if not isinstance(payload, dict):
            raise TypeError("CarrierFieldPhysical.rebuild() expects a dict payload.")
        return cls(
            E0_MV_per_cm=float(payload["E0_MV_per_cm"]),
            omega_L_fs_inv=float(payload["omega_L_fs_inv"]),
            phase_rad=float(payload.get("phase_rad", 0.0)),
            name=str(payload.get("name", "carrier_field_physical")),
            metadata=dict(payload.get("metadata") or {}),
        )


@dataclass(frozen=True)
class GaussianCarrierFieldPhysical(FieldPhyRoot):
    """Gaussian envelope lab-frame carrier field。

    物理约定：
    `E(t_fs) = 2 E0 exp[-(t_fs-center)^2/(2 sigma^2)] cos(omega_L t_fs + phase)`。
    """

    E0_MV_per_cm: float
    omega_L_fs_inv: float
    center_fs: float
    sigma_fs: float
    phase_rad: float = 0.0
    name: str = "gaussian_carrier_field_physical"
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.sigma_fs <= 0:
            raise ValueError("sigma_fs must be positive.")

    def physical_E_MV_per_cm(self, t_fs: np.ndarray) -> np.ndarray:
        envelope = np.exp(-((t_fs - float(self.center_fs)) ** 2) / (2.0 * float(self.sigma_fs) ** 2))
        return 2.0 * float(self.E0_MV_per_cm) * envelope * np.cos(
            float(self.omega_L_fs_inv) * t_fs + float(self.phase_rad)
        )

    def __repr__(self) -> str:
        return (
            "GaussianCarrierFieldPhysical("
            f"E0_MV_per_cm={self.E0_MV_per_cm!r}, "
            f"omega_L_fs_inv={self.omega_L_fs_inv!r}, "
            f"center_fs={self.center_fs!r}, "
            f"sigma_fs={self.sigma_fs!r}, "
            f"phase_rad={self.phase_rad!r})"
        )

    def to_dict(self) -> dict[str, Any]:
        metadata = _metadata_copy(self.metadata)
        return {
            "class": self.__class__.__name__,
            "repr": repr(self),
            "name": self.name,
            "time_unit": self.time_unit,
            "field_unit": self.field_unit,
            "rebuildable": True,
            "E0_MV_per_cm": float(self.E0_MV_per_cm),
            "peak_E_MV_per_cm": 2.0 * float(self.E0_MV_per_cm),
            "omega_L_fs_inv": float(self.omega_L_fs_inv),
            "phase_rad": float(self.phase_rad),
            "center_fs": float(self.center_fs),
            "sigma_fs": float(self.sigma_fs),
            "envelope": "gaussian",
            "expression": "E(t_fs) = 2 E0 exp[-(t_fs-center)^2/(2 sigma^2)] cos(omega_L t_fs + phase)",
            "amplitude_convention": "E0_MV_per_cm is E0 in E(t)=2E0 f(t) cos(...).",
            "description": metadata.get("description"),
            "metadata": metadata,
        }

    @classmethod
    def rebuild(cls, payload):
        if not isinstance(payload, dict):
            raise TypeError("GaussianCarrierFieldPhysical.rebuild() expects a dict payload.")
        return cls(
            E0_MV_per_cm=float(payload["E0_MV_per_cm"]),
            omega_L_fs_inv=float(payload["omega_L_fs_inv"]),
            center_fs=float(payload["center_fs"]),
            sigma_fs=float(payload["sigma_fs"]),
            phase_rad=float(payload.get("phase_rad", 0.0)),
            name=str(payload.get("name", "gaussian_carrier_field_physical")),
            metadata=dict(payload.get("metadata") or {}),
        )


@dataclass(frozen=True)
class CompositeLabFieldPhysical(FieldPhyRoot):
    """多个 lab-frame 物理电场的线性叠加。"""

    fields: tuple[FieldPhyRoot, ...]
    name: str = "composite_lab_field_physical"
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        for field in self.fields:
            if not isinstance(field, FieldPhyRoot):
                raise TypeError("CompositeLabFieldPhysical.fields must contain FieldPhyRoot instances.")

    def physical_E_MV_per_cm(self, t_fs: np.ndarray) -> np.ndarray:
        if not self.fields:
            return np.zeros_like(t_fs, dtype=float)
        total = np.zeros_like(t_fs, dtype=float)
        for field in self.fields:
            total = total + field(t_fs)
        return total

    def __repr__(self) -> str:
        return "CompositeLabFieldPhysical(fields=(" + ", ".join(repr(field) for field in self.fields) + "))"

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
            "fields": [field.to_dict() for field in self.fields],
            "expression": "E_total(t_fs) = sum_k E_k(t_fs)",
            "description": metadata.get("description"),
            "metadata": metadata,
        }

    @classmethod
    def rebuild(cls, payload):
        if not isinstance(payload, dict):
            raise TypeError("CompositeLabFieldPhysical.rebuild() expects a dict payload.")
        fields = tuple(rebuild_physical_field(item) for item in payload["fields"])
        return cls(
            fields=fields,
            name=str(payload.get("name", "composite_lab_field_physical")),
            metadata=dict(payload.get("metadata") or {}),
        )


@dataclass(frozen=True)
class _CodeFieldAdapter:
    """内部 solver-unit adapter。

    它把 `t_code -> t_fs`、`E_MV_per_cm -> E_code`，只供 core/model/solvers
    内部使用，不是用户侧输入类。
    """

    field_phy: FieldPhyRoot
    normalizer: Any
    solver: Any
    reference_field_MV_per_cm: float
    name: str = "internal_code_field_adapter"

    def __call__(self, t_code):
        t_fs = self.normalizer.denormalize_time_array(np.asarray(t_code, dtype=float), self.solver)
        E_MV_per_cm = self.field_phy(t_fs)
        return self.normalizer.normalize_field_MV_per_cm(
            E_MV_per_cm,
            reference_field_MV_per_cm=self.reference_field_MV_per_cm,
        )

    def physical(self, t_fs):
        return self.field_phy(t_fs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "class": self.__class__.__name__,
            "domain": "solver_code",
            "time_unit": "code",
            "field_unit": "code",
            "time_scale_fs": float(self.solver.time_scale_fs),
            "field_scale": "E_code = E_MV_per_cm / reference_field_MV_per_cm",
            "reference_field_MV_per_cm": float(self.reference_field_MV_per_cm),
            "source_field": self.field_phy.to_dict(),
        }

    def to_expr(self) -> str:
        return f"E_code(t_code) = E_phys(t_fs) / {self.reference_field_MV_per_cm:g} MV/cm"


def make_code_field_adapter(
    field_phy: FieldPhyRoot,
    normalizer,
    solver,
    *,
    reference_field_MV_per_cm: float,
):
    if not isinstance(field_phy, FieldPhyRoot):
        raise TypeError("field_phy must be a FieldPhyRoot instance.")
    return _CodeFieldAdapter(
        field_phy=field_phy,
        normalizer=normalizer,
        solver=solver,
        reference_field_MV_per_cm=float(reference_field_MV_per_cm),
    )


def rebuild_physical_field(payload) -> FieldPhyRoot:
    if not isinstance(payload, dict):
        raise TypeError("rebuild_physical_field() expects a dict payload.")
    class_name = payload.get("class")
    registry = {
        "CarrierFieldPhysical": CarrierFieldPhysical,
        "GaussianCarrierFieldPhysical": GaussianCarrierFieldPhysical,
        "CompositeLabFieldPhysical": CompositeLabFieldPhysical,
    }
    if class_name not in registry:
        raise ValueError(f"Unknown or non-rebuildable physical field class: {class_name!r}.")
    return registry[class_name].rebuild(payload)


def default_field_from_physical_params(physical_params, normalizer=None) -> FieldPhyRoot:
    """从 `NLevelPhysicalParams` 生成默认物理 lab-frame field。"""

    if normalizer is None:
        raise ValueError("normalizer is required to convert laser_energy_eV to omega_L_fs_inv.")
    omega_fs_inv = float(normalizer.energy_eV_to_fs_inv(physical_params.laser_energy_eV))
    metadata = {
        "laser_energy_eV": physical_params.laser_energy_eV,
        "source": "NLevelPhysicalParams default field",
    }
    if physical_params.input_description is not None:
        metadata["description"] = physical_params.input_description
    if physical_params.input_metadata is not None:
        metadata["user_metadata"] = dict(physical_params.input_metadata)
    if physical_params.pulse_sigma_fs is None:
        return CarrierFieldPhysical(
            E0_MV_per_cm=physical_params.field_MV_per_cm,
            omega_L_fs_inv=omega_fs_inv,
            name="default_physical_carrier_field",
            metadata=metadata,
        )
    if physical_params.pulse_center_fs is None:
        raise ValueError("pulse_center_fs is required when pulse_sigma_fs is set.")
    return GaussianCarrierFieldPhysical(
        E0_MV_per_cm=physical_params.field_MV_per_cm,
        omega_L_fs_inv=omega_fs_inv,
        center_fs=physical_params.pulse_center_fs,
        sigma_fs=physical_params.pulse_sigma_fs,
        name="default_physical_gaussian_carrier_field",
        metadata=metadata,
    )


__all__ = [
    "FieldPhyRoot",
    "FieldPhyCustomed",
    "CarrierFieldPhysical",
    "GaussianCarrierFieldPhysical",
    "CompositeLabFieldPhysical",
    "default_field_from_physical_params",
    "rebuild_physical_field",
]
