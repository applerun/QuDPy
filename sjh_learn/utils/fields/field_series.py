"""物理多脉冲 field 组合工具。

本模块定义物理单位下的 field series。它们仍然是 `FieldPhyRoot`，因此可直接
传入 `NLevelPhysicalParams(..., field=...)`。Normalizer 不解析 TA / 2DES
细节，只通过 `FieldPhyRoot` 的通用接口获取电场、metadata 和 auto-scale
候选。
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from itertools import product
from typing import Any

import numpy as np

from .lab_fields import (
    FieldPhyRoot,
    GaussianCarrierFieldPhysical,
    make_default_gaussian_carrier_field,
)


def _metadata_copy(metadata: dict[str, Any] | None) -> dict[str, Any]:
    return dict(metadata or {})


def _is_scan_value(value: Any) -> bool:
    if isinstance(value, (str, bytes, dict)):
        return False
    if isinstance(value, FieldPhyRoot):
        return False
    return isinstance(value, Iterable)


def _scan_items(params: dict[str, Any]) -> tuple[list[str], list[list[Any]]]:
    names: list[str] = []
    values: list[list[Any]] = []
    for key, value in params.items():
        if _is_scan_value(value):
            names.append(key)
            values.append(list(value))
    return names, values


def _iter_scan_params(params: dict[str, Any]) -> Iterator[dict[str, Any]]:
    names, values = _scan_items(params)
    if not names:
        yield dict(params)
        return

    for combo in product(*values):
        item = dict(params)
        for key, value in zip(names, combo):
            item[key] = value
        yield item


@dataclass(frozen=True)
class FieldPhySeries(FieldPhyRoot):
    """多个物理 field 的线性叠加。

    `FieldPhySeries` 是 physical field 层的组合对象，不是 solver code-unit
    field。它支持按 index 或 subfield name 提取子场。
    """

    fields: tuple[FieldPhyRoot, ...]
    sub_field_names: tuple[str, ...] | None = None
    name: str = "field_phy_series"
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.fields:
            raise ValueError("FieldPhySeries requires at least one subfield.")

        for field in self.fields:
            if not isinstance(field, FieldPhyRoot):
                raise TypeError("FieldPhySeries.fields must contain FieldPhyRoot instances.")

        if self.sub_field_names is None:
            names = []
            for idx, field in enumerate(self.fields):
                payload = field.to_dict()
                names.append(str(payload.get("name") or f"field_{idx}"))
            object.__setattr__(self, "sub_field_names", tuple(names))
        else:
            if len(self.sub_field_names) != len(self.fields):
                raise ValueError("sub_field_names length must match fields length.")
            if len(set(self.sub_field_names)) != len(self.sub_field_names):
                raise ValueError("sub_field_names must be unique.")

    @property
    def reference_MV_per_cm(self) -> float | None:
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


@dataclass(frozen=True)
class TAField(FieldPhySeries):
    """Transient absorption 常用 pump-probe field."""

    probe_delay_fs: float = 0.0

    @property
    def probe_delay(self) -> float:
        """probe 相对于 pump 的延迟，单位 fs。"""

        return float(self.probe_delay_fs)

    @property
    def pump_tau(self) -> None:
        """TA 默认只有一个 pump，因此没有 inter-pump delay。"""

        return None

    def to_dict(self) -> dict[str, Any]:
        payload = super().to_dict()
        payload["probe_delay_fs"] = float(self.probe_delay_fs)
        payload["pump_tau_fs"] = None
        return payload


@dataclass(frozen=True)
class TwoDESField(FieldPhySeries):
    """2DES 常用 pump1-pump2-probe field."""

    pump_tau_fs: float = 0.0
    probe_delay_fs: float = 0.0

    @property
    def probe_delay(self) -> float:
        """probe 相对于 pump sequence 的延迟，单位 fs。"""

        return float(self.probe_delay_fs)

    @property
    def pump_tau(self) -> float:
        """pump1 和 pump2 之间的 inter-pump delay，单位 fs。"""

        return float(self.pump_tau_fs)

    def to_dict(self) -> dict[str, Any]:
        payload = super().to_dict()
        payload["probe_delay_fs"] = float(self.probe_delay_fs)
        payload["pump_tau_fs"] = float(self.pump_tau_fs)
        return payload


def make_ta_gaussian_field(
    *,
    probe_delay_fs: float,
    pump_E0_MV_per_cm: float,
    probe_E0_MV_per_cm: float,
    pump_laser_energy_eV: float,
    probe_laser_energy_eV: float | None = None,
    pump_center_fs: float = 0.0,
    pump_sigma_fs: float = 10.0,
    probe_sigma_fs: float | None = None,
    pump_phase_rad: float = 0.0,
    probe_phase_rad: float = 0.0,
    name: str = "ta_gaussian_field",
    metadata: dict[str, Any] | None = None,
) -> TAField:
    """生成 TA 常用 Gaussian pump-probe field。"""

    probe_laser_energy = pump_laser_energy_eV if probe_laser_energy_eV is None else probe_laser_energy_eV
    probe_sigma = pump_sigma_fs if probe_sigma_fs is None else probe_sigma_fs
    probe_center_fs = float(pump_center_fs) + float(probe_delay_fs)

    pump = make_default_gaussian_carrier_field(
        E0_MV_per_cm=float(pump_E0_MV_per_cm),
        laser_energy_eV=float(pump_laser_energy_eV),
        pulse_center_fs=float(pump_center_fs),
        pulse_sigma_fs=float(pump_sigma_fs),
        phase_rad=float(pump_phase_rad),
        name="pump",
        metadata={"role": "pump", "parent_field": name},
    )
    probe = make_default_gaussian_carrier_field(
        E0_MV_per_cm=float(probe_E0_MV_per_cm),
        laser_energy_eV=float(probe_laser_energy),
        pulse_center_fs=probe_center_fs,
        pulse_sigma_fs=float(probe_sigma),
        phase_rad=float(probe_phase_rad),
        name="probe",
        metadata={"role": "probe", "parent_field": name},
    )

    payload = _metadata_copy(metadata)
    payload.setdefault("experiment", "TA")
    return TAField(
        fields=(pump, probe),
        sub_field_names=("pump", "probe"),
        name=name,
        metadata=payload,
        probe_delay_fs=float(probe_delay_fs),
    )


def make_pump_probe_field_from_templates(
    *,
    pump_template: FieldPhyRoot,
    probe_template: FieldPhyRoot,
    delay_fs: float,
    probe_center_fs: float = 0.0,
    name: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> TAField:
    """Build a pump-probe field by time-shifting zero-centered templates.

    The probe is fixed at ``probe_center_fs`` and the pump is placed by
    ``pump_center_fs = probe_center_fs - delay_fs``.
    """

    if not isinstance(pump_template, FieldPhyRoot):
        raise TypeError("pump_template must be a FieldPhyRoot instance.")
    if not isinstance(probe_template, FieldPhyRoot):
        raise TypeError("probe_template must be a FieldPhyRoot instance.")

    delay = float(delay_fs)
    probe_center = float(probe_center_fs)
    pump_center = probe_center - delay
    field_name = name or "pump_probe_template_field"

    pump = pump_template.time_shifted(
        pump_center,
        name="pump",
        metadata={"role": "pump", "parent_field": field_name},
    )
    probe = probe_template.time_shifted(
        probe_center,
        name="probe",
        metadata={"role": "probe", "parent_field": field_name},
    )

    payload = _metadata_copy(metadata)
    payload.setdefault("experiment", "TA")
    payload.update(
        {
            "delay_fs": delay,
            "probe_delay_fs": delay,
            "probe_center_fs": probe_center,
            "pump_center_fs": pump_center,
            "center_rule": "pump_center_fs = probe_center_fs - delay_fs",
            "template_convention": "pump/probe templates are expected to be centered at 0 fs.",
        }
    )
    return TAField(
        fields=(pump, probe),
        sub_field_names=("pump", "probe"),
        name=field_name,
        metadata=payload,
        probe_delay_fs=delay,
    )


def make_ta_field_from_templates(
    *,
    pump_template: FieldPhyRoot,
    probe_template: FieldPhyRoot,
    probe_delay_fs: float | None = None,
    delay_fs: float | None = None,
    probe_center_fs: float = 0.0,
    name: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> TAField:
    """Compatibility alias for template-based TA pump-probe fields."""

    if delay_fs is None and probe_delay_fs is None:
        raise TypeError("Either delay_fs or probe_delay_fs must be provided.")
    if delay_fs is not None and probe_delay_fs is not None and float(delay_fs) != float(probe_delay_fs):
        raise ValueError("delay_fs and probe_delay_fs must agree when both are provided.")
    delay = float(delay_fs if delay_fs is not None else probe_delay_fs)
    return make_pump_probe_field_from_templates(
        pump_template=pump_template,
        probe_template=probe_template,
        delay_fs=delay,
        probe_center_fs=probe_center_fs,
        name=name,
        metadata=metadata,
    )


def make_twodes_gaussian_field(
    *,
    pump_tau_fs: float,
    probe_delay_fs: float,
    pump1_E0_MV_per_cm: float,
    pump2_E0_MV_per_cm: float,
    probe_E0_MV_per_cm: float,
    pump1_laser_energy_eV: float,
    pump2_laser_energy_eV: float | None = None,
    probe_laser_energy_eV: float | None = None,
    pump1_center_fs: float = 0.0,
    pump1_sigma_fs: float = 10.0,
    pump2_sigma_fs: float | None = None,
    probe_sigma_fs: float | None = None,
    pump1_phase_rad: float = 0.0,
    pump2_phase_rad: float = 0.0,
    probe_phase_rad: float = 0.0,
    name: str = "twodes_gaussian_field",
    metadata: dict[str, Any] | None = None,
) -> TwoDESField:
    """生成 2DES 常用 Gaussian pump1-pump2-probe field。"""

    pump2_laser_energy = pump1_laser_energy_eV if pump2_laser_energy_eV is None else pump2_laser_energy_eV
    probe_laser_energy = pump1_laser_energy_eV if probe_laser_energy_eV is None else probe_laser_energy_eV
    pump2_sigma = pump1_sigma_fs if pump2_sigma_fs is None else pump2_sigma_fs
    probe_sigma = pump1_sigma_fs if probe_sigma_fs is None else probe_sigma_fs

    pump2_center_fs = float(pump1_center_fs) + float(pump_tau_fs)
    probe_center_fs = pump2_center_fs + float(probe_delay_fs)

    pump1 = make_default_gaussian_carrier_field(
        E0_MV_per_cm=float(pump1_E0_MV_per_cm),
        laser_energy_eV=float(pump1_laser_energy_eV),
        pulse_center_fs=float(pump1_center_fs),
        pulse_sigma_fs=float(pump1_sigma_fs),
        phase_rad=float(pump1_phase_rad),
        name="pump1",
        metadata={"role": "pump1", "parent_field": name},
    )
    pump2 = make_default_gaussian_carrier_field(
        E0_MV_per_cm=float(pump2_E0_MV_per_cm),
        laser_energy_eV=float(pump2_laser_energy),
        pulse_center_fs=pump2_center_fs,
        pulse_sigma_fs=float(pump2_sigma),
        phase_rad=float(pump2_phase_rad),
        name="pump2",
        metadata={"role": "pump2", "parent_field": name},
    )
    probe = make_default_gaussian_carrier_field(
        E0_MV_per_cm=float(probe_E0_MV_per_cm),
        laser_energy_eV=float(probe_laser_energy),
        pulse_center_fs=probe_center_fs,
        pulse_sigma_fs=float(probe_sigma),
        phase_rad=float(probe_phase_rad),
        name="probe",
        metadata={"role": "probe", "parent_field": name},
    )

    payload = _metadata_copy(metadata)
    payload.setdefault("experiment", "2DES")
    return TwoDESField(
        fields=(pump1, pump2, probe),
        sub_field_names=("pump1", "pump2", "probe"),
        name=name,
        metadata=payload,
        pump_tau_fs=float(pump_tau_fs),
        probe_delay_fs=float(probe_delay_fs),
    )


def iter_ta_gaussian_fields(**kwargs) -> Iterator[tuple[dict[str, Any], TAField]]:
    """扫描 TA Gaussian field 参数。

    任意非字符串 iterable 参数都会被视作扫描维度；多个 iterable 参数使用
    Cartesian product。每次 yield `(scan_params, field)`。
    """

    for params in _iter_scan_params(kwargs):
        yield params, make_ta_gaussian_field(**params)


def iter_twodes_gaussian_fields(**kwargs) -> Iterator[tuple[dict[str, Any], TwoDESField]]:
    """扫描 2DES Gaussian field 参数。

    任意非字符串 iterable 参数都会被视作扫描维度；多个 iterable 参数使用
    Cartesian product。每次 yield `(scan_params, field)`。
    """

    for params in _iter_scan_params(kwargs):
        yield params, make_twodes_gaussian_field(**params)


__all__ = [
    "FieldPhySeries",
    "TAField",
    "TwoDESField",
    "make_pump_probe_field_from_templates",
    "make_ta_field_from_templates",
    "make_ta_gaussian_field",
    "make_twodes_gaussian_field",
    "iter_ta_gaussian_fields",
    "iter_twodes_gaussian_fields",
]
