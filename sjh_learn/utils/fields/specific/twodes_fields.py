from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..field_series import FieldPhySeries, _metadata_copy
from ..lab_fields import FieldPhyRoot


@dataclass(frozen=True)
class TwoDESField(FieldPhySeries):
    """2DES 常用 pump1-pump2-probe field。

    本类只记录 2DES 的命名语义、inter-pump delay 和 probe delay；线性叠加行为
    由 ``FieldPhySeries`` 提供。
    """

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


def make_twodes_field_from_templates(
    *,
    pump1_template: FieldPhyRoot,
    pump2_template: FieldPhyRoot,
    probe_template: FieldPhyRoot,
    pump_tau_fs: float,
    probe_delay_fs: float,
    pump1_center_fs: float = 0.0,
    name: str = "twodes_field",
    metadata: dict[str, Any] | None = None,
) -> TwoDESField:
    """由中心在 0 fs 的 pulse 模板生成 2DES pump1-pump2-probe field。"""

    pump1_center = float(pump1_center_fs)
    pump2_center = pump1_center + float(pump_tau_fs)
    probe_center = pump2_center + float(probe_delay_fs)

    pump1 = pump1_template.time_shifted(pump1_center, name="pump1")
    pump2 = pump2_template.time_shifted(pump2_center, name="pump2")
    probe = probe_template.time_shifted(probe_center, name="probe")

    payload = _metadata_copy(metadata)
    payload.setdefault("experiment", "2DES")
    payload.update(
        {
            "pump_tau_fs": float(pump_tau_fs),
            "probe_delay_fs": float(probe_delay_fs),
            "pump1_center_fs": float(pump1_center),
            "pump2_center_fs": float(pump2_center),
            "probe_center_fs": float(probe_center),
            "template_convention": "pulse templates are expected to be centered at 0 fs.",
        }
    )

    return TwoDESField(
        fields=(pump1, pump2, probe),
        sub_field_names=("pump1", "pump2", "probe"),
        name=name,
        metadata=payload,
        pump_tau_fs=float(pump_tau_fs),
        probe_delay_fs=float(probe_delay_fs),
    )


__all__ = [
    "TwoDESField",
    "make_twodes_field_from_templates",
]
