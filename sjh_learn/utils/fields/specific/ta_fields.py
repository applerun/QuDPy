from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..field_series import FieldPhySeries, _metadata_copy
from ..lab_fields import FieldPhyRoot


@dataclass(frozen=True)
class TAField(FieldPhySeries):
    """Transient absorption 常用 pump-probe field。

    本类只记录 TA 的命名语义和 probe delay；线性叠加行为由
    ``FieldPhySeries`` 提供。
    """

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


def make_ta_field_from_templates(
    *,
    pump_template: FieldPhyRoot,
    probe_template: FieldPhyRoot,
    probe_delay_fs: float,
    probe_center_fs: float = 0.0,
    name: str = "ta_field",
    metadata: dict[str, Any] | None = None,
) -> TAField:
    """由中心在 0 fs 的 pump/probe 模板生成 TA pump-probe field。

    delay 约定：``probe_delay_fs > 0`` 表示 pump 早于 probe。
    """

    pump_center_fs = float(probe_center_fs) - float(probe_delay_fs)
    probe_center = float(probe_center_fs)

    pump = pump_template.time_shifted(pump_center_fs, name="pump")
    probe = probe_template.time_shifted(probe_center, name="probe")

    payload = _metadata_copy(metadata)
    payload.setdefault("experiment", "TA")
    payload.update(
        {
            "probe_delay_fs": float(probe_delay_fs),
            "pump_center_fs": float(pump_center_fs),
            "probe_center_fs": float(probe_center),
            "template_convention": "pump/probe templates are expected to be centered at 0 fs.",
        }
    )

    return TAField(
        fields=(pump, probe),
        sub_field_names=("pump", "probe"),
        name=name,
        metadata=payload,
        probe_delay_fs=float(probe_delay_fs),
    )


__all__ = [
    "TAField",
    "make_ta_field_from_templates",
]
