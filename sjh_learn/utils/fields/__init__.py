"""用户侧输入场 public API。

普通用户只应从这里导入物理单位下的 `FieldPhyRoot` 子类。solver code-unit
callable 由 `ParaNormalizer` 和 core solver 内部生成，不作为 public API 导出。
"""

from .lab_fields import (
    CarrierFieldPhysical,
    CompositeLabFieldPhysical,
    FieldPhyCustomed,
    FieldPhyRoot,
    GaussianCarrierFieldPhysical,
    default_field_from_physical_params,
    rebuild_physical_field,
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
