"""用户侧输入场 public API。

普通用户只应从这里导入物理单位下的 `FieldPhyRoot` 子类。solver code-unit
callable 由 `ParaNormalizer` 和 core solver 内部生成，不作为 public API 导出。
"""

from .lab_fields import (
    CarrierFieldPhysical,
    FieldPhyCustomed,
    FieldPhyRoot,
    GaussianCarrierFieldPhysical,
    make_default_carrier_field,
    make_default_gaussian_carrier_field,
    rebuild_physical_field,
)
from .field_series import (
    FieldPhySeries,
    TAField,
    TwoDESField,
    iter_ta_gaussian_fields,
    iter_twodes_gaussian_fields,
    make_ta_gaussian_field,
    make_twodes_gaussian_field,
)

__all__ = [
    "FieldPhyRoot",
    "FieldPhyCustomed",
    "CarrierFieldPhysical",
    "GaussianCarrierFieldPhysical",
    "make_default_carrier_field",
    "make_default_gaussian_carrier_field",
    "rebuild_physical_field",
    "FieldPhySeries",
    "TAField",
    "TwoDESField",
    "make_ta_gaussian_field",
    "make_twodes_gaussian_field",
    "iter_ta_gaussian_fields",
    "iter_twodes_gaussian_fields",
]
