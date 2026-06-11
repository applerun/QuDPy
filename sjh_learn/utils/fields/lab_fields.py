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

HBAR_J_S = 1.054571817e-34
E_CHARGE_C = 1.602176634e-19
FS_TO_S = 1e-15
EV_TO_FS_INV = (E_CHARGE_C / HBAR_J_S) * FS_TO_S


def _metadata_copy(metadata: dict[str, Any] | None) -> dict[str, Any]:
	return dict(metadata or {})


def _energy_eV_to_fs_inv(energy_eV: float) -> float:
	"""把 photon energy 从 eV 转成 angular frequency，单位 fs^-1。"""

	return float(energy_eV) * EV_TO_FS_INV


class FieldPhyRoot(ABC):
	"""物理电场基类。

	子类的 `physical_E_MV_per_cm(t_fs)` 必须支持 numpy array 输入，并返回
	与 `t_fs` 相同 shape 的真实 lab-frame 电场，单位为 MV/cm。这里不做
	scalar-only callable 的自动兼容；shape 不匹配时直接 fail-fast。

	Normalizer 只依赖本基类暴露的通用接口，不应根据具体 field 类型
	进行分支处理。若某个自定义 field 有自己的快时间尺度，例如脉冲宽度、
	调制频率、重复频率或其它 envelope time scale，应通过
	`normalization_rate_candidates_fs_inv` 主动提供给 normalizer。
	"""

	time_unit = "fs"
	field_unit = "MV/cm"

	def __call__(self, t_fs):
		t_array = np.asarray(t_fs, dtype = float)
		values = np.asarray(self.physical_E_MV_per_cm(t_array), dtype = float)
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

	@property
	def reference_MV_per_cm(self) -> float | None:
		"""返回 field 归一化参考幅度，单位 MV/cm。

		`ParaNormalizer.make_code_field()` 使用该值把真实电场归一化为
		`E_code(t)=E_MV_per_cm(t)/reference_MV_per_cm`。自定义 field 如果
		要进入 solver 主线，必须提供非零 reference。
		"""

		return None

	@property
	def normalization_rate_candidates_fs_inv(self) -> tuple[float, ...]:
		"""返回 field 建议的 auto-scale 速率候选，单位 fs^-1。

		该 property 只给 `ParaNormalizer` 提供数值尺度提示，不改变物理
		电场，也不参与 Hamiltonian 物理定义。默认返回空 tuple。

		自定义 field 如果包含明确的时间尺度，例如 pulse bandwidth
		`1 / sigma_fs`、调制频率、重复频率或其它快 envelope 变化尺度，
		建议覆盖该 property 并返回对应的正速率候选。
		"""

		return ()

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

	用户应继承本类并实现 `physical_E_MV_per_cm(t_fs)`、`__repr__()`。
	如需从 metadata/checkpoint 恢复，还应实现 `@classmethod rebuild(...)`。

	若该 field 有明显的快时间尺度，建议覆盖
	`normalization_rate_candidates_fs_inv`。该属性应返回 fs^-1 候选速率，
	例如 `1 / pulse_width_fs` 或 modulation angular frequency。
	"""

	@classmethod
	def rebuild(cls, payload):
		raise NotImplementedError(
			"Custom physical fields are not rebuildable unless the subclass implements rebuild()."
		)


@dataclass(frozen = True)
class CarrierFieldPhysical(FieldPhyRoot):
	"""CW lab-frame carrier field。

	物理约定：`E(t_fs) = 2 E0 cos(omega_L t_fs + phase)`，其中
	`E0_MV_per_cm` 是输入场强参数，单位 MV/cm。`phase_rad` 是该
	carrier 的 optical phase。
	"""

	E0_MV_per_cm: float
	omega_L_fs_inv: float
	phase_rad: float = 0.0
	name: str = "carrier_field_physical"
	metadata: dict[str, Any] | None = None

	@property
	def reference_MV_per_cm(self) -> float | None:
		return float(self.E0_MV_per_cm)

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
			"laser_energy_eV": metadata.get("laser_energy_eV"),
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
			E0_MV_per_cm = float(payload["E0_MV_per_cm"]),
			omega_L_fs_inv = float(payload["omega_L_fs_inv"]),
			phase_rad = float(payload.get("phase_rad", 0.0)),
			name = str(payload.get("name", "carrier_field_physical")),
			metadata = dict(payload.get("metadata") or {}),
		)


@dataclass(frozen = True)
class GaussianCarrierFieldPhysical(FieldPhyRoot):
	"""Gaussian envelope lab-frame carrier field。

	物理约定：
	`E(t_fs) = 2 E0 exp[-(t_fs-center)^2/(2 sigma^2)] cos(omega_L t_fs + phase)`。

	`phase_rad` 是该 pulse 的 carrier / optical phase。多脉冲实验中的
	相对相位应通过不同 subfield 的 `phase_rad` 表达。这里不额外定义
	envelope phase；如果后续需要 complex envelope，应新增专门的 field
	class，而不是在此 real lab-frame Gaussian field 中加入特化参数。
	"""

	E0_MV_per_cm: float
	omega_L_fs_inv: float
	center_fs: float
	sigma_fs: float
	phase_rad: float = 0.0
	name: str = "gaussian_carrier_field_physical"
	metadata: dict[str, Any] | None = None

	@property
	def reference_MV_per_cm(self) -> float | None:
		return float(self.E0_MV_per_cm)

	def __post_init__(self) -> None:
		if self.sigma_fs <= 0:
			raise ValueError("sigma_fs must be positive.")

	@property
	def normalization_rate_candidates_fs_inv(self) -> tuple[float, ...]:
		"""返回 Gaussian envelope bandwidth 对应的 auto-scale 候选。"""

		return (1.0 / float(self.sigma_fs),)

	def physical_E_MV_per_cm(self, t_fs: np.ndarray) -> np.ndarray:
		envelope = np.exp(
			-((t_fs - float(self.center_fs)) ** 2)
			/ (2.0 * float(self.sigma_fs) ** 2)
		)
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
			"laser_energy_eV": metadata.get("laser_energy_eV"),
			"phase_rad": float(self.phase_rad),
			"center_fs": float(self.center_fs),
			"sigma_fs": float(self.sigma_fs),
			"pulse_center_fs": float(self.center_fs),
			"pulse_sigma_fs": float(self.sigma_fs),
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
			E0_MV_per_cm = float(payload["E0_MV_per_cm"]),
			omega_L_fs_inv = float(payload["omega_L_fs_inv"]),
			center_fs = float(payload["center_fs"]),
			sigma_fs = float(payload["sigma_fs"]),
			phase_rad = float(payload.get("phase_rad", 0.0)),
			name = str(payload.get("name", "gaussian_carrier_field_physical")),
			metadata = dict(payload.get("metadata") or {}),
		)


def make_default_carrier_field(
		*,
		E0_MV_per_cm: float,
		laser_energy_eV: float,
		phase_rad: float = 0.0,
		name: str = "explicit_carrier_field",
		metadata: dict[str, Any] | None = None,
) -> CarrierFieldPhysical:
	"""显式构造 CW carrier field。"""

	payload = _metadata_copy(metadata)
	payload["laser_energy_eV"] = float(laser_energy_eV)
	payload.setdefault("source", "explicit field helper")
	return CarrierFieldPhysical(
		E0_MV_per_cm = float(E0_MV_per_cm),
		omega_L_fs_inv = _energy_eV_to_fs_inv(laser_energy_eV),
		phase_rad = float(phase_rad),
		name = name,
		metadata = payload,
	)


def make_default_gaussian_carrier_field(
		*,
		E0_MV_per_cm: float,
		laser_energy_eV: float,
		pulse_center_fs: float,
		pulse_sigma_fs: float,
		phase_rad: float = 0.0,
		name: str = "explicit_gaussian_carrier_field",
		metadata: dict[str, Any] | None = None,
) -> GaussianCarrierFieldPhysical:
	"""显式构造 Gaussian carrier field。

	`E0_MV_per_cm` 是表达式 `E(t)=2E0 f(t) cos(...)` 中的 E0；
	`pulse_sigma_fs` 是 Gaussian envelope 的 sigma，单位 fs。
	"""

	payload = _metadata_copy(metadata)
	payload["laser_energy_eV"] = float(laser_energy_eV)
	payload.setdefault("source", "explicit field helper")
	return GaussianCarrierFieldPhysical(
		E0_MV_per_cm = float(E0_MV_per_cm),
		omega_L_fs_inv = _energy_eV_to_fs_inv(laser_energy_eV),
		center_fs = float(pulse_center_fs),
		sigma_fs = float(pulse_sigma_fs),
		phase_rad = float(phase_rad),
		name = name,
		metadata = payload,
	)


@dataclass(frozen = True)
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
		t_fs = self.normalizer.denormalize_time_array(
			np.asarray(t_code, dtype = float), self.solver
		)
		E_MV_per_cm = self.field_phy(t_fs)
		return self.normalizer.normalize_field_MV_per_cm(
			E_MV_per_cm,
			reference_field_MV_per_cm = self.reference_field_MV_per_cm,
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
		field_phy = field_phy,
		normalizer = normalizer,
		solver = solver,
		reference_field_MV_per_cm = float(reference_field_MV_per_cm),
	)


def rebuild_physical_field(payload) -> FieldPhyRoot:
	if not isinstance(payload, dict):
		raise TypeError("rebuild_physical_field() expects a dict payload.")
	class_name = payload.get("class")
	registry = {
		"CarrierFieldPhysical": CarrierFieldPhysical,
		"GaussianCarrierFieldPhysical": GaussianCarrierFieldPhysical,
	}
	if class_name not in registry:
		raise ValueError(f"Unknown or non-rebuildable physical field class: {class_name!r}.")
	return registry[class_name].rebuild(payload)


__all__ = [
	"FieldPhyRoot",
	"FieldPhyCustomed",
	"CarrierFieldPhysical",
	"GaussianCarrierFieldPhysical",
	"make_default_carrier_field",
	"make_default_gaussian_carrier_field",
	"rebuild_physical_field",
]
