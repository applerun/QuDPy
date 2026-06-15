"""瞬态吸收实验的 workflow 配置。

本模块只定义 TA delay-scan 的配置对象，不保存具体物理系统参数，也不保存
pump/probe field template。物理系统应由 base NLevelPhysicalParams 提供，
field template 应由 runner 或 case builder 显式传入。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np

from sjh_learn.utils.fields.field_windows import FieldActiveWindowSettings
from sjh_learn.utils.io import make_json_safe

TA_EXPERIMENT_NAME = "ta_piecewise_delay_scan"

TaDelayModePolicy = Literal[
	"auto",
	"full_only",
]

DarkTlistMode = Literal[
	"endpoints",
	"coarse",
]

TaDelayCaseMode = Literal[
	"full_overlap",
	"pump_dark_probe",
	"pure_probe",
]


@dataclass(frozen = True)
class DelayScanSettings:
	"""TA delay-scan 设置。

	delay 约定：
		- probe 默认固定在 probe_center_fs；
		- pump_center_fs = probe_center_fs - delay_fs；
		- delay_fs > 0 表示 pump 早于 probe；
		- delay_fs < 0 表示 probe 早于 pump。

	auto 模式下，delay case 的实际模式由 pump/probe active window 决定：
		- pump 和 probe 亮区重叠：full_overlap；
		- pump 亮区完全早于 probe 亮区：pump_dark_probe；
		- probe 亮区完全早于 pump 亮区：pure_probe。
	"""

	probe_delays_fs: Sequence[float] = (
		-100.0,
		-60.0,
		-30.0,
		0.0,
		30.0,
		60.0,
		100.0,
		200.0,
		500.0,
		1000.0,
		2000.0,
		5000.0,
		10000.0,
	)
	probe_center_fs: float = 0.0
	delay_mode_policy: TaDelayModePolicy = "auto"

	# pure_probe 指 probe 在 pump 到来前已经完成读出，此时差谱理论上为零。
	# 默认不为这些 case 保存重复 dynamics，只输出零差谱或引用 probe-only reference。
	save_pure_probe_cases: bool = False

	def __post_init__(self) -> None:
		delays = np.asarray(self.probe_delays_fs, dtype = float)
		if delays.ndim != 1:
			raise ValueError("probe_delays_fs must be a 1D sequence or 1D numpy.ndarray.")
		object.__setattr__(self, "probe_delays_fs", tuple(float(x) for x in delays))
		object.__setattr__(self, "probe_center_fs", float(self.probe_center_fs))

	def pump_center_fs_for_delay(self, delay_fs: float) -> float:
		"""返回给定 delay 下 pump center 的时间位置。"""

		return float(self.probe_center_fs) - float(delay_fs)


@dataclass(frozen = True)
class PropagationSettings:
	"""TA case 传播设置。"""

	# lab_exact solver dt。不要随意放大；lab-frame 显式保留 optical carrier。
	dt_fs: float = 0.2

	# full pump+probe case 的最大点数保护。
	max_full_case_points: int = 80000

	# pump/probe segment 的最大点数保护。
	max_segment_points: int = 20000

	# pump_dark_probe 模式下，probe readout 段相对 probe active window 的额外扩展。
	probe_readout_extra_padding_fs: float = 0.0

	def __post_init__(self) -> None:
		if self.dt_fs <= 0:
			raise ValueError("dt_fs must be positive.")
		if self.max_full_case_points < 2:
			raise ValueError("max_full_case_points must be >= 2.")
		if self.max_segment_points < 2:
			raise ValueError("max_segment_points must be >= 2.")
		if self.probe_readout_extra_padding_fs < 0:
			raise ValueError("probe_readout_extra_padding_fs must be non-negative.")

		object.__setattr__(self, "dt_fs", float(self.dt_fs))
		object.__setattr__(self, "probe_readout_extra_padding_fs", float(self.probe_readout_extra_padding_fs))


@dataclass(frozen = True)
class AbsorptionSettings:
	"""从时间域 polarization 计算 absorption spectrum 的设置。"""

	number_density_m3: float = 1.0e24
	window: str | None = "hann"
	subtract_mean: bool = True
	rel_threshold: float = 1e-6
	zero_padding_factor: int = 4

	def __post_init__(self) -> None:
		if self.number_density_m3 <= 0:
			raise ValueError("number_density_m3 must be positive.")
		if self.rel_threshold <= 0:
			raise ValueError("rel_threshold must be positive.")
		if self.zero_padding_factor < 1:
			raise ValueError("zero_padding_factor must be >= 1.")

		object.__setattr__(self, "number_density_m3", float(self.number_density_m3))
		object.__setattr__(self, "rel_threshold", float(self.rel_threshold))
		object.__setattr__(self, "zero_padding_factor", int(self.zero_padding_factor))


@dataclass(frozen = True)
class OutputSettings:
	"""TA workflow 输出设置。"""

	output_dir: Path | None | str = None
	use_checkpoints: bool = True
	force_run: bool = False
	save_case_previews: bool = False

	# delay case 数量可能很多，因此保留单独分层。
	delay_case_dir_name: str = "res_per_delay"

	def __post_init__(self) -> None:
		if self.output_dir is not None:
			object.__setattr__(self, "output_dir", Path(self.output_dir))
		if not self.delay_case_dir_name:
			raise ValueError("delay_case_dir_name must be non-empty.")


@dataclass(frozen = True)
class TaDelayScanSettings:
	"""TA delay-scan 总配置。

	本类只聚合 workflow settings。具体物理系统参数来自 base NLevelPhysicalParams，
	pump/probe field template 由 runner 或 case builder 显式传入。
	"""

	experiment_name: str = TA_EXPERIMENT_NAME
	delay_scan: DelayScanSettings = field(default_factory = DelayScanSettings)
	field_window: FieldActiveWindowSettings = field(default_factory = FieldActiveWindowSettings)
	propagation: PropagationSettings = field(default_factory = PropagationSettings)
	absorption: AbsorptionSettings = field(default_factory = AbsorptionSettings)
	output: OutputSettings = field(default_factory = OutputSettings)
	metadata: dict[str, Any] = field(default_factory = dict)

	def __post_init__(self) -> None:
		if not self.experiment_name:
			raise ValueError("experiment_name must be non-empty.")
		object.__setattr__(self, "metadata", dict(self.metadata))

	@property
	def probe_delays_fs(self) -> tuple[float, ...]:
		"""便捷访问 delay 列表。"""

		return tuple(self.delay_scan.probe_delays_fs)

	@property
	def probe_center_fs(self) -> float:
		"""便捷访问 probe center。"""

		return float(self.delay_scan.probe_center_fs)

	def pump_center_fs_for_delay(self, delay_fs: float) -> float:
		"""返回给定 delay 下 pump center 的时间位置。"""

		return self.delay_scan.pump_center_fs_for_delay(delay_fs)


if __name__ == "__main__":
	import json
	from dataclasses import asdict

	import numpy as np

	# 自定义 delay 扫描：支持 numpy.ndarray，__post_init__ 会转为 tuple[float, ...]
	custom_delays = np.concatenate(
		[
			np.linspace(-300.0, 100.0, 9),
			np.array([200.0, 500.0, 1000.0, 2000.0]),
		]
	)

	settings = TaDelayScanSettings(
		experiment_name = "debug_ta_delay_scan",
		delay_scan = DelayScanSettings(
			probe_delays_fs = custom_delays,
			probe_center_fs = 0.0,
			delay_mode_policy = "auto",
			save_pure_probe_cases = False,
		),
		field_window = FieldActiveWindowSettings(
			rel_threshold = 1e-3,
			padding_fs = 2.0,
			dt_fs = 0.2,
			t_start_fs = -1500.0,
			t_end_fs = 1500.0,
		),
		propagation = PropagationSettings(
			dt_fs = 0.2,
			max_full_case_points = 80000,
			max_segment_points = 20000,
			probe_readout_extra_padding_fs = 20.0,
		),
		absorption = AbsorptionSettings(
			number_density_m3 = 1.0e24,
			window = "hann",
			subtract_mean = True,
			rel_threshold = 1e-6,
			zero_padding_factor = 4,
		),
		output = OutputSettings(
			output_dir = "./outputs/debug_ta_delay_scan",
			use_checkpoints = True,
			force_run = False,
			save_case_previews = False,
			delay_case_dir_name = "res_per_delay",
		),
		metadata = {
			"purpose": "manual settings debug",
			"note": "Only validates settings construction; no solver is executed.",
		},
	)

	print("Created custom TaDelayScanSettings.")
	print(f"experiment_name: {settings.experiment_name}")
	print(f"probe_delays_fs type: {type(settings.probe_delays_fs)}")
	print(f"probe_delays_fs: {settings.probe_delays_fs}")
	print(f"probe_center_fs: {settings.probe_center_fs}")
	print(f"pump center at delay=200 fs: {settings.pump_center_fs_for_delay(200.0)} fs")
	print(f"output_dir: {settings.output.output_dir}")
	print(f"delay case dir name: {settings.output.delay_case_dir_name}")

	# dataclass -> dict。注意这里应只包含 settings，不包含 field template 或 NLevelPhysicalParams。
	payload = asdict(settings)


	print("\nJSON-like settings preview:")
	print(json.dumps(make_json_safe(payload), indent = 2, ensure_ascii = False))
