"""把用户给定的 pulse template 组装成 TA delay case。"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from sjh_learn.utils.core import (
    NLevelPhysicalParams,
    PureDephasingChannel,
    RelaxationChannel,
)
from sjh_learn.utils.fields import FieldPhyCustomed, FieldPhySeries

from .pulse_scheduling import (
    check_time_points,
    classify_delay_mode,
    compute_full_overlap_window,
    compute_piecewise_windows,
    compute_pulse_centers,
    safe_delay_tag,
)
from .ta_settings import TaDelayScanSettings
from .ta_specs import TaDelayCaseSpec, TaSegmentSpec


class ZeroFieldPhysical(FieldPhyCustomed):
    """零电场，用于 piecewise TA 的 dark segment。"""

    def __init__(
        self,
        *,
        reference_MV_per_cm: float,
        name: str = "zero_field_physical",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._reference_MV_per_cm = float(reference_MV_per_cm)
        if self._reference_MV_per_cm <= 0:
            raise ValueError("reference_MV_per_cm must be positive.")
        self.name = str(name)
        self.metadata = dict(metadata or {})

    @property
    def reference_MV_per_cm(self) -> float:
        return self._reference_MV_per_cm

    def physical_E_MV_per_cm(self, t_fs: np.ndarray) -> np.ndarray:
        return np.zeros_like(t_fs, dtype=float)

    def __repr__(self) -> str:
        return f"ZeroFieldPhysical(reference_MV_per_cm={self._reference_MV_per_cm!r})"

    def to_dict(self) -> dict[str, Any]:
        return {
            "class": self.__class__.__name__,
            "repr": repr(self),
            "name": self.name,
            "time_unit": self.time_unit,
            "field_unit": self.field_unit,
            "rebuildable": False,
            "reference_MV_per_cm": self._reference_MV_per_cm,
            "expression": "E(t_fs) = 0",
            "metadata": dict(self.metadata),
        }


class TaCaseAssembler:
    """TA case builder。

    该类只负责构造 field、NLevelPhysicalParams 和 SegmentSpec，不运行 solver，
    也不写文件。
    """

    def __init__(self, settings: TaDelayScanSettings, *, checkpoints_dir: Path | None = None) -> None:
        self.settings = settings
        self.checkpoints_dir = None if checkpoints_dir is None else Path(checkpoints_dir)

    def _checkpoint_path(self, key: str) -> Path | None:
        if not self.settings.use_checkpoints or self.checkpoints_dir is None:
            return None
        return self.checkpoints_dir / f"{key}.ckp"

    def _require_templates(self) -> tuple[Any, Any]:
        if self.settings.pump_template is None:
            raise ValueError("settings.pump_template is required.")
        if self.settings.probe_template is None:
            raise ValueError("settings.probe_template is required.")
        return self.settings.pump_template, self.settings.probe_template

    def _shift_field(self, field: Any, center_fs: float, *, name: str):
        if not hasattr(field, "time_shifted"):
            raise TypeError("TA field templates must provide time_shifted(shift_fs).")
        return field.time_shifted(float(center_fs), name=name)

    def _field_series(self, fields: tuple[Any, ...], *, name: str, metadata: dict[str, Any]):
        """构造多个物理场的线性叠加。

        优先使用已有的 FieldPhySeries，避免在 experiments 层重复定义 SumField。
        """

        try:
            return FieldPhySeries(fields=fields, name=name, metadata=metadata)
        except TypeError:
            # 兼容较旧的 FieldPhySeries 构造函数；metadata 会写入 physical params。
            return FieldPhySeries(fields)

    def make_dark_field(self) -> ZeroFieldPhysical:
        pump_template, probe_template = self._require_templates()
        refs = [
            getattr(pump_template, "reference_MV_per_cm", None),
            getattr(probe_template, "reference_MV_per_cm", None),
            1e-12,
        ]
        reference = max(abs(float(x)) for x in refs if x is not None)
        return ZeroFieldPhysical(
            reference_MV_per_cm=reference,
            metadata={
                "experiment_name": self.settings.experiment_name,
                "role": "dark",
                "description": "零光场，用于无场精确传播。",
            },
        )

    def make_shifted_pump_probe_fields(self, delay_fs: float) -> tuple[Any, Any, Any, float, float]:
        pump_template, probe_template = self._require_templates()
        pump_center, probe_center = compute_pulse_centers(
            delay_fs=float(delay_fs),
            probe_center_fs=self.settings.probe_center_fs,
        )

        pump = self._shift_field(
            pump_template,
            pump_center,
            name=f"pump_center_{pump_center:g}_fs",
        )
        probe = self._shift_field(
            probe_template,
            probe_center,
            name=f"probe_center_{probe_center:g}_fs",
        )
        pump_probe = self._field_series(
            (pump, probe),
            name=f"pump_probe_delay_{safe_delay_tag(delay_fs)}_fs",
            metadata={
                "role": "pump_probe",
                "delay_fs": float(delay_fs),
                "pump_center_fs": pump_center,
                "probe_center_fs": probe_center,
            },
        )
        return pump, probe, pump_probe, pump_center, probe_center

    def make_physical_params(
        self,
        *,
        field: Any,
        t_start_fs: float,
        t_end_fs: float,
        dt_fs: float,
        case_name: str,
        description: str,
        segment_metadata: dict[str, Any],
    ) -> NLevelPhysicalParams:
        s = self.settings
        return NLevelPhysicalParams(
            energies_eV=tuple(float(x) for x in s.energies_eV),
            dipole_matrix_D=tuple(tuple(float(v) for v in row) for row in s.dipole_matrix_D),
            t_start_fs=float(t_start_fs),
            t_end_fs=float(t_end_fs),
            dt_fs=float(dt_fs),
            field=field,
            basis=tuple(s.basis),
            relaxation_channels=(
                RelaxationChannel(
                    name="relaxation_2_to_1",
                    from_level=2,
                    to_level=1,
                    T1_fs=float(s.T1_2_to_1_fs),
                ),
                RelaxationChannel(
                    name="relaxation_1_to_0",
                    from_level=1,
                    to_level=0,
                    T1_fs=float(s.T1_1_to_0_fs),
                ),
            ),
            pure_dephasing_channels=(
                PureDephasingChannel(
                    name="pure_dephasing_level_1",
                    level=1,
                    Tphi_fs=float(s.Tphi_1_fs),
                ),
                PureDephasingChannel(
                    name="pure_dephasing_level_2",
                    level=2,
                    Tphi_fs=float(s.Tphi_2_fs),
                ),
            ),
            solver_mode="lab_exact",
            input_description=description,
            input_metadata={
                "experiment_name": s.experiment_name,
                "case_name": case_name,
                "number_density_m3": float(s.number_density_m3),
                **segment_metadata,
            },
        )

    def build_probe_only_segment(self) -> TaSegmentSpec:
        _, probe_template = self._require_templates()
        s = self.settings
        probe = self._shift_field(
            probe_template,
            s.probe_center_fs,
            name=f"probe_center_{s.probe_center_fs:g}_fs",
        )
        t_start = s.probe_center_fs - s.probe_window_half_width_fs
        t_end = s.probe_center_fs + s.probe_window_half_width_fs
        check_time_points(
            key="probe_only",
            t_start_fs=t_start,
            t_end_fs=t_end,
            dt_fs=s.dt_fs,
            max_points=s.max_segment_points,
        )
        params = self.make_physical_params(
            field=probe,
            t_start_fs=t_start,
            t_end_fs=t_end,
            dt_fs=s.dt_fs,
            case_name="probe_only",
            description="Probe-only reference segment.",
            segment_metadata={"role": "probe_only", "segment_role": "probe_only"},
        )
        return TaSegmentSpec(
            segment_key="probe_only",
            role="probe_only",
            field=probe,
            t_start_fs=float(t_start),
            t_end_fs=float(t_end),
            dt_fs=float(s.dt_fs),
            params=params,
            checkpoint_path=self._checkpoint_path("probe_only"),
        )

    def build_delay_case(self, delay_fs: float) -> TaDelayCaseSpec:
        s = self.settings
        delay = float(delay_fs)
        pump, probe, pump_probe, pump_center, probe_center = self.make_shifted_pump_probe_fields(delay)
        mode = classify_delay_mode(
            delay_fs=delay,
            piecewise_min_positive_delay_fs=s.piecewise_min_positive_delay_fs,
        )
        case_key = f"delay_{safe_delay_tag(delay)}_fs_pump_probe"

        if mode == "full_overlap":
            window = compute_full_overlap_window(
                pump_center_fs=pump_center,
                probe_center_fs=probe_center,
                pump_half_width_fs=s.pump_window_half_width_fs,
                probe_half_width_fs=s.probe_window_half_width_fs,
                extra_padding_fs=s.full_overlap_extra_padding_fs,
                dt_fs=s.dt_fs,
            )
            check_time_points(
                key=case_key,
                t_start_fs=window.t_start_fs,
                t_end_fs=window.t_end_fs,
                dt_fs=s.dt_fs,
                max_points=s.max_full_case_points,
            )
            params = self.make_physical_params(
                field=pump_probe,
                t_start_fs=window.t_start_fs,
                t_end_fs=window.t_end_fs,
                dt_fs=s.dt_fs,
                case_name=case_key,
                description=f"Full lab-frame pump-probe case for delay={delay:g} fs.",
                segment_metadata={
                    "role": "pump_probe",
                    "mode": "full_overlap",
                    "delay_fs": delay,
                    "pump_center_fs": pump_center,
                    "probe_center_fs": probe_center,
                },
            )
            segment = TaSegmentSpec(
                segment_key=case_key,
                role="full",
                field=pump_probe,
                t_start_fs=window.t_start_fs,
                t_end_fs=window.t_end_fs,
                dt_fs=float(s.dt_fs),
                params=params,
                checkpoint_path=self._checkpoint_path(case_key),
            )
            return TaDelayCaseSpec(
                case_key=case_key,
                delay_fs=delay,
                mode="full_overlap",
                pump_center_fs=pump_center,
                probe_center_fs=probe_center,
                segments=(segment,),
            )

        windows = compute_piecewise_windows(
            pump_center_fs=pump_center,
            probe_center_fs=probe_center,
            pump_half_width_fs=s.pump_window_half_width_fs,
            probe_half_width_fs=s.probe_window_half_width_fs,
            dt_fs=s.dt_fs,
        )
        check_time_points(
            key=f"{case_key}_pump",
            t_start_fs=windows.pump_start_fs,
            t_end_fs=windows.pump_end_fs,
            dt_fs=s.dt_fs,
            max_points=s.max_segment_points,
        )
        check_time_points(
            key=f"{case_key}_probe",
            t_start_fs=windows.probe_start_fs,
            t_end_fs=windows.probe_end_fs,
            dt_fs=s.dt_fs,
            max_points=s.max_segment_points,
        )

        pump_params = self.make_physical_params(
            field=pump,
            t_start_fs=windows.pump_start_fs,
            t_end_fs=windows.pump_end_fs,
            dt_fs=s.dt_fs,
            case_name=case_key,
            description=f"Piecewise pump segment for delay={delay:g} fs.",
            segment_metadata={
                "role": "pump_probe",
                "mode": "piecewise",
                "segment_role": "pump",
                "delay_fs": delay,
                "pump_center_fs": pump_center,
                "probe_center_fs": probe_center,
            },
        )
        dark_params = self.make_physical_params(
            field=self.make_dark_field(),
            t_start_fs=windows.dark_start_fs,
            t_end_fs=windows.dark_end_fs,
            dt_fs=max(windows.dark_end_fs - windows.dark_start_fs, s.dt_fs),
            case_name=case_key,
            description=f"Piecewise dark segment for delay={delay:g} fs.",
            segment_metadata={
                "role": "pump_probe",
                "mode": "piecewise",
                "segment_role": "dark",
                "delay_fs": delay,
                "pump_center_fs": pump_center,
                "probe_center_fs": probe_center,
            },
        )
        probe_params = self.make_physical_params(
            field=probe,
            t_start_fs=windows.probe_start_fs,
            t_end_fs=windows.probe_end_fs,
            dt_fs=s.dt_fs,
            case_name=case_key,
            description=f"Piecewise probe/readout segment for delay={delay:g} fs.",
            segment_metadata={
                "role": "pump_probe",
                "mode": "piecewise",
                "segment_role": "probe_readout",
                "delay_fs": delay,
                "pump_center_fs": pump_center,
                "probe_center_fs": probe_center,
            },
        )

        segments = (
            TaSegmentSpec(
                segment_key=f"{case_key}_pump",
                role="pump",
                field=pump,
                t_start_fs=windows.pump_start_fs,
                t_end_fs=windows.pump_end_fs,
                dt_fs=float(s.dt_fs),
                params=pump_params,
                checkpoint_path=self._checkpoint_path(f"{case_key}_pump"),
            ),
            TaSegmentSpec(
                segment_key=f"{case_key}_dark",
                role="dark",
                field=dark_params.field,
                t_start_fs=windows.dark_start_fs,
                t_end_fs=windows.dark_end_fs,
                dt_fs=float(dark_params.dt_fs),
                params=dark_params,
                checkpoint_path=self._checkpoint_path(f"{case_key}_dark"),
            ),
            TaSegmentSpec(
                segment_key=case_key,
                role="probe",
                field=probe,
                t_start_fs=windows.probe_start_fs,
                t_end_fs=windows.probe_end_fs,
                dt_fs=float(s.dt_fs),
                params=probe_params,
                checkpoint_path=self._checkpoint_path(case_key),
            ),
        )
        return TaDelayCaseSpec(
            case_key=case_key,
            delay_fs=delay,
            mode="piecewise",
            pump_center_fs=pump_center,
            probe_center_fs=probe_center,
            segments=segments,
        )
