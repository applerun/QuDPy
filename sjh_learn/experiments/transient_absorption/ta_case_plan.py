"""TA delay-case plan 生成模块。

本模块只根据 pump/probe physical field template 与 active-window 设置生成
delay-case plan：active/dark pieces、piece field、readout_window 和 metadata。
它不运行 solver，不做 dark propagation，不计算 absorption，也不写输出文件。
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from typing import Any, Literal

from sjh_learn.utils.core.piecewise_propagation import (
    PropagationPiece,
    contains_window,
)
from sjh_learn.utils.fields import FieldPhyRoot, FieldPhySeries
from sjh_learn.utils.fields.field_windows import (
    ActiveWindow,
    FieldActiveWindowSettings,
    detect_active_windows,
)


SignalPolicy = Literal["normal", "zero_difference"]
TaDelayCaseLabel = Literal["full_overlap", "pump_dark_probe", "pure_probe"]


@dataclass(frozen=True)
class TaPulseCenters:
    """一个 delay case 中 pump/probe 的中心时间。"""

    delay_fs: float
    pump_center_fs: float
    probe_center_fs: float

    def to_dict(self) -> dict[str, float]:
        return {
            "delay_fs": float(self.delay_fs),
            "pump_center_fs": float(self.pump_center_fs),
            "probe_center_fs": float(self.probe_center_fs),
        }


@dataclass(frozen=True)
class TaDelayCasePlan:
    """单个 TA delay case 的传播计划。

    pieces 定义 active/dark 传播顺序；readout_window 定义后续吸收谱截取窗口。
    """

    delay_fs: float
    case_name: str
    pump_center_fs: float
    probe_center_fs: float
    pieces: tuple[PropagationPiece, ...]
    readout_window: ActiveWindow
    signal_policy: SignalPolicy
    case_label: TaDelayCaseLabel
    propagation_threshold: float | None = None
    readout_threshold: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        pieces = tuple(self.pieces)
        if not pieces:
            raise ValueError("TaDelayCasePlan.pieces must contain at least one piece.")
        active_pieces = tuple(piece for piece in pieces if piece.kind == "active")
        if not active_pieces:
            raise ValueError("TaDelayCasePlan requires at least one active piece.")
        if not any(contains_window(piece.window, self.readout_window) for piece in active_pieces):
            raise ValueError("readout_window must be contained in one active propagation piece.")
        object.__setattr__(self, "pieces", pieces)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def readout_piece(self) -> PropagationPiece:
        """返回包含 readout_window 的 active piece。"""

        matches = tuple(
            piece
            for piece in self.pieces
            if piece.kind == "active" and contains_window(piece.window, self.readout_window)
        )
        if len(matches) != 1:
            raise ValueError("readout_window must be contained in exactly one active piece.")
        return matches[0]

    def to_dict(self) -> dict[str, object]:
        return {
            "delay_fs": float(self.delay_fs),
            "case_name": self.case_name,
            "pump_center_fs": float(self.pump_center_fs),
            "probe_center_fs": float(self.probe_center_fs),
            "pieces": [piece.to_dict() for piece in self.pieces],
            "readout_window": self.readout_window.to_dict(),
            "signal_policy": self.signal_policy,
            "case_label": self.case_label,
            "propagation_threshold": self.propagation_threshold,
            "readout_threshold": self.readout_threshold,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class TaDelayScanPlan:
    """TA delay scan 的懒执行计划。

    iter_delay_cases() 每次只生成一个 TaDelayCasePlan，避免一次性保存所有
    shifted field。
    """

    delays_fs: tuple[float, ...]
    pump_template: FieldPhyRoot
    probe_template: FieldPhyRoot
    field_window_settings: FieldActiveWindowSettings
    probe_center_fs: float = 0.0
    readout_extra_padding_fs: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "delays_fs", tuple(float(delay) for delay in self.delays_fs))

    def iter_delay_cases(self) -> Iterator[TaDelayCasePlan]:
        """逐个生成 delay case plan。"""

        for delay_fs in self.delays_fs:
            yield make_delay_case_plan(
                delay_fs=delay_fs,
                pump_template=self.pump_template,
                probe_template=self.probe_template,
                field_window_settings=self.field_window_settings,
                probe_center_fs=self.probe_center_fs,
                readout_extra_padding_fs=self.readout_extra_padding_fs,
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "delays_fs": [float(delay) for delay in self.delays_fs],
            "probe_center_fs": float(self.probe_center_fs),
            "readout_extra_padding_fs": float(self.readout_extra_padding_fs),
        }


def compute_pulse_centers(
    *,
    delay_fs: float,
    probe_center_fs: float = 0.0,
) -> tuple[float, float]:
    """根据 TA delay 约定计算 pump/probe center。

    delay_fs = probe_center_fs - pump_center_fs；delay_fs > 0 表示 pump 早于 probe。
    """

    probe_center = float(probe_center_fs)
    pump_center = probe_center - float(delay_fs)
    return pump_center, probe_center


def make_delay_case_name(
    *,
    delay_fs: float,
    case_label: TaDelayCaseLabel,
) -> str:
    """生成稳定的 delay case 名称。"""

    delay = float(delay_fs)
    sign = "p" if delay >= 0 else "m"
    value = abs(delay)
    if value.is_integer():
        delay_token = f"{sign}{int(value)}fs"
    else:
        delay_token = f"{sign}{value:g}fs".replace(".", "p")
    return f"delay_{delay_token}_{case_label}"


def infer_signal_policy_from_pieces(pieces: tuple[PropagationPiece, ...]) -> SignalPolicy:
    """根据第一个 active piece 推断信号策略。"""

    for piece in pieces:
        if piece.kind != "active":
            continue
        return "zero_difference" if piece.window.name == "probe" else "normal"
    raise ValueError("Cannot infer signal policy without an active propagation piece.")


def infer_case_label_from_pieces(pieces: tuple[PropagationPiece, ...]) -> TaDelayCaseLabel:
    """根据 pieces 推断 TA case label。"""

    active_names = tuple(piece.window.name for piece in pieces if piece.kind == "active")
    if not active_names:
        raise ValueError("Cannot infer case_label without active pieces.")
    first = active_names[0]
    if len(active_names) == 1 and first == "pump_and_probe":
        return "full_overlap"
    if first == "pump" and any(name is not None and "probe" in name.split("_and_") for name in active_names[1:]):
        return "pump_dark_probe"
    if first == "probe":
        return "pure_probe"
    if first == "pump_and_probe":
        return "full_overlap"
    raise ValueError(f"Cannot infer TA case_label from active window names: {active_names!r}.")


def _window_field_for_name(
    *,
    window_name: str | None,
    pump_field: FieldPhyRoot,
    probe_field: FieldPhyRoot,
    pump_probe_field: FieldPhySeries,
) -> FieldPhyRoot | FieldPhySeries:
    if window_name == "pump":
        return pump_field
    if window_name == "probe":
        return probe_field
    if window_name == "pump_and_probe":
        return pump_probe_field
    if window_name is not None:
        parts = tuple(part for part in str(window_name).split("_and_") if part)
        if set(parts).issubset({"pump", "probe"}) and len(parts) >= 2:
            return pump_probe_field
    raise ValueError(f"Unsupported TA propagation window name: {window_name!r}.")


def _active_piece_from_window(
    *,
    window: ActiveWindow,
    pump_field: FieldPhyRoot,
    probe_field: FieldPhyRoot,
    pump_probe_field: FieldPhySeries,
) -> PropagationPiece:
    name = window.name
    if name is None:
        raise ValueError("Propagation ActiveWindow.name is required for TA piece naming.")
    return PropagationPiece(
        piece_name=f"active_{name}",
        kind="active",
        window=window,
        field=_window_field_for_name(
            window_name=name,
            pump_field=pump_field,
            probe_field=probe_field,
            pump_probe_field=pump_probe_field,
        ),
    )


def _dark_piece_between(prev: PropagationPiece, next_piece: PropagationPiece) -> PropagationPiece:
    prev_name = prev.window.name
    next_name = next_piece.window.name
    if prev_name is None or next_name is None:
        raise ValueError("Dark piece naming requires neighboring window names.")
    if float(next_piece.window.start_fs) < float(prev.window.end_fs):
        raise ValueError(f"Cannot create dark piece between overlapping windows: {prev.window!r}, {next_piece.window!r}.")
    dark_name = f"{prev_name}_to_{next_name}"
    return PropagationPiece(
        piece_name=f"dark_{dark_name}",
        kind="dark",
        window=ActiveWindow(
            start_fs=float(prev.window.end_fs),
            end_fs=float(next_piece.window.start_fs),
            name=dark_name,
        ),
        field=None,
    )


def _pieces_from_propagation_windows(
    *,
    propagation_windows: tuple[ActiveWindow, ...],
    pump_field: FieldPhyRoot,
    probe_field: FieldPhyRoot,
    pump_probe_field: FieldPhySeries,
) -> tuple[PropagationPiece, ...]:
    if not propagation_windows:
        raise ValueError("No propagation active windows detected for TA delay case.")

    active_pieces = tuple(
        _active_piece_from_window(
            window=window,
            pump_field=pump_field,
            probe_field=probe_field,
            pump_probe_field=pump_probe_field,
        )
        for window in sorted(propagation_windows, key=lambda item: item.start_fs)
    )
    pieces: list[PropagationPiece] = [active_pieces[0]]
    for prev, current in zip(active_pieces, active_pieces[1:]):
        pieces.append(_dark_piece_between(prev, current))
        pieces.append(current)
    return tuple(pieces)


def _detect_single_probe_readout_window(
    *,
    probe_field: FieldPhyRoot,
    field_window_settings: FieldActiveWindowSettings,
    readout_extra_padding_fs: float,
) -> tuple[ActiveWindow, float]:
    """独立检测 probe-only readout window。"""

    readout_settings = replace(field_window_settings, force_single_window=True)
    probe_windows, threshold = detect_active_windows(
        probe_field,
        settings=readout_settings,
        field_names=("probe",),
    )
    if len(probe_windows) != 1:
        raise ValueError(
            "TA readout detection requires exactly one probe window, "
            f"got {len(probe_windows)} windows."
        )
    return probe_windows[0].padded(float(readout_extra_padding_fs)), float(threshold)


def make_delay_case_plan(
    *,
    delay_fs: float,
    pump_template: FieldPhyRoot,
    probe_template: FieldPhyRoot,
    field_window_settings: FieldActiveWindowSettings,
    probe_center_fs: float = 0.0,
    readout_extra_padding_fs: float = 0.0,
    case_name: str | None = None,
) -> TaDelayCasePlan:
    """生成单个 delay 的完整 TA case plan。"""

    pump_center_fs, probe_center_fs = compute_pulse_centers(
        delay_fs=delay_fs,
        probe_center_fs=probe_center_fs,
    )
    pump_field = pump_template.time_shifted(pump_center_fs, name="pump")
    probe_field = probe_template.time_shifted(probe_center_fs, name="probe")
    pump_probe_field = FieldPhySeries(
        fields=(pump_field, probe_field),
        sub_field_names=("pump", "probe"),
        name=f"pump_probe_delay_{float(delay_fs):g}_fs",
        metadata={
            "role": "ta_pump_probe_composite",
            "delay_fs": float(delay_fs),
            "pump_center_fs": float(pump_center_fs),
            "probe_center_fs": float(probe_center_fs),
        },
    )

    propagation_windows, propagation_threshold = detect_active_windows(
        pump_probe_field,
        settings=field_window_settings,
    )
    readout_window, readout_threshold = _detect_single_probe_readout_window(
        probe_field=probe_field,
        field_window_settings=field_window_settings,
        readout_extra_padding_fs=readout_extra_padding_fs,
    )
    pieces = _pieces_from_propagation_windows(
        propagation_windows=propagation_windows,
        pump_field=pump_field,
        probe_field=probe_field,
        pump_probe_field=pump_probe_field,
    )
    signal_policy = infer_signal_policy_from_pieces(pieces)
    if signal_policy == "zero_difference":
        first_active = next(piece for piece in pieces if piece.kind == "active")
        pieces = (first_active,)
    case_label = infer_case_label_from_pieces(pieces)
    resolved_case_name = case_name or make_delay_case_name(delay_fs=delay_fs, case_label=case_label)

    return TaDelayCasePlan(
        delay_fs=float(delay_fs),
        case_name=resolved_case_name,
        pump_center_fs=float(pump_center_fs),
        probe_center_fs=float(probe_center_fs),
        pieces=pieces,
        readout_window=readout_window,
        signal_policy=signal_policy,
        case_label=case_label,
        propagation_threshold=float(propagation_threshold),
        readout_threshold=float(readout_threshold),
        metadata={
            "propagation_windows": [window.to_dict() for window in propagation_windows],
        },
    )


def make_delay_scan_plan(
    *,
    delays_fs: tuple[float, ...] | list[float],
    pump_template: FieldPhyRoot,
    probe_template: FieldPhyRoot,
    field_window_settings: FieldActiveWindowSettings,
    probe_center_fs: float = 0.0,
    readout_extra_padding_fs: float = 0.0,
) -> TaDelayScanPlan:
    """构造整个 delay scan 的懒执行计划。"""

    return TaDelayScanPlan(
        delays_fs=tuple(float(delay) for delay in delays_fs),
        pump_template=pump_template,
        probe_template=probe_template,
        field_window_settings=field_window_settings,
        probe_center_fs=float(probe_center_fs),
        readout_extra_padding_fs=float(readout_extra_padding_fs),
    )


__all__ = [
    "SignalPolicy",
    "TaDelayCaseLabel",
    "TaPulseCenters",
    "TaDelayCasePlan",
    "TaDelayScanPlan",
    "contains_window",
    "compute_pulse_centers",
    "make_delay_case_name",
    "make_delay_case_plan",
    "make_delay_scan_plan",
    "infer_signal_policy_from_pieces",
    "infer_case_label_from_pieces",
]
