"""通用 active/dark piecewise propagation 执行抽象。

本模块只负责底层 piece 序列执行，不包含 TA delay、readout、absorption、
difference、phase cycling 等实验语义。用户主 API 不应直接暴露这些对象；
它们是 experiment workflow 内部共享的执行结构。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from sjh_learn.utils.fields import FieldPhyRoot, FieldPhySeries
from sjh_learn.utils.fields.field_windows import ActiveWindow


PieceKind = Literal["active", "dark"]


def _field_debug_name(field_obj: FieldPhyRoot | FieldPhySeries | None) -> str | None:
    if field_obj is None:
        return None
    return str(getattr(field_obj, "name", None) or field_obj.__class__.__name__)


@dataclass(frozen=True)
class PropagationPiece:
    """通用 active/dark 传播片段。

    active piece 必须携带 physical field；dark piece 必须 ``field=None``。
    本对象不包含 readout、absorption、signal policy 或 experiment label。
    """

    piece_name: str
    kind: PieceKind
    window: ActiveWindow
    field: FieldPhyRoot | FieldPhySeries | None = None

    def __post_init__(self) -> None:
        if not self.piece_name:
            raise ValueError("piece_name must be non-empty.")
        if self.kind == "active" and self.field is None:
            raise ValueError("active propagation piece requires field.")
        if self.kind == "dark" and self.field is not None:
            raise ValueError("dark propagation piece must have field=None.")

    @property
    def time_window(self) -> ActiveWindow:
        """返回该 piece 的传播时间窗。"""

        return self.window

    def to_dict(self) -> dict[str, object]:
        return {
            "piece_name": self.piece_name,
            "kind": self.kind,
            "window": self.window.to_dict(),
            "field_name": _field_debug_name(self.field),
            "field_repr": None if self.field is None else repr(self.field),
        }


def contains_window(outer: ActiveWindow, inner: ActiveWindow) -> bool:
    """判断 ``inner`` 是否完全落在 ``outer`` 内。"""

    return float(outer.start_fs) <= float(inner.start_fs) and float(inner.end_fs) <= float(outer.end_fs)


def extract_final_state(result: Any) -> Any:
    """从一段传播结果中提取下一段初态。"""

    if isinstance(result, PieceDynamicsResult):
        return extract_final_state(result.result)
    if isinstance(result, dict) and "final_state" in result:
        return result["final_state"]
    states = getattr(result, "states", None)
    if states:
        return states[-1]
    return result


@dataclass(frozen=True)
class PieceDynamicsResult:
    """把一个 propagation piece 与对应 raw dynamics result 绑定。"""

    piece: PropagationPiece
    result: Any

    def to_dict(self) -> dict[str, object]:
        result_type = self.result.__class__.__name__
        return {
            "piece": self.piece.to_dict(),
            "result_summary": {
                "result_type": result_type,
                "has_states": bool(getattr(self.result, "states", None)),
            },
        }


@dataclass(frozen=True)
class ActiveWindowDynamicsResult(PieceDynamicsResult):
    """active piece 的 dynamics result wrapper。"""

    def __post_init__(self) -> None:
        if self.piece.kind != "active":
            raise ValueError("ActiveWindowDynamicsResult requires piece.kind == 'active'.")


@dataclass(frozen=True)
class DarkWindowDynamicsResult(PieceDynamicsResult):
    """dark piece 的 dynamics result wrapper。"""

    def __post_init__(self) -> None:
        if self.piece.kind != "dark":
            raise ValueError("DarkWindowDynamicsResult requires piece.kind == 'dark'.")


@dataclass(frozen=True)
class PieceDynamicsResultSeries:
    """一组连续 piece dynamics results。"""

    piece_results: tuple[PieceDynamicsResult, ...]
    atol_fs: float = 1e-9

    def __post_init__(self) -> None:
        results = tuple(self.piece_results)
        if not results:
            raise ValueError("PieceDynamicsResultSeries requires at least one piece result.")

        atol = float(self.atol_fs)
        for prev, current in zip(results, results[1:]):
            prev_end = float(prev.piece.window.end_fs)
            curr_start = float(current.piece.window.start_fs)
            if abs(prev_end - curr_start) > atol:
                raise ValueError(
                    "PieceDynamicsResultSeries requires contiguous piece windows, "
                    f"got prev_end={prev_end:g}, curr_start={curr_start:g}."
                )

        object.__setattr__(self, "piece_results", results)
        object.__setattr__(self, "atol_fs", atol)

    @property
    def final_state(self) -> Any:
        """返回最后一段传播结果的 final state。"""

        return extract_final_state(self.piece_results[-1])

    def stitch(self):
        """把连续 piece raw trajectories 拼接成一个导出用 DynamicsResult。

        `DynamicsResult` 仍然表示单段 raw trajectory；本方法只在 IO/plotting
        或回归比较需要一条连续轨迹时生成派生结果。相邻 piece 的边界点只保留
        一份，避免 CSV/NPZ 中重复保存同一物理时刻。
        """

        from sjh_learn.utils.core.results import DynamicsResult

        raw_results = [item.result for item in self.piece_results]
        for raw in raw_results:
            if not isinstance(raw, DynamicsResult):
                raise TypeError("PieceDynamicsResultSeries.stitch() requires DynamicsResult raw piece results.")

        times_parts = []
        times_fs_parts = []
        states = []
        for index, raw in enumerate(raw_results):
            start = 0 if index == 0 else 1
            times_parts.append(raw.times[start:])
            if raw.times_fs is None:
                times_fs_parts.append(None)
            else:
                times_fs_parts.append(raw.times_fs[start:])
            states.extend(raw.states[start:])

        times = np.concatenate(times_parts)
        if any(part is None for part in times_fs_parts):
            times_fs = None
        else:
            times_fs = np.concatenate(times_fs_parts)

        source = raw_results[0]
        drive_source = next((raw for raw in raw_results if raw.drive is not None), source)
        metadata = dict(source.metadata)
        metadata["piecewise"] = {
            "n_pieces": len(self.piece_results),
            "pieces": [item.piece.to_dict() for item in self.piece_results],
        }
        result = DynamicsResult(
            mode=source.mode,
            times=times,
            times_fs=times_fs,
            states=states,
            parameters=source.parameters,
            physical_params=source.physical_params,
            solver_params=source.solver_params,
            metadata=metadata,
            source_mode=source.source_mode,
            drive=drive_source.drive,
            drive_dict=drive_source.drive_dict,
            drive_expr=drive_source.drive_expr,
            drive_name=drive_source.drive_name,
        )
        result.sanity_checks = getattr(source, "sanity_checks", {})
        return result

    def save_ckp(self, file):
        """保存 piecewise result checkpoint。"""

        from pathlib import Path
        import pickle

        path = Path(file)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return path

    @classmethod
    def from_ckp(cls, file):
        """读取 piecewise result checkpoint。"""

        from pathlib import Path
        import pickle

        path = Path(file)
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        if not isinstance(payload, cls):
            raise TypeError("checkpoint does not contain a PieceDynamicsResultSeries.")
        return payload

    def __getattr__(self, name: str) -> Any:
        """临时把常用单轨迹读操作委托给 stitch()。

        下游 IO/plotting 正在迁移到遍历 piece_results；在这之前，读取型
        `DynamicsResult` API 通过拼接轨迹工作。不要在这里加入求解逻辑。
        """

        return getattr(self.stitch(), name)

    def to_dict(self) -> dict[str, object]:
        return {
            "piece_results": [item.to_dict() for item in self.piece_results],
            "atol_fs": float(self.atol_fs),
        }


def wrap_piece_result(piece: PropagationPiece, raw_result: Any) -> PieceDynamicsResult:
    """按 piece kind 包装 raw dynamics result。"""

    if piece.kind == "active":
        return ActiveWindowDynamicsResult(piece=piece, result=raw_result)
    if piece.kind == "dark":
        return DarkWindowDynamicsResult(piece=piece, result=raw_result)
    raise RuntimeError(f"Unsupported propagation piece kind: {piece.kind!r}.")


def make_piece_result_series(piece_results: tuple[PieceDynamicsResult, ...]) -> PieceDynamicsResultSeries:
    """构造连续 piece result series。"""

    return PieceDynamicsResultSeries(piece_results=piece_results)


__all__ = [
    "PieceKind",
    "PropagationPiece",
    "contains_window",
    "extract_final_state",
    "PieceDynamicsResult",
    "ActiveWindowDynamicsResult",
    "DarkWindowDynamicsResult",
    "PieceDynamicsResultSeries",
    "wrap_piece_result",
    "make_piece_result_series",
]
