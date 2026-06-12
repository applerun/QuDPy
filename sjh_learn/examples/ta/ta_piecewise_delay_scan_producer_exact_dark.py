#!/usr/bin/env python3
"""Piecewise three-level TA delay-scan producer.

This script is a simulation producer for long-delay TA workflows.

QuDPy side:
    - build per-delay simulation cases
    - run lab-frame dynamics with checkpoints
    - save one readout DynamicsResult per delay under simulation/res_per_delay
    - save one difference spectrum per delay under real/final_output

UFANSYS side:
    - read the per-delay spectra / case outputs
    - handle non-uniform time axes and energy axes
    - make final TA maps, kinetics, and publication figures

For long positive delays, the pump+probe case is run piecewise:

    pump segment -> dark segment -> probe/readout segment

Each segment is propagated by run_case. The next segment receives the previous
segment's final density matrix as rho0.

For pulse-overlap delays, the case falls back to a full pump+probe lab_exact run,
because the Hamiltonian genuinely contains pump and probe at the same time.

No RWA/envelope approximation is introduced.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import csv
import json
import math
import sys
from typing import Any, Literal

import numpy as np
from qutip import Qobj, basis, liouvillian, operator_to_vector, vector_to_operator

if __package__ is None or __package__ == "":
    # Intended path:
    #   QuDPy/sjh_learn/examples/ta/ta_piecewise_delay_scan_producer.py
    # so parents[3] is the repository root.
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from sjh_learn.utils.core import (
    NLevelPhysicalParams,
    ParaNormalizer,
    PureDephasingChannel,
    RelaxationChannel,
    run_case,
)
from sjh_learn.utils.core.results import DynamicsResult
from sjh_learn.utils.fields import FieldPhyCustomed, make_ta_gaussian_field
from sjh_learn.utils.io import save_result_case
from sjh_learn.utils.spectroscopy import lab_frame_fft_response, polarization_C_per_m2


EXAMPLE_NAME = "ta_piecewise_delay_scan_producer"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / EXAMPLE_NAME


def _json_safe(value: Any) -> Any:
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _json_safe(value.to_dict())
    if hasattr(value, "__dataclass_fields__"):
        return _json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, complex):
        return {"real": float(np.real(value)), "imag": float(np.imag(value))}
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _safe_tag(value: float) -> str:
    if abs(value) < 1e-12:
        value = 0.0
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    return text.replace("-", "m").replace(".", "p")


def _align_floor(value: float, dt: float) -> float:
    return math.floor(float(value) / float(dt)) * float(dt)


def _align_ceil(value: float, dt: float) -> float:
    return math.ceil(float(value) / float(dt)) * float(dt)


def _n_points(t_start_fs: float, t_end_fs: float, dt_fs: float) -> int:
    return int(round((float(t_end_fs) - float(t_start_fs)) / float(dt_fs))) + 1


class ZeroFieldPhysical(FieldPhyCustomed):
    """Zero lab-frame field with a non-zero reference scale for normalization."""

    def __init__(
        self,
        *,
        reference_MV_per_cm: float,
        name: str = "zero_field_physical",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._reference_MV_per_cm = float(reference_MV_per_cm)
        if self._reference_MV_per_cm <= 0:
            raise ValueError("reference_MV_per_cm must be positive for ZeroFieldPhysical.")
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


@dataclass(frozen=True)
class TaPiecewiseConfig:
    example_name: str = EXAMPLE_NAME

    # Include long positive delays; UFANSYS should handle non-uniform output axes.
    probe_delays_fs: tuple[float, ...] = (
        -100.0, -60.0, -30.0, 0.0, 30.0, 60.0, 100.0,
        200.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0,
    )

    probe_center_fs: float = 0.0

    pump_E0_MV_per_cm: float = 0.18
    probe_E0_MV_per_cm: float = 0.012
    pump_laser_energy_eV: float = 1.55
    probe_laser_energy_eV: float = 1.65
    pump_sigma_fs: float = 12.0
    probe_sigma_fs: float = 7.0
    pump_phase_rad: float = 0.0
    probe_phase_rad: float = 0.0

    # lab_exact dt. Do not increase this casually; it must resolve optical carrier.
    dt_fs: float = 0.2

    # Full lab_exact is used in overlap region. Beyond this threshold, use piecewise.
    piecewise_min_positive_delay_fs: float = 120.0

    # Segment windows.
    pulse_padding_sigma_factor: float = 8.0
    probe_readout_padding_sigma_factor: float = 10.0

    # Long dark segment output policy.
    # "endpoints" writes only start/end times to avoid dense dark trajectories.
    dark_tlist_mode: Literal["endpoints", "coarse"] = "coarse"
    dark_coarse_dt_fs: float = 50.0
    max_dark_interval_fs: float = 50.0

    # Guardrails.
    max_full_case_points: int = 80000
    max_segment_points: int = 20000

    # Three-level ladder demonstration system.
    basis: tuple[str, ...] = ("g", "e", "f")
    energies_eV: tuple[float, ...] = (0.0, 1.55, 3.30)
    dipole_matrix_D: tuple[tuple[float, ...], ...] = (
        (0.0, 5.0, 0.0),
        (5.0, 0.0, 4.0),
        (0.0, 4.0, 0.0),
    )

    T1_2_to_1_fs: float = 150.0
    T1_1_to_0_fs: float = 350.0
    Tphi_1_fs: float = 90.0
    Tphi_2_fs: float = 80.0

    number_density_m3: float = 1.0e24
    window: str | None = "hann"
    subtract_mean: bool = True
    rel_threshold: float = 1e-6
    zero_padding_factor: int = 4

    use_checkpoints: bool = True
    force_run: bool = False
    save_case_previews: bool = False


@dataclass(frozen=True)
class SegmentSpec:
    segment_key: str
    role: Literal["full", "pump", "dark", "probe", "probe_only"]
    field: Any
    t_start_fs: float
    t_end_fs: float
    dt_fs: float
    params: NLevelPhysicalParams
    checkpoint_path: Path | None


@dataclass(frozen=True)
class DelayCaseSpec:
    case_key: str
    delay_fs: float
    mode: Literal["full_overlap", "piecewise"]
    pump_center_fs: float
    probe_center_fs: float
    segments: tuple[SegmentSpec, ...]


@dataclass
class DelaySpectrumResult:
    case_key: str
    delay_fs: float
    mode: str
    readout_result: Any
    readout_response: dict[str, np.ndarray]
    probe_reference_response: dict[str, np.ndarray]
    spectrum_csv: Path


class TaPiecewiseExp:
    def __init__(
        self,
        config: TaPiecewiseConfig | None = None,
        *,
        output_dir: Path | None = None,
        normalizer: ParaNormalizer | None = None,
    ) -> None:
        self.config = TaPiecewiseConfig() if config is None else config
        self.output_dir = DEFAULT_OUTPUT_DIR if output_dir is None else Path(output_dir)
        self.normalizer = ParaNormalizer(auto_scale=True) if normalizer is None else normalizer

    @property
    def simulation_dir(self) -> Path:
        return self.output_dir / "simulation"

    @property
    def final_output_dir(self) -> Path:
        return self.output_dir / "real" / "final_output"

    @property
    def checkpoints_dir(self) -> Path:
        return self.simulation_dir / "checkpoints"

    def make_ta_field(self, delay_fs: float):
        c = self.config
        pump_center_fs = float(c.probe_center_fs) - float(delay_fs)
        return make_ta_gaussian_field(
            probe_delay_fs=float(delay_fs),
            pump_E0_MV_per_cm=float(c.pump_E0_MV_per_cm),
            probe_E0_MV_per_cm=float(c.probe_E0_MV_per_cm),
            pump_laser_energy_eV=float(c.pump_laser_energy_eV),
            probe_laser_energy_eV=float(c.probe_laser_energy_eV),
            pump_center_fs=pump_center_fs,
            pump_sigma_fs=float(c.pump_sigma_fs),
            probe_sigma_fs=float(c.probe_sigma_fs),
            pump_phase_rad=float(c.pump_phase_rad),
            probe_phase_rad=float(c.probe_phase_rad),
            name=f"ta_delay_{delay_fs:g}_fs",
            metadata={
                "example_name": c.example_name,
                "probe_delay_fs": float(delay_fs),
                "time_anchor": "probe",
                "pump_center_fs": pump_center_fs,
                "probe_center_fs": float(c.probe_center_fs),
            },
        )

    def make_probe_field(self):
        return self.make_ta_field(0.0)["probe"]

    def make_dark_field(self) -> ZeroFieldPhysical:
        c = self.config
        ref = max(abs(float(c.pump_E0_MV_per_cm)), abs(float(c.probe_E0_MV_per_cm)), 1e-12)
        return ZeroFieldPhysical(
            reference_MV_per_cm=ref,
            metadata={
                "example_name": c.example_name,
                "role": "dark",
                "description": "Zero optical field used for dark evolution.",
            },
        )

    def physical_params(
        self,
        *,
        field,
        t_start_fs: float,
        t_end_fs: float,
        dt_fs: float,
        case_name: str,
        description: str,
        segment_metadata: dict[str, Any],
    ) -> NLevelPhysicalParams:
        c = self.config
        return NLevelPhysicalParams(
            energies_eV=tuple(float(x) for x in c.energies_eV),
            dipole_matrix_D=tuple(tuple(float(v) for v in row) for row in c.dipole_matrix_D),
            t_start_fs=float(t_start_fs),
            t_end_fs=float(t_end_fs),
            dt_fs=float(dt_fs),
            field=field,
            basis=tuple(c.basis),
            relaxation_channels=(
                RelaxationChannel(
                    name="relaxation_2_to_1",
                    from_level=2,
                    to_level=1,
                    T1_fs=float(c.T1_2_to_1_fs),
                ),
                RelaxationChannel(
                    name="relaxation_1_to_0",
                    from_level=1,
                    to_level=0,
                    T1_fs=float(c.T1_1_to_0_fs),
                ),
            ),
            pure_dephasing_channels=(
                PureDephasingChannel(
                    name="pure_dephasing_level_1",
                    level=1,
                    Tphi_fs=float(c.Tphi_1_fs),
                ),
                PureDephasingChannel(
                    name="pure_dephasing_level_2",
                    level=2,
                    Tphi_fs=float(c.Tphi_2_fs),
                ),
            ),
            solver_mode="lab_exact",
            input_description=description,
            input_metadata={
                "example_name": c.example_name,
                "case_name": case_name,
                "number_density_m3": float(c.number_density_m3),
                **segment_metadata,
            },
        )

    def _ckp_path(self, key: str) -> Path | None:
        if not self.config.use_checkpoints:
            return None
        path = self.checkpoints_dir / f"{key}.ckp"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _check_points(self, *, key: str, t_start: float, t_end: float, dt: float, max_points: int) -> None:
        n = _n_points(t_start, t_end, dt)
        if n > max_points:
            raise ValueError(
                f"{key} requires {n} time points: t_start={t_start:g}, "
                f"t_end={t_end:g}, dt={dt:g}. This exceeds max_points={max_points}. "
                "Use piecewise propagation, a shorter saved window, or a different producer config."
            )

    def build_probe_only_spec(self) -> SegmentSpec:
        c = self.config
        probe_field = self.make_probe_field()
        pad = float(c.probe_readout_padding_sigma_factor) * float(c.probe_sigma_fs)
        t_start = _align_floor(float(c.probe_center_fs) - pad, c.dt_fs)
        t_end = _align_ceil(float(c.probe_center_fs) + pad, c.dt_fs)
        self._check_points(key="probe_only", t_start=t_start, t_end=t_end, dt=c.dt_fs, max_points=c.max_segment_points)

        params = self.physical_params(
            field=probe_field,
            t_start_fs=t_start,
            t_end_fs=t_end,
            dt_fs=c.dt_fs,
            case_name="probe_only",
            description="Probe-only reference window.",
            segment_metadata={
                "role": "probe_only",
                "segment_role": "probe_only",
                "probe_center_fs": float(c.probe_center_fs),
            },
        )
        return SegmentSpec(
            segment_key="probe_only",
            role="probe_only",
            field=probe_field,
            t_start_fs=t_start,
            t_end_fs=t_end,
            dt_fs=float(c.dt_fs),
            params=params,
            checkpoint_path=self._ckp_path("probe_only"),
        )

    def build_delay_case_spec(self, delay_fs: float) -> DelayCaseSpec:
        c = self.config
        delay = float(delay_fs)
        ta_field = self.make_ta_field(delay)
        pump_field = ta_field["pump"]
        probe_field = ta_field["probe"]
        pump_center = float(c.probe_center_fs) - delay
        probe_center = float(c.probe_center_fs)

        use_piecewise = delay > float(c.piecewise_min_positive_delay_fs)
        case_key = f"delay_{_safe_tag(delay)}_fs_pump_probe"

        pulse_pad = float(c.pulse_padding_sigma_factor) * max(float(c.pump_sigma_fs), float(c.probe_sigma_fs))
        probe_pad = float(c.probe_readout_padding_sigma_factor) * float(c.probe_sigma_fs)

        if not use_piecewise:
            t_start = _align_floor(min(pump_center, probe_center) - pulse_pad, c.dt_fs)
            t_end = _align_ceil(max(pump_center, probe_center) + pulse_pad, c.dt_fs)
            self._check_points(key=case_key, t_start=t_start, t_end=t_end, dt=c.dt_fs, max_points=c.max_full_case_points)

            params = self.physical_params(
                field=ta_field,
                t_start_fs=t_start,
                t_end_fs=t_end,
                dt_fs=c.dt_fs,
                case_name=case_key,
                description=f"Full lab_exact pump+probe overlap/short-delay run for delay={delay:g} fs.",
                segment_metadata={
                    "role": "pump_probe",
                    "mode": "full_overlap",
                    "delay_fs": delay,
                    "pump_center_fs": pump_center,
                    "probe_center_fs": probe_center,
                },
            )
            segment = SegmentSpec(
                segment_key=case_key,
                role="full",
                field=ta_field,
                t_start_fs=t_start,
                t_end_fs=t_end,
                dt_fs=float(c.dt_fs),
                params=params,
                checkpoint_path=self._ckp_path(case_key),
            )
            return DelayCaseSpec(
                case_key=case_key,
                delay_fs=delay,
                mode="full_overlap",
                pump_center_fs=pump_center,
                probe_center_fs=probe_center,
                segments=(segment,),
            )

        # Piecewise non-overlap positive delay.
        pump_start = _align_floor(pump_center - pulse_pad, c.dt_fs)
        pump_end = _align_ceil(pump_center + pulse_pad, c.dt_fs)
        probe_start = _align_floor(probe_center - probe_pad, c.dt_fs)
        probe_end = _align_ceil(probe_center + probe_pad, c.dt_fs)

        if pump_end >= probe_start:
            raise ValueError(
                f"delay={delay:g} fs was classified as piecewise, but pump_end={pump_end:g} "
                f"is not earlier than probe_start={probe_start:g}."
            )

        self._check_points(
            key=f"{case_key}_pump",
            t_start=pump_start,
            t_end=pump_end,
            dt=c.dt_fs,
            max_points=c.max_segment_points,
        )
        self._check_points(
            key=f"{case_key}_probe",
            t_start=probe_start,
            t_end=probe_end,
            dt=c.dt_fs,
            max_points=c.max_segment_points,
        )

        pump_params = self.physical_params(
            field=pump_field,
            t_start_fs=pump_start,
            t_end_fs=pump_end,
            dt_fs=c.dt_fs,
            case_name=case_key,
            description=f"Piecewise TA pump segment for delay={delay:g} fs.",
            segment_metadata={
                "role": "pump_probe",
                "mode": "piecewise",
                "segment_role": "pump",
                "delay_fs": delay,
                "pump_center_fs": pump_center,
                "probe_center_fs": probe_center,
            },
        )

        dark_start = pump_end
        dark_end = probe_start
        dark_duration = dark_end - dark_start

        # Do not let mesolve integrate a multi-ps dark interval in a single output step.
        # QuTiP's scipy integrator has an internal nsteps limit per interval between
        # adjacent tlist points; an endpoint-only 5-10 ps dark segment can trigger
        # "Excess work done on this call". We therefore cap the dark output interval.
        # This is still cheap because the dark window has no optical carrier.
        if c.dark_tlist_mode == "endpoints":
            dark_dt = min(float(c.max_dark_interval_fs), dark_duration)
        else:
            dark_dt = min(float(c.dark_coarse_dt_fs), float(c.max_dark_interval_fs), dark_duration)

        if dark_dt <= 0:
            raise ValueError(f"Invalid dark_dt={dark_dt:g} fs for delay={delay:g} fs.")

        dark_params = self.physical_params(
            field=self.make_dark_field(),
            t_start_fs=dark_start,
            t_end_fs=dark_end,
            dt_fs=dark_dt,
            case_name=case_key,
            description=f"Piecewise TA dark segment for delay={delay:g} fs.",
            segment_metadata={
                "role": "pump_probe",
                "mode": "piecewise",
                "segment_role": "dark",
                "delay_fs": delay,
                "pump_center_fs": pump_center,
                "probe_center_fs": probe_center,
                "dark_tlist_mode": c.dark_tlist_mode,
            },
        )

        probe_params = self.physical_params(
            field=probe_field,
            t_start_fs=probe_start,
            t_end_fs=probe_end,
            dt_fs=c.dt_fs,
            case_name=case_key,
            description=f"Piecewise TA probe/readout segment for delay={delay:g} fs.",
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
            SegmentSpec(
                segment_key=f"{case_key}_pump",
                role="pump",
                field=pump_field,
                t_start_fs=pump_start,
                t_end_fs=pump_end,
                dt_fs=float(c.dt_fs),
                params=pump_params,
                checkpoint_path=self._ckp_path(f"{case_key}_pump"),
            ),
            SegmentSpec(
                segment_key=f"{case_key}_dark",
                role="dark",
                field=dark_params.field,
                t_start_fs=dark_start,
                t_end_fs=dark_end,
                dt_fs=float(dark_dt),
                params=dark_params,
                checkpoint_path=self._ckp_path(f"{case_key}_dark"),
            ),
            SegmentSpec(
                segment_key=case_key,
                role="probe",
                field=probe_field,
                t_start_fs=probe_start,
                t_end_fs=probe_end,
                dt_fs=float(c.dt_fs),
                params=probe_params,
                checkpoint_path=self._ckp_path(case_key),
            ),
        )
        return DelayCaseSpec(
            case_key=case_key,
            delay_fs=delay,
            mode="piecewise",
            pump_center_fs=pump_center,
            probe_center_fs=probe_center,
            segments=segments,
        )

    def _dark_c_ops_physical(self) -> list[Qobj]:
        """Collapse operators in physical fs^-1 units for exact dark propagation."""
        c = self.config
        n_levels = len(c.energies_eV)
        c_ops: list[Qobj] = []

        relaxation_channels = (
            {"from_level": 2, "to_level": 1, "rate_fs_inv": 1.0 / float(c.T1_2_to_1_fs)},
            {"from_level": 1, "to_level": 0, "rate_fs_inv": 1.0 / float(c.T1_1_to_0_fs)},
        )
        for channel in relaxation_channels:
            rate = float(channel["rate_fs_inv"])
            if rate <= 0:
                continue
            from_level = int(channel["from_level"])
            to_level = int(channel["to_level"])
            c_ops.append(np.sqrt(rate) * (basis(n_levels, to_level) * basis(n_levels, from_level).dag()))

        pure_dephasing_channels = (
            {"level": 1, "rate_fs_inv": 1.0 / float(c.Tphi_1_fs)},
            {"level": 2, "rate_fs_inv": 1.0 / float(c.Tphi_2_fs)},
        )
        for channel in pure_dephasing_channels:
            rate = float(channel["rate_fs_inv"])
            if rate <= 0:
                continue
            level = int(channel["level"])
            ket = basis(n_levels, level)
            c_ops.append(np.sqrt(rate) * (ket * ket.dag()))

        return c_ops

    def run_dark_segment_exact(self, segment: SegmentSpec, *, rho0: Qobj | None):
        """Run the dark segment by exact Liouvillian propagation.

        The dark segment has no optical field, but lab-frame coherences still rotate
        at optical transition frequencies under H0. Using mesolve with a sparse
        dark tlist can therefore fail with "Excess work done". A time-independent
        Liouvillian exponential avoids that integrator problem and is the intended
        dark-window implementation.
        """
        if rho0 is None:
            raise ValueError("Dark segment requires rho0 from the previous pump segment.")

        ckp = segment.checkpoint_path
        if ckp is not None and ckp.exists() and not self.config.force_run:
            print(f"Loading checkpoint: {ckp}")
            return DynamicsResult.from_ckp(ckp)
        if ckp is not None and self.config.force_run:
            print(f"force_run=True, running exact dark propagation and ignoring checkpoint: {ckp}")
        elif ckp is not None:
            print(f"Checkpoint not found, running exact dark propagation: {ckp}")

        duration_fs = float(segment.t_end_fs) - float(segment.t_start_fs)
        if duration_fs < 0:
            raise ValueError(f"Dark segment has negative duration: {duration_fs:g} fs.")

        energies_fs_inv = np.asarray(
            ParaNormalizer.energy_eV_to_fs_inv(np.asarray(self.config.energies_eV, dtype=float)),
            dtype=float,
        )
        h0 = Qobj(np.diag(energies_fs_inv.astype(np.complex128)))
        L = liouvillian(h0, self._dark_c_ops_physical())
        rho_vec_final = (L * duration_fs).expm() * operator_to_vector(rho0)
        rho_final = vector_to_operator(rho_vec_final)

        times_fs = np.asarray([segment.t_start_fs, segment.t_end_fs], dtype=float)
        result = DynamicsResult(
            mode="dark_exact",
            times=times_fs.copy(),
            times_fs=times_fs,
            states=[rho0, rho_final],
            parameters=None,
            physical_params=segment.params,
            solver_params=None,
            metadata={
                "segment_role": "dark",
                "propagation": "exact_liouvillian_expm",
                "duration_fs": duration_fs,
                "note": "No optical field is present; H0 and Lindblad terms are propagated exactly as a time-independent Liouvillian.",
            },
        )
        result.sanity_checks = {
            "trace_error_small": {
                "value": result.max_trace_error(),
                "threshold": 1e-8,
                "passed": bool(result.max_trace_error() < 1e-8),
            },
            "hermiticity_error_small": {
                "value": result.max_hermiticity_error(),
                "threshold": 1e-8,
                "passed": bool(result.max_hermiticity_error() < 1e-8),
            },
        }

        if ckp is not None:
            ckp.parent.mkdir(parents=True, exist_ok=True)
            print(f"Saving checkpoint: {ckp}")
            result.save_ckp(ckp)

        return result

    def run_segment(self, segment: SegmentSpec, *, rho0=None):
        if segment.role == "dark":
            return self.run_dark_segment_exact(segment, rho0=rho0)
        return run_case(
            segment.params,
            normalizer=self.normalizer,
            rho0=rho0,
            load_ckp=segment.checkpoint_path,
            save_ckp=segment.checkpoint_path,
            force_run=bool(self.config.force_run),
        )

    def run_delay_case(self, spec: DelayCaseSpec):
        rho = None
        segment_results = {}
        for segment in spec.segments:
            result = self.run_segment(segment, rho0=rho)
            segment_results[segment.segment_key] = result
            rho = result.states[-1]
        readout_result = segment_results[spec.segments[-1].segment_key]
        return readout_result, segment_results

    def polarization_from_result(self, result) -> np.ndarray:
        physical = result.physical_params
        if physical is None:
            raise ValueError("DynamicsResult.physical_params is required.")
        return polarization_C_per_m2(
            result.density_array(),
            physical.dipole_matrix_D,
            float(self.config.number_density_m3),
        )

    def response_from_result(self, result, probe_field) -> dict[str, np.ndarray]:
        c = self.config
        t_fs = np.asarray(result.times_fs, dtype=float)
        E_probe = np.asarray(probe_field(t_fs), dtype=float)
        P_t = self.polarization_from_result(result)
        return lab_frame_fft_response(
            t_fs=t_fs,
            E_MV_per_cm=E_probe,
            P_C_per_m2=P_t,
            rhoij=result.matrix_element(0, 1),
            window=c.window,
            subtract_mean=bool(c.subtract_mean),
            rel_threshold=float(c.rel_threshold),
            zero_padding_factor=int(c.zero_padding_factor),
        )

    @staticmethod
    def omega_im_p_over_e(response: dict[str, np.ndarray]) -> np.ndarray:
        return np.asarray(response["omega_fs_inv"], dtype=float) * np.imag(response["P_over_E"])

    def save_case_result(self, *, key: str, result) -> Path:
        cases_root = self.simulation_dir
        written = save_result_case(
            result,
            cases_root / "res_per_delay",
            output_data=True,
            output_preview=bool(self.config.save_case_previews),
            case_name=key,
            example_name=self.config.example_name,
            condition_name="ta_delay_scan",
            append_results_csv=True,
        )
        return Path(written.get("case_dir", cases_root / "res_per_delay" / key))

    def save_difference_spectrum(
        self,
        *,
        spec: DelayCaseSpec,
        readout_response: dict[str, np.ndarray],
        probe_reference_response: dict[str, np.ndarray],
    ) -> Path:
        out_dir = self.final_output_dir / "difference_spectra"
        out_dir.mkdir(parents=True, exist_ok=True)

        energy = np.asarray(readout_response["energy_eV"], dtype=float)
        omega = np.asarray(readout_response["omega_fs_inv"], dtype=float)
        s_readout = self.omega_im_p_over_e(readout_response)

        probe_energy = np.asarray(probe_reference_response["energy_eV"], dtype=float)
        s_probe_ref = self.omega_im_p_over_e(probe_reference_response)
        p_readout = np.asarray(readout_response["P_over_E"], dtype=np.complex128)
        p_probe = np.asarray(probe_reference_response["P_over_E"], dtype=np.complex128)

        # UFANSYS can do more rigorous axis handling. Here we save a per-delay
        # practical difference spectrum on the readout energy axis.
        if energy.shape == probe_energy.shape and np.allclose(energy, probe_energy):
            s_probe_on_axis = s_probe_ref
            p_probe_on_axis = p_probe
        else:
            overlap_min = max(float(np.min(energy)), float(np.min(probe_energy)))
            overlap_max = min(float(np.max(energy)), float(np.max(probe_energy)))
            mask = (energy >= overlap_min) & (energy <= overlap_max)
            if not np.any(mask):
                raise ValueError(f"No shared energy range for {spec.case_key}.")
            energy = energy[mask]
            omega = omega[mask]
            s_readout = s_readout[mask]
            p_readout = p_readout[mask]
            s_probe_on_axis = np.interp(energy, probe_energy, s_probe_ref)
            p_probe_on_axis = (
                np.interp(energy, probe_energy, np.real(p_probe))
                + 1j * np.interp(energy, probe_energy, np.imag(p_probe))
            )

        s_diff = s_readout - s_probe_on_axis
        rows = []
        for idx in range(energy.size):
            rows.append(
                {
                    "case_key": spec.case_key,
                    "delay_fs": float(spec.delay_fs),
                    "mode": spec.mode,
                    "energy_eV": float(energy[idx]),
                    "omega_fs_inv": float(omega[idx]),
                    "S_readout": float(s_readout[idx]),
                    "S_probe_reference": float(s_probe_on_axis[idx]),
                    "S_difference": float(s_diff[idx]),
                    "Re_P_over_E_readout": float(np.real(p_readout[idx])),
                    "Im_P_over_E_readout": float(np.imag(p_readout[idx])),
                    "Re_P_over_E_probe_reference": float(np.real(p_probe_on_axis[idx])),
                    "Im_P_over_E_probe_reference": float(np.imag(p_probe_on_axis[idx])),
                }
            )

        path = out_dir / f"{spec.case_key}_difference_spectrum.csv"
        _write_rows(path, rows)
        return path

    def case_spec_rows(self, probe_spec: SegmentSpec, delay_specs: list[DelayCaseSpec]) -> list[dict[str, Any]]:
        rows = [
            {
                "case_key": probe_spec.segment_key,
                "delay_fs": "",
                "mode": "probe_only",
                "segment_key": probe_spec.segment_key,
                "segment_role": probe_spec.role,
                "t_start_fs": probe_spec.t_start_fs,
                "t_end_fs": probe_spec.t_end_fs,
                "dt_fs": probe_spec.dt_fs,
                "n_points": _n_points(probe_spec.t_start_fs, probe_spec.t_end_fs, probe_spec.dt_fs),
                "checkpoint_path": "" if probe_spec.checkpoint_path is None else str(probe_spec.checkpoint_path),
            }
        ]
        for spec in delay_specs:
            for segment in spec.segments:
                rows.append(
                    {
                        "case_key": spec.case_key,
                        "delay_fs": spec.delay_fs,
                        "mode": spec.mode,
                        "segment_key": segment.segment_key,
                        "segment_role": segment.role,
                        "t_start_fs": segment.t_start_fs,
                        "t_end_fs": segment.t_end_fs,
                        "dt_fs": segment.dt_fs,
                        "n_points": _n_points(segment.t_start_fs, segment.t_end_fs, segment.dt_fs),
                        "checkpoint_path": "" if segment.checkpoint_path is None else str(segment.checkpoint_path),
                    }
                )
        return rows

    def run(self) -> dict[str, Any]:
        self.simulation_dir.mkdir(parents=True, exist_ok=True)
        self.final_output_dir.mkdir(parents=True, exist_ok=True)

        probe_spec = self.build_probe_only_spec()
        delay_specs = [self.build_delay_case_spec(delay) for delay in self.config.probe_delays_fs]

        _write_rows(self.simulation_dir / "case_specs.csv", self.case_spec_rows(probe_spec, delay_specs))

        probe_ref_result = self.run_segment(probe_spec)
        probe_ref_case_dir = self.save_case_result(key="probe_only", result=probe_ref_result)
        probe_ref_response = self.response_from_result(probe_ref_result, probe_spec.field)

        delay_results: list[DelaySpectrumResult] = []
        segment_summary: dict[str, Any] = {
            "probe_only": {
                "case_dir": probe_ref_case_dir,
                "checkpoint_path": probe_spec.checkpoint_path,
            },
            "delays": {},
        }

        all_spectrum_rows: list[dict[str, Any]] = []
        for spec in delay_specs:
            print(f"Running {spec.case_key} ({spec.mode})...")
            readout_result, segment_results = self.run_delay_case(spec)
            case_dir = self.save_case_result(key=spec.case_key, result=readout_result)

            # Use the readout segment's own probe field.
            readout_probe_field = spec.segments[-1].field
            readout_response = self.response_from_result(readout_result, readout_probe_field)
            spectrum_path = self.save_difference_spectrum(
                spec=spec,
                readout_response=readout_response,
                probe_reference_response=probe_ref_response,
            )

            # Build index rows from saved spectrum for convenience.
            with spectrum_path.open("r", newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    all_spectrum_rows.append(dict(row))

            delay_results.append(
                DelaySpectrumResult(
                    case_key=spec.case_key,
                    delay_fs=spec.delay_fs,
                    mode=spec.mode,
                    readout_result=readout_result,
                    readout_response=readout_response,
                    probe_reference_response=probe_ref_response,
                    spectrum_csv=spectrum_path,
                )
            )

            segment_summary["delays"][spec.case_key] = {
                "delay_fs": spec.delay_fs,
                "mode": spec.mode,
                "readout_case_dir": case_dir,
                "difference_spectrum_csv": spectrum_path,
                "segments": [
                    {
                        "segment_key": seg.segment_key,
                        "role": seg.role,
                        "t_start_fs": seg.t_start_fs,
                        "t_end_fs": seg.t_end_fs,
                        "dt_fs": seg.dt_fs,
                        "checkpoint_path": seg.checkpoint_path,
                        "max_trace_error": float(segment_results[seg.segment_key].max_trace_error()),
                        "max_hermiticity_error": float(segment_results[seg.segment_key].max_hermiticity_error()),
                    }
                    for seg in spec.segments
                ],
            }

        if all_spectrum_rows:
            _write_rows(self.final_output_dir / "all_difference_spectra.csv", all_spectrum_rows)

        _write_json(self.simulation_dir / "segment_summary.json", segment_summary)

        meta = {
            "example_name": self.config.example_name,
            "description": "Piecewise lab-frame TA simulation producer. Analysis and final map assembly are intended for UFANSYS.",
            "output_policy": {
                "simulation_outputs": "Raw QuDPy dynamics, checkpoints, segment summaries, and case data.",
                "real_final_outputs": "Per-delay difference spectra that approximate final experimental-style output.",
            },
            "directories": {
                "simulation": self.simulation_dir,
                "simulation_res_per_delay": self.simulation_dir / "res_per_delay",
                "simulation_checkpoints": self.checkpoints_dir,
                "real_final_output": self.final_output_dir,
                "difference_spectra": self.final_output_dir / "difference_spectra",
            },
            "config": self.config,
            "notes": [
                "Full lab_exact is used for overlap/short delays.",
                "Piecewise pump-dark-probe propagation is used for long positive delays.",
                "Dark segment uses zero field and receives rho0 from the pump segment.",
                "Probe/readout segment receives rho0 from the dark segment.",
                "UFANSYS should handle cross-delay resampling, map construction, and publication plotting.",
            ],
        }
        _write_json(self.output_dir / "meta.json", meta)

        return {
            "probe_reference_case_dir": probe_ref_case_dir,
            "delay_results": delay_results,
            "meta": self.output_dir / "meta.json",
            "case_specs": self.simulation_dir / "case_specs.csv",
            "segment_summary": self.simulation_dir / "segment_summary.json",
            "all_difference_spectra": self.final_output_dir / "all_difference_spectra.csv",
        }


def main() -> None:
    output_dir = DEFAULT_OUTPUT_DIR
    exp = TaPiecewiseExp(output_dir=output_dir)
    outputs = exp.run()

    print("Piecewise TA delay-scan producer finished.")
    print(f"output directory        : {output_dir}")
    print(f"case specs             : {outputs['case_specs']}")
    print(f"segment summary        : {outputs['segment_summary']}")
    print(f"all difference spectra : {outputs['all_difference_spectra']}")


if __name__ == "__main__":
    main()
