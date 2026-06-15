"""TA delay-scan 的顶层 runner。"""

from __future__ import annotations

from pathlib import Path
import csv
from typing import Any

from sjh_learn.utils.core import ParaNormalizer, run_case
from sjh_learn.utils.io import save_result_case, write_json
from sjh_learn.utils.spectroscopy import lab_frame_absorption_response

from .case_assembly import TaCaseAssembler
from .dark_propagation import run_dark_segment_exact
from .pulse_scheduling import count_time_points
from .ta_settings import TaDelayScanSettings
from .ta_specs import TaDelayCaseSpec, TaDelayScanOutputs, TaDelayResultRecord, TaSegmentSpec


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class TaDelayScanRunner:
    """TA delay-scan 顶层编排器。

    该类负责：构建 case、运行 solver、计算 absorption、保存 delay-wise
    difference spectrum。field 构造、dark 精确传播和吸收谱计算分别由对应模块
    或 utils 函数完成。
    """

    def __init__(
        self,
        settings: TaDelayScanSettings,
        *,
        normalizer: ParaNormalizer | None = None,
    ) -> None:
        self.settings = settings
        self.normalizer = ParaNormalizer(auto_scale=True) if normalizer is None else normalizer

        default_output = Path.cwd() / "outputs" / settings.experiment_name
        self.output_dir = default_output if settings.output_dir is None else Path(settings.output_dir)
        self.simulation_dir = self.output_dir / "simulation"
        self.res_per_delay_dir = self.simulation_dir / "res_per_delay"
        self.checkpoints_dir = self.simulation_dir / "checkpoints"
        self.final_output_dir = self.output_dir / "real" / "final_output"
        self.difference_spectra_dir = self.final_output_dir / "difference_spectra"

        self.assembler = TaCaseAssembler(settings, checkpoints_dir=self.checkpoints_dir)

    def output_paths(self) -> TaDelayScanOutputs:
        return TaDelayScanOutputs(
            output_dir=self.output_dir,
            simulation_dir=self.simulation_dir,
            res_per_delay_dir=self.res_per_delay_dir,
            checkpoints_dir=self.checkpoints_dir,
            final_output_dir=self.final_output_dir,
            case_specs_csv=self.simulation_dir / "case_specs.csv",
            segment_summary_json=self.simulation_dir / "segment_summary.json",
            all_difference_spectra_csv=self.final_output_dir / "all_difference_spectra.csv",
            workflow_metadata_json=self.output_dir / "workflow_metadata.json",
        )

    def _ensure_dirs(self) -> None:
        self.res_per_delay_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.difference_spectra_dir.mkdir(parents=True, exist_ok=True)

    def run_segment(self, segment: TaSegmentSpec, *, rho0=None):
        if segment.role == "dark":
            return run_dark_segment_exact(segment, settings=self.settings, rho0=rho0)

        return run_case(
            segment.params,
            normalizer=self.normalizer,
            rho0=rho0,
            load_ckp=segment.checkpoint_path,
            save_ckp=segment.checkpoint_path,
            force_run=bool(self.settings.force_run),
        )

    def run_delay_case(self, spec: TaDelayCaseSpec):
        rho = None
        segment_results = {}
        for segment in spec.segments:
            result = self.run_segment(segment, rho0=rho)
            segment_results[segment.segment_key] = result
            rho = result.states[-1]
        readout_result = segment_results[spec.segments[-1].segment_key]
        return readout_result, segment_results

    def polarization_from_result(self, result) -> np.ndarray:
        from sjh_learn.utils.spectroscopy import polarization_C_per_m2

        physical = result.physical_params
        if physical is None:
            raise ValueError("DynamicsResult.physical_params is required.")
        return polarization_C_per_m2(
            result.density_array(),
            physical.dipole_matrix_D,
            float(self.settings.number_density_m3),
        )

    def absorption_from_result(self, result, probe_field) -> dict[str, np.ndarray]:
        t_fs = np.asarray(result.times_fs, dtype=float)
        E_probe = np.asarray(probe_field(t_fs), dtype=float)
        P_t = self.polarization_from_result(result)
        return lab_frame_absorption_response(
            t_fs=t_fs,
            E_MV_per_cm=E_probe,
            P_C_per_m2=P_t,
            window=self.settings.window,
            subtract_mean=bool(self.settings.subtract_mean),
            rel_threshold=float(self.settings.rel_threshold),
            zero_padding_factor=int(self.settings.zero_padding_factor),
            return_intermediates=False,
        )

    def save_case_result(self, *, key: str, result) -> Path:
        written = save_result_case(
            result,
            self.res_per_delay_dir,
            output_data=True,
            output_preview=bool(self.settings.save_case_previews),
            case_name=key,
            example_name=self.settings.experiment_name,
            condition_name="ta_delay_scan",
            append_results_csv=True,
        )
        if isinstance(written, dict) and "case_dir" in written:
            return Path(written["case_dir"])
        return self.res_per_delay_dir / key

    def save_difference_spectrum(
        self,
        *,
        spec: TaDelayCaseSpec,
        readout_abs: dict[str, np.ndarray],
        probe_ref_abs: dict[str, np.ndarray],
    ) -> Path:
        energy = np.asarray(readout_abs["energy_eV"], dtype=float)
        omega = np.asarray(readout_abs["omega_fs_inv"], dtype=float)
        readout = np.asarray(readout_abs["absorption"], dtype=float)

        ref_energy = np.asarray(probe_ref_abs["energy_eV"], dtype=float)
        ref = np.asarray(probe_ref_abs["absorption"], dtype=float)

        if energy.shape == ref_energy.shape and np.allclose(energy, ref_energy):
            ref_on_axis = ref
        else:
            overlap_min = max(float(np.min(energy)), float(np.min(ref_energy)))
            overlap_max = min(float(np.max(energy)), float(np.max(ref_energy)))
            mask = (energy >= overlap_min) & (energy <= overlap_max)
            if not np.any(mask):
                raise ValueError(f"No shared energy range for {spec.case_key}.")
            energy = energy[mask]
            omega = omega[mask]
            readout = readout[mask]
            ref_on_axis = np.interp(energy, ref_energy, ref)

        delta_absorption = readout - ref_on_axis
        rows = [
            {
                "case_key": spec.case_key,
                "delay_fs": float(spec.delay_fs),
                "mode": spec.mode,
                "energy_eV": float(energy[i]),
                "omega_fs_inv": float(omega[i]),
                "absorption_readout": float(readout[i]),
                "absorption_probe_reference": float(ref_on_axis[i]),
                "delta_absorption": float(delta_absorption[i]),
            }
            for i in range(energy.size)
        ]

        path = self.difference_spectra_dir / f"{spec.case_key}_difference_spectrum.csv"
        _write_rows(path, rows)
        return path

    def case_spec_rows(self, probe_spec: TaSegmentSpec, delay_specs: list[TaDelayCaseSpec]) -> list[dict[str, Any]]:
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
                "n_points": count_time_points(probe_spec.t_start_fs, probe_spec.t_end_fs, probe_spec.dt_fs),
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
                        "n_points": count_time_points(segment.t_start_fs, segment.t_end_fs, segment.dt_fs),
                        "checkpoint_path": "" if segment.checkpoint_path is None else str(segment.checkpoint_path),
                    }
                )
        return rows

    def run(self) -> dict[str, Any]:
        self._ensure_dirs()
        paths = self.output_paths()

        probe_spec = self.assembler.build_probe_only_segment()
        delay_specs = [self.assembler.build_delay_case(delay) for delay in self.settings.probe_delays_fs]
        _write_rows(paths.case_specs_csv, self.case_spec_rows(probe_spec, delay_specs))

        probe_ref_result = self.run_segment(probe_spec)
        probe_ref_case_dir = self.save_case_result(key=probe_spec.segment_key, result=probe_ref_result)
        probe_ref_abs = self.absorption_from_result(probe_ref_result, probe_spec.field)

        all_spectrum_rows: list[dict[str, Any]] = []
        delay_records: list[TaDelayResultRecord] = []
        segment_summary: dict[str, Any] = {
            "probe_only": {
                "case_dir": probe_ref_case_dir,
                "checkpoint_path": probe_spec.checkpoint_path,
            },
            "delays": {},
        }

        for spec in delay_specs:
            readout_result, segment_results = self.run_delay_case(spec)
            case_dir = self.save_case_result(key=spec.case_key, result=readout_result)
            readout_abs = self.absorption_from_result(readout_result, spec.segments[-1].field)
            spectrum_path = self.save_difference_spectrum(
                spec=spec,
                readout_abs=readout_abs,
                probe_ref_abs=probe_ref_abs,
            )

            with spectrum_path.open("r", newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    all_spectrum_rows.append(dict(row))

            delay_records.append(
                TaDelayResultRecord(
                    case_key=spec.case_key,
                    delay_fs=spec.delay_fs,
                    mode=spec.mode,
                    case_dir=case_dir,
                    difference_spectrum_csv=spectrum_path,
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
            _write_rows(paths.all_difference_spectra_csv, all_spectrum_rows)

        write_json(paths.segment_summary_json, segment_summary)
        write_json(
            paths.workflow_metadata_json,
            {
                "experiment_name": self.settings.experiment_name,
                "metadata_role": "workflow_summary",
                "description": "TA delay-scan producer。最终 map 和 publication-level plotting 交给后处理层。",
                "directories": {
                    "output_dir": self.output_dir,
                    "simulation_dir": self.simulation_dir,
                    "res_per_delay_dir": self.res_per_delay_dir,
                    "checkpoints_dir": self.checkpoints_dir,
                    "final_output_dir": self.final_output_dir,
                    "difference_spectra_dir": self.difference_spectra_dir,
                },
                "settings": self.settings,
            },
        )

        return {
            "outputs": paths,
            "probe_reference_case_dir": probe_ref_case_dir,
            "delay_records": delay_records,
        }
