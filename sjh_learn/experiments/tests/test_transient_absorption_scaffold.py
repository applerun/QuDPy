"""transient_absorption scaffold 的轻量级 unittest。

这些测试只检查 settings、active-window planning、piece assembly 和 result
wrapper，不运行 QuTiP solver，也不生成输出文件。
"""

from __future__ import annotations

from collections.abc import Iterator
import unittest

import numpy as np

from sjh_learn.experiments.transient_absorption import ta_case_plan as ta_case_plan_module
from sjh_learn.experiments.transient_absorption.ta_case_plan import (
    TaDelayCasePlan,
    contains_window,
    compute_pulse_centers,
    make_delay_case_plan,
    make_delay_scan_plan,
)
from sjh_learn.experiments.transient_absorption.ta_settings import (
    DelayScanSettings,
    TaDelayScanSettings,
)
from sjh_learn.utils.core import (
    DynamicsResult,
    NLevelPhysicalParams,
    ParaNormalizer,
    run_case,
)
from sjh_learn.utils.core.piecewise_propagation import (
    ActiveWindowDynamicsResult,
    DarkWindowDynamicsResult,
    PieceDynamicsResultSeries,
    PropagationPiece,
)
from sjh_learn.utils.fields import FieldPhyRoot, make_default_gaussian_carrier_field
from sjh_learn.utils.fields.field_windows import ActiveWindow, FieldActiveWindowSettings, detect_active_windows
from sjh_learn.utils.spectroscopy import lab_frame_absorption_response, polarization_C_per_m2


class DummyPulse(FieldPhyRoot):
    """用于 scaffold test 的矩形物理脉冲。"""

    def __init__(self, *, half_width_fs: float = 10.0, amplitude_MV_per_cm: float = 1.0) -> None:
        self.half_width_fs = float(half_width_fs)
        self.amplitude_MV_per_cm = float(amplitude_MV_per_cm)

    @property
    def reference_MV_per_cm(self) -> float:
        return abs(self.amplitude_MV_per_cm)

    def physical_E_MV_per_cm(self, t_fs: np.ndarray) -> np.ndarray:
        active = np.abs(t_fs) <= self.half_width_fs
        return np.where(active, self.amplitude_MV_per_cm, 0.0)

    def __repr__(self) -> str:
        return (
            "DummyPulse("
            f"half_width_fs={self.half_width_fs!r}, "
            f"amplitude_MV_per_cm={self.amplitude_MV_per_cm!r})"
        )


def absorption_from_dynamics_result(
    result: DynamicsResult,
    readout_window: ActiveWindow,
    *,
    number_density_m3: float = 1.0e24,
) -> dict[str, np.ndarray]:
    """从 stitched DynamicsResult 计算 readout window 内的 lab-frame absorption。

    absorption 属于 spectroscopy / experiment 后处理层；这里的 helper 只在
    smoke test 中把 result 拆成后处理函数需要的数组，不把该逻辑放回 core。
    """

    if result.physical_params is None:
        raise ValueError("physical_params is required for absorption smoke test.")
    if result.times_fs is None:
        raise ValueError("times_fs is required for lab-frame absorption smoke test.")
    t_fs = np.asarray(result.times_fs, dtype=float)
    mask = (t_fs >= float(readout_window.start_fs)) & (t_fs <= float(readout_window.end_fs))
    if int(np.count_nonzero(mask)) < 8:
        raise ValueError("readout_window must contain enough time points for FFT smoke test.")
    polarization = polarization_C_per_m2(
        result.density_array(),
        result.physical_params.dipole_matrix_D,
        number_density_m3=number_density_m3,
    )
    field = result.field_MV_per_cm_values()
    if field is None:
        raise ValueError("lab-frame field values are required for absorption smoke test.")
    return lab_frame_absorption_response(
        t_fs=t_fs[mask],
        E_MV_per_cm=np.asarray(field, dtype=float)[mask],
        P_C_per_m2=polarization[mask],
        window="hann",
        subtract_mean=True,
        rel_threshold=1e-8,
        zero_padding_factor=4,
    )


class TestTransientAbsorptionScaffold(unittest.TestCase):
    def window_settings(self) -> FieldActiveWindowSettings:
        return FieldActiveWindowSettings(
            rel_threshold=0.5,
            padding_fs=0.0,
            dt_fs=1.0,
            t_start_fs=-80.0,
            t_end_fs=80.0,
            merge_gap_fs=0.0,
            force_single_window=False,
        )

    def test_settings_construct(self) -> None:
        settings = TaDelayScanSettings()

        self.assertGreater(len(settings.probe_delays_fs), 0)
        self.assertIsInstance(settings.delay_scan, DelayScanSettings)
        self.assertIsInstance(settings.field_window, FieldActiveWindowSettings)

    def test_numpy_delay_input_is_normalized(self) -> None:
        settings = TaDelayScanSettings(
            delay_scan=DelayScanSettings(
                probe_delays_fs=np.array([-100.0, 0.0, 100.0]),
                probe_center_fs=0.0,
            )
        )

        self.assertIsInstance(settings.probe_delays_fs, tuple)
        self.assertEqual(settings.probe_delays_fs, (-100.0, 0.0, 100.0))

    def test_pump_center_delay_convention(self) -> None:
        settings = TaDelayScanSettings(
            delay_scan=DelayScanSettings(
                probe_delays_fs=[200.0],
                probe_center_fs=0.0,
            )
        )

        self.assertEqual(settings.pump_center_fs_for_delay(200.0), -200.0)
        self.assertEqual(compute_pulse_centers(delay_fs=200.0, probe_center_fs=0.0), (-200.0, 0.0))

    def test_detect_active_windows_preserves_field_names(self) -> None:
        pump = DummyPulse(half_width_fs=3.0).time_shifted(-20.0, name="pump")
        probe = DummyPulse(half_width_fs=3.0).time_shifted(20.0, name="probe")

        windows, _threshold = detect_active_windows(
            (pump, probe),
            settings=self.window_settings(),
            field_names=("pump", "probe"),
        )

        self.assertEqual(tuple(window.name for window in windows), ("pump", "probe"))
        padded = windows[0].padded(1.0)
        self.assertEqual(padded.name, "pump")
        self.assertEqual(padded.to_dict()["name"], "pump")

    def test_overlap_case_has_single_active_pump_and_probe_piece(self) -> None:
        plan = make_delay_case_plan(
            delay_fs=0.0,
            pump_template=DummyPulse(half_width_fs=8.0),
            probe_template=DummyPulse(half_width_fs=8.0),
            field_window_settings=self.window_settings(),
        )

        self.assertIsInstance(plan, TaDelayCasePlan)
        self.assertEqual(plan.case_label, "full_overlap")
        self.assertEqual(plan.signal_policy, "normal")
        self.assertEqual(len(plan.pieces), 1)
        self.assertEqual(plan.pieces[0].piece_name, "active_pump_and_probe")
        self.assertEqual(plan.pieces[0].kind, "active")
        self.assertIsNotNone(plan.pieces[0].field)
        self.assertTrue(contains_window(plan.pieces[0].window, plan.readout_window))

    def test_pump_before_probe_case_has_active_dark_active_pieces(self) -> None:
        plan = make_delay_case_plan(
            delay_fs=40.0,
            pump_template=DummyPulse(half_width_fs=5.0),
            probe_template=DummyPulse(half_width_fs=5.0),
            field_window_settings=self.window_settings(),
            probe_center_fs=0.0,
        )

        self.assertEqual(plan.case_label, "pump_dark_probe")
        self.assertEqual(plan.signal_policy, "normal")
        self.assertEqual(tuple(piece.kind for piece in plan.pieces), ("active", "dark", "active"))
        self.assertEqual(plan.pieces[0].window.name, "pump")
        self.assertEqual(plan.pieces[1].kind, "dark")
        self.assertIsNone(plan.pieces[1].field)
        self.assertEqual(plan.pieces[2].window.name, "probe")
        self.assertIsNotNone(plan.pieces[0].field)
        self.assertIsNotNone(plan.pieces[2].field)
        self.assertTrue(contains_window(plan.pieces[2].window, plan.readout_window))

    def test_probe_before_pump_case_only_keeps_probe_piece(self) -> None:
        plan = make_delay_case_plan(
            delay_fs=-40.0,
            pump_template=DummyPulse(half_width_fs=5.0),
            probe_template=DummyPulse(half_width_fs=5.0),
            field_window_settings=self.window_settings(),
            probe_center_fs=0.0,
        )

        self.assertEqual(plan.case_label, "pure_probe")
        self.assertEqual(plan.signal_policy, "zero_difference")
        self.assertEqual(len(plan.pieces), 1)
        self.assertEqual(plan.pieces[0].window.name, "probe")
        self.assertEqual(plan.pieces[0].kind, "active")
        self.assertIsNotNone(plan.pieces[0].field)
        self.assertTrue(contains_window(plan.pieces[0].window, plan.readout_window))
        self.assertFalse(hasattr(plan.pieces[0], "active_fields"))
        self.assertFalse(hasattr(plan.pieces[0], "is_readout"))
        self.assertFalse(hasattr(plan.pieces[0], "execute"))

    def test_delay_case_plan_json_records_pieces_without_full_field_payload(self) -> None:
        plan = make_delay_case_plan(
            delay_fs=40.0,
            pump_template=DummyPulse(half_width_fs=5.0),
            probe_template=DummyPulse(half_width_fs=5.0),
            field_window_settings=self.window_settings(),
        )
        payload = plan.to_dict()

        self.assertIn("pieces", payload)
        self.assertIn("readout_window", payload)
        self.assertIn("field_name", payload["pieces"][0])
        self.assertNotIn("field", payload["pieces"][0])

    def test_propagation_piece_validation(self) -> None:
        with self.assertRaises(ValueError):
            PropagationPiece(
                piece_name="bad_active",
                kind="active",
                window=ActiveWindow(0.0, 1.0, name="pump"),
                field=None,
            )

        with self.assertRaises(ValueError):
            PropagationPiece(
                piece_name="bad_dark",
                kind="dark",
                window=ActiveWindow(0.0, 1.0, name="dark"),
                field=DummyPulse(),
            )

    def test_piece_dynamics_result_wrappers_validate_kind(self) -> None:
        active_piece = PropagationPiece(
            piece_name="active_pump",
            kind="active",
            window=ActiveWindow(0.0, 1.0, name="pump"),
            field=DummyPulse(),
        )
        dark_piece = PropagationPiece(
            piece_name="dark_gap",
            kind="dark",
            window=ActiveWindow(1.0, 2.0, name="pump_to_probe"),
            field=None,
        )

        self.assertIsInstance(
            ActiveWindowDynamicsResult(piece=active_piece, result={"final_state": "rho1"}),
            ActiveWindowDynamicsResult,
        )
        self.assertIsInstance(
            DarkWindowDynamicsResult(piece=dark_piece, result={"final_state": "rho2"}),
            DarkWindowDynamicsResult,
        )
        with self.assertRaises(ValueError):
            ActiveWindowDynamicsResult(piece=dark_piece, result={})
        with self.assertRaises(ValueError):
            DarkWindowDynamicsResult(piece=active_piece, result={})

    def test_piece_dynamics_result_series_checks_contiguity_and_final_state(self) -> None:
        active_piece = PropagationPiece(
            piece_name="active_pump",
            kind="active",
            window=ActiveWindow(0.0, 1.0, name="pump"),
            field=DummyPulse(),
        )
        dark_piece = PropagationPiece(
            piece_name="dark_gap",
            kind="dark",
            window=ActiveWindow(1.0, 2.0, name="pump_to_probe"),
            field=None,
        )
        active_result = ActiveWindowDynamicsResult(piece=active_piece, result={"final_state": "rho1"})
        dark_result = DarkWindowDynamicsResult(piece=dark_piece, result={"final_state": "rho2"})

        series = PieceDynamicsResultSeries(piece_results=(active_result, dark_result))
        self.assertEqual(series.final_state, "rho2")

        broken_piece = PropagationPiece(
            piece_name="active_probe",
            kind="active",
            window=ActiveWindow(3.0, 4.0, name="probe"),
            field=DummyPulse(),
        )
        with self.assertRaises(ValueError):
            PieceDynamicsResultSeries(
                piece_results=(
                    active_result,
                    ActiveWindowDynamicsResult(piece=broken_piece, result={}),
                )
            )

    def test_piecewise_stitch_supports_absorption_smoke_with_fewer_points(self) -> None:
        field = make_default_gaussian_carrier_field(
            E0_MV_per_cm=0.01,
            laser_energy_eV=1.0,
            pulse_center_fs=0.0,
            pulse_sigma_fs=5.0,
        )
        physical = NLevelPhysicalParams(
            energies_eV=(0.0, 1.0),
            dipole_matrix_D=((0.0, 1.0), (1.0, 0.0)),
            t_start_fs=-80.0,
            t_end_fs=80.0,
            dt_fs=0.5,
            field=field,
        )
        normalizer = ParaNormalizer(time_scale_fs=1.0, auto_scale=False)
        readout_window = ActiveWindow(start_fs=-18.0, end_fs=18.0, name="probe")

        full_series = run_case(physical, normalizer=normalizer, piecewise=False)
        full_result = full_series.stitch()
        absorption_full = absorption_from_dynamics_result(full_result, readout_window)

        piecewise_settings = FieldActiveWindowSettings(
            rel_threshold=1e-3,
            padding_fs=12.0,
            dt_fs=0.5,
            t_start_fs=physical.t_start_fs,
            t_end_fs=physical.t_end_fs,
            merge_gap_fs=3.0,
            force_single_window=True,
        )
        piecewise_series = run_case(
            physical,
            normalizer=normalizer,
            piecewise=True,
            piecewise_settings=piecewise_settings,
        )
        piecewise_result = piecewise_series.stitch()
        absorption_piecewise = absorption_from_dynamics_result(piecewise_result, readout_window)

        self.assertIsInstance(full_series, PieceDynamicsResultSeries)
        self.assertIsInstance(piecewise_series, PieceDynamicsResultSeries)
        self.assertIsInstance(full_result, DynamicsResult)
        self.assertIsInstance(piecewise_result, DynamicsResult)
        self.assertGreater(len(absorption_full["absorption"]), 0)
        self.assertGreater(len(absorption_piecewise["absorption"]), 0)

        full_solver_points = sum(
            len(piece_result.result.times)
            for piece_result in full_series.piece_results
            if piece_result.piece.kind == "active"
        )
        piecewise_solver_points = sum(
            len(piece_result.result.times)
            for piece_result in piecewise_series.piece_results
            if piece_result.piece.kind == "active"
        )
        self.assertLess(piecewise_solver_points, full_solver_points)

        self.assertGreaterEqual(len(absorption_full["energy_eV"]), 8)
        self.assertGreaterEqual(len(absorption_piecewise["energy_eV"]), 8)
        full_peak = float(absorption_full["energy_eV"][np.nanargmax(np.abs(absorption_full["absorption"]))])
        piecewise_peak = float(
            absorption_piecewise["energy_eV"][np.nanargmax(np.abs(absorption_piecewise["absorption"]))]
        )
        self.assertAlmostEqual(full_peak, piecewise_peak, delta=0.05)

    def test_delay_scan_plan_iter_delay_cases_is_generator(self) -> None:
        scan_plan = make_delay_scan_plan(
            delays_fs=[0.0, 40.0, -40.0],
            pump_template=DummyPulse(half_width_fs=5.0),
            probe_template=DummyPulse(half_width_fs=5.0),
            field_window_settings=self.window_settings(),
        )
        iterator = scan_plan.iter_delay_cases()

        self.assertIsInstance(iterator, Iterator)
        first = next(iterator)
        second = next(iterator)
        self.assertEqual(first.case_label, "full_overlap")
        self.assertEqual(second.case_label, "pump_dark_probe")
        self.assertNotIn("delay_cases", scan_plan.to_dict())

    def test_readout_detection_raises_if_not_single_window(self) -> None:
        original = ta_case_plan_module.detect_active_windows

        def fake_detect_active_windows(field_or_fields, *, settings, field_names=None):
            if field_names == ("probe",) and settings.force_single_window:
                return (
                    ActiveWindow(-10.0, -5.0, name="probe"),
                    ActiveWindow(5.0, 10.0, name="probe"),
                ), 0.1
            return original(field_or_fields, settings=settings, field_names=field_names)

        ta_case_plan_module.detect_active_windows = fake_detect_active_windows
        try:
            with self.assertRaises(ValueError):
                make_delay_case_plan(
                    delay_fs=40.0,
                    pump_template=DummyPulse(half_width_fs=5.0),
                    probe_template=DummyPulse(half_width_fs=5.0),
                    field_window_settings=self.window_settings(),
                )
        finally:
            ta_case_plan_module.detect_active_windows = original


if __name__ == "__main__":
    unittest.main()
