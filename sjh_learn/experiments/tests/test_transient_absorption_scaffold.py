"""transient_absorption scaffold 的轻量级 unittest。

这些测试只检查 settings、delay 约定和 scaffold import，不运行 QuTiP solver，
也不生成输出文件。
"""

from __future__ import annotations

import unittest

import numpy as np

from sjh_learn.experiments.transient_absorption.case_assembly import TaCaseAssembler
from sjh_learn.experiments.transient_absorption.pulse_scheduling import (
    classify_delay_mode,
    compute_pulse_centers,
)
from sjh_learn.experiments.transient_absorption.ta_settings import (
    DelayScanSettings,
    TaDelayScanSettings,
)
from sjh_learn.utils.fields.field_windows import FieldActiveWindowSettings


class TestTransientAbsorptionScaffold(unittest.TestCase):
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

    def test_field_window_settings_import(self) -> None:
        settings = TaDelayScanSettings(
            field_window=FieldActiveWindowSettings(
                rel_threshold=1e-3,
                padding_fs=1.0,
                dt_fs=0.2,
                t_start_fs=-500.0,
                t_end_fs=500.0,
            )
        )

        self.assertEqual(settings.field_window.t_start_fs, -500.0)
        self.assertEqual(settings.field_window.t_end_fs, 500.0)
        self.assertEqual(settings.field_window.padding_fs, 1.0)

    def test_case_assembler_scaffold_constructs(self) -> None:
        settings = TaDelayScanSettings(
            delay_scan=DelayScanSettings(
                probe_delays_fs=[30.0],
                probe_center_fs=0.0,
            )
        )
        assembler = TaCaseAssembler(settings)

        self.assertIs(assembler.settings, settings)

    def test_pulse_scheduling_mode_classification(self) -> None:
        self.assertEqual(
            classify_delay_mode(delay_fs=30.0, piecewise_min_positive_delay_fs=120.0),
            "full_overlap",
        )
        self.assertEqual(
            classify_delay_mode(delay_fs=200.0, piecewise_min_positive_delay_fs=120.0),
            "piecewise",
        )


if __name__ == "__main__":
    unittest.main()
