#!/usr/bin/env python3
"""两能级光学 Bloch demo。

这个脚本只负责三件事：
1. 定义参数扫描范围
2. 调用 `sjh_learn.utils` 里的计算函数
3. 保存图像并打印结果摘要
"""

from __future__ import annotations

from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sjh_learn.utils import (
    ParaNormalizer,
    PhysicalParameterSweep,
    PhysicalParams,
    default_output_path,
    run_physical_parameter_sweep,
    save_parameter_summary,
    save_comparison_plot,
)


BASE_PHYSICAL_PARAMS = PhysicalParams(
    energy_gap_eV=1.55,
    laser_energy_eV=1.55,
    dipole_D=3.0,
    field_MV_per_cm=0.3,
    t_start_fs=0.0,
    t_end_fs=1000.0,
    dt_fs=0.5,
    T1_fs=500.0,
    T2_fs=None,
    Tphi_fs=300.0,
    pulse_center_fs=None,
    pulse_sigma_fs=None,
)

PHYSICAL_SWEEP = PhysicalParameterSweep(
    base_params=BASE_PHYSICAL_PARAMS,
    field_MV_per_cm_values=(0.5, 1, 2),
    laser_energy_eV_values=(1.55, 1.57),
)

NORMALIZER = ParaNormalizer(time_scale_fs=None, auto_scale=True)

OUTPUT_DIR = Path(__file__).resolve().parent / "optical_bloch_plots"
SUMMARY_PATH = OUTPUT_DIR / "parameter_summary.json"


def main() -> None:
    """运行参数扫描并保存图像。"""
    print("两能级光学 Bloch 演化示例（真实物理参数输入）")
    print(f"energy_gap_eV     : {BASE_PHYSICAL_PARAMS.energy_gap_eV}")
    print(f"laser_energy_eV   : {BASE_PHYSICAL_PARAMS.laser_energy_eV}")
    print(f"dipole_D          : {BASE_PHYSICAL_PARAMS.dipole_D}")
    print(f"field_MV_per_cm   : {list(PHYSICAL_SWEEP.field_MV_per_cm_values)}")
    print(f"laser scan eV     : {list(PHYSICAL_SWEEP.laser_energy_eV_values)}")
    print(f"time range fs     : {BASE_PHYSICAL_PARAMS.t_start_fs} -> {BASE_PHYSICAL_PARAMS.t_end_fs}")
    print(f"dt_fs             : {BASE_PHYSICAL_PARAMS.dt_fs}")
    print(f"T1_fs             : {BASE_PHYSICAL_PARAMS.T1_fs}")
    print(f"Tphi_fs           : {BASE_PHYSICAL_PARAMS.Tphi_fs}")

    results = run_physical_parameter_sweep(PHYSICAL_SWEEP, normalizer=NORMALIZER)

    print("\n逐个参数点结果：")
    saved_paths: list[Path] = []
    for result in results:
        output_path = default_output_path(OUTPUT_DIR, result)
        save_comparison_plot(result, output_path)
        saved_paths.append(output_path)

        print(
            f"\nE0 = {result.parameters.field_amplitude:.4f}, "
            f"Delta = {result.parameters.detuning:.4f}"
        )
        print(f"输出图像         : {output_path}")
        if result.solver_params is not None:
            print(NORMALIZER.summary_text(result.physical_params, result.solver_params))
        for key, value in result.summary_dict().items():
            print(f"{key:<18}: {value}")
        print(f"sanity_checks      : {result.sanity_checks}")

    print("\n已生成图像：")
    for output_path in saved_paths:
        print(output_path)

    summary_path = save_parameter_summary(results, SUMMARY_PATH)
    print(f"\n参数摘要已保存   : {summary_path}")


if __name__ == "__main__":
    main()
