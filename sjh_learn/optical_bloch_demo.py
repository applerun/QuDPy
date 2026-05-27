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
    ParameterSweep,
    default_output_path,
    run_parameter_sweep,
    save_comparison_plot,
)



SWEEP = ParameterSweep(
    t_final=120.0,
    dt=0.01,
    hbar=1.0,
    epsilon_1=0.0,
    dipole=0.08,
    omega_drive=1.0,
    field_amplitudes=(0.5, 1.0, 2.0),
    detunings=(0.0, 0.2),
)

OUTPUT_DIR = Path("optical_bloch_plots")


def main() -> None:
    """运行参数扫描并保存图像。"""
    print("两能级光学 Bloch 演化示例")
    print(f"总时间           : {SWEEP.t_final}")
    print(f"时间步长         : {SWEEP.dt}")
    print(f"hbar             : {SWEEP.hbar}")
    print(f"驱动频率         : {SWEEP.omega_drive}")
    print(f"驱动振幅列表     : {list(SWEEP.field_amplitudes)}")
    print(f"失谐量列表       : {list(SWEEP.detunings)}")

    results = run_parameter_sweep(SWEEP)

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
        for key, value in result.summary_dict().items():
            print(f"{key:<18}: {value}")

    print("\n已生成图像：")
    for output_path in saved_paths:
        print(output_path)


if __name__ == "__main__":
    main()
