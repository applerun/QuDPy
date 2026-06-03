#!/usr/bin/env python3
"""Two-level optical Bloch demo.

本文件的定位：
1. 不是 solver 核心代码；
2. 是一个 demo / example script；
3. 作用是：
   - 定义一个 two-level system 的物理参数；
   - 做几个 field / laser 参数组合的扫描；
   - 分别跑 lab-frame、rotating-view、RWA 三种结果；
   - 画比较图；
   - 保存 csv / npz / json 结果。

最核心主线是：

    physical_params
        -> NORMALIZER.normalize()
        -> run_case(solver_mode="lab_exact")
        -> make_rotating_view()
        -> run_case(solver_mode="rwa")
        -> save / plot

也就是说：
- 物理参数在这里定义；
- 真正的数值求解由 run_case() 根据 solver_mode 分流；
- 这个文件主要是“组织运行流程”。
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
from time import perf_counter

import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# 0. 允许直接运行 example 文件
# ----------------------------------------------------------------------
# 如果你直接用：
#
#     python examples/optical_bloch_demo.py
#
# 而不是作为 package 运行，那么相对导入可能失败。
# 这一段的作用是把项目根目录临时加入 sys.path。
#
# 这不是物理主线，只是为了方便直接运行脚本。
if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ----------------------------------------------------------------------
# 1. 导入项目内部核心工具
# ----------------------------------------------------------------------
# 这些才是 demo 真正调用的库函数。
#
# NLevelPhysicalParams:
#   用户层面的物理参数容器，比如能级、偶极矩、光场强度、时间范围、T1/Tphi。
#
# ParaNormalizer:
#   把物理单位转换成 solver 内部使用的 code units。
#
# PhysicalParameterSweep:
#   用来定义参数扫描。不是必须，只是为了批量跑多个 field / laser case。
#
# RelaxationChannel / PureDephasingChannel:
#   Lindblad collapse operator 的物理参数描述。
#
# run_case(solver_mode="lab_exact"):
#   跑 lab-frame exact Hamiltonian。
#
# make_rotating_view:
#   不重新求解，只是把 lab-frame 结果变换到 rotating frame 视角。
#
# run_case(solver_mode="rwa"):
#   跑 RWA Hamiltonian。
#
# run_case 内部会完成 physical-to-solver 参数转换:
#   把 normalizer 输出的 solver 参数包装成 run_*_case 需要的参数对象。
from sjh_learn.utils.core import (
    NLevelPhysicalParams,
    ParaNormalizer,
    PhysicalParameterSweep,
    PureDephasingChannel,
    RelaxationChannel,
    make_rotating_view,
    run_case,
)


# ----------------------------------------------------------------------
# 2. 导入保存和画图工具
# ----------------------------------------------------------------------
# 这些不是物理主线，只是 demo 输出需要。
#
# QuantumResultIO:
#   把每个 result 保存成 npz / csv / json。
#
# default_output_path:
#   根据 result 自动生成输出图片路径。
#
# save_figure:
#   保存 matplotlib figure。
#
# save_parameter_summary:
#   汇总所有 case 的参数。
#
# save_results_components_long:
#   把 lab / rotating / RWA 的 density components 保存成长表 csv。
from sjh_learn.utils.io import (
    QuantumResultIO,
    default_output_path,
    save_figure,
    save_parameter_summary,
    save_results_components_long,
)


# plot_density_components:
#   画 population、coherence、drive / field 等分量。
from sjh_learn.utils.plotting import plot_density_components


# ----------------------------------------------------------------------
# 3. 定义基础物理参数
# ----------------------------------------------------------------------
# 这是这个 demo 最重要的输入。
#
# 这里定义的是一个 two-level system:
#
#   |g> energy = 0 eV
#   |e> energy = 1.55 eV
#
# 偶极矩矩阵：
#
#       [[0,   3],
#        [3,   0]]
#
# 表示只有 g <-> e transition dipole，单位 Debye。
#
# laser_energy_eV = 1.55 表示入射光能量。
# field_MV_per_cm = 0.3 表示光场强度。
# t_start_fs, t_end_fs, dt_fs 定义模拟时间网格。
#
# relaxation_channels:
#   e -> g 的 population relaxation，T1 = 500 fs。
#
# pure_dephasing_channels:
#   excited state 的 pure dephasing，Tphi = 300 fs。
#
# pulse_center_fs / pulse_sigma_fs = None:
#   这里不是 Gaussian pulse，而是连续/常包络驱动。
BASE_PHYSICAL_PARAMS = NLevelPhysicalParams(
    basis=("g", "e"),
    energies_eV=(0.0, 1.55),
    dipole_matrix_D=((0.0, 3.0), (3.0, 0.0)),
    laser_energy_eV=1.55,
    field_MV_per_cm=0.3,
    t_start_fs=0.0,
    t_end_fs=1000.0,
    dt_fs=0.05,
    relaxation_channels=(
        RelaxationChannel(
            name="relaxation_1_to_0",
            from_level=1,
            to_level=0,
            T1_fs=500.0,
        ),
    ),
    pure_dephasing_channels=(
        PureDephasingChannel(
            name="pure_dephasing_level_1",
            level=1,
            Tphi_fs=300.0,
        ),
    ),
    pulse_center_fs=None,
    pulse_sigma_fs=None,
)


# ----------------------------------------------------------------------
# 4. 定义参数扫描
# ----------------------------------------------------------------------
# 这部分不是必须的。
#
# 如果你只想跑一个 case，可以完全不需要 PhysicalParameterSweep。
#
# 现在这个 demo 会跑：
#
#   field = 0.5, 1, 2 MV/cm
#   laser = 1.55, 1.57 eV
#
# 所以总共是：
#
#   3 * 2 = 6 个 physical cases
#
# 每个 physical case 又会生成：
#
#   lab-frame
#   rotating-view
#   RWA
#
# 三条结果。
PHYSICAL_SWEEP = PhysicalParameterSweep(
    base_params=BASE_PHYSICAL_PARAMS,
    field_MV_per_cm_values=(0.5, 1, 2),
    laser_energy_eV_values=(1.55, 1.57),
)


# ----------------------------------------------------------------------
# 5. 定义单位归一化器
# ----------------------------------------------------------------------
# 这是从“物理单位”到“solver code units”的桥。
#
# time_scale_fs=None, auto_scale=True:
#   让 normalizer 自动选择合适的 time scale。
#
# 物理意义：
#   用户输入 eV、Debye、MV/cm、fs；
#   solver 不直接用这些单位，而是转成内部单位后再求解。
NORMALIZER = ParaNormalizer(time_scale_fs=None, auto_scale=True)


# ----------------------------------------------------------------------
# 6. 定义输出路径和结果保存器
# ----------------------------------------------------------------------
# 这些不是物理主线，只是为了保存 demo 输出。
OUTPUT_DIR = Path(__file__).resolve().parent / "optical_bloch_plots"
SUMMARY_PATH = OUTPUT_DIR / "parameter_summary.json"
RESULT_IO = QuantumResultIO(str(OUTPUT_DIR / "quantum_results_single"))


# ----------------------------------------------------------------------
# 7. 小工具函数：取 T1 / Tphi
# ----------------------------------------------------------------------
# 这两个函数只是为了：
#   1. 打印参数；
#   2. 写进 figure title。
#
# 它们不参与求解。
# 如果觉得累赘，完全可以内联到 print 和 title 里。
def _t1_fs(physical: NLevelPhysicalParams):
    return physical.relaxation_channels[0].T1_fs if physical.relaxation_channels else None


def _tphi_fs(physical: NLevelPhysicalParams):
    return physical.pure_dephasing_channels[0].Tphi_fs if physical.pure_dephasing_channels else None


# ----------------------------------------------------------------------
# 8. 单个物理参数点的核心求解函数
# ----------------------------------------------------------------------
# 这是本文件最关键的函数。
#
# 输入：
#   physical_params:
#       一个 NLevelPhysicalParams 对象。
#
# 输出：
#   lab:
#       lab-frame exact simulation result。
#
#   rotating:
#       从 lab 结果变换得到的 rotating-frame view。
#       注意：这个不是重新求解，只是视角变换。
#
#   rwa:
#       RWA Hamiltonian 下重新求解得到的 result。
#
# 主线：
#
#   1. physical_params
#          ↓
#   2. NORMALIZER.normalize()
#          ↓
#   3. run_case(solver_mode="lab_exact")
#          ↓
#   4. make_rotating_view()
#          ↓
#   5. make_rotating_view()
#          ↓
#   6. run_case(solver_mode="rwa")
#
def run_one_physical_point(physical_params: NLevelPhysicalParams):
    # Step 1:
    # 把物理参数转成 solver 内部参数。
    #
    # 例如：
    #   eV -> fs^-1
    #   Debye * field -> Rabi frequency
    #   T1 / Tphi -> collapse rates
    lab = run_case(replace(physical_params, solver_mode="lab_exact"), normalizer=NORMALIZER)

    # Step 2:
    # run_case 内部根据 solver_mode 完成归一化、参数构造和单轨迹求解。
    #
    # parameters 可以理解为“已经准备好给 solver 使用的一组 optical simulation params”。
    # Solver normalization is handled inside run_case().

    # Step 3:
    # 跑 lab-frame exact Hamiltonian。
    #
    # lab-frame 中，光场通常形如：
    #
    #   E(t) = 2 E0 cos(omega t + phase)
    #
    # Hamiltonian 里保留快速振荡项。
    # lab_exact result has already been produced by run_case().

    # Step 4:
    # 从 lab-frame 结果生成 rotating-frame view。
    #
    # 注意：
    #   这个函数通常不是重新积分 ODE；
    #   而是把 rho12(t) 等 coherence 乘上旋转因子。
    #
    # 作用：
    #   让你更容易观察慢变量 envelope。
    rotating = make_rotating_view(lab)

    # Step 5:
    # 跑 RWA Hamiltonian。
    #
    # RWA 会去掉 counter-rotating term，只保留慢变耦合项。
    # 它和 lab-frame exact 的对比可以用来检查 RWA 近似是否合理。
    rwa = run_case(replace(physical_params, solver_mode="rwa"), normalizer=NORMALIZER)

    # Step 6:
    # 给 RWA result 补充 physical / solver metadata。
    #
    # 这里不是求解需要，而是为了后续画图、保存、summary。
    # RWA metadata is attached by run_case().

    return lab, rotating, rwa


# ----------------------------------------------------------------------
# 9. 保存 lab / rotating / RWA 三列比较图
# ----------------------------------------------------------------------
# 这个函数也不是物理核心。
#
# 它的作用是把三种结果画到同一个 figure 里：
#
#   column 1: lab frame
#   column 2: rotating view
#   column 3: RWA
#
# 每一列通常包含：
#   population
#   coherence
#   drive / field
#
def save_comparison_figure(lab, rotating, rwa, output_path: Path) -> Path:
    # 3 行 3 列：
    #   3 列对应 lab / rotating / RWA；
    #   3 行由 plot_density_components 内部约定使用。
    fig, axes = plt.subplots(3, 3, figsize=(12.6, 6.3), sharex="col")

    # 画 lab-frame result。
    plot_density_components(
        lab,
        axes=axes[:, 0],
        include_drive=True,
        title="Lab frame",
    )

    # 画 rotating-frame view。
    plot_density_components(
        rotating,
        axes=axes[:, 1],
        include_drive=True,
        title="Rotating view",
    )

    # 画 RWA result。
    plot_density_components(
        rwa,
        axes=axes[:, 2],
        include_drive=True,
        title="RWA",
    )

    # 从 result 里取 metadata，用于 figure title。
    physical = lab.physical_params
    solver = lab.solver_params

    # 如果 metadata 存在，就把主要物理量写进标题。
    # 这部分只是为了读图方便，不影响求解。
    if physical is not None and solver is not None:
        fig.suptitle(
            "Two-Level Optical Bloch Comparison\n"
            f"Eg={physical.energy_gap_eV:.4f} eV, "
            f"EL={physical.laser_energy_eV:.4f} eV, "
            f"Field={physical.field_MV_per_cm:.4f} MV/cm, "
            f"mu01={physical.dipole_matrix_D[0][1]:.4f} D\n"
            f"T1={_t1_fs(physical)}, "
            f"Tphi={_tphi_fs(physical)}, "
            f"Delta={solver.detuning_fs_inv:.6g} fs^-1, "
            f"Rabi={solver.rabi_fs_inv:.6g} fs^-1"
        )
    else:
        fig.suptitle("Two-Level Optical Bloch Comparison")

    fig.tight_layout()

    # 保存图片。
    save_figure(fig, output_path, dpi=160)

    # 关闭 figure，避免批量循环时内存占用过多。
    plt.close(fig)

    return output_path


# ----------------------------------------------------------------------
# 10. 顶层运行入口
# ----------------------------------------------------------------------
# main() 的职责：
#
#   1. 打印基础参数；
#   2. 展开参数扫描；
#   3. 对每个参数点调用 run_one_physical_point()；
#   4. 画图；
#   5. 保存 csv / npz / json；
#   6. 打印 summary。
#
# main() 不是物理核心。
# 真正的物理求解核心在 run_one_physical_point() 里面。
def main() -> None:
    print("Two-level optical Bloch demo with explicit single-mode cases")

    # 打印基础参数，方便确认当前跑的是什么系统。
    print(f"energy_gap_eV     : {BASE_PHYSICAL_PARAMS.energy_gap_eV}")
    print(f"laser_energy_eV   : {BASE_PHYSICAL_PARAMS.laser_energy_eV}")
    print(f"dipole_matrix_D   : {BASE_PHYSICAL_PARAMS.dipole_matrix_D}")
    print(f"field_MV_per_cm   : {list(PHYSICAL_SWEEP.field_MV_per_cm_values)}")
    print(f"laser scan eV     : {list(PHYSICAL_SWEEP.laser_energy_eV_values)}")
    print(f"time range fs     : {BASE_PHYSICAL_PARAMS.t_start_fs} -> {BASE_PHYSICAL_PARAMS.t_end_fs}")
    print(f"dt_fs             : {BASE_PHYSICAL_PARAMS.dt_fs}")
    print(f"T1_fs             : {_t1_fs(BASE_PHYSICAL_PARAMS)}")
    print(f"Tphi_fs           : {_tphi_fs(BASE_PHYSICAL_PARAMS)}")

    # 如果 sweep 中显式给了 field 列表，就用列表；
    # 否则只用 base_params 里的单个 field。
    field_values = (
        PHYSICAL_SWEEP.field_MV_per_cm_values
        or (PHYSICAL_SWEEP.base_params.field_MV_per_cm,)
    )

    # 如果 sweep 中显式给了 laser 列表，就用列表；
    # 否则只用 base_params 里的单个 laser energy。
    laser_values = (
        PHYSICAL_SWEEP.laser_energy_eV_values
        or (PHYSICAL_SWEEP.base_params.laser_energy_eV,)
    )

    print("\nSolve timing:")

    # saved_paths:
    #   保存所有输出图片路径，只用于最后打印。
    saved_paths: list[Path] = []

    # all_results:
    #   收集所有 lab / rotating / RWA results，
    #   用于最后生成 parameter_summary.json。
    all_results = []

    case_index = 0
    total_cases = len(field_values) * len(laser_values)

    # 记录总耗时。
    sweep_start = perf_counter()

    # ------------------------------------------------------------------
    # 双重循环：
    #
    #   outer: laser_energy_eV
    #   inner: field_MV_per_cm
    #
    # 每组参数都会生成一个新的 physical_params。
    # ------------------------------------------------------------------
    for laser_energy_eV in laser_values:
        for field_MV_per_cm in field_values:
            case_index += 1

            # 用 dataclasses.replace 基于 base_params 创建一个新参数对象。
            #
            # 这样不会修改 BASE_PHYSICAL_PARAMS 本身。
            physical_params = replace(
                PHYSICAL_SWEEP.base_params,
                laser_energy_eV=laser_energy_eV,
                field_MV_per_cm=field_MV_per_cm,
            )

            # 记录单个 case 的求解时间。
            case_start = perf_counter()

            # ------------------------------
            # 核心调用：
            # 对当前 physical_params 同时得到：
            #   lab-frame result
            #   rotating view
            #   RWA result
            # ------------------------------
            lab, rotating, rwa = run_one_physical_point(physical_params)

            solve_elapsed_s = perf_counter() - case_start

            # 自动生成输出图片路径。
            output_path = default_output_path(OUTPUT_DIR, lab)

            # 保存三列比较图。
            save_comparison_figure(lab, rotating, rwa, output_path)

            # 保存三种结果的 components 长表。
            #
            # 这个 csv 适合后续用 pandas 分析。
            save_results_components_long(
                [lab, rotating, rwa],
                output_path.with_name(f"{output_path.stem}_components.csv"),
            )

            saved_paths.append(output_path)

            # 保存每一个 result 的完整数据。
            #
            # output_data=True:
            #   保存具体时序数据。
            #
            # output_preview=False:
            #   不额外保存 preview。
            #
            # save_npz / save_csv / save_json:
            #   同时保存不同格式，便于后续读取和检查。
            #
            # 这部分不是主线。如果只想快速看图，可以暂时删掉。
            for result in (lab, rotating, rwa):
                RESULT_IO.save_case(
                    result,
                    output_data=True,
                    output_preview=False,
                    save_npz=True,
                    save_csv=True,
                    save_json=True,
                )
                all_results.append(result)

            # 打印当前 case 的运行信息。
            print(
                f"case {case_index}/{total_cases} "
                f"field={field_MV_per_cm:g} MV/cm, "
                f"laser={laser_energy_eV:g} eV -> "
                f"{solve_elapsed_s:.3f} s"
            )
            print(f"output figure      : {output_path}")

            # summary_dict() 通常包含 final population / coherence 等摘要。
            print(f"lab final summary  : {lab.summary_dict()}")
            print(f"rwa final summary  : {rwa.summary_dict()}")

    # 打印总耗时。
    total_elapsed_s = perf_counter() - sweep_start
    print(f"\ntotal solve time  : {total_elapsed_s:.3f} s")

    # 打印所有生成的图片路径。
    print("\nGenerated comparison figures:")
    for output_path in saved_paths:
        print(output_path)

    # 保存所有 result 的参数汇总。
    summary_path = save_parameter_summary(all_results, SUMMARY_PATH)
    print(f"\nparameter summary  : {summary_path}")
    print(f"result data root   : {RESULT_IO.outdir}")


# ----------------------------------------------------------------------
# 11. Python 脚本入口
# ----------------------------------------------------------------------
# 只有直接运行本文件时，才会执行 main()。
#
# 如果这个文件被 import，则不会自动运行。
if __name__ == "__main__":
    main()
