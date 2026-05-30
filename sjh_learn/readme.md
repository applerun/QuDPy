# `sjh_learn` 学习代码说明

当前代码现在有两条学习路径：

- `two-level optical Bloch path`
- `multi-level lab-frame path`

两条路径都统一使用 QuTiP 的 `mesolve(H, rho0, tlist, c_ops, e_ops)`，不再使用手写 Liouvillian、密度矩阵向量化或 RK4 作为正式主流程。

## 当前结构

```text
sjh_learn/
├─ optical_bloch_demo.py
├─ multilevel_demo.py
├─ optical_bloch_plots/
├─ readme.md
├─ document.md
└─ utils/
   ├─ __init__.py
   ├─ normalization.py
   ├─ parameters.py
   ├─ fields.py
   ├─ model.py
   ├─ multilevel.py
   ├─ solvers.py
   ├─ results.py
   ├─ checks.py
   ├─ plotting.py
   └─ io.py
```

## Two-Level Path

两能级路径当前支持：

- lab frame exact simulation
- exact rotating-frame transformation
- RWA simulation
- physical parameter normalization
- pure dephasing / population relaxation
- parameter sweep
- comparison plotting

当前物理约定保持不变：

- `detuning = omega_eg - omega_L`
- `E(t) = 2 E0 cos(omega t)`
- RWA 非对角项使用 `-g`
- `g = mu E0 / hbar`
- `C_phi = sqrt(gamma_phi / 2) * sigma_z`
- `C_down = sqrt(gamma1) * sigma_minus`

绘图默认优先使用真实时间轴：

- 如果 `result.times_fs` 存在，横轴使用 `Time (fs)`
- 否则回退到内部时间 `result.times`

图标题默认显示真实物理量，内部无量纲参数继续保存到 `parameter_summary.json`。

## Multi-Level Path

多能级路径当前支持：

- arbitrary `N`-level energies
- arbitrary `N x N` dipole matrix
- one or multiple `FieldConfig` fields
- exact lab-frame `mesolve`
- user-defined collapse channels
  - `relaxation`
  - `pure_dephasing`
  - `custom`

多能级路径当前还不支持：

- multi-level RWA
- rotating-frame transformation
- automatic physical-unit normalization
- nonlinear spectroscopy pulse-sequence response functions

也就是说，现在的 multi-level 部分是“最小可运行的 exact lab-frame framework”，用于避免后续代码被 two-level 写死，但还不是完整的多能级光谱框架。

单位约定：

- `energies`、`dipole_matrix`、`field amplitude`、`gamma` 当前都应已经是内部求解单位
- 当前 multi-level path 不自动处理 `eV / fs / Debye / MV/cm` 的物理归一化
- `hbar` 字段当前主要是约定保留，正式 multi-level lab-frame 路径没有额外用它做单位换算

pure dephasing 约定：

- `CollapseChannel(kind="pure_dephasing", rate=gamma, dephase_level=i)` 使用
  - `C = sqrt(gamma) |i><i|`
- 这里的 `gamma` 是 Lindblad collapse rate
- 它不一定直接等于某个指定 coherence 的总 `T2` 衰减率
- 对 `rho_ij`，多个 projector dephasing channels 的贡献通常是 `(gamma_i + gamma_j) / 2`

## 电场层

`utils/fields.py` 现在专门负责外加电场定义：

- `FieldConfig`
- `envelope_value`
- `electric_field_value`
- `electric_field_array`
- `total_electric_field_value`
- `total_electric_field_array`

目前支持：

- `constant` envelope
- `gaussian` envelope

后续可以在这里继续扩展 chirp、脉冲列、更多 envelope 形式，而不需要把波形逻辑写死在 `model.py` 或 `multilevel.py`。

## 推荐导入方式

```python
from sjh_learn.utils.normalization import PhysicalParams, SolverParams, ParaNormalizer
from sjh_learn.utils.parameters import OpticalBlochParameters, PhysicalParameterSweep, ParameterSweep
from sjh_learn.utils.fields import FieldConfig, electric_field_value, total_electric_field_value
from sjh_learn.utils.model import build_lab_hamiltonian, build_rwa_hamiltonian, build_c_ops
from sjh_learn.utils.multilevel import (
    MultiLevelParameters,
    CollapseChannel,
    build_multilevel_lab_hamiltonian,
    build_multilevel_c_ops,
    run_multilevel_case,
)
from sjh_learn.utils.results import OpticalBlochResult, MultiLevelResult
from sjh_learn.utils.plotting import save_comparison_plot, save_multilevel_plot
```

## 如何运行

Two-level demo：

```bash
conda activate qutip-clean
python /Users/shenjianghao/PycharmProjects/QuDPy/sjh_learn/optical_bloch_demo.py
```

Multi-level demo：

```bash
conda activate qutip-clean
python /Users/shenjianghao/PycharmProjects/QuDPy/sjh_learn/multilevel_demo.py
```

N=2 equivalence check：

```bash
conda activate qutip-clean
python /Users/shenjianghao/PycharmProjects/QuDPy/sjh_learn/n2_equivalence_check.py
```

## 当前边界

当前代码适合：

- 学 two-level optical Bloch dynamics
- 学 lab / rotating / RWA 的区别
- 学 physical normalization
- 试最小多能级实验室系数值演化

当前还不适合直接当成：

- 通用多能级 RWA 框架
- 真正的多脉冲非线性光谱响应函数框架
- 自动处理真实多能级物理量归一化的生产代码

当前代码里还有两个明确的“下一步可以收束的重复点”：

- `solvers.py` 和 `multilevel.py` 各自维护了一份 lab-frame `mesolve` 包装；未来可以抽象成通用 `simulate_lab_frame_system(system, rho0=None)`
- `OpticalBlochResult` 和 `MultiLevelResult` 有一部分重复接口；未来可以抽象成 `QuantumResult` base class
## Result IO / QuantumResultIO

`utils/io.py` 现在同时负责传统图像路径和数值结果管理。新增的 `QuantumResultIO`
借鉴了 `NOPAExpRes` 的思路：求解器只负责物理计算，IO manager 负责 `results.csv`、
逐 case 目录、数值数据、`meta.json` 和可选预览图。

默认输出结构：

```text
outdir/
├─ results.csv
└─ res_per_case/
   └─ <case_name>/
      ├─ data/
      ├─ figs/
      └─ meta.json
```

Two-level `OpticalBlochResult` 支持保存：

- `data/density_lab.npz`
- `data/density_rotating.npz`
- `data/density_rwa.npz`
- `data/components_lab.csv`
- `data/components_rotating.csv`
- `data/components_rwa.csv`
- `meta.json`
- `figs/preview.png`，可选低 dpi 预览图
- `figs/comparison.png`，可选高 dpi 完整图

Multi-level `MultiLevelResult` 支持保存：

- `data/density_lab.npz`
- `data/populations.csv`
- `data/selected_elements.csv`，当用户指定矩阵元时输出
- `meta.json`
- `figs/preview.png`，可选低 dpi 预览图

保存开关：

```python
from sjh_learn.utils import QuantumResultIO

io = QuantumResultIO("output_dir")
io.save_case(
    result,
    output_data=True,
    output_preview=True,
    output_full_figure=False,
    save_npz=True,
    save_csv=True,
    save_json=True,
    preview_dpi=120,
    full_dpi=300,
)
```

说明：

- density matrix / population / coherence 都是无量纲量。
- CSV 和 NPZ 中的主时间轴保存为 `time_fs`，优先使用真实物理时间 fs。
- `time_code` 也会写入 NPZ，便于追踪内部求解单位。
- 完整数据和预览图可以分别通过 `output_data`、`output_preview`、
  `output_full_figure` 开关控制。

## Single-Trajectory Dynamics Architecture

当前推荐架构已经调整为“一次求解只产生一个 result”：

- `lab_exact` 和 `rwa` 是两个独立 simulation case。
- `rotating_view` 不是独立求解，而是由 `lab_exact` result 通过幺正变换派生。
- solver 层一次只返回一个 `DynamicsResult`。
- result 层一次只保存一条 density-matrix trajectory。
- `plotting.py` 的新函数只画到 matplotlib `fig/axes` 并返回句柄，不默认保存。
- 顶层脚本负责组合多个 result、排版、加总标题和保存最终图。
- `io.py` 只保存 result 数字数据、metadata 和调用方已经构造好的 figure。

推荐 two-level 用法：

```python
lab = run_lab_case(parameters)
rotating = make_rotating_view(lab)
rwa = run_rwa_case(parameters)

fig, axes = plt.subplots(2, 3)
plot_density_components(lab, axes=axes[:, 0])
plot_density_components(rotating, axes=axes[:, 1])
plot_density_components(rwa, axes=axes[:, 2])
save_figure(fig, output_path, dpi=160)
```

推荐 IO 用法：

```python
save_result_case(lab, outdir, output_data=True)
save_result_case(rwa, outdir, output_data=True)
save_result_case(lab, outdir, output_preview=True, fig=fig)
```

`save_comparison_plot()` 和 `save_multilevel_plot()` 仍保留为兼容 wrapper，但新代码应优先
使用 `plot_populations()`、`plot_coherences()`、`plot_density_components()`、
`plot_multilevel_components()` 后由顶层脚本或 `io.save_figure()` 保存。
