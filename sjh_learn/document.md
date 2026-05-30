# `sjh_learn` 审核文档

这个文档用于同步记录 `sjh_learn` 当前代码结构、主要类/函数、输入输出和职责，方便后续审核。

## 当前目录

```text
sjh_learn/
├─ __init__.py
├─ document.md
├─ optical_bloch_demo.py
├─ multilevel_demo.py
├─ n2_equivalence_check.py
├─ optical_bloch_plots/
├─ readme.md
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

## 1. 当前两条主路径

### Two-Level Path
- 输入：`PhysicalParams` 或 `OpticalBlochParameters`
- 输出：`OpticalBlochResult`
- 支持：
  - exact lab frame
  - exact rotating frame
  - RWA
  - physical normalization
  - pure dephasing / relaxation
  - 参数扫描
  - 三列对比图

### Multi-Level Path
- 输入：`MultiLevelParameters`
- 输出：`MultiLevelResult`
- 当前支持：
  - 最小 multi-level exact lab-frame simulation
  - `N x N` energies / dipole matrix
  - 一个或多个 `FieldConfig`
  - `relaxation / pure_dephasing / custom` collapse channels
- 当前不支持：
  - multi-level RWA
  - rotating-frame 自动变换
  - 多能级真实物理量自动归一化
  - 非线性光谱响应函数

## 2. 入口脚本

### `sjh_learn/optical_bloch_demo.py`
- 作用：two-level 参数扫描、出图、保存 `parameter_summary.json`

### `sjh_learn/multilevel_demo.py`
- 作用：三能级 exact lab-frame 最小示例

### `sjh_learn/n2_equivalence_check.py`
- 作用：比较 two-level exact lab-frame 与等价 `N=2` multi-level 路径
- 输出：
  - `rho11/rho22/rho12/rho21` 最大差异
  - `overall_max_difference`

## 3. `normalization.py`

### `PhysicalParams`
- 用户输入真实物理量：
  - `energy_gap_eV`
  - `laser_energy_eV`
  - `dipole_D`
  - `field_MV_per_cm`
  - `t_start_fs/t_end_fs/dt_fs`
  - `T1_fs/T2_fs/Tphi_fs`

### `SolverParams`
- 内部求解量：
  - `omega_eg/omega_L/detuning/rabi`
  - `gamma1/gamma_phi/gamma2`
  - `tlist`

### `ParaNormalizer`
- 作用：
  - 真实物理量 -> `fs^-1`
  - `fs^-1` -> 内部无量纲代码单位
  - 时间轴反变换回 `fs`

审核点：
- density matrix / population / coherence 不反变换
- `Hamiltonian / detuning / rabi / gamma / tlist` 共用同一内部时间尺度
- QuTiP 默认 `hbar = 1`

## 4. `parameters.py`

### `OpticalBlochParameters`
- two-level 内部求解参数
- 重要字段：
  - `detuning`
  - `dipole`
  - `field_amplitude`
  - `omega_drive`
  - `pulse_center`
  - `pulse_sigma`
  - `fields`
  - `gamma1`
  - `gamma_phi`
  - `tlist`
  - `times_fs`

### `PhysicalParameterSweep`
- two-level 真实物理量扫描

### `ParameterSweep`
- two-level legacy 内部参数扫描

## 5. `fields.py`

### `FieldConfig`
- 字段：
  - `amplitude`
  - `omega`
  - `phase`
  - `envelope`
  - `center`
  - `sigma`
  - `name`

### 主要函数
- `envelope_value(t, field)`
- `electric_field_value(t, field)`
  - `E(t) = 2 * amplitude * envelope(t) * cos(omega * t + phase)`
- `electric_field_array(times, field)`
- `total_electric_field_value(t, fields)`
- `total_electric_field_array(times, fields)`

## 6. `model.py`

### 作用
- 保存 two-level 模型定义
- 电场波形不再写死在本文件，而是委托给 `fields.py`

### 主要函数
- `default_field_config(parameters)`
  - 把旧 two-level 参数转换成默认 `FieldConfig`
- `parameter_fields(parameters)`
  - 若 `fields is None`，从 `field_amplitude / omega_drive / pulse_center / pulse_sigma` 自动构造默认场
  - 若 `fields` 已给定，则优先使用 `fields`
- `build_lab_hamiltonian(parameters)`
  - 驱动项：`H_int(t) = -mu * E(t)`
- `build_rwa_hamiltonian(parameters)`
- `build_c_ops(parameters)`
  - `C_phi = sqrt(gamma_phi / 2) * sigma_z`
  - `C_down = sqrt(gamma1) * sigma_minus`

## 7. `multilevel.py`

### `MultiLevelParameters`
- 字段：
  - `energies`
  - `dipole_matrix`
  - `t_start/t_end/dt`
  - `fields`
  - `hbar`
  - `level_labels`
  - `collapse_channels`
  - `tlist`
  - `times_fs`

单位说明：
- `energies`、`dipole_matrix`、`field amplitude`、`gamma` 当前都应已经是内部求解单位
- 当前 multi-level path 不自动处理 `eV / fs / Debye / MV/cm` 的物理归一化
- `hbar` 字段当前仅作为约定保留，multi-level 主路径没有额外用它做单位换算

### `CollapseChannel`
- `kind="relaxation"`
  - `C = sqrt(gamma) |j><i|`
- `kind="pure_dephasing"`
  - `C = sqrt(gamma) |i><i|`
- `kind="custom"`
  - `C = sqrt(gamma) * operator`

pure dephasing 的 rate 说明：
- `CollapseChannel(kind="pure_dephasing", rate=gamma, dephase_level=i)` 中的 `gamma` 是 Lindblad collapse rate
- 它不一定直接等于某个特定 coherence 的总 `T2` 衰减率
- 对于 `rho_ij`，多个 projector dephasing channels 的贡献通常是 `(gamma_i + gamma_j) / 2`

### 主要函数
- `build_multilevel_static_hamiltonian(parameters)`
  - `H0 = diag(energies)`
- `build_multilevel_dipole_operator(parameters)`
- `build_multilevel_lab_hamiltonian(parameters)`
  - `H = [H0, [-mu_operator, coeff_function]]`
- `build_multilevel_c_ops(parameters)`
- `initial_multilevel_density_matrix(n_levels, occupied_level=0)`
- `coherent_multilevel_density_matrix(amplitudes)`
- `simulate_multilevel_lab_frame(parameters, rho0=None)`
- `run_multilevel_case(parameters, rho0=None)`
- `optical_bloch_to_multilevel_parameters(parameters)`
  - 把 two-level exact lab-frame 参数映射成等价 `N=2` multi-level 系统
- `two_level_multilevel_equivalence_check(parameters)`
  - 比较 two-level exact lab-frame 与 `N=2` multi-level 结果差异

下一步建议：
- `multilevel.py` 和 `solvers.py` 当前都各自维护了一份 lab-frame `mesolve` 包装
- 下一步应抽象为通用 `simulate_lab_frame_system(system, rho0=None)`

## 8. `results.py`

### `OpticalBlochResult`
- 保存 two-level 的 lab / rotating / RWA 结果

### `MultiLevelResult`
- 保存 multi-level 的 exact lab-frame 结果

当前重复点：
- 两个结果类都维护了：
  - `times/times_fs`
  - `max_trace_error`
  - `max_hermiticity_error`
  - `summary_dict`
  - `plot_times_and_label`

下一步建议：
- 未来应抽象出 `QuantumResult` base class
- 当前先保留两套结果类，避免破坏 two-level 既有接口

## 9. `solvers.py`

### 正式主流程
- 实验室系：
```python
H = build_lab_hamiltonian(parameters)
c_ops = build_c_ops(parameters)
result = mesolve(H, rho0, tlist, c_ops, e_ops)
```

- RWA：
```python
H_rwa = build_rwa_hamiltonian(parameters)
c_ops = build_c_ops(parameters)
result = mesolve(H_rwa, rho0, tlist, c_ops, e_ops)
```

不再使用：
- 手写 Liouvillian
- density matrix vectorization
- RK4
- direct RWA Liouvillian

## 10. `checks.py`

### 作用
- 保存 two-level sanity checks

### 当前检查
- `trace_error_small`
- `hermiticity_error_small`
- `zero_field_closed_system_auxiliary`
- `pure_dephasing_auxiliary`
- `population_relaxation_auxiliary`

## 11. `plotting.py`

### `save_comparison_plot(result, output)`
- 保存 two-level 三列图：
  - lab frame
  - exact rotating frame
  - RWA

### `save_multilevel_plot(result, output, populations=None, coherences=None)`
- 保存 multi-level exact lab-frame 图
- 规则：
  - 横轴优先使用 `result.times_fs`
  - 若 `times_fs` 不存在，回退到 `result.times`
  - population 默认画所有对角元
  - coherence 由用户指定，当前画实部和虚部

## 12. `io.py`

### `default_output_path(output_dir, result)`
- 输出：two-level 对比图默认路径

### `save_parameter_summary(results, output)`
- 输出：`parameter_summary.json`
- 当前主要用于 two-level 参数摘要和 sanity checks 保存

## 13. 当前需要人工核对的物理边界

- two-level 约定保持不变：
  - `detuning = omega_eg - omega_L`
  - `E(t) = 2 E0 cos(omega t)`
  - RWA 非对角项为 `-g`
  - `g = mu E0 / hbar`
- multi-level 当前只有 exact lab-frame，尚未定义统一的 multi-level RWA 约定
- multi-level pure dephasing 采用 projector collapse 约定，和 two-level `sigma_z` 约定并不完全同形，但可以通过等价映射复现 two-level lab-frame 动力学
## 14. Result IO / QuantumResultIO

`io.py` 已从简单输出辅助模块增强为结果管理模块，保留原有
`default_output_path()` 和 `save_parameter_summary()`，并新增
`QuantumResultIO` / `ResultManager`。

设计边界参考 `NOPAExpRes`：

- 求解流程仍只负责生成 `OpticalBlochResult` / `MultiLevelResult`。
- `QuantumResultIO` 只负责文件夹、`results.csv`、逐 case 数据、`meta.json`
  和可选预览图。
- 不在 IO 层改变 Hamiltonian、`c_ops`、RWA、`FieldConfig` 或任何物理约定。

### `OpticalBlochResult` 新增接口

- `available_frames()`：返回 `lab / rotating / rwa`。
- `to_npz_dict()`：返回 `time_fs`、`time_code` 和三个 frame 的密度矩阵。
- `components_dataframe(frame)`：返回指定 frame 的表格，列为
  `time_fs, rho11, rho22, Re_rho12, Im_rho12, Re_rho21, Im_rho21`。
- `metadata_dict()`：返回可写入 JSON 的 metadata。

### `MultiLevelResult` 新增接口

- `to_npz_dict()`：返回 `time_fs`、`time_code` 和 `density_lab`。
- `populations_dataframe()`：返回所有对角布居随时间变化。
- `selected_elements_dataframe(elements)`：返回用户指定矩阵元的实部和虚部。
- `metadata_dict()`：返回可写入 JSON 的 metadata。

### 输出结构

```text
outdir/
├─ results.csv
└─ res_per_case/
   └─ <case_name>/
      ├─ data/
      ├─ figs/
      └─ meta.json
```

Two-level 每个 case 可输出：

- `data/density_lab.npz`
- `data/density_rotating.npz`
- `data/density_rwa.npz`
- `data/components_lab.csv`
- `data/components_rotating.csv`
- `data/components_rwa.csv`
- `meta.json`
- `figs/preview.png`
- `figs/comparison.png`，仅在 `output_full_figure=True` 时输出

Multi-level 每个 case 可输出：

- `data/density_lab.npz`
- `data/populations.csv`
- `data/selected_elements.csv`，仅在用户指定 `selected_elements` 时输出
- `meta.json`
- `figs/preview.png`

### 单位约定

- density matrix、population、coherence 均为无量纲量。
- CSV 主时间列为 `time_fs`。
- NPZ 同时保存 `time_fs` 和 `time_code`。
- 当 `times_fs` 存在时，`time_fs` 是真实物理时间 fs。
- 当某些内部测试没有真实物理时间轴时，`time_fs` 回退为当前 result 的时间轴，
  并在 metadata 中用 `time_fs_is_physical=False` 标记。

### 图像开关

`QuantumResultIO.save_case()` 中：

- `output_data` 控制是否保存 NPZ/CSV。
- `output_preview` 控制是否保存低 dpi `preview.png`，默认 dpi 为 120。
- `output_full_figure` 控制是否保存高 dpi 完整图，默认 dpi 为 300。
- `save_npz / save_csv / save_json` 分别控制不同数字文件。

## 15. Single-Trajectory Solver Contract

当前 solver / result / plotting / IO 的职责边界如下：

1. `lab_exact` 和 `rwa` 是两个独立 simulation case。
2. `rotating_view` 是 `lab_exact` result 的派生视图，不调用 `mesolve`。
3. solver 层一次只返回一个 `DynamicsResult`。
4. result 层一次只保存一条轨迹。
5. plotting 层返回 matplotlib `fig, axes`，不默认保存 PNG。
6. 顶层脚本负责组合多个 result、排版、加总标题、保存最终图。
7. `io.py` 只保存数字数据、metadata 和已经构造好的 figure。

### 新 result 类型

`DynamicsResult` 字段：

- `mode`：例如 `lab_exact`、`rwa`、`rotating_view`、`multilevel_lab`
- `times`
- `times_fs`
- `states`
- `parameters`
- `physical_params`
- `solver_params`
- `metadata`
- `source_mode`

主要方法：

- `density_array()`
- `dimension()`
- `populations()`
- `matrix_element(i, j)`
- `matrix_elements(pairs)`
- `selected_elements(elements)`
- `max_trace_error()`
- `max_hermiticity_error()`
- `summary_dict()`
- `metadata_dict()`
- `to_npz_dict()`
- `components_dataframe()`，仅 two-level result 使用

### 单模式求解函数

- `run_lab_case(parameters, rho0=None) -> DynamicsResult`
- `run_rwa_case(parameters, rho0=None) -> DynamicsResult`
- `make_rotating_view(lab_result) -> DynamicsResult`
- `run_multilevel_lab_case(parameters, rho0=None) -> DynamicsResult`

`run_lab_case` 内部使用 `build_lab_hamiltonian + build_c_ops + mesolve`。
`run_rwa_case` 内部使用 `build_rwa_hamiltonian + build_c_ops + mesolve`。
`make_rotating_view` 只对 `lab_result.states` 做 `rotate_density_trajectory`。

### 绘图函数

新增低层 plotting API：

- `plot_populations(result, ax=None, ...)`
- `plot_coherences(result, ax=None, ...)`
- `plot_density_components(result, axes=None, ...)`
- `plot_multilevel_components(result, axes=None, ...)`

如果传入 `ax/axes`，函数画在传入坐标轴上；否则内部创建 figure。
所有函数返回 `fig, axes`，不直接保存文件。

`save_comparison_plot()` 和 `save_multilevel_plot()` 仅作为 deprecated compatibility wrapper
保留，新代码应由顶层脚本拼图并调用 `save_figure()`。

### IO 函数

`io.py` 提供：

- `save_result_data(result, output_dir, save_npz=True, save_csv=True, save_json=True)`
- `save_figure(fig, output_path, dpi=120)`
- `save_result_case(result, output_dir, output_data=True, output_preview=False, fig=None, preview_dpi=120)`

`io.py` 不决定 lab / rotating / RWA 如何排版；只有传入 `fig` 时才保存图像。
