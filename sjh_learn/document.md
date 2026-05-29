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
