# sjh_learn 架构记录（详细版）

本文档在当前 `document.md` 的基础上扩展，重点记录各模块职责、主要函数用途、变量含义、`DynamicsResult` 字段解释，以及数据保存逻辑。当前目标是让代码既能服务 two-level optical Bloch 学习，又尽量保持对 multi-level 扩展友好。

## 1. 总体设计原则

### 1.1 单轨迹结果对象

标准结果对象是 `DynamicsResult`。一个 `DynamicsResult` 只表示一条 density-matrix trajectory，也就是一次 simulation case 的结果。

例如：

```python
lab = run_lab_case(parameters)
rotating = make_rotating_view(lab)
rwa = run_rwa_case(parameters)
```

这三行会生成三个独立结果：

```text
lab.mode      = "lab_exact"
rotating.mode = "rotating_view"
rwa.mode      = "rwa"
```

含义：

- `lab_exact`：实验室坐标系中直接求解含时 Hamiltonian；
- `rotating_view`：由 `lab_exact` 后处理变换得到，不重新调用 `mesolve`；
- `rwa`：直接求解 RWA 后的有效 Hamiltonian。

这种设计避免了一个 result 同时装多个物理情形，使数据保存、绘图和复现更清晰。

### 1.2 分层设计

程序按以下流程组织：

```text
物理参数设置
→ input field / input drive 构建
→ 单位转换
→ Hamiltonian 和 collapse operators 构建
→ QuTiP mesolve 求解
→ DynamicsResult 封装
→ plotting 绘图
→ IO 保存数据、metadata 和图像
```

各模块职责：

| 模块 | 文件 | 职责 |
|---|---|---|
| 参数定义 | `parameters.py` | 定义物理输入参数和 solver 参数 |
| 单位转换 | `normalization.py` | eV、fs、Debye、MV/cm 与 solver code unit 转换 |
| 光场/drive | `fields.py` | 定义 Lab-frame 电场和 RWA drive |
| 模型构造 | `model.py` | 构建 Hamiltonian、collapse operators、rotating transform |
| 求解器 | `solvers.py`、`multilevel.py` | 调用 QuTiP 求解或构造派生 result |
| 结果对象 | `results.py` | 保存 density matrix trajectory 和通用访问接口 |
| 绘图 | `plotting.py` | 根据 result 画图，返回 matplotlib 句柄 |
| IO | `io.py` | 保存数据、metadata 和 figure |
| 检查 | `checks.py` | 等价性检查和 sanity check |
| 示例 | `examples/` | 组织具体模拟条件和参数扫描 |

## 2. 当前目录结构

```text
sjh_learn/
├─ optical_bloch_demo.py
├─ multilevel_demo.py
├─ n2_equivalence_check.py
├─ examples/
│  ├─ rwa_common.py
│  ├─ rwa_01_field_strength.py
│  ├─ rwa_02_dephasing.py
│  ├─ rwa_03_redistribution.py
│  └─ rwa_04_dephasing_and_redistribution.py
└─ utils/
   ├─ fields.py
   ├─ model.py
   ├─ solvers.py
   ├─ multilevel.py
   ├─ results.py
   ├─ plotting.py
   ├─ io.py
   ├─ checks.py
   ├─ normalization.py
   └─ parameters.py
```

说明：

- `optical_bloch_demo.py`：Lab-frame / rotating view / RWA 三种结果对比；
- `multilevel_demo.py`：multi-level lab-frame 示例；
- `n2_equivalence_check.py`：检查 two-level lab-frame 与 N=2 multi-level lab-frame 一致性；
- `examples/rwa_*.py`：RWA-only 参数扫描和物理现象验证；
- `utils/`：核心模块。

## 3. `fields.py`

`fields.py` 定义输入光场和 RWA drive。它们都是 callable class，即可以像函数一样使用：

```python
value = field_or_drive(t)
```

### 3.1 通用接口

每个 field / drive 对象通常支持以下方法。

#### `__call__(t)`

功能：返回时间 `t` 处的场或 drive 值。

变量：

- `t`：求解器时间。可能是 fs，也可能是 solver code time，具体看对象的 `time_unit` 和 metadata。
- 返回值：当前时间点的电场值或 RWA drive 值。

#### `to_dict()`

功能：把对象保存为可重建的字典。

用途：

- 写入 `debug_meta.json`；
- 记录输入场或 drive 的参数；
- 后续复现实验条件。

常见字段：

- `class`：类名；
- `name`：对象名；
- `amplitude`：幅度；
- `domain`：例如 `lab_frame`、`RWA`、`solver_code`；
- `time_unit`：时间单位；
- `amplitude_unit`：幅度单位。

#### `from_dict()`

功能：从 `to_dict()` 生成的字典重建对象。

说明：

- 不使用 `eval()`；
- 新增 field / drive 类时，需要同步支持重建。

#### `to_expr()`

功能：生成面向人的表达式字符串。

用途：

- 写入日志；
- 写入 metadata；
- 用于 PPT 或文档说明。

注意：`to_expr()` 不是可靠重建方式，可靠重建应使用 `to_dict()`。

### 3.2 `ConstantDrive`

RWA 中的常数慢变量 drive。

表达式：

```text
g(t) = g0
```

或在部分文档中写为：

```text
Omega(t) = Omega0
```

重要变量：

- `amplitude`：drive 幅度；
- `name`：drive 名称，例如 `rwa_cw_drive`；
- `domain`：常见为 `solver_code` 或 `RWA`；
- `time_unit`：时间单位；
- `amplitude_unit`：幅度单位，例如 `code` 或 `fs^-1`。

物理含义：CW 光在 RWA 中去掉 optical carrier 后，保留下来的就是常数慢变量耦合。

### 3.3 `GaussianDrive`

RWA 中带高斯包络的慢变量 drive。

表达式：

```text
g(t) = g0 exp[-(t - t0)^2 / (2 sigma^2)]
```

重要变量：

- `amplitude`：峰值 drive；
- `t0` 或 `center`：脉冲中心；
- `sigma`：高斯宽度；
- `name`：drive 名称；
- `time_unit`：时间单位；
- `amplitude_unit`：drive 单位。

物理含义：RWA 中去掉载波，保留脉冲包络对应的有效耦合。

### 3.4 `CarrierField`

Lab-frame 真实电场。

表达式：

```text
E(t) = 2 E0 cos(omega_L t + phase)
```

重要变量：

- `E0` 或 `amplitude`：电场幅度参数；
- `omega_L`：激光角频率；
- `phase`：初始相位；
- `field_unit`：电场单位，例如 `MV/cm`；
- `time_unit`：时间单位。

当前常用约定：

```text
field_MV_per_cm is E0 in E(t) = 2 E0 cos(omega_L t + phase)
```

因此若 `field_MV_per_cm = 0.1`，峰值电场为 `0.2 MV/cm`。

### 3.5 `GaussianCarrierField`

带高斯包络的 Lab-frame 真实电场。

表达式：

```text
E(t) = 2 E0 exp[-(t - t0)^2 / (2 sigma^2)] cos(omega_L t + phase)
```

变量：

- `E0`：电场幅度参数；
- `t0`：脉冲中心；
- `sigma`：脉冲宽度；
- `omega_L`：载波角频率；
- `phase`：载波相位。

与 `GaussianDrive` 的区别：

- `GaussianCarrierField` 是 Lab-frame 真实电场，包含 `cos(omega_L t)`；
- `GaussianDrive` 是 RWA 慢变量 drive，不包含 optical carrier。

### 3.6 `CompositeField`

多个 field 的叠加。

表达式：

```text
E_total(t) = E_1(t) + E_2(t) + ...
```

用途：

- 多脉冲叠加；
- pump-probe 扩展；
- 后续 multi-pulse simulation。

## 4. `parameters.py`

### 4.1 `PhysicalParams`

`PhysicalParams` 保存用户输入的真实物理参数。

常见字段：

| 字段 | 含义 | 单位 |
|---|---|---|
| `energy_gap_eV` | 两能级能量差 | eV |
| `laser_energy_eV` | 激光光子能量 | eV |
| `dipole_D` | 跃迁偶极矩 | Debye |
| `field_MV_per_cm` | 电场幅度参数 `E0` | MV/cm |
| `T1_fs` | population relaxation time | fs |
| `Tphi_fs` | pure dephasing time | fs |
| `t_start_fs` | 模拟起始时间 | fs |
| `t_end_fs` | 模拟结束时间 | fs |
| `dt_fs` | 输出时间间隔 | fs |
| `pulse_center_fs` | 脉冲中心 | fs |
| `pulse_sigma_fs` | 脉冲宽度 | fs |

注意：

- `T1_fs=None` 表示不加入 T1 relaxation；
- `Tphi_fs=None` 表示不加入 pure dephasing；
- `field_MV_per_cm` 是输入幅度参数，不一定等于瞬时峰值场。

### 4.2 `SolverParams`

`SolverParams` 是求解器内部使用的参数，通常已经完成单位转换。

常见字段：

| 字段 | 含义 |
|---|---|
| `t_start` | solver 起始时间 |
| `t_end` / `t_final` | solver 结束时间 |
| `dt` | solver 输出时间步长 |
| `tlist` | solver 时间数组 |
| `epsilon_1`、`epsilon_2` | 能级参数或角频率参数 |
| `detuning` | RWA detuning，通常为 code unit |
| `field_amplitude` | solver unit 下的 drive / field amplitude |
| `omega_drive` | 激光角频率或 drive 频率 |
| `gamma1` | T1 relaxation rate in code unit |
| `gamma_phi` | pure dephasing rate in code unit |
| `gamma2` | total coherence decay rate in code unit |
| `pulse_center` | code unit 下的 pulse center |
| `pulse_sigma` | code unit 下的 pulse width |

注意：

- `SolverParams` 服务 solver，不适合直接作为人工检查实验条件的主要入口；
- 人工检查优先看 `meta.json`；
- 完整 solver 参数保存在 `debug_meta.json`。

## 5. `normalization.py`

`normalization.py` 负责真实物理单位到 solver code unit 的转换。

### 5.1 `ParaNormalizer`

功能：将 `PhysicalParams` 转换为 solver 可用参数。

典型流程：

```text
PhysicalParams
→ derived physical quantities in fs^-1
→ SolverParams in code unit
```

常见转换：

```text
omega_eg_fs_inv = energy_gap_eV / hbar_eV_fs
omega_L_fs_inv  = laser_energy_eV / hbar_eV_fs
```

其中：

```text
hbar_eV_fs ≈ 0.6582 eV*fs
```

失谐量：

```text
detuning_fs_inv = omega_eg_fs_inv - omega_L_fs_inv
```

也可能根据代码约定使用相反符号，metadata 中应保持一致。

RWA coupling：

```text
g(t) = mu E0 f(t) / hbar
```

如果 Hamiltonian 非对角项写作 `-g`，则共振无耗散时：

```text
rho_11(t) = sin^2(g t)
population_rabi_period = pi / g
```

弛豫速率：

```text
gamma1_fs_inv     = 1 / T1_fs
gamma_phi_fs_inv  = 1 / Tphi_fs
gamma2_fs_inv     = gamma1_fs_inv / 2 + gamma_phi_fs_inv
```

code unit：

```text
quantity_code = quantity_fs_inv * time_scale_fs
```

### 5.2 当前限制

当前 `ParaNormalizer` 主要服务 two-level physical path。

Multi-level path 目前默认输入已经是 solver-ready / code-unit：

```text
MultiLevelParameters are solver-ready/code-unit inputs.
```

也就是说，multi-level 中的 energies、dipole matrix、fields 暂不自动从 eV、Debye、MV/cm 转换。

## 6. `model.py`

`model.py` 负责构建 Hamiltonian、collapse operators 和 rotating-frame 变换。

### 6.1 `build_lab_hamiltonian(...)`

功能：构建 Lab-frame 含时 Hamiltonian。

物理形式：

```text
H(t) = H0 + H_int(t)
H_int(t) = -mu E(t)
```

变量：

- `H0`：无场 Hamiltonian；
- `E(t)`：Lab-frame 真实电场；
- `mu`：跃迁偶极矩；
- 返回值：可传给 QuTiP `mesolve` 的 Hamiltonian。

特点：

- 保留 optical carrier；
- 对时间步长要求高；
- 适合做 RWA 的对照验证。

### 6.2 `build_rwa_hamiltonian(...)`

功能：构建 RWA 有效 Hamiltonian。

常见形式：

```text
H_RWA = -Delta |e><e| - g(t)(|e><g| + |g><e|)
```

变量：

- `Delta`：失谐；
- `g(t)`：RWA drive / coupling；
- 返回值：RWA 下传给 `mesolve` 的 Hamiltonian。

特点：

- 去掉 optical carrier；
- 只保留慢变量 drive；
- 更适合参数扫描和长时间模拟。

### 6.3 `build_c_ops(...)`

功能：构建 Lindblad collapse operators。

T1 relaxation：

```text
C_down = sqrt(gamma1) |g><e|
```

作用：把 excited-state population 拉回 ground state，同时贡献 coherence decay。

Pure dephasing：

```text
C_phi = sqrt(gamma_phi / 2) sigma_z
```

作用：主要破坏 coherence，不直接导致 excited-to-ground population transfer。

### 6.4 `rotating_frame_unitary(...)`

功能：生成 rotating-frame 的幺正变换矩阵。

典型形式：

```text
U(t) = exp(-i omega_L t |e><e|)
```

变量：

- `t`：时间；
- `omega_L`：参考旋转频率，通常来自 laser energy。

### 6.5 `rotate_density_trajectory(...)`

功能：把 Lab-frame density matrix trajectory 转换到 rotating frame。

典型形式：

```text
rho_rot(t) = U^(t) rho_lab(t) U(t)
```

这里 `U^` 表示 `U` 的 Hermitian conjugate，即通常写作 `U dagger`。

注意：

- 这是后处理变换；
- 不重新求解；
- 不等于 RWA；
- 没有丢弃 counter-rotating terms。

## 7. `solvers.py`

### 7.1 `run_lab_case(parameters, rho0=None) -> DynamicsResult`

功能：运行 two-level Lab-frame exact simulation。

流程：

```text
PhysicalParams
→ ParaNormalizer
→ build_lab_hamiltonian
→ build_c_ops
→ mesolve
→ DynamicsResult(mode="lab_exact")
```

参数：

- `parameters`：物理参数；
- `rho0`：初始密度矩阵，默认通常为基态。

返回：

- `DynamicsResult`，`mode="lab_exact"`。

结果特点：

- 保留真实载波；
- 使用 Lab-frame field；
- 不使用 RWA drive。

### 7.2 `run_rwa_case(parameters, rho0=None, drive=None) -> DynamicsResult`

功能：运行 two-level RWA simulation。

流程：

```text
PhysicalParams
→ ParaNormalizer
→ build RWA drive
→ build_rwa_hamiltonian
→ build_c_ops
→ mesolve
→ DynamicsResult(mode="rwa")
```

参数：

- `parameters`：物理参数；
- `rho0`：初始密度矩阵；
- `drive`：可选 RWA drive。若不传，CW 情况通常构造 `ConstantDrive`。

返回：

- `DynamicsResult`，`mode="rwa"`。

结果特点：

- 使用有效 RWA Hamiltonian；
- 不包含 optical carrier；
- 实际进入 Hamiltonian 的是慢变量 `g(t)`。

### 7.3 `make_rotating_view(lab_result) -> DynamicsResult`

功能：从 Lab-frame result 派生 rotating-frame view。

流程：

```text
lab_result.states
→ rotating-frame unitary transform
→ DynamicsResult(mode="rotating_view", source_mode="lab_exact")
```

特点：

- 不调用 `mesolve`；
- 不使用 RWA Hamiltonian；
- 只是换表象观察同一条 Lab-frame trajectory。

## 8. `multilevel.py`

### 8.1 当前定位

`multilevel.py` 负责 multi-level lab-frame simulation。当前主要用途是：

- 验证 N-level density matrix 框架；
- 验证 N=2 multilevel 与 two-level lab-frame 一致；
- 为后续 multi-level physical normalization 和 multi-level RWA 做准备。

当前不做：

- multi-level physical normalization；
- multi-level RWA；
- automatic selected transitions；
- thermal redistribution。

### 8.2 `MultiLevelParameters`

常见字段：

| 字段 | 含义 |
|---|---|
| `energies` | N 个能级的 solver-ready energy / angular frequency |
| `dipole_matrix` | N x N 偶极矩矩阵，当前默认 solver-ready |
| `fields` | field 配置或 field 对象 |
| `collapse_channels` | collapse channel 配置 |
| `tlist` | 时间数组 |
| `rho0` | 初始密度矩阵 |

注意：当前这些输入不自动理解为 eV / Debye / MV/cm。

### 8.3 `simulate_multilevel_lab_frame(...)`

功能：运行 multi-level Lab-frame simulation。

需要检查：

- `dipole_matrix` 是否为 N x N；
- `rho0` 是否为 N x N；
- collapse operators 是否为 N x N；
- `tlist` 是否有效。

返回：

- `DynamicsResult(mode="multilevel_lab")`。

metadata 中应明确：

```text
multilevel_units = solver_ready_or_code_unit
```

## 9. `results.py`

`results.py` 是最重要的结果层，核心类是 `DynamicsResult`。

### 9.1 `DynamicsResult` 的职责

`DynamicsResult` 负责：

- 保存 density matrix trajectory；
- 保存 mode、times、states、metadata；
- 提供 dimension-aware 的通用访问接口；
- 生成 `density.npz`、`components.csv`、`populations.csv` 所需数据；
- 计算 trace 和 Hermiticity sanity check。

`DynamicsResult` 不负责：

- 求解微分方程；
- 绘图；
- 保存文件；
- 判断物理结论；
- 固定 two-level 的 `rho22/rho12` 解释。

### 9.2 `DynamicsResult` 保存字段详解

#### `mode`

类型：`str`

含义：当前 result 的来源或求解方式。

常见值：

| mode | 含义 |
|---|---|
| `lab_exact` | two-level Lab-frame exact simulation |
| `rotating_view` | 由 lab result 派生的 rotating-frame view |
| `rwa` | two-level RWA simulation |
| `multilevel_lab` | multi-level Lab-frame simulation |

#### `times`

类型：array-like

含义：solver 内部时间数组。

用途：

- 传给 drive callable 计算 `drive(t)`；
- 保存到 `density.npz`；
- debug。

注意：`times` 不一定永远等于真实 fs 时间。

#### `times_fs`

类型：array-like 或 `None`

含义：用户侧物理时间轴，单位 fs。

用途：

- plot 横坐标；
- `components.csv` 的 `time_fs`；
- `meta.json` 的 `time_range_fs`。

注意：multi-level 当前可能只是 solver-ready 时间，需要结合 `multilevel_units` 判断。

#### `states`

类型：list of density matrices，通常是 QuTiP `Qobj`

含义：每个时间点的 density matrix。

结构：

```text
states[k] = rho(t_k)
```

其中 `rho(t_k)` 是 N x N matrix。

#### `parameters`

类型：object 或 dict

含义：创建 result 时传入的原始参数或 solver 参数。

用途：debug 和追踪来源。

注意：人工检查实验条件优先看 `meta.json`，不要只看 `parameters`。

#### `physical_params`

类型：dict / object / `None`

含义：物理输入参数。

常见字段：

- `energy_gap_eV`
- `laser_energy_eV`
- `dipole_D`
- `field_MV_per_cm`
- `T1_fs`
- `Tphi_fs`
- `t_start_fs`
- `t_end_fs`
- `dt_fs`

注意：multi-level 当前可能没有真实 physical params。

#### `solver_params`

类型：dict / object

含义：solver 实际使用的内部参数。

可能包含：

- `detuning`
- `field_amplitude`
- `omega_drive`
- `gamma1`
- `gamma_phi`
- `gamma2`
- `tlist`

注意：这些多为 code unit，主要写入 `debug_meta.json`。

#### `metadata`

类型：dict

含义：额外信息。

用途：

- 保存 example_name；
- 保存 condition_name；
- 保存 case_name；
- 保存 source / derived info；
- 保存 sanity checks。

#### `source_mode`

类型：`str` 或 `None`

含义：如果 result 是派生结果，记录来源 mode。

例：

```text
rotating_view.source_mode = "lab_exact"
```

#### `drive_dict`

类型：dict 或 `None`

含义：RWA drive 的可重建字典。

主要用于 `rwa` result。

注意：

- 可能保存的是 code-unit amplitude；
- human-readable metadata 中应另有 `input_drive` 或 derived physical summary。

#### `drive_expr`

类型：str 或 `None`

含义：drive 的人类可读表达式。

注意：需要明确单位，不能默认认为是物理单位。

#### `drive_name`

类型：str 或 `None`

含义：drive 名称，用于 metadata 和 plot label。

### 9.3 主要方法详解

#### `density_array()`

功能：返回完整 density matrix trajectory。

返回形状：

```text
(n_time, N, N)
```

含义：

```text
density[k, i, j] = rho_ij(t_k)
```

用途：

- 保存 `density.npz`；
- 后续计算任意 observable；
- polarization；
- trace / purity / spectrum analysis。

#### `dimension()`

功能：返回 Hilbert space dimension。

返回：

```text
N
```

用途：

- 判断 two-level / multi-level；
- 自动导出所有 populations 和 coherences；
- metadata 中记录 dimension。

#### `populations()`

功能：返回所有主对角 population。

返回形状：

```text
(n_time, N)
```

含义：

```text
populations[k, i] = Re(rho_ii(t_k))
```

用途：

- `populations.csv`；
- `plot_populations()`；
- 结果摘要。

#### `matrix_element(i, j)`

功能：返回指定矩阵元随时间变化。

参数：

- `i`：行索引，zero-based；
- `j`：列索引，zero-based。

返回：

```text
complex array, shape = (n_time,)
```

例：

```text
matrix_element(0, 1) → rho_01(t)
```

#### `matrix_elements(pairs)`

功能：批量返回多个矩阵元。

参数：

```python
pairs = [(0, 1), (0, 2), (1, 2)]
```

返回：

```text
dict[(i, j)] = rho_ij(t)
```

#### `selected_elements(elements)`

功能：返回用户指定的矩阵元集合。

用途：

- 自定义分析；
- 高维体系中只导出部分 elements；
- selected plot。

#### `drive_values(times=None)`

功能：计算 drive 在指定时间点的值。

参数：

- `times`：如果为 `None`，默认使用 `result.times`。

返回：

- drive value array；
- 如果没有 drive，可能返回 `None`。

注意：evaluate drive 用 solver time；绘图横坐标用 `times_fs`。

#### `max_trace_error()`

功能：计算最大 trace error。

定义：

```text
max_t |Tr(rho(t)) - 1|
```

用途：检查数值是否保迹。

#### `max_hermiticity_error()`

功能：计算最大 Hermiticity error。

定义：

```text
max_t ||rho(t) - rho(t)^dagger||
```

用途：检查 density matrix 是否保持 Hermitian。

#### `summary_dict()`

功能：生成简短结果摘要。

常见内容：

- `mode`
- `dimension`
- final populations
- time range
- trace error
- Hermiticity error

用途：`meta.json` 和控制台输出。

#### `metadata_dict()`

功能：生成完整 debug metadata。

用途：写入 `debug_meta.json`。

内容可以包括：

- full parameters；
- full tlist；
- full times_fs；
- drive_dict；
- solver internals；
- sanity checks。

#### `to_npz_dict()`

功能：生成保存到 `density.npz` 的数组字典。

通常包含：

- `times`
- `times_fs`
- `density`

#### `components_dataframe()`

功能：生成 dimension-aware 的可读时间序列表。

输出逻辑：

1. 时间列：

```text
time_fs
```

2. 所有主对角元素：

```text
rho_00, rho_11, ..., rho_N-1_N-1
```

3. 所有上三角非对角 coherence：

对每个 `i < j`：

```text
Re_rho_ij
Im_rho_ij
abs_rho_ij
phase_rho_ij
phase_rho_ij_unwrapped
```

4. 可选输入列：

```text
drive_code
drive_fs_inv
field_MV_per_cm
```

注意：

- 使用 zero-based index；
- 只保存上三角 coherence，避免 Hermitian 冗余；
- phase 在 `abs_rho_ij` 很小时可设为 `NaN`；
- `phase_unwrapped` 用于观察连续相位演化。

#### `populations_dataframe()`

功能：生成 population-only dataframe。

输出：

```text
time_fs, rho_00, rho_11, ..., rho_N-1_N-1
```

#### `selected_elements_dataframe(elements)`

功能：生成用户指定 matrix elements 的 dataframe。

对角元素保存 `rho_ii`；非对角元素保存 Re / Im / abs / phase / unwrapped phase。

## 10. `plotting.py`

### 10.1 `plot_drive(result, ax=None, ...)`

功能：绘制 RWA drive。

- evaluate drive 用 `result.times`；
- x 轴用 `result.times_fs`；
- y 轴建议显示 `fs^-1`，不要未标注地显示 code unit。

### 10.2 `plot_field(field, times, ax=None, ...)`

功能：绘制 Lab-frame 电场。

- 用于 `E(t)`；
- y 轴单位通常是 `MV/cm`；
- 长时间窗口下 carrier 很密，必要时稀疏采样或 zoom。

### 10.3 `plot_populations(result, ax=None, ...)`

功能：绘制所有 populations。

默认画：

```text
rho_00, rho_11, ..., rho_N-1_N-1
```

支持任意 N。

### 10.4 `plot_coherences(result, ax=None, pairs=None, max_pairs=None, component="abs", ...)`

功能：绘制 coherence。

参数：

- `pairs`：指定 coherence pairs；
- `max_pairs`：最多画几个 pair；
- `component`：`abs`、`real`、`imag`、`phase` 等。

注意：高维体系 coherence 数量为 `N(N-1)/2`，可能需要限制显示数量。

### 10.5 `plot_density_components(result, axes=None, include_drive=False, ...)`

功能：组合绘制 input、population、coherence。

用于 two-level demo 或 RWA examples。

### 10.6 `plot_multilevel_components(result, axes=None, ...)`

功能：multi-level 简洁预览。

推荐三行：

1. populations；
2. coherence amplitudes；
3. coherence phases。

### 10.7 `build_preview_figure(result, ...)`

功能：为单个 result 生成低清 preview。

它只生成 figure，不保存。保存由 `io.py` 完成。

## 11. `io.py`

### 11.1 case 输出结构

```text
case_dir/
├─ data/
│  ├─ density.npz
│  ├─ components.csv
│  └─ populations.csv
├─ figs/
│  ├─ preview.png
│  └─ full.png
├─ meta.json
└─ debug_meta.json
```

### 11.2 `save_result_data(result, output_dir, save_npz=True, save_csv=True, save_json=True)`

功能：保存 result 的数字数据。

保存：

- `density.npz`：完整 density matrix trajectory；
- `components.csv`：dimension-aware 展开表；
- `populations.csv`：population-only 表；
- `meta.json`：简洁 metadata；
- `debug_meta.json`：完整 debug metadata。

### 11.3 `save_figure(fig, output_path, dpi=120)`

功能：保存 matplotlib figure。

### 11.4 `save_result_case(...)`

功能：保存一个完整 simulation case，包括 data、metadata 和 figures。

### 11.5 `QuantumResultIO`

功能：管理统一输出根目录，并提供 `save_case()` 等便捷接口。

### 11.6 `save_results_components_long(results, output_path, ...)`

功能：把多个 result 的 `components_dataframe()` 合并成长表，生成 comparison 数据。

用途：

- `comparison_components.csv`；
- 复现 comparison plot；
- pandas 分组分析。

## 12. Metadata 保存逻辑

每个 case 保存两份 metadata。

### 12.1 `meta.json`

用途：给人看。

特点：

- 简洁；
- 物理单位优先；
- 不保存完整 `tlist`；
- 不保存完整 `times_fs`；
- 不展开完整 code parameters。

推荐结构：

```json
{
  "result_type": "DynamicsResult",
  "example_name": "...",
  "condition_name": "...",
  "case_name": "...",
  "mode": "rwa",
  "source_mode": null,
  "inputs_physical": {},
  "derived_physical": {},
  "solver_code_summary": {},
  "trajectory_summary": {},
  "component_export": {},
  "multilevel_units": {},
  "output_files": {}
}
```

#### `inputs_physical`

保存用户输入的物理条件，例如：

- `energy_gap_eV`
- `laser_energy_eV`
- `detuning_eV`
- `wavelength_nm`
- `dipole_D`
- `field_MV_per_cm`
- `T1_fs`
- `Tphi_fs`
- `t_start_fs`
- `t_end_fs`
- `dt_fs`
- `pulse_center_fs`
- `pulse_sigma_fs`

#### `derived_physical`

保存派生物理量，例如：

- `omega_eg_fs_inv`
- `omega_L_fs_inv`
- `detuning_fs_inv`
- `rabi_fs_inv`
- `gamma1_fs_inv`
- `gamma_phi_fs_inv`
- `gamma2_fs_inv`
- `T2_effective_fs`
- `population_rabi_period_fs`

#### `solver_code_summary`

保存少量 code-unit 标量，例如：

- `time_scale_fs`
- `detuning_code`
- `field_amplitude_code`
- `omega_drive_code`
- `gamma1_code`
- `gamma_phi_code`
- `gamma2_code`
- `t_start_code`
- `t_end_code`
- `dt_code`

#### `trajectory_summary`

保存结果摘要，例如：

- `n_time_points`
- `time_range_fs`
- final populations
- selected final coherences
- `max_trace_error`
- `max_hermiticity_error`

#### `component_export`

说明 CSV 保存约定。

推荐内容：

```json
{
  "dimension": 3,
  "component_indexing": "zero_based",
  "saved_populations": "all diagonal elements",
  "saved_coherences": "upper triangular off-diagonal elements only",
  "coherence_components": ["real", "imag", "abs", "phase", "phase_unwrapped"],
  "phase_mask_threshold": 1e-8
}
```

### 12.2 `debug_meta.json`

用途：给程序调试用。

特点：

- 完整；
- 可以很长；
- 可以包含 code unit；
- 可以包含完整 `tlist` 和 `times_fs`。

常见内容：

- full `metadata_dict()`；
- full `parameters_code`；
- full `tlist`；
- full `times_fs`；
- `drive_dict`；
- `drive_expr`；
- `physical_params`；
- `solver_params_fs_inv`；
- `solver_params_code`；
- sanity checks。

## 13. CSV 保存逻辑

### 13.1 `density.npz`

完整保存：

```text
times
times_fs
density
```

这是后续重新计算任何 observable 的主要数据源。

### 13.2 `components.csv`

保存可读的展开结果：

```text
time_fs
rho_00, rho_11, ...
Re_rho_01, Im_rho_01, abs_rho_01, phase_rho_01, phase_rho_01_unwrapped
...
```

逻辑：

- 保存所有 diagonal populations；
- 保存所有 upper-triangular coherences；
- 非对角 coherence 不保存下三角，避免冗余；
- 使用 zero-based index。

### 13.3 `populations.csv`

只保存：

```text
time_fs, rho_00, rho_11, ..., rho_N-1_N-1
```

### 13.4 `comparison_components.csv`

保存 comparison plot 对应的长表数据。

通常额外包含：

- `example_name`
- `condition_name`
- `case_name`
- `mode`
- `label`

### 13.5 `results.csv`

每个 case 一行，保存摘要。

常见列：

- `example_name`
- `condition_name`
- `case_name`
- `field_MV_per_cm`
- `peak_E_MV_per_cm`
- `energy_gap_eV`
- `laser_energy_eV`
- `detuning_eV`
- `detuning_fs_inv`
- `rabi_fs_inv`
- `rabi_code`
- `T1_fs`
- `Tphi_fs`
- `gamma1_fs_inv`
- `gamma_phi_fs_inv`
- `gamma2_fs_inv`
- `max_rho_11`
- `final_rho_11`
- `max_abs_rho_01`
- `final_abs_rho_01`

## 14. `checks.py`

`checks.py` 放验证逻辑。

### 14.1 two-level helper

two-level helper 可以存在于 `checks.py` 或 `examples/rwa_common.py`，但不应放在 `DynamicsResult` 中。

原因：`DynamicsResult` 是 N-level 通用对象，不应知道 `rho_11` 是 excited population 或 `rho_01` 是 two-level coherence。

### 14.2 N=2 equivalence check

功能：比较 two-level lab-frame 和 N=2 multilevel lab-frame。

常见比较对象：

- `rho_00`
- `rho_11`
- `rho_01`
- `rho_10`
- full density matrix trajectory

用途：重构后的回归测试。

## 15. 顶层脚本与 examples

### 15.1 `optical_bloch_demo.py`

功能：生成 Lab-frame、rotating view、RWA 三列对比图。

典型流程：

```python
lab = run_lab_case(parameters)
rotating = make_rotating_view(lab)
rwa = run_rwa_case(parameters)
```

图像结构：

- 行：input field / drive、population、coherence；
- 列：Lab-frame、Rotating view、RWA。

### 15.2 `multilevel_demo.py`

功能：运行 N-level lab-frame 示例，并验证 dimension-aware export。

应检查：

- `components.csv` 是否包含所有 diagonal 和 upper-triangular coherence；
- `meta.json` 是否有 `dimension` 和 `component_export`；
- `multilevel_units` 是否明确说明当前输入是 solver-ready。

### 15.3 RWA examples

共有：

- `rwa_01_field_strength.py`：场强影响；
- `rwa_02_dephasing.py`：pure dephasing；
- `rwa_03_redistribution.py`：T1 relaxation / redistribution；
- `rwa_04_dephasing_and_redistribution.py`：两种弛豫共同作用。

共同约定：

- 每个 simulation case 是一个 `DynamicsResult`；
- 每个 case 保存 data、metadata、preview；
- 每个 example 保存 comparison plot、comparison data 和 results.csv；
- two-level observables 由 `rwa_common.py` 的 example-level helper 提取；
- 不使用 `DynamicsResult.components()`。

## 16. Multi-level compatibility notes

已经完成：

- `components_dataframe()` 支持任意 N；
- `populations_dataframe()` 支持任意 N；
- `plot_populations()` 支持任意 N；
- `plot_coherences()` 支持任意 N；
- multi-level preview 支持 populations / coherence abs / coherence phase；
- `meta.json` 记录 `component_export`；
- `multilevel_lab` metadata 标注 solver-ready/code-unit inputs。

尚未实现：

- multi-level physical normalization；
- multi-level RWA；
- selected transition automation；
- thermal redistribution；
- response-function workflow。

## 17. 后续 TODO

### 17.1 以 Hamm 第二章为主线

下一步不只是继续堆功能，而是以 Hamm 第二章为理论主线，将 density matrix dynamics 与光谱信号联系起来。

### 17.2 Gaussian pulse

从 CW excitation 扩展到 Gaussian pulse。

重点：

- Lab-frame：`GaussianCarrierField`；
- RWA：`GaussianDrive`；
- 观察 pulse width、pulse area、peak field 对 population 和 coherence 的影响。

### 17.3 Polarization

新增：

```text
P(t) = Tr[mu rho(t)]
```

对于 two-level，`P(t)` 主要由 `rho_01` 和 `rho_10` 决定。

### 17.4 Absorption-like response

可行路线：

- 扫描 `laser_energy_eV` 或 detuning；
- 求 long-time 或 steady-state response；
- 提取 `Im[P/E]`、`Im[rho_01]` 或类似吸收响应；
- 比较 dephasing 和 T1 relaxation 对谱线宽度、峰值和线型的影响。

### 17.5 2DES 前置基础

当前程序不能直接生成标准 2DES，但可以作为学习 2DES 的基础层。

后续若进入 2DES，需要新增：

- multi-pulse sequence；
- coherence time / waiting time / detection time；
- third-order response；
- Liouville pathways；
- phase matching 或 phase cycling；
- Fourier transform along t1 and t3。

## 18. 维护规则

1. `DynamicsResult` 保持 dimension-aware，不再加入 two-level-only helper。
2. two-level analysis 放在 example/check 层。
3. `density.npz` 保存完整数据，`components.csv` 保存可读摘要。
4. `meta.json` 给人看，`debug_meta.json` 给调试用。
5. 新增任何字段必须写清单位。
6. 新增 field / drive 类型时，必须实现 `__call__()`、`to_dict()`、`from_dict()` 和 `to_expr()`。
7. 新增 multi-level 功能时，必须说明是否支持 physical normalization。
8. 每次重构后至少运行：

```text
conda --no-plugins run -n quantum python -m compileall sjh_learn
conda --no-plugins run -n quantum python sjh_learn\multilevel_demo.py
conda --no-plugins run -n quantum python sjh_learn\n2_equivalence_check.py
conda --no-plugins run -n quantum python sjh_learn\optical_bloch_demo.py
conda --no-plugins run -n quantum python sjh_learn\examples\rwa_01_field_strength.py
conda --no-plugins run -n quantum python sjh_learn\examples\rwa_02_dephasing.py
conda --no-plugins run -n quantum python sjh_learn\examples\rwa_03_redistribution.py
conda --no-plugins run -n quantum python sjh_learn\examples\rwa_04_dephasing_and_redistribution.py
```
