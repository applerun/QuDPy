# sjh_learn 架构记录

本文档记录当前 `sjh_learn` 的代码职责边界。当前阶段的目标是保持 two-level / multilevel 已有物理约定不变，同时把求解、结果、绘图和 IO 收口到清晰的单轨迹结构。

## 目录

```text
sjh_learn/
├─ bin/
│  ├─ multilevel_demo.py
│  └─ n2_equivalence_check.py
├─ examples/
│  └─ absorption/
│     ├─ absorption_01_chi_analytic.py
│     ├─ cw_pulse_absorption_compare.py
│     └─ three_level_absorption_lab_exact.py
├─ scratch/
│  └─ validate_*.py
└─ utils/
   ├─ fields/
   │  ├─ __init__.py
   │  ├─ field_series.py
   │  ├─ lab_fields.py
   ├─ model.py
   ├─ solvers.py
   ├─ results.py
   ├─ plotting.py
   ├─ io.py
   ├─ checks.py
   ├─ normalization.py
   └─ parameters.py
```

## 核心约定

1. 标准结果对象是 `DynamicsResult`。
2. 一个 `DynamicsResult` 只表示一条 density-matrix trajectory。
3. `lab_exact` 和 `rwa` 是独立 simulation case。
4. `rotating_view` 是 `lab_exact` 的派生视图，不调用 `mesolve`。
5. solver 一次只返回一个 result。
6. plotting 只返回 matplotlib `fig/axes`，不默认保存。
7. 顶层脚本负责组合多个 result 和保存最终图。
8. `io.py` 只保存数字数据、metadata 和已经构造好的 figure。

## fields/

`utils/fields/` 按物理含义和单位边界拆分：

- `lab_fields.py` 保存用户侧 lab-frame physical field：`CarrierFieldPhysical` 和 `GaussianCarrierFieldPhysical`。
- `field_series.py` 保存 physical-layer 多场组合：`FieldPhySeries`、`TAField` 和 `TwoDESField`。
- `__init__.py` re-export 当前 physical field API。

用户侧 field class 只使用物理单位。`CarrierFieldPhysical.__call__(t_fs)` 返回单位为 `MV/cm` 的 `E(t)`。solver 内部 code-unit callable 只能由 `ParaNormalizer.make_code_field()` 从 physical field 生成；旧 solver-unit field / drive 类已经删除。

The physical lab-frame field and the RWA drive are not the same object:

```text
E(t) = 2 E0 f(t) cos(omega_L t + phase)
g(t) = mu E0 f(t) / hbar
```

RWA 中保留的是 slow drive / coupling `g(t)`，不是 optical carrier。

输入场是 physical-layer callable class，不使用 `eval`。

主要类：

- `CarrierFieldPhysical`: `E(t_fs) = 2 E0 cos(omega_L t_fs + phase)`
- `GaussianCarrierFieldPhysical`: 高斯包络载波场
- `FieldPhySeries`: physical-layer 多个 field 相加
- `TAField` / `TwoDESField`: TA / 2DES 常用多脉冲 field

每个对象支持 `__call__(t_fs)`、`to_dict()` 和 `rebuild(...)`。solver code-unit callable 只由 `ParaNormalizer.make_code_field()` 从 physical field 生成。

## solvers.py

two-level 求解入口：

- `run_lab_case(parameters, rho0=None) -> DynamicsResult`
- `run_rwa_case(parameters, rho0=None, drive=None) -> DynamicsResult`
- `make_rotating_view(lab_result) -> DynamicsResult`

`run_lab_case` 使用 `build_lab_hamiltonian + build_c_ops + mesolve`。`run_rwa_case` 使用 callable `drive` 构造 time-dependent RWA Hamiltonian；CW 情况默认构造 `ConstantDrive`，pulse 预留 `GaussianDrive`。`make_rotating_view` 只对 `lab_result.states` 做幺正变换。

RWA 中实际进入 Hamiltonian 的是慢变量耦合 `Omega(t)`，不是 lab frame 的快速 optical carrier。若从真实物理参数出发，`dipole_matrix_D` 与 `field_MV_per_cm` 会通过 `ParaNormalizer` 转换为物理耦合矩阵和内部 solver code-unit 耦合矩阵。

## results.py

`DynamicsResult` 保存：

- `mode`
- `times`
- `times_fs`
- `states`
- `parameters`
- `physical_params`
- `solver_params`
- `metadata`
- `source_mode`
- `drive_dict`
- `drive_expr`
- `drive_name`

主要方法：

- `density_array()`
- `dimension()`
- `populations()`
- `matrix_element(i, j)`
- `matrix_elements(pairs)`
- `selected_elements(elements)`
- `drive_values(times=None)`
- `max_trace_error()`
- `max_hermiticity_error()`
- `summary_dict()`
- `metadata_dict()`
- `to_npz_dict()`
- `components_dataframe()`
- `populations_dataframe()`
- `selected_elements_dataframe(elements)`

`DynamicsResult` 不提供 `plot()` 或 `save_*()` 方法，避免 result 层依赖 matplotlib 或文件系统。

## plotting.py

绘图函数：

- `plot_drive(result, ax=None, ...)`
- `plot_field(field, times, ax=None, ...)`
- `plot_populations(result, ax=None, ...)`
- `plot_coherences(result, ax=None, ...)`
- `plot_density_components(result, axes=None, include_drive=False, ...)`
- `plot_multilevel_components(result, axes=None, ...)`
- `build_preview_figure(result, ...)`

如果传入 `ax/axes`，函数画到传入坐标轴；否则内部创建 figure。所有函数返回 matplotlib 句柄，不直接保存文件。

`build_preview_figure(result)` 默认生成 3 行低清预览内容：

1. input drive / field
2. population
3. coherence

RWA result 第一行画 `Omega(t)`；lab result 第一行画 `E(t)`，会按需要稀疏采样，避免载波振荡过密。

## io.py

IO 函数：

- `save_result_data(result, output_dir, save_npz=True, save_csv=True, save_json=True)`
- `save_figure(fig, output_path, dpi=120)`
- `save_result_case(result, output_dir, output_data=True, output_preview=False, preview_fig=None, full_fig=None, ...)`
- `QuantumResultIO`

case 输出结构：

```text
case_dir/
├─ data/
│  ├─ density.npz
│  ├─ components.csv
│  └─ populations.csv
├─ figs/
│  ├─ preview.png
│  └─ full.png
└─ meta.json
```

`meta.json` 包含 mode、`physical_params`、`solver_params_fs_inv`、`solver_params_code`、`drive_dict`、`drive_expr`、trace error 和 Hermiticity error。`components.csv` 现在是 dimension-aware：输出 `time_fs`、所有 diagonal populations（例如 `rho_00`）、所有 upper-triangular coherences（例如 `Re_rho_01`、`Im_rho_01`、`abs_rho_01`、`phase_rho_01`），并在可用时输出 `drive_code`、`drive_fs_inv`、`field_MV_per_cm`。dipole expectation、polarization、FFT response 和吸收功相关量只由 analysis 层输出。

density matrix、population、coherence 都是无量纲量；时间轴优先保存真实 fs。

## Spectroscopy Observables

第一层谱学 observable 放在 `utils/analysis/observables.py`，属于 analysis 层；它们从 `DynamicsResult` 的 density matrix trajectory 派生，不改变 solver/result 架构。

- `dipole_expectation_D(rho_t, dipole_matrix_D)`：计算 `p(t)=Tr[rho(t) mu]`，单位 Debye。
- `polarization_C_per_m2(rho_t, dipole_matrix_D, number_density_m3)`：计算可选宏观 polarization，单位 `C/m^2`；`number_density_m3` 的单位是 `m^-3`。
- `chi_two_level_linear(...)`：two-level analytic linear susceptibility，用作 linear-response reference。

矩阵指标约定为 `Tr(rho mu)=sum_ij rho_ij mu_ji`，实现使用 `np.einsum("tij,ji->t", rho_t, mu)`。observable 必须使用用户侧物理偶极矩 `dipole_matrix_D`，不能使用 `coupling_matrix_code`，因为后者已经包含光场和 code-unit 归一化。

simulation 输出的 `components.csv` 只保存 density matrix、population、coherence 以及输入 drive/field 相关列；dipole expectation、polarization、FFT response 和吸收功相关量只在 analysis 层输出，例如 `analysis_components.csv` 和 `fft_response.csv`。

## 顶层脚本

`bin/multilevel_demo.py` 使用 `NLevelPhysicalParams` 构造普通 N=3 physical system，并显式构造 field 对象后进入统一 N-level mainline。

`bin/n2_equivalence_check.py` 继续保留验证意义，但不再依赖旧的 solver-ready multilevel route；它检查 N=2 physical mainline 与显式 `ParaNormalizer -> solver` 路径的一致性。

`examples/absorption/` 放稳定 absorption 示例；`scratch/` 只放临时验证脚本。RWA 路径默认禁用，legacy 对比分支只能显式开启后使用。
- only RWA
- `field_MV_per_cm = [0.1, 0.2, 0.5, 1.0]`

每个 case 保存：

- `data/density.npz`
- `data/components.csv`
- `meta.json`
- `figs/preview.png`

总图 `comparison.png` 至少包含 `Omega(t)`、`rho_11(t)`、`abs_rho_01(t)`。`results.csv` 记录 `field_MV_per_cm`、`rabi_fs_inv`、`rabi_code`、`max_rho_11`、`final_rho_11`、`max_abs_rho_01`、`final_abs_rho_01`。

预期物理结果：场强越大，Rabi frequency 越大，excited-state population `rho_11` 振荡周期越短；无 relaxation / dephasing 时振荡不应衰减；共振且强度足够时 `rho_11` 可接近 1。

## RWA Examples

- `rwa_01_field_strength.py`: field strength controls Rabi frequency.
- `rwa_02_dephasing.py`: pure dephasing damps coherence and Rabi oscillations without adding T1 population relaxation.
- `rwa_03_redistribution.py`: T1 relaxation / redistribution damps excited-state population.
- `rwa_04_dephasing_and_redistribution.py`: combined dephasing and redistribution.

CW-input examples are grouped under `examples/cw_input/`. Gaussian femtosecond-pulse examples are grouped under `examples/gau_pulse/`:

- `pulse_01_T1_Tphi_dependence.py`: fixed Gaussian pulse, then scan Tphi, T1, and the four dissipation scenarios.
- `pulse_02_width_dependence.py`: scan Gaussian pulse width under free, dephasing, and redistribution scenarios.
- `pulse_03_field_strength_dependence.py`: scan field strength under free, dephasing, and redistribution scenarios.

The Gaussian pulse examples are still RWA-only. They use physical `pulse_center_fs` and `pulse_sigma_fs`, and the RWA Hamiltonian receives the slow Gaussian coupling `g(t) = mu E0 exp[-(t - t0)^2 / (2 sigma^2)] / hbar`. The solver sets `mesolve` `max_step = dt` so narrow time-dependent pulses are resolved by the integrator.

Current `redistribution` is intentionally simplified to excited-to-ground T1 relaxation in the RWA examples. Bidirectional redistribution and thermal redistribution are future extensions, and upward transitions are not implemented in this round. All RWA examples run only `run_rwa_case`, each simulation case is one `DynamicsResult`, and the input drive is saved in metadata, preview figures, and `comparison_components.csv`.

`field_MV_per_cm` is the physical input field amplitude. The Rabi frequency is obtained from `mu E / hbar`. In RWA plots, the first row defaults to `Omega(t)` in `fs^-1`. In lab-frame plots, the first row defaults to the physical electric field `E(t)` in `MV/cm`; if code-unit diagnostics are shown instead, they are labeled explicitly as code units.

Each case writes two metadata files. `meta.json` is a short human-readable summary with `example_name`, `condition_name`, `case_name`, physical N-level inputs, physical field/drive information, derived physical rates, trajectory summary, and output-file paths. `debug_meta.json` keeps the full raw `DynamicsResult.metadata_dict()` payload, including full code parameters, `tlist`, `times_fs`, code-unit drive metadata, solver internals, and sanity checks.

`rwa_02_dephasing.py`, `rwa_03_redistribution.py`, and `rwa_04_dephasing_and_redistribution.py` now each generate three condition groups: `resonant_strong`, `resonant_weak`, and `detuned_weak`. The `field_MV_per_cm = 0.1` cases are meant to show weak-drive dynamics, while `detuned_weak` shows how off-resonant driving changes population transfer and coherence response.

RWA comparison plots now contain four rows: `Omega(t)`, `rho_11(t)`, `abs_rho_01(t)`, and `phase(rho_01)`. The phase trace is unwrapped, and low-amplitude regions with very small `abs(rho_01)` are masked with `NaN` because the phase there is not numerically meaningful. RWA example comparisons also use colormap gradients rather than matplotlib's default color cycle.

## Unit Conventions

- `NLevelPhysicalParams` 是当前用户侧标准物理系统对象。two-level system 只是 `N=2` 的普通 N-level system，multilevel system 是普通 `N>2` system，不再由核心层的 `dipole_D`、`T1_fs`、`Tphi_fs`、`T2_fs` 标量字段表示。
- `NLevelSolverParams` 是内部 solver 参数容器，保存 N-level matrices、channel lists 和 code-unit 时间/频率；普通示例不直接构造它。
- `basis` 可选保存能级名称；`energies_eV` 保存长度为 N 的能级列表；`dipole_matrix_D` 保存沿选定 optical polarization 投影后的 N x N 偶极矩矩阵，单位为 Debye。
- `relaxation_channels` 定义 population relaxation：`C_{to <- from} = sqrt(rate) |to><from|`，每个 channel 可用 `T1_fs` 或 `rate_fs_inv` 指定速率。
- `pure_dephasing_channels` 定义 level projector pure dephasing：`C_level^phi = sqrt(rate) |level><level|`，每个 channel 可用 `Tphi_fs` 或 `rate_fs_inv` 指定速率。
- 普通 N=2 示例如果需要“`dipole_D` / `T1_fs` / `Tphi_fs`”这样的教学参数，会在 example-local helper 中翻译成 `dipole_matrix_D` 和 channel list；这些不再是 core model API。
- `ParaNormalizer` converts those physical units into solver code units for internal time, frequency, drive, and decay rates.
- Solver and model code are allowed to use code units internally.
- User-facing field/drive classes use physical units only: `E0_MV_per_cm`, `laser_energy_eV` or `omega_L_fs_inv`, `phase_rad`, `pulse_center_fs`, `pulse_sigma_fs`, and RWA `amplitude_fs_inv`.
- Plotting, CSV export, and example summaries default to physical units when available.
- Code-unit outputs are kept only as metadata or clearly labeled diagnostic fields such as `drive_code`.
- Density matrix, populations, and coherences are dimensionless.

## Dynamics Analysis

- `DynamicsResult.save_ckp(path)` 保存一次模拟的完整 checkpoint；`DynamicsResult.from_ckp(path)` 从 `.ckp` 文件恢复 result。`.ckp` 是内部 checkpoint / 后处理缓存，不保证跨版本长期稳定，不应加载不可信来源文件，也不是替代 `density.npz`、`components.csv`、`meta.json`、`debug_meta.json` 的归档格式。
- `DynamicsAnalysis.from_dynamics_res(result)` 从内存 result 创建分析对象；`DynamicsAnalysis.from_ckp(path)` 从 checkpoint 创建分析对象。
- analysis 层只依赖 `DynamicsResult` API，不调用 solver 内部 code-unit 输入，也不改变 solver/result 主架构。
- analysis 默认使用通用 N-level polarization：`P(t)=number_density_m3 * Tr[rho(t) mu_D] * DEBYE_TO_C_M`，其中 `mu_D` 来自 `physical_params.dipole_matrix_D`，`number_density_m3` 必须显式传入。
- `Tr[rho(t) mu_D]` 的实现使用 `np.einsum("tij,ji->t", rho_t, dipole_matrix_D)`；若物理偶极矩期望值的虚部超过容差，analysis 会直接报错，不会静默丢弃。
- 常见 two-level 0-1 近似公式 `P(t)=2 n mu_01 Re[rho_01(t)]` 只作为文档说明保留；analysis API 不提供 two-level 专用 polarization 函数，避免误用于 N-level 体系。
- `rho_over_E = fft_rho12 / fft_E` 是 coherence response-like quantity，不是 `chi` 或 absorption。`P_over_E = P_fft / fft_E` 和 `omega_Im_P_over_E` 更接近 polarization response / 吸收功方向的分析量，但仍依赖 Fourier convention 和线性响应条件。
- `analysis_components.csv` 是增强时域表，保留 dimension-aware density columns，并增加 dipole / polarization 列；`fft_response.csv` 保存 `frequency_fs_inv`、`angular_frequency_fs_inv`、`energy_eV`、`fft_rho12`、`fft_E`、`rho_over_E`、`P_fft`、`P_over_E` 和 `omega_Im_P_over_E`。
- `analysis_metadata.json` 记录 `example_name`、`case_name`、`input_field_class`、`physical_params`、`solver_params_fs_inv`、分析参数和输出文件路径。

## Multi-Level Compatibility Notes

- `components_dataframe()` and `components.csv` are dimension-aware. They save all diagonal populations as `rho_00`, `rho_11`, ... and all upper-triangular coherences as `rho_ij` components.
- Each coherence exports real, imaginary, absolute value, masked phase, and unwrapped masked phase columns.
- `populations.csv` contains all diagonal elements; `density.npz` contains the complete density-matrix trajectory.
- `DynamicsResult` no longer exposes a two-level-only `components()` helper; result access should use `populations()`, `matrix_element()`, `matrix_elements()`, or dataframe exporters.
- The two-level RWA examples still use two-level-specific summaries and plots by design; their `rho_11` / `rho_01` extraction is kept in example-level helpers, not in the generic result export.
- 官方 multilevel route 是 `NLevelPhysicalParams -> ParaNormalizer -> NLevelSolverParams -> model.py -> solvers.py -> DynamicsResult`。two-level system 是普通 `N=2`，multilevel system 是普通 `N>2`；旧 solver-ready multilevel 路径已移除，普通示例不再直接构造 code-unit multilevel 输入。
