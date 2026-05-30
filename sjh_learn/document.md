# sjh_learn 架构记录

本文档记录当前 `sjh_learn` 的代码职责边界。当前阶段的目标是保持 two-level / multilevel 已有物理约定不变，同时把求解、结果、绘图和 IO 收口到清晰的单轨迹结构。

## 目录

```text
sjh_learn/
├─ optical_bloch_demo.py
├─ multilevel_demo.py
├─ n2_equivalence_check.py
├─ examples/
│  ├─ rwa_common.py
│  └─ rwa_01_field_strength.py
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

## 核心约定

1. 标准结果对象是 `DynamicsResult`。
2. 一个 `DynamicsResult` 只表示一条 density-matrix trajectory。
3. `lab_exact` 和 `rwa` 是独立 simulation case。
4. `rotating_view` 是 `lab_exact` 的派生视图，不调用 `mesolve`。
5. solver 一次只返回一个 result。
6. plotting 只返回 matplotlib `fig/axes`，不默认保存。
7. 顶层脚本负责组合多个 result 和保存最终图。
8. `io.py` 只保存数字数据、metadata 和已经构造好的 figure。

## fields.py

输入场和 RWA drive 都是 callable class，不使用 `eval`。

主要类：

- `ConstantDrive`: `Omega(t) = Omega0`
- `GaussianDrive`: `Omega(t) = Omega0 exp[-(t - t0)^2 / (2 sigma^2)]`
- `CarrierField`: `E(t) = 2A cos(omega t + phase)`
- `GaussianCarrierField`: 高斯包络载波场
- `CompositeField`: 多个 field 相加

每个对象支持 `__call__(t)`、`to_dict()`、`from_dict()`、`to_expr()`。`to_dict()` 用于可靠保存和重建，`to_expr()` 只用于人类可读日志。

## solvers.py

two-level 求解入口：

- `run_lab_case(parameters, rho0=None) -> DynamicsResult`
- `run_rwa_case(parameters, rho0=None, drive=None) -> DynamicsResult`
- `make_rotating_view(lab_result) -> DynamicsResult`

`run_lab_case` 使用 `build_lab_hamiltonian + build_c_ops + mesolve`。`run_rwa_case` 使用 callable `drive` 构造 time-dependent RWA Hamiltonian；CW 情况默认构造 `ConstantDrive`，pulse 预留 `GaussianDrive`。`make_rotating_view` 只对 `lab_result.states` 做幺正变换。

RWA 中实际进入 Hamiltonian 的是慢变量耦合 `Omega(t)`，不是 lab frame 的快速 optical carrier。若从真实物理参数出发，`field_MV_per_cm + dipole_D` 仍通过 `ParaNormalizer` 转换为 `rabi_fs_inv / rabi_code`。

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

`meta.json` 包含 mode、`physical_params`、`solver_params_fs_inv`、`solver_params_code`、`drive_dict`、`drive_expr`、trace error 和 Hermiticity error。`components.csv` 对 two-level result 输出 `time_fs`、`rho11`、`rho22`、`Re_rho12`、`Im_rho12`、`abs_rho12`，并在可用时输出 `drive_code`、`drive_fs_inv` 和 `field_MV_per_cm`。

density matrix、population、coherence 都是无量纲量；时间轴优先保存真实 fs。

## 顶层脚本

`optical_bloch_demo.py` 由脚本显式决定要跑哪些 case：

```python
lab = run_lab_case(parameters)
rotating = make_rotating_view(lab)
rwa = run_rwa_case(parameters)
```

脚本创建 3x3 对比图，列为 Lab frame、Rotating view、RWA，行为 input drive、population、coherence。最终图像由脚本调用 `save_figure()` 保存。

`multilevel_demo.py` 仍只运行 exact lab-frame multilevel case。

`n2_equivalence_check.py` 继续用于检查 two-level lab frame 与 N=2 multilevel lab frame 的一致性。

## RWA Example

`examples/rwa_01_field_strength.py` 是第一个 RWA-only example：

- detuning = 0
- no T1
- no Tphi
- only RWA
- `field_MV_per_cm = [0.1, 0.2, 0.5, 1.0]`

每个 case 保存：

- `data/density.npz`
- `data/components.csv`
- `meta.json`
- `figs/preview.png`

总图 `comparison.png` 至少包含 `Omega(t)`、`rho22(t)`、`abs_rho12(t)`。`results.csv` 记录 `field_MV_per_cm`、`rabi_fs_inv`、`rabi_code`、`max_rho22`、`final_rho22`、`max_abs_rho12`、`final_abs_rho12`。

预期物理结果：场强越大，Rabi frequency 越大，`rho22` 振荡周期越短；无 relaxation / dephasing 时振荡不应衰减；共振且强度足够时 `rho22` 可接近 1。

## RWA Examples

- `rwa_01_field_strength.py`: field strength controls Rabi frequency.
- `rwa_02_dephasing.py`: pure dephasing damps coherence and Rabi oscillations without adding T1 population relaxation.
- `rwa_03_redistribution.py`: T1 relaxation / redistribution damps excited-state population.
- `rwa_04_dephasing_and_redistribution.py`: combined dephasing and redistribution.

Current `redistribution` is intentionally simplified to excited-to-ground T1 relaxation in the RWA examples. Bidirectional redistribution and thermal redistribution are future extensions, and upward transitions are not implemented in this round. All RWA examples run only `run_rwa_case`, each simulation case is one `DynamicsResult`, and the input drive is saved in metadata, preview figures, and `comparison_components.csv`.

`field_MV_per_cm` is the physical input field amplitude. The Rabi frequency is obtained from `mu E / hbar`. In RWA plots, the first row defaults to `Omega(t)` in `fs^-1`. In lab-frame plots, the first row defaults to the physical electric field `E(t)` in `MV/cm`; if code-unit diagnostics are shown instead, they are labeled explicitly as code units.

## Unit Conventions

- `PhysicalParams` uses physical units such as `time_fs`, `field_MV_per_cm`, `rabi_fs_inv`, and `gamma_fs_inv`.
- `ParaNormalizer` converts those physical units into solver code units for internal time, frequency, drive, and decay rates.
- Solver and model code are allowed to use code units internally.
- Plotting, CSV export, and example summaries default to physical units when available.
- Code-unit outputs are kept only as metadata or clearly labeled diagnostic fields such as `drive_code`.
- Density matrix, populations, and coherences are dimensionless.
