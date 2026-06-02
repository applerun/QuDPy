# sjh_learn

`sjh_learn` 是当前用来学习 two-level optical Bloch dynamics 的小型 QuTiP 项目。现在的主线已经收口到单轨迹架构：一次求解只产生一个 `DynamicsResult`，顶层脚本负责决定要跑哪些 case、怎样拼图、怎样保存最终图像。

## 当前架构

- 标准结果对象是 `DynamicsResult`。
- 一个 `DynamicsResult` 只表示一条 density-matrix trajectory。
- `lab_exact` 和 `rwa` 是两个独立 simulation case。
- `rotating_view` 不是独立求解，而是由 `lab_exact` 结果通过幺正变换派生。
- solver 层一次只返回一个 result，不同时运行 lab / rotating / RWA。
- result 层只保存数值轨迹和 metadata，不依赖 matplotlib。
- plotting 层只画到 matplotlib `fig/axes` 上并返回句柄，不默认保存文件。
- 顶层脚本负责组合多个 result、排版、加总标题、保存最终图。
- `io.py` 只负责保存 result 数字数据、metadata 和已经构造好的 figure。

## Field / Drive

`utils/fields/` 按物理含义和单位边界拆分：

- `utils/fields/lab_fields.py`：用户侧 lab-frame physical field，包括 `CarrierFieldPhysical`、`GaussianCarrierFieldPhysical` 和 `CompositeLabFieldPhysical`。
- `utils/fields/rwa_drives.py`：用户侧 physical RWA slow drive / coupling，包括 `ConstantRwaDrivePhysical`、`GaussianRwaDrivePhysical` 和 `make_rwa_drive_from_physical_field()`。
- `utils/fields/solver_inputs.py`：solver 内部 code-unit callable，例如 `CodeCarrierField`、`CodeConstantDrive` 和 `CodeGaussianDrive`。
- `utils/fields/__init__.py`：公共 re-export。兼容名称 `CarrierField`、`ConstantDrive` 指向用户侧 physical class；solver 内部显式导入 `Code*` class。

用户侧 field/drive class 只使用物理单位。`CarrierFieldPhysical.__call__(t_fs)` 返回单位为 `MV/cm` 的 `E(t)`，`ConstantRwaDrivePhysical.__call__(t_fs)` 返回单位为 `fs^-1` 的 `g(t)`。普通示例应设置 `NLevelPhysicalParams` 或 physical field/drive object；`amplitude_code`、`time_unit="code"` 和 `domain="solver_code"` 只属于 `ParaNormalizer`、solver/model 内部或 `debug_meta.json`。

The lab-frame field and RWA drive are intentionally different objects:

```text
E(t) = 2 E0 f(t) cos(omega_L t + phase)
g(t) = mu E0 f(t) / hbar
```

RWA 中保留的是 slow drive / coupling `g(t)`，不是 optical carrier。

`utils/fields/` 现在使用 callable class 描述输入场和 RWA 慢变量 drive，不使用 `eval` 作为重建机制。

已支持：

- `ConstantDrive`: RWA 中的 CW drive，`Omega(t) = Omega0`。
- `GaussianDrive`: RWA 中的高斯 envelope。
- `CarrierField`: lab frame 载波场，`E(t) = 2A cos(omega t + phase)`。
- `GaussianCarrierField`: lab frame 高斯包络载波场。
- `CompositeField`: 多个 field 相加。

每个 field/drive 支持：

- `__call__(t)`: 接受 scalar 或 numpy array。
- `to_dict()` / `from_dict()`: 用于可靠保存和重建。
- `to_expr()` / `__repr__()`: 用于人类可读日志。

RWA 中真正进入 Hamiltonian 的是慢变量耦合 `Omega(t)`，不是快速振荡 optical carrier。CW RWA drive 在 preview 中是一条水平线；pulse RWA drive 在 preview 中显示 envelope。

## 结果和单位

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
- input drive/field 的 `to_dict()` 和 `to_expr()`

density matrix、population、coherence 都是无量纲量。CSV 中主时间轴保存为真实 `time_fs`；NPZ 中同时保存 `time_fs` 和内部求解用的 `time_code`，方便回溯归一化。

## Spectroscopy Observables

`utils/analysis/observables.py` 提供第一层谱学 observable，属于 analysis 层：

- `dipole_expectation_D(rho_t, dipole_matrix_D)` 计算 `p(t)=Tr[rho(t) mu]`，单位 Debye。
- `polarization_C_per_m2(rho_t, dipole_matrix_D, number_density_m3)` 计算 `P(t)=N p(t)`，单位 `C/m^2`，其中 `number_density_m3` 的单位是 `m^-3`。
- `chi_two_level_linear(...)` 给出 two-level analytic linear-response susceptibility，用作后续数值谱学结果的参考。

迹的指标约定是 `Tr(rho mu)=sum_ij rho_ij mu_ji`，代码使用 `np.einsum("tij,ji->t", rho_t, mu)`。这里必须使用物理输入 `dipole_matrix_D`，不能使用已经包含光场强度和 code-unit 归一化的 `coupling_matrix_code`。

`DynamicsResult.components_dataframe()` 和 simulation 输出的 `components.csv` 只保存 density matrix、population、coherence 以及输入 drive/field 相关列；dipole expectation、polarization、FFT response 和吸收功相关量只在 analysis 层输出，例如 `analysis_components.csv` 和 `fft_response.csv`。

## 绘图和 IO

`utils/plotting.py` 提供低层绘图函数：

- `plot_drive(result, ax=None, ...)`
- `plot_field(field, times, ax=None, ...)`
- `plot_populations(result, ax=None, ...)`
- `plot_coherences(result, ax=None, ...)`
- `plot_density_components(result, axes=None, include_drive=False, ...)`
- `plot_multilevel_components(result, axes=None, ...)`
- `build_preview_figure(result, ...)`

所有绘图函数只返回 `fig, ax/axes`，不保存 PNG。低清 preview 由 `plotting.py` 构建，由 `io.py` 保存。

`utils/io.py` 提供：

- `save_result_data(result, output_dir, save_npz=True, save_csv=True, save_json=True)`
- `save_figure(fig, output_path, dpi=120)`
- `save_result_case(result, output_dir, output_data=True, output_preview=False, ...)`
- `QuantumResultIO`

默认 case 输出结构：

```text
outdir/
├─ results.csv
└─ res_per_case/
   └─ <case_name>/
      ├─ data/
      │  ├─ density.npz
      │  ├─ components.csv
      │  └─ populations.csv
      ├─ figs/
      │  ├─ preview.png
      │  └─ full.png
      └─ meta.json
```

完整数据和预览图可以分别开关。`output_preview=True` 时，如果没有传入 `preview_fig`，IO 层会调用 `build_preview_figure(result)` 生成低清预览图；result 对象本身不画图。

## Demo 和 Example

`optical_bloch_demo.py` 显式执行：

```python
lab = run_lab_case(parameters)
rotating = make_rotating_view(lab)
rwa = run_rwa_case(parameters)
```

然后由顶层脚本创建 3x3 comparison figure：第一行 input drive / field，第二行 population，第三行 coherence。

`examples/rwa_01_field_strength.py` 是第一个 RWA-only example。它只运行 RWA，不计算 lab frame，用不同 `field_MV_per_cm` 验证场强越大、Rabi 振荡越快。每个 case 保存 `density.npz`、`components.csv`、`meta.json` 和低清 `preview.png`，总图保存为 `comparison.png`。

`rwa_02_dephasing.py`、`rwa_03_redistribution.py` 和 `rwa_04_dephasing_and_redistribution.py` 现在都包含三组条件：`resonant_strong`、`resonant_weak`、`detuned_weak`。其中 `field_MV_per_cm = 0.1` 用于观察弱驱动下的动力学，`detuned_weak` 用于观察非共振条件下 population transfer 和 coherence response 的变化。每个 condition group 都会保存自己的 `comparison.png`、`comparison_components.csv` 和 `results.csv`。

RWA comparison 图现在包含四行：`Omega(t)`、`\rho_{11}(t)`、`|\rho_{01}(t)|` 和 `phase(\rho_{01})`。当 `abs(rho_01)` 很小时，相位会通过 NaN mask 和 unwrap 处理来避免无意义的跳变。RWA examples 的 comparison 曲线使用 colormap 渐变色，而不是 matplotlib 默认颜色循环。

## 运行检查

```powershell
conda --no-plugins run -n quantum python -m compileall sjh_learn
conda --no-plugins run -n quantum python sjh_learn\n2_equivalence_check.py
conda --no-plugins run -n quantum python sjh_learn\multilevel_demo.py
conda --no-plugins run -n quantum python sjh_learn\optical_bloch_demo.py
conda --no-plugins run -n quantum python sjh_learn\examples\rwa_01_field_strength.py
```

当前阶段不引入 UFSS、多能级 RWA、复杂真实脉冲或吸收光谱；重点是 two-level RWA 基础 example、input drive preview 和结果输出整理。

## RWA Examples

- `rwa_01_field_strength.py`: field strength controls Rabi frequency.
- `rwa_02_dephasing.py`: pure dephasing damps coherence and Rabi oscillations without adding T1 population relaxation.
- `rwa_03_redistribution.py`: T1 relaxation / redistribution damps excited-state population.
- `rwa_04_dephasing_and_redistribution.py`: combined dephasing and redistribution.

CW-input examples live under `examples/cw_input/`. Gaussian femtosecond-pulse examples live under `examples/gau_pulse/`:

- `pulse_01_T1_Tphi_dependence.py`: fixed Gaussian pulse, then scan Tphi, T1, and the four dissipation scenarios.
- `pulse_02_width_dependence.py`: scan Gaussian pulse width under free, dephasing, and redistribution scenarios.
- `pulse_03_field_strength_dependence.py`: scan field strength under free, dephasing, and redistribution scenarios.

The Gaussian pulse examples remain RWA-only. They use `pulse_center_fs` and `pulse_sigma_fs` in `NLevelPhysicalParams`; the RWA Hamiltonian receives the slow Gaussian coupling `g(t) = mu E0 exp[-(t - t0)^2 / (2 sigma^2)] / hbar`. The solver constrains `mesolve` with `max_step = dt` so narrow time-dependent pulses are not skipped by adaptive stepping.

Current `redistribution` is intentionally simplified to excited-to-ground T1 relaxation in the RWA examples. Bidirectional redistribution and thermal redistribution are future extensions, and upward transitions are not implemented in this round. All RWA examples run only `run_rwa_case`, each simulation case is one `DynamicsResult`, and the input drive is saved in metadata, preview figures, and `comparison_components.csv`.

`field_MV_per_cm` is the physical input field amplitude. The Rabi frequency is obtained from `mu E / hbar`. In RWA plots, the first row defaults to `Omega(t)` in `fs^-1`. In lab-frame plots, the first row defaults to the physical electric field `E(t)` in `MV/cm`; if code-unit diagnostics are shown instead, they are labeled explicitly as code units.

Each case writes two metadata files. `meta.json` is a short human-readable summary with `example_name`, `condition_name`, `case_name`, physical N-level inputs, physical field/drive information, derived physical rates, trajectory summary, and output-file paths. `debug_meta.json` keeps the full raw `DynamicsResult.metadata_dict()` payload, including full code parameters, `tlist`, `times_fs`, code-unit drive metadata, solver internals, and sanity checks.

## Unit Conventions

- 现在用户侧标准物理系统对象是 `NLevelPhysicalParams`。two-level system 不再是核心层的特殊标量模型，而是 `N=2` 的普通 N-level system；multilevel system 也是普通 `N>2` system。
- `dipole_matrix_D` 是沿选定 optical polarization 投影后的偶极矩矩阵，单位是 Debye。光场幅度仍用 `field_MV_per_cm`，归一化时会由 `ParaNormalizer` 转换为 coupling matrix。
- population relaxation 由 `relaxation_channels` 定义，每个通道表示 `C_{to <- from} = sqrt(rate) |to><from|`。通道可用 `T1_fs` 或 `rate_fs_inv` 指定速率。
- pure dephasing 由 `pure_dephasing_channels` 定义，每个通道表示 `C_level^phi = sqrt(rate) |level><level|`。通道可用 `Tphi_fs` 或 `rate_fs_inv` 指定速率。
- 标量 `dipole_D`、`T1_fs`、`Tphi_fs`、`T2_fs` 不再是核心物理模型输入；如果某个 N=2 example 需要这些概念，会在 example-local helper 中把它们翻译成 `dipole_matrix_D` 和 channel list。
- `NLevelPhysicalParams` 使用真实物理单位，例如 `energies_eV`、`dipole_matrix_D`、`field_MV_per_cm`、`time_fs` 和 `fs^-1` 速率。
- `NLevelSolverParams` 是内部 solver 参数容器，保存 N-level matrices、channel lists 和 code-unit 时间/频率；普通用户示例不直接构造它。
- `ParaNormalizer` 把这些物理单位转换为 solver 内部使用的 code unit，包括时间、频率、drive 和 decay rate。
- `model.py` 构造 N-level 的 `H0`、`H_int(t)` 和 Lindblad `c_ops`；`solvers.py` 每次只返回一条 `DynamicsResult` 轨迹。
- 用户侧 field/drive 类只使用物理单位：`E0_MV_per_cm`、`laser_energy_eV` 或 `omega_L_fs_inv`、`phase_rad`、`pulse_center_fs`、`pulse_sigma_fs`，以及 RWA `amplitude_fs_inv`。
- Plotting, CSV export, and example summaries default to physical units when available.
- Code-unit outputs are kept only as metadata or clearly labeled diagnostic fields such as `drive_code`.
- Density matrix, populations, and coherences are dimensionless.

## Dynamics Analysis

- `DynamicsResult.save_ckp(path)` 可以把一次模拟得到的完整 result 保存为 `.ckp` checkpoint；`DynamicsResult.from_ckp(path)` 可以从 checkpoint 重新加载 result。`.ckp` 是内部 checkpoint / 后处理缓存，不保证跨版本长期稳定，不应加载不可信来源文件，也不是替代 `density.npz`、`components.csv`、`meta.json`、`debug_meta.json` 的归档格式。
- `utils/analysis/DynamicsAnalysis` 是 analysis 层对象，可由 `DynamicsAnalysis.from_dynamics_res(result)` 或 `DynamicsAnalysis.from_ckp(path)` 创建。
- analysis 层只读取 `DynamicsResult` 的公开 API，不调用 solver，也不使用 solver-internal code-unit 输入。
- analysis 默认使用通用 N-level polarization：`P(t)=number_density_m3 * Tr[rho(t) mu_D] * DEBYE_TO_C_M`，其中 `mu_D` 来自 `physical_params.dipole_matrix_D`，`number_density_m3` 必须显式传入。
- `Tr[rho(t) mu_D]` 的实现使用 `np.einsum("tij,ji->t", rho_t, dipole_matrix_D)`；若物理偶极矩期望值的虚部超过容差，analysis 会直接报错，不会静默丢弃。
- 常见 two-level 0-1 近似公式 `P(t)=2 n mu_01 Re[rho_01(t)]` 只作为文档说明保留；analysis API 不提供 two-level 专用 polarization 函数，避免误用于 N-level 体系。
- `rho_over_E = fft_rho12 / fft_E` 是 coherence response-like quantity，不是 `chi` 或 absorption。`P_over_E = P_fft / fft_E` 和 `omega_Im_P_over_E` 更接近 polarization response / 吸收功方向的分析量，但仍依赖 Fourier convention 和线性响应条件。
- 频域 CSV 同时保存 `frequency_fs_inv`、`angular_frequency_fs_inv` 和 `energy_eV`；时域 CSV 保存为 `analysis_components.csv`，避免和 simulation output 的 `components.csv` 混淆。
- 默认分析输出目录是 `outputs/analysis/<example_name>/<case_name>/`，包含 `analysis_components.csv`、`fft_response.csv`、`figs/polarization_time.png`、`figs/fft_response.png` 和 `analysis_metadata.json`。

## Multi-Level Result Export

- `components.csv` is dimension-aware. It saves all diagonal density-matrix elements as populations (`rho_00`, `rho_11`, ...).
- Upper-triangular off-diagonal elements are saved as coherences with `Re_rho_ij`, `Im_rho_ij`, `abs_rho_ij`, `phase_rho_ij`, and `phase_rho_ij_unwrapped` columns, using zero-based indices.
- `populations.csv` saves every diagonal population, and `density.npz` always contains the full density-matrix trajectory.
- `DynamicsResult` is a dimension-aware result object and does not provide a two-level-only `components()` helper.
- Two-level demos and RWA examples remain intentionally two-level specific where they compare `rho_11` and `rho_01`; that extraction now lives in example-level helpers.
- 官方 multilevel 路线已经统一为 `NLevelPhysicalParams`：two-level system 是普通 `N=2`，multilevel system 是普通 `N>2`。旧的 solver-ready multilevel 路径已移除，普通示例不再直接构造 code-unit multilevel 输入。
