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

`utils/fields/` is split by physical meaning and unit boundary:

- `utils/fields/lab_fields.py`: user-facing lab-frame physical fields, including `CarrierFieldPhysical`, `GaussianCarrierFieldPhysical`, and `CompositeLabFieldPhysical`.
- `utils/fields/rwa_drives.py`: user-facing physical RWA slow drives / couplings, including `ConstantRwaDrivePhysical`, `GaussianRwaDrivePhysical`, and `make_rwa_drive_from_physical_field()`.
- `utils/fields/solver_inputs.py`: solver-internal code-unit callables such as `CodeCarrierField`, `CodeConstantDrive`, and `CodeGaussianDrive`.
- `utils/fields/__init__.py`: public re-exports. Backward-compatible names such as `CarrierField` and `ConstantDrive` now point to physical user-facing classes; solver internals import `Code*` classes explicitly.

User-facing field/drive classes use physical units only. `CarrierFieldPhysical.__call__(t_fs)` returns `E(t)` in `MV/cm`, and `ConstantRwaDrivePhysical.__call__(t_fs)` returns `g(t)` in `fs^-1`. Ordinary examples should set `PhysicalParams` or physical field/drive objects; `amplitude_code`, `time_unit="code"`, and `domain="solver_code"` belong only to `ParaNormalizer`, solver/model internals, `debug_meta.json`, or `solver_code_summary`.

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

Current `redistribution` is intentionally simplified to excited-to-ground T1 relaxation in the RWA examples. Bidirectional redistribution and thermal redistribution are future extensions, and upward transitions are not implemented in this round. All RWA examples run only `run_rwa_case`, each simulation case is one `DynamicsResult`, and the input drive is saved in metadata, preview figures, and `comparison_components.csv`.

`field_MV_per_cm` is the physical input field amplitude. The Rabi frequency is obtained from `mu E / hbar`. In RWA plots, the first row defaults to `Omega(t)` in `fs^-1`. In lab-frame plots, the first row defaults to the physical electric field `E(t)` in `MV/cm`; if code-unit diagnostics are shown instead, they are labeled explicitly as code units.

Each case writes two metadata files. `meta.json` is a short human-readable summary with `example_name`, `condition_name`, `case_name`, physical inputs, derived physical rates, a compact code-unit summary, trajectory summary, and output-file paths. `debug_meta.json` keeps the full raw `DynamicsResult.metadata_dict()` payload, including full code parameters, `tlist`, `times_fs`, drive metadata, solver internals, and sanity checks.

## Unit Conventions

- `PhysicalParams` uses physical units such as `time_fs`, `field_MV_per_cm`, `rabi_fs_inv`, and `gamma_fs_inv`.
- `ParaNormalizer` converts those physical units into solver code units for internal time, frequency, drive, and decay rates.
- Solver and model code are allowed to use code units internally.
- User-facing field/drive classes use physical units only: `E0_MV_per_cm`, `laser_energy_eV` or `omega_L_fs_inv`, `phase_rad`, `pulse_center_fs`, `pulse_sigma_fs`, and RWA `amplitude_fs_inv`.
- Plotting, CSV export, and example summaries default to physical units when available.
- Code-unit outputs are kept only as metadata or clearly labeled diagnostic fields such as `drive_code`.
- Density matrix, populations, and coherences are dimensionless.

## Multi-Level Result Export

- `components.csv` is dimension-aware. It saves all diagonal density-matrix elements as populations (`rho_00`, `rho_11`, ...).
- Upper-triangular off-diagonal elements are saved as coherences with `Re_rho_ij`, `Im_rho_ij`, `abs_rho_ij`, `phase_rho_ij`, and `phase_rho_ij_unwrapped` columns, using zero-based indices.
- `populations.csv` saves every diagonal population, and `density.npz` always contains the full density-matrix trajectory.
- `DynamicsResult` is a dimension-aware result object and does not provide a two-level-only `components()` helper.
- Two-level demos and RWA examples remain intentionally two-level specific where they compare `rho_11` and `rho_01`; that extraction now lives in example-level helpers.
- Multi-level physical normalization is not implemented yet; the current multi-level path assumes solver-ready/code-unit inputs.
