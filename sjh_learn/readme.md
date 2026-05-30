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

`utils/fields.py` 现在使用 callable class 描述输入场和 RWA 慢变量 drive，不使用 `eval` 作为重建机制。

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
