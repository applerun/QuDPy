# QuDPy 当前架构说明

本文档记录当前 field / normalizer / solver API 的正式边界。当前 QuDPy
主线是 lab-frame / exact solver；RWA 只保留为 legacy diagnostic 说明和
后处理辅助，默认不可调用。

## 总体主线

```text
FieldPhyRoot / FieldPhySeries
    -> NLevelPhysicalParams(..., field=field)
    -> ParaNormalizer
    -> _CodeFieldAdapter
    -> NLevelSolverParams
    -> lab_exact solver
    -> DynamicsResult
    -> io.py / plotting.py / spectroscopy postprocessing
```

用户侧只构造 physical field。solver/model 层只接收一个 code-unit callable：

```python
args["field"]
```

多脉冲求和必须在 physical field 层完成，例如用 `FieldPhySeries`、`TAField`
或 `TwoDESField`。solver 层不接受 `args["fields"]`，也不接受多个 code-unit
field callable。

## 职责边界

`core/` 只负责参数容器、单位归一化、Hamiltonian / collapse operator 构造、
`mesolve` 调用和 `DynamicsResult`。参数扫描、batch run、delay scan 和 power
scan 不属于 core。

`fields/` 只负责用户侧 physical field。时间单位是 `fs`，电场单位是
`MV/cm`。内部 code-unit field 只能由 `ParaNormalizer.make_code_field()` 生成。

`examples/` 和上层 workflow 负责定义 case、循环扫描参数、调用 `run_case()`、
保存结果和构造具体 figure。

`spectroscopy/` 负责从 density matrix trajectory、输入场和 polarization 派生
谱学后处理量，例如 dipole expectation、polarization、FFT response 和 analytic
linear theory comparison。

`plotting.py` 只提供 QuDPy 自身的物理绘图 helper，不负责实验数据分析。

## 推荐入口

普通模拟应显式构造 field，再传入 `NLevelPhysicalParams`：

```python
field = make_default_gaussian_carrier_field(
    E0_MV_per_cm=0.05,
    laser_energy_eV=1.625,
    pulse_center_fs=0.0,
    pulse_sigma_fs=8.0,
)

params = NLevelPhysicalParams(
    energies_eV=(0.0, 1.625),
    dipole_matrix_D=((0.0, 3.0), (3.0, 0.0)),
    t_start_fs=-50.0,
    t_end_fs=80.0,
    dt_fs=0.05,
    field=field,
)

result = run_case(params)
```

参数扫描应放在上层脚本中，用普通 `for` loop 显式构造每个 case：

```python
results = []
for scan_params, field in iter_ta_gaussian_fields(...):
    params = NLevelPhysicalParams(
        energies_eV=...,
        dipole_matrix_D=...,
        t_start_fs=...,
        t_end_fs=...,
        dt_fs=...,
        field=field,
    )
    results.append((scan_params, run_case(params)))
```

core 不再提供 `ParameterSweep` 抽象。

## Field 和 normalizer 边界

`FieldPhyRoot.reference_MV_per_cm` 是 core 数值接口，不是 metadata。它同时用于：

```text
coupling_matrix_fs_inv = mu_D * reference_MV_per_cm * constant
E_code(t) = E_phys(t) / reference_MV_per_cm
```

这两处必须使用同一个 reference，否则 Hamiltonian 中的 `mu * E(t)` 会被错误缩放。

`FieldPhyRoot.normalization_rate_candidates_fs_inv` 是 field 给 auto-scale 的速率候选，
单位 `fs^-1`。它可以返回多个候选，例如 Gaussian envelope bandwidth、调制频率
或重复频率。它只影响 code-unit 数值尺度，不改变物理场。

`ParaNormalizer` 不从 `field.to_dict()` 读取核心数值接口，也不根据
`GaussianCarrierFieldPhysical`、`CarrierFieldPhysical` 或 `FieldPhySeries` 等具体类型
分支。field-specific 信息必须通过 `FieldPhyRoot` 的正式 property 暴露。

`omega_L_fs_inv` 和 `detuning_fs_inv` 可作为 field metadata、transition table 或
debug metadata 中的 optional 字段出现，但 lab_exact 主路径不要求 field 有单一
carrier frequency，normalizer 也不再用单一 carrier detuning 作为主路径尺度。

## RWA 状态

RWA solver-unit drive path 已删除或禁用。默认 `FORCE_RWA=False`。如果用户尝试运行：

```python
NLevelPhysicalParams(..., solver_mode="rwa")
```

会在进入求解前失败，并提示 RWA 是 legacy path。不要自动 fallback 到 lab_exact。

`utils/spectroscopy/rwa.py` 只保留 legacy diagnostic 后处理函数，用于已有 RWA-like
trajectory 的比较诊断；它不应成为 core 主线依赖。

## 已删除的 legacy API

以下接口不再属于当前 API，也不应在新代码中恢复：

- `args["fields"]`
- `CodeCarrierField`
- `CodeGaussianCarrierField`
- `CodeCompositeField`
- `CodeConstantDrive`
- `CodeGaussianDrive`
- `solver_input_from_dict`
- `fields/solver_inputs`
- `fields/legacy_solver_inputs`
- `lab_fields_code`
- `rwa_drives_code`
- RWA solver-unit drive path
- `PhysicalParameterSweep`
- `run_physical_parameter_sweep`
- `ParameterSweep`
- `run_parameter_sweep`

## Breaking Changes

- `NLevelPhysicalParams` 不再保存顶层 `field_MV_per_cm`、`laser_energy_eV`、
  `pulse_center_fs` 或 `pulse_sigma_fs`；这些信息来自 field 对象。
- solver/model 层不再接收多个 code-unit field callable。
- normalizer 不再要求单一 `omega_L_fs_inv`。
- 参数扫描从 core 移到 examples / workflow。
- RWA solver-unit drive 默认不可用。

## 不应恢复的旧路径

不要重新引入 solver-unit field class 作为用户 API。不要让 normalizer 从
`field.to_dict()` 猜 core 数值量。不要在 solver 层对多个 field 求和。不要把
参数扫描抽象放回 core。不要用 RWA 作为默认或自动 fallback 路径。
