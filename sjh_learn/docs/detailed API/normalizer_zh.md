# ParaNormalizer 当前说明

`ParaNormalizer` 把 `NLevelPhysicalParams` 转换成 solver 内部使用的 `SolverParams`
和 code-unit field adapter。它只做单位换算和 code-unit scaling，不定义物理模型，
不构造 Hamiltonian，不构造 collapse operator，也不做 spectroscopy 后处理。

## 输入和输出

用户侧输入使用真实物理单位：

| 物理量 | 用户单位 | normalizer 中间单位 | code unit |
| --- | --- | --- | --- |
| time | `fs` | `fs` | `t_code = t_fs / time_scale_fs` |
| energy | `eV` | `fs^-1` | `omega_code = omega_fs_inv * time_scale_fs` |
| dipole | `Debye` | 与 field 合成 coupling | `coupling_code = coupling_fs_inv * time_scale_fs` |
| field | `MV/cm` | `E_phys(t)` | `E_code(t) = E_phys(t) / E_ref` |
| relaxation rate | `fs^-1` 或 `T1_fs` | `fs^-1` | `rate_code = rate_fs_inv * time_scale_fs` |
| pure dephasing rate | `fs^-1` 或 `Tphi_fs` | `fs^-1` | `rate_code = rate_fs_inv * time_scale_fs` |

`NLevelSolverParams` 和 `_CodeFieldAdapter` 是内部对象，普通 example 不应直接构造。

## Field Reference

`ParaNormalizer.field_reference_MV_per_cm(field)` 只读取：

```python
field.reference_MV_per_cm
```

它不从 `field.to_dict()` 读取 `E0_MV_per_cm`，也不根据具体 field 类型分支。

同一个 `reference_MV_per_cm` 同时用于：

```text
coupling_matrix_fs_inv = dipole_matrix_D * reference_MV_per_cm * DIPOLE_FIELD_TO_RABI_FS_INV
E_code(t) = E_phys_MV_per_cm(t) / reference_MV_per_cm
```

这两者必须一致。因为 lab-frame Hamiltonian 中实际进入的是：

```text
H_int(t) = - coupling_matrix_code * E_code(t)
```

若 coupling matrix 使用一个 reference，而 field adapter 使用另一个 reference，
则 `mu * E(t)` 的物理幅度会错误。

如果 `reference_MV_per_cm is None` 或为 0，normalizer 会直接报错。自定义 field
如果要进入 solver 主线，必须提供非零 `reference_MV_per_cm`。

## rate_candidates 和 auto_scale

`rate_candidates` 是 auto-scale 使用的数值尺度候选，单位 `fs^-1`。当前来源包括：

- `coupling_matrix_fs_inv` 中非零元素的模长；
- relaxation channels 的 `rate_fs_inv`；
- pure dephasing channels 的 `rate_fs_inv`；
- `field.normalization_rate_candidates_fs_inv` 提供的 field-specific candidates。

如果用户显式设置 `time_scale_fs`，normalizer 直接使用该值。

如果 `auto_scale=True` 且没有显式 `time_scale_fs`，normalizer 选择：

```text
time_scale_fs = 1 / max(rate_candidates)
```

若没有正的 candidates，则退回到 `energies_fs_inv` 中的非零能量尺度；仍没有则使用
`time_scale_fs = 1.0`。

## auto_scale 为什么不改变物理结果

code unit 只是重新选择数值单位。例如物理演化中出现：

```text
omega_fs_inv * t_fs
```

归一化后变成：

```text
(omega_fs_inv * time_scale_fs) * (t_fs / time_scale_fs)
```

两者相同。relaxation、dephasing 和 coupling 也用同一个 `time_scale_fs` 一致缩放，
所以 auto-scale 不改变真实物理演化，只改变 solver 内部数值量级。

## omega_L_fs_inv 和 detuning_fs_inv

`omega_L_fs_inv` 不再是 normalizer 主路径要求。lab_exact 可以使用没有单一 carrier
frequency 的 field，例如 multi-pulse series 或自定义 broadband field。

`omega_L_fs_inv` 和 `detuning_fs_inv` 可以作为 optional metadata / debug 字段出现：

- 单频 field 的 `to_dict()` 可以记录 `omega_L_fs_inv` 和 `laser_energy_eV`。
- `transition_table` 可以在 field 有单频 metadata 时写出对应 detuning。
- `debug_meta.json` 可以保留 optional `omega_L_fs_inv=None` 或
  `detuning_fs_inv=None`。

不要为了 metadata 恢复 normalizer 对单一 carrier frequency 的依赖。

## dt_fs warning 现状

当前 normalizer 会检查 `dt_fs > 0`、`t_end_fs > t_start_fs`，并构造 code-unit
`tlist`。如果需要更严格的采样质量提示，应基于当前 `time_scale_fs`、field-specific
速率候选和用户的 `dt_fs` 增加显式 warning；不要把它和 Hamiltonian 物理公式耦合。

## 禁止的 normalizer 方向

不要在 normalizer 中按具体 field 类型分支，例如：

- `isinstance(field, GaussianCarrierFieldPhysical)`
- `isinstance(field, CarrierFieldPhysical)`
- `isinstance(field, FieldPhySeries)`
- `isinstance(field, TAField)`
- `isinstance(field, TwoDESField)`

field-specific 数值信息应通过 `FieldPhyRoot.reference_MV_per_cm` 和
`FieldPhyRoot.normalization_rate_candidates_fs_inv` 暴露。metadata 只用于输出和重建，
不作为 core 数值接口。
