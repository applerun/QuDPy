# Field 物理电场接口说明

## 1. Field 是什么？

`FieldPhyRoot` 是 QuDPy 用户侧物理电场的基类。它描述真实 lab-frame
电场，时间单位固定为 `fs`，电场单位固定为 `MV/cm`。

用户侧应直接构造物理 field，例如：

```python
field = make_default_gaussian_carrier_field(
    E0_MV_per_cm=0.1,
    laser_energy_eV=1.625,
    pulse_center_fs=0.0,
    pulse_sigma_fs=5.0,
)
```

然后传入：

```python
params = NLevelPhysicalParams(
    energies_eV=...,
    dipole_matrix_D=...,
    field=field,
    ...
)
```

`NLevelPhysicalParams` 不再保存 `field_MV_per_cm`、`laser_energy_eV`、
`pulse_center_fs`、`pulse_sigma_fs` 等顶层光场标量。

## 2. FieldPhyRoot 的基本要求

自定义 field 至少需要实现：

```python
def physical_E_MV_per_cm(self, t_fs: np.ndarray) -> np.ndarray:
    ...
```

要求：

- 输入 `t_fs` 是 numpy array，单位 fs；
- 返回值必须和 `t_fs` shape 相同；
- 返回值是真实 lab-frame 电场，单位 MV/cm；
- 不要返回 code-unit field；
- 不要在 field 中处理 density matrix 或 solver 参数。

## 3. Normalizer 如何使用 field？

`ParaNormalizer` 只使用 `FieldPhyRoot` 基类暴露的通用接口：

- `field(t_fs)` / `physical_E_MV_per_cm(t_fs)`；
- `field.to_dict()`；
- `field.normalization_rate_candidates_fs_inv`。

Normalizer 不判断 field 是 CW、Gaussian、FieldSeries、TAField、TwoDESField
还是用户自定义类。不要在 normalizer 中添加针对具体 field 类型的分支。

## 4. normalization_rate_candidates_fs_inv

`normalization_rate_candidates_fs_inv` 是 field 给 normalizer 的 auto-scale
建议，单位为 `fs^-1`。它只用于数值单位缩放，不改变物理模型。

默认实现返回：

```python
()
```

如果 field 有明确的快时间尺度，建议覆盖该 property。例如 Gaussian pulse
可以返回：

```python
(1.0 / sigma_fs,)
```

自定义 field 可以根据以下量提供候选：

- 脉冲宽度对应的 bandwidth，例如 `1 / pulse_width_fs`；
- 包络调制频率；
- 重复频率；
- 其它会影响数值变化速度的 envelope time scale。

不要把该 property 当作物理参数。它只是帮助 normalizer 选择
`time_scale_fs`。

## 5. to_dict 的作用

`to_dict()` 用于 metadata、debug metadata 和可选 rebuild 信息。建议包含：

- `class`
- `name`
- `time_unit`
- `field_unit`
- `rebuildable`
- `E0_MV_per_cm` 或 reference field amplitude
- `omega_L_fs_inv`，如果当前 solver 需要单一 carrier frequency
- field-specific parameters
- `expression`
- `metadata`

对自定义 field，如果不能从 metadata 恢复，应设置：

```python
"rebuildable": False
```

## 6. 内置 Field

### CarrierFieldPhysical

CW lab-frame carrier：

```text
E(t) = 2 E0 cos(omega_L t + phase)
```

`phase_rad` 是 carrier / optical phase。

### GaussianCarrierFieldPhysical

Gaussian envelope lab-frame carrier：

```text
E(t) = 2 E0 exp[-(t-center)^2/(2 sigma^2)] cos(omega_L t + phase)
```

`phase_rad` 仍然是该 pulse 的 carrier / optical phase。

这里不定义独立的 envelope phase。若后续需要 complex envelope，应新增专门
field class，而不是在 real lab-frame Gaussian field 中加入过于特化的参数。

## 7. 多脉冲 FieldSeries

TA 和 2DES 中有多个 pulse。它们应作为 physical field 层的组合对象表达：

```python
field = make_ta_gaussian_field(...)
field = make_twodes_gaussian_field(...)
```

FieldSeries 本身仍然是 `FieldPhyRoot`，因此可直接传入
`NLevelPhysicalParams(..., field=field)`。

Normalizer 不解析 TA / 2DES 细节。它只看到一个可调用的 physical field 和一组
auto-scale candidates。

## 8. 自定义 Field 示例

```python
class MyField(FieldPhyCustomed):
    def __init__(self, amplitude, modulation_frequency):
        self.amplitude = amplitude
        self.modulation_frequency = modulation_frequency

    def physical_E_MV_per_cm(self, t_fs):
        return self.amplitude * np.sin(self.modulation_frequency * t_fs)

    @property
    def normalization_rate_candidates_fs_inv(self):
        return (abs(self.modulation_frequency),)

    def __repr__(self):
        return (
            "MyField("
            f"amplitude={self.amplitude!r}, "
            f"modulation_frequency={self.modulation_frequency!r})"
        )

    def to_dict(self):
        return {
            "class": self.__class__.__name__,
            "repr": repr(self),
            "time_unit": "fs",
            "field_unit": "MV/cm",
            "rebuildable": False,
            "amplitude": self.amplitude,
            "modulation_frequency": self.modulation_frequency,
        }
```

## 9. 设计边界

Field 负责描述外加物理电场。它不负责：

- Hamiltonian 构造；
- Lindblad channel 构造；
- density matrix 后处理；
- spectroscopy pathway 管理；
- absorption / 2DES 数据分析；
- plotting style。

这些功能应分别放在 core、spectroscopy、analysis 或 UFANSYS 中。
