# 长延迟 TA 的分段传播与暗场窗口识别说明

## 1. 背景问题

在当前 QuDPy 的 `lab_exact` 主线中，光场以真实 lab-frame electric field 的形式进入哈密顿量：

\[
H(t)=H_0-\mu E(t)
\]

这里的 \(E(t)\) 显式保留 optical carrier。因此，只要光场不为零，时间步长 `dt_fs` 就必须足够小，以解析载波振荡。以 1.55 eV 光场为例，光学周期约为：

\[
T_\mathrm{opt}\approx \frac{4.1357}{1.55}\ \mathrm{fs}\approx 2.67\ \mathrm{fs}
\]

如果希望每个周期至少有 10–20 个采样点，那么 `dt_fs=0.1–0.2 fs` 是比较合理的范围。这个要求对于 pump/probe interaction window 是必要的；但对于长延迟 TA，如果 delay 达到 5000 fs 或 10000 fs，pump 和 probe 之间的大部分时间并没有有效光场。若仍然用 `dt_fs=0.2 fs` 对整个 pump-probe 时间跨度做统一传播，就会产生大量没有必要的时间点和 density matrix 输出。

因此，长延迟 TA 的主要数值问题不是“把 `dt` 调大”，而是应该避免在无光场区域使用 optical-scale dense time grid。

## 2. 分段传播的基本思想

分段传播（piecewise propagation）的核心思想是：一个物理 delay case 仍然对应一个实验条件，但在数值传播时将其拆成几个连续 segment。后一段的初始 density matrix 由前一段的最终 density matrix 给出。

对于正 delay 的 pump-probe case，可以写成：

```text
pump segment -> dark segment -> probe/readout segment
```

对应的 density matrix 传递为：

```text
rho_initial
  -> pump segment
  -> rho_after_pump
  -> dark segment
  -> rho_before_probe
  -> probe/readout segment
  -> probe-window trajectory
```

这里需要强调：这些 segment 不应该被理解为多个独立实验 case。一个 delay 仍然是一个 case；pump、dark、probe 只是该 case 内部的连续传播阶段。

## 3. 哪些区间仍然必须使用 lab-frame full propagation

只要电场不可以忽略，就应继续使用完整 lab-frame Hamiltonian：

\[
H(t)=H_0-\mu E(t)
\]

这些区间包括：

1. pump interaction window；
2. probe interaction window；
3. pump 和 probe 有时间重叠的区域；
4. 少周期脉冲或 shaped pulse 中电场尾部仍然不可忽略的区域；
5. 任何自动检测认为 \(|E(t)|\) 高于 dark threshold 的区域。

这些区间不能用 RWA 替代，也不应使用粗时间步长。原因是 pump 和 probe 可能不是同一个载波，脉冲可能是少周期脉冲，后续 2DES 还可能涉及多脉冲相位、延迟和频率组合。为了保持物理定义清楚，QuDPy 当前主线应继续保留 lab-frame carrier。

## 4. 暗传播的物理含义

暗场窗口（dark window）是指光场幅值在整个区间内都足够小，可以近似认为：

\[
E(t)\approx 0
\]

此时哈密顿量退化为 field-free Hamiltonian：

\[
H_\mathrm{dark}=H_0
\]

如果同时包含 relaxation 和 pure dephasing，density matrix 满足 Lindblad master equation：

\[
\frac{d\rho}{dt}
=
-i[H_0,\rho]
+
\sum_k\mathcal{D}[c_k]\rho
\]

其中

\[
\mathcal{D}[c_k]\rho
=
c_k\rho c_k^\dagger
-
\frac{1}{2}
\left(c_k^\dagger c_k\rho+\rho c_k^\dagger c_k\right)
\]

由于 \(H_0\) 和 collapse operators 在 dark window 内是 time-independent 的，dark evolution 可以形式上写成：

\[
\rho(t+\Delta t)
=
\exp(\mathcal{L}_\mathrm{dark}\Delta t)\rho(t)
\]

其中 \(\mathcal{L}_\mathrm{dark}\) 是 field-free Lindblad Liouvillian。这个过程不是 RWA，也不是 envelope approximation；它只是利用“该时间段没有有效光场”这一事实，把 time-dependent optical Hamiltonian 简化为 time-independent dark Liouvillian。

## 5. 为什么 dark 段不应简单用 endpoint-only `mesolve`

一个看似简单的做法是，对 dark segment 仍然调用 `run_case()` / `mesolve()`，但只输出两个点：

```text
t_dark = [t_dark_start, t_dark_end]
```

这样虽然输出点很少，但并不一定稳定。原因是即使 \(E(t)=0\)，field-free Hamiltonian \(H_0\) 仍然会让 coherence 按 lab-frame energy gap 旋转。对于几千 fs 的单个大 interval，scipy integrator 可能需要很多内部积分步，超过默认 `nsteps`，从而报错：

```text
Excess work done on this call. Try increasing the nsteps parameter.
```

因此，长 dark window 更适合使用 time-independent Liouvillian propagator，或者至少使用足够细的 coarse dark grid。对于 TA 长延迟场景，通常只关心 probe 到来前的最终状态 \(\rho_\mathrm{before\ probe}\)，不需要保存完整暗场轨迹，所以 exact Liouvillian propagation 更合适。

## 6. 当前脚本中暗传播的数值角色

在当前 `ta_piecewise_delay_scan_producer` 的设计中：

```text
pump segment:
  使用 lab_exact，保留 pump carrier，小 dt。

dark segment:
  使用 field-free Liouvillian exact propagation，不调用 optical-field mesolve。

probe/readout segment:
  使用 lab_exact，保留 probe carrier，小 dt。
```

最终每个 delay 保存一个 readout `DynamicsResult`，即 probe/readout segment 的轨迹。TA 差谱由 readout response 减去 probe-only reference response 得到。跨 delay 的统一时间轴、统一能量轴、插值、TA map 和 kinetic trace 留给 UFANSYS 处理。

## 7. 暗场窗口不应只由 pulse center 和 sigma 决定

早期实现中可以用 Gaussian pulse 的中心和宽度粗略定义：

```text
pump window: pump_center ± padding_sigma * pump_sigma
probe window: probe_center ± padding_sigma * probe_sigma
dark window: pump_end -> probe_start
```

但这个方法不够通用。未来可能有任意自定义电场、非 Gaussian pulse、chirped pulse、pulse train、2DES 多脉冲场等。此时“center ± sigma”并不一定能准确表示电场是否已经足够小。

更稳妥的定义应基于实际电场幅值，而不是基于某种特定脉冲模型。

## 8. 基于电场幅值的 dark window 判定

建议定义参考场强：

\[
E_\mathrm{ref}
=
\max
\left(
\max_t |E_\mathrm{pump}(t)|,\ 
\max_t |E_\mathrm{probe}(t)|
\right)
\]

然后设置 dark threshold：

\[
E_\mathrm{dark}
=
\epsilon E_\mathrm{ref}
\]

其中默认可以取：

\[
\epsilon = 10^{-3}
\]

也就是：

```text
E_dark = max(max(abs(pump)), max(abs(probe))) / 1000
```

当某一连续时间区间内始终满足：

\[
|E_\mathrm{total}(t)| < E_\mathrm{dark}
\]

则可以把该区间视为 dark window。

这个判据有两个优点：第一，它不依赖 Gaussian 假设；第二，它可以自然适配自定义 field、TAField、FieldPhySeries 和未来 TwoDESField。

## 9. 推荐的 dark window 检测流程

对于一个候选 pump-probe 时间范围：

1. 先生成或接收完整 field object，例如 `TAField` 或 `FieldPhySeries`。
2. 在参考时间范围内采样 pump field、probe field 或多个 reference fields。
3. 计算：

```python
E_ref = max(max(abs(E_pump(t))), max(abs(E_probe(t))))
```

4. 设置：

```python
E_dark_threshold = E_ref * dark_threshold_fraction
```

默认：

```python
dark_threshold_fraction = 1e-3
```

5. 在参考时间范围内采样 total field：

```python
E_total = field(t)
```

6. 得到布尔掩码：

```python
is_dark = abs(E_total) < E_dark_threshold
```

7. 寻找所有连续 `is_dark == True` 的区间。
8. 删除短于 `min_dark_duration_fs` 的区间。
9. 在这些 dark intervals 中选择位于 pump-active window 和 probe-active window 之间的那一段。
10. 如果找不到可靠 dark interval，则 fail-safe：使用 full lab_exact propagation，而不是强行 piecewise。

## 10. 这个逻辑应放在哪里

不建议把 dark window detection 放进 core solver，例如：

```text
sjh_learn/utils/core/solvers.py
sjh_learn/utils/core/model.py
```

原因是 core solver 的职责应保持简单：

```text
给定参数、哈密顿量、时间轴和初始态，完成传播。
```

而 dark window detection 回答的是另一个问题：

```text
给定一个实验电场序列，哪些时间区间可以认为是 field-free？
```

这属于 experiment workflow 或 field analysis 层，而不是 Hamiltonian solver 层。

更合适的位置是：

```text
sjh_learn/experiments/common/field_windows.py
```

如果暂时还没有 `experiments/common`，也可以先放在：

```text
sjh_learn/utils/fields/window_detection.py
```

但从长期结构看，TA 和 2DES 都会使用这一套逻辑，因此建议放在 `experiments/common` 中。

## 11. 建议的通用 API

可以先定义一个通用窗口对象：

```python
@dataclass(frozen=True)
class FieldWindow:
    role: str
    t_start_fs: float
    t_end_fs: float
    metadata: dict[str, Any]
```

然后提供参考场强估计：

```python
def estimate_field_reference_amplitude(
    fields: Sequence[FieldPhyRoot],
    *,
    t_start_fs: float,
    t_end_fs: float,
    dt_sample_fs: float,
) -> float:
    ...
```

以及 dark window 检测：

```python
def detect_dark_windows(
    field: FieldPhyRoot,
    *,
    reference_fields: Sequence[FieldPhyRoot],
    t_start_fs: float,
    t_end_fs: float,
    dt_sample_fs: float,
    dark_threshold_fraction: float = 1e-3,
    min_dark_duration_fs: float = 20.0,
) -> list[FieldWindow]:
    ...
```

其中 `field` 是用于判断 dark 的总场，`reference_fields` 用来估计参考幅值。对于 TA，可以传入 pump、probe 作为 reference fields，用 pump+probe total field 作为检测对象。对于 2DES，可以传入所有 pulse subfields 作为 reference fields，用完整多脉冲 field 作为检测对象。

## 12. TA 中如何使用检测结果

对于一个 delay case：

```text
1. 构造 pump field、probe field 和 total TA field。
2. 检测 total field 的 dark intervals。
3. 找到位于 pump-active 区域和 probe-active 区域之间的 dark interval。
4. 如果 dark interval 可靠：
   使用 pump-dark-probe piecewise propagation。
5. 如果 dark interval 不可靠：
   使用 full lab_exact pump+probe propagation。
```

这样可以避免仅凭 delay 大小或 pulse sigma 做判断。比如某些 shaped pulse 尾部很长，即使 nominal delay 很大，也可能不满足 dark threshold，此时应该自动退回 full propagation 或至少发出 warning。

## 13. 输出和 metadata 要求

由于 dark window 判定是数值近似，必须写入 metadata。每个 piecewise case 应记录：

```text
dark_threshold_fraction
E_ref_MV_per_cm
E_dark_threshold_MV_per_cm
dark_t_start_fs
dark_t_end_fs
dark_duration_fs
dark_detection_dt_sample_fs
dark_detection_method
whether_piecewise_used
fallback_reason
```

这样 UFANSYS 或后续论文作图时可以追溯每个 delay 是如何分段的。

推荐输出结构：

```text
outputs/ta_piecewise_delay_scan_producer/
├─ simulation/
│  ├─ case_specs.csv
│  ├─ segment_summary.json
│  ├─ checkpoints/
│  └─ res_per_delay/
└─ real/
   └─ final_output/
      └─ difference_spectra/
```

其中 simulation 目录保存 QuDPy 原始模拟结果和中间信息；real/final_output 保存每个 delay 接近实验输出形式的差谱。跨 delay 的 map 和进一步分析交给 UFANSYS。

## 14. 适用范围和限制

piecewise dark propagation 适用于长正 delay 中 pump 和 probe 明确分离、且二者之间电场幅值始终低于阈值的情况。

不适用于以下情况：

1. pump 和 probe 明显重叠；
2. coherent artifact 区域；
3. 电场尾部或 shaped pulse 在中间区间仍不可忽略；
4. 需要显式模拟中间弱场持续作用的情形；
5. dark window detection 无法找到可靠连续区间。

这些情况下应使用 full lab_exact propagation。

## 15. 小结

长延迟 TA 的核心矛盾是：有光场时必须用足够小的 `dt` 解析 optical carrier；无光场时不应浪费 optical-scale dense time grid。piecewise propagation 的合理实现不是引入 RWA，而是在电场确实可忽略的 dark window 中使用 field-free Lindblad evolution。

更稳健的下一步不是继续用 pulse center 和 sigma 人工切分，而是实现基于实际电场幅值的 dark window detector。这个 detector 应位于 experiment/common 或 field-analysis 层，而不是 solver core。solver core 保持“给定参数就传播”的职责，是否启用 piecewise dark propagation 由 TA / 2DES experiment workflow 显式决定。
