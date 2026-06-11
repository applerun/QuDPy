# Normalizer 说明：物理单位到 solver code unit 的转换

## 1. 这是一个什么功能？

`ParaNormalizer` 的功能是把用户输入的物理参数转换成 solver 内部使用的
code-unit 参数。它只负责**单位换算和数值尺度归一化**，不负责定义物理模型，
也不负责构造 Hamiltonian、collapse operator 或 spectroscopy 后处理。

用户侧输入尽量保持真实物理单位，例如：

- 能量：eV
- 时间：fs
- 电场：MV/cm
- 偶极矩：Debye
- relaxation / dephasing 时间常数：fs

solver 内部则使用无量纲 code unit。核心转换关系是选定一个时间尺度
`T0 = time_scale_fs`，然后把所有频率、速率和时间转换到 code unit：

- `t_code = t_fs / T0`
- `omega_code = omega_fs_inv * T0`
- `gamma_code = gamma_fs_inv * T0`
- `Omega_code = Omega_fs_inv * T0`

这样做的目的不是改变物理方程，而是让进入数值 solver 的量级更稳定，
避免某些量过大或过小导致数值表现差。

## 2. Normalizer 不做什么？

`ParaNormalizer` 不应该：

- 改变能级结构；
- 改变偶极矩矩阵；
- 改变光场的物理形式；
- 改变 Hamiltonian 构造公式；
- 改变 Lindblad relaxation / dephasing 物理通道；
- 改变 spectroscopy 或 absorption-like spectrum 后处理；
- 把 N-level system 隐式简化成 two-level system。

也就是说，normalization 只改变**单位表示**，不改变 simulation 的物理内容。

## 3. 当前 normalization 的主要流程

一次 `normalize(p: NLevelPhysicalParams)` 的核心流程是：

1. 检查用户输入的物理参数是否合法。
2. 将 `energies_eV` 转换为 `energies_fs_inv`。
3. 从显式 `field` 对象读取归一化所需的光场信息：
   - `E0_MV_per_cm`
   - `omega_L_fs_inv`
   - Gaussian field 的 `pulse_center_fs` 和 `pulse_sigma_fs`，如果存在。
4. 根据 `dipole_matrix_D` 和 field reference amplitude 计算
   `coupling_matrix_fs_inv`。
5. 将 relaxation 和 pure dephasing channel 转成 `rate_fs_inv`。
6. 收集 `rate_candidates`，用于自动选择 `time_scale_fs`。
7. 用 `time_scale_fs` 把时间轴、能量、耦合矩阵和速率转换成 code unit。
8. 返回 `SolverParams`，供 solver 构造 Hamiltonian 和 collapse operators。

## 4. `rate_candidates` 是什么？

`rate_candidates` 是 auto-scale 使用的一组“典型频率或速率候选值”，单位是
`fs^-1`。它不是新的物理模型参数，也不直接参与 Hamiltonian 或 Lindblad
operator 的物理定义。

它的唯一用途是帮助 `_choose_time_scale_fs(...)` 自动选择一个合适的
`time_scale_fs`。

当前候选值主要包括：

1. 非零光场耦合强度  
   来自 `coupling_matrix_fs_inv` 中所有非零元素的模长：

   ```text
   abs(coupling_matrix_fs_inv[i, j])
   ```

2. relaxation rates  
   来自每个 relaxation channel 的：

   ```text
   gamma_1 = 1 / T1
   ```

   或用户显式提供的 `rate_fs_inv`。

3. pure dephasing rates  
   来自每个 pure dephasing channel 的：

   ```text
   gamma_phi = 1 / Tphi
   ```

   或用户显式提供的 `rate_fs_inv`。

4. 失谐量候选  
   当前代码中保留了一个二能级兼容式 heuristic：

   ```text
   abs(energies_fs_inv[1] - energies_fs_inv[0] - omega_L_fs_inv)
   ```

   这只是 auto-scale 的数值尺度候选，不改变模拟主线。对严格 N-level
   体系，后续可以改为扫描所有 `i < j` transition，或直接删除这个候选。

5. Gaussian pulse bandwidth scale  
   如果 field 中存在 `pulse_sigma_fs`，则加入：

   ```text
   1 / pulse_sigma_fs
   ```

这些候选值的作用是告诉 normalizer：当前问题里有哪些可能比较快的时间尺度。

## 5. `time_scale_fs` 如何得出？

如果用户显式指定了 `time_scale_fs`，则直接使用用户给定值。

如果没有指定，且 `auto_scale=True`，则 normalizer 会从 `rate_candidates`
中取正值，并选择其中最大的速率：

```text
rate_ref = max(rate_candidates)
```

然后令：

```text
time_scale_fs = 1 / rate_ref
```

这样最大的典型速率在 code unit 中大约变成 1：

```text
rate_ref_code = rate_ref * time_scale_fs ≈ 1
```

这使 solver 内部的主要速率和频率尺度更适中。

如果 `rate_candidates` 为空，则 fallback 到 `energies_fs_inv` 中的非零能量。
如果仍然没有可用尺度，则使用：

```text
time_scale_fs = 1.0
```

## 6. 为什么这个 auto-scale 不改变物理结果？

因为所有相关量都按同一个 `T0 = time_scale_fs` 做一致变换：

```text
t_code = t_fs / T0
omega_code = omega_fs_inv * T0
gamma_code = gamma_fs_inv * T0
Omega_code = Omega_fs_inv * T0
```

物理演化中真正出现的是类似下面的组合：

```text
omega_fs_inv * t_fs
```

归一化后变成：

```text
(omega_fs_inv * T0) * (t_fs / T0)
```

二者相同。因此 auto-scale 只改变 solver 内部的数值单位，不改变真实物理演化。

## 7. 关于失谐量 candidate 的说明

当前 normalization 中仍有一个固定 `0 -> 1` 的 detuning candidate：

```text
abs(energies_fs_inv[1] - energies_fs_inv[0] - omega_L_fs_inv)
```

它来自早期 two-level 模型，作用只是给 auto-scale 提供一个额外的速率尺度。
它不决定哪个 transition 被驱动，也不决定 Hamiltonian 的耦合结构。

对 N-level system，更严格的做法应该是：

- 遍历所有 `i < j` transition；
- 可选地只考虑 `abs(mu_ij)` 或 `abs(coupling_ij)` 非零的 transition；
- 取与激光频率最近的 detuning；
- 或者完全不把 detuning 纳入 auto-scale，只使用 coupling、relaxation、
  dephasing 和 pulse bandwidth。

但因为它只影响 code-unit scale，而不改变物理主线，所以可以作为低优先级
重构项处理。

## 8. 单位转换表

| 物理量                  |                    用户输入单位 |                                          中间物理求解单位 |                      内部代码单位 | 转换关系                                                  |
| -------------------- | ------------------------: | ------------------------------------------------: | --------------------------: | ----------------------------------------------------- |
| 时间                   |                        fs |                                                fs |               dimensionless | $t_{\mathrm{code}}=t_{\mathrm{fs}}/T_0$               |
| 时间步长                 |                        fs |                                                fs |               dimensionless | $dt_{\mathrm{code}}=dt_{\mathrm{fs}}/T_0$             |
| 能级差                  |                        eV |                                $\mathrm{fs}^{-1}$ |               dimensionless | $\omega_{eg}=E_{eg}/\hbar$，再乘 $T_0$                   |
| 激光能量                 |                        eV |                                $\mathrm{fs}^{-1}$ |               dimensionless | $\omega_L=E_L/\hbar$，再乘 $T_0$                         |
| 失谐量                  |               eV 或由两频率差得到 |                                $\mathrm{fs}^{-1}$ |               dimensionless | $\Delta=\omega_{eg}-\omega_L$，再乘 $T_0$                |
| 偶极矩                  |                     Debye |                              参与计算 $\mu E_0/\hbar$ |                   通常并入 Rabi | 与电场合成 $\Omega$                                        |
| 电场强度                 |                     MV/cm |                              参与计算 $\mu E_0/\hbar$ | 通常并入 Rabi / field amplitude | 与偶极矩合成 $\Omega$                                       |
| Rabi frequency       |      由 $\mu E_0/\hbar$ 得到 |                                $\mathrm{fs}^{-1}$ |               dimensionless | $\Omega_{\mathrm{code}}=\Omega_{\mathrm{fs}^{-1}}T_0$ |
| $T_1$                |                        fs |       $\gamma_1=\frac{1}{T_1}$，$\mathrm{fs}^{-1}$ |               dimensionless | $\gamma_{1,\mathrm{code}}=\gamma_1T_0$                |
| $T_\phi$             |                        fs | $\gamma_\phi=\frac{1}{T_\phi}$，$\mathrm{fs}^{-1}$ |               dimensionless | $\gamma_{\phi,\mathrm{code}}=\gamma_\phi T_0$         |
| $T_2$                |                        fs |       $\gamma_2=\frac{1}{T_2}$，$\mathrm{fs}^{-1}$ |               dimensionless | $\gamma_{2,\mathrm{code}}=\gamma_2T_0$                |
| 密度矩阵                 |                       无量纲 |                                               无量纲 |                         无量纲 | 不反变换                                                  |
| population           |                       无量纲 |                                               无量纲 |                         无量纲 | 不反变换                                                  |
| coherence            |                       无量纲 |                                               无量纲 |                         无量纲 | 不反变换                                                  |
| collapse operator 系数 | $\sqrt{\mathrm{fs}^{-1}}$ |                         $\sqrt{\mathrm{fs}^{-1}}$ |   $\sqrt{\text{code rate}}$ | 系数是 $\sqrt{\gamma_{\mathrm{code}}}$                   |
