# exp201 — FNO + GS 物理残差（做法1 RHS），数据源替换：freegs → gspack2_TRAE

> 实验日期：2026-08-24 ｜ 状态：**完成**（N=500，seed 1，data_gspack2_v1 DN+SN 混合）
> 数据：`data_gspack2_v1/dn + sn` 逗号拼接（各 500/500/500，混合 1000/1000/1000）
> 对照：**exp103**（同口径 freegs data_v5，0.703%）；exp011（混合纯 MSE 0.894%）
> 结论速览：**gspack2_TRAE 生成的数据与 data_v5 分布等价（输入/场/频谱/收敛
> 全部统计一致、模型交叉评估互通），但可达误差下限更高——test 整体
> rel L2 1.20%（DN 桶 1.06% / SN 桶 1.33%）vs exp103 的 0.70%（+0.50pp，
> 超出 ±0.2pp 数据等价判据）**；交叉评估证明差异是数据侧固有（exp103 模型
> 在 g2 的 train/test 上都只到 1.11/1.12%，train/test 无差），不是训练失败
> 或子集运气；物理残差机制本身在 g2 数据上依然成立（GS 残差 pred/truth
> 同量级关系，X 点定位亚厘米）

## 1. 目标

验证 gspack2_TRAE（freegs 风格平衡求解包，gspack v2.0.0，CPU）可作为 PINO
数据源：**管线零改动**（train/evaluate/data/model/loss 全部复用），仅把
data_v5（freegs）换成 data_gspack2_v1（gspack，MAST 11 线圈 1:1 复刻），
重跑 exp103 的做法1（RHS 单阶段物理残差）对照实验。数据 schema 26 键与
data_v5 逐键一致（见 `data_gspack2_v1/README.md`）。

**损失 = MSE(psi_plasma) + w_pde · ‖Δ\*ψ_pred + μ0·R·J_data‖²（core mask 内）**，
w_pde=0.1，与 exp103 完全相同。

## 2. 输入通道（18 通道，与 exp103 完全同构）

| # | 通道 | 含义/单位 | 来源 |
|---|---|---|---|
| 1 | R | 网格 R 坐标（[-1,1]） | 固定 MAST 65×65（= v5 网格，array_equal） |
| 2 | Z | 网格 Z 坐标（[-1,1]） | 固定 MAST 65×65 |
| 3 | Ip | 等离子体电流 (A) | params[0] |
| 4 | paxis | 磁轴压强 (Pa) | params[1] |
| 5 | fvac | 真空通量函数 f (Wb/m) | params[2] |
| 6 | alpha_m | 剖面形状指数 m | params[3] |
| 7 | alpha_n | 剖面形状指数 n | params[4] |
| 8–17 | I_P2U … I_P6L | 上下偏滤器线圈电流 (A) | coil_currents[0:10] |
| 18 | I_P1 (Solenoid) | 中心螺线管电流 (A) | coil_currents[10] |

`config` 字段（dn=0 / sn=1）存在但不加载（use_config=False，同 exp103）；
位形信息由 11 线圈电流自推断。

## 3. coil 分离与评估口径（同 exp103）

网络只预测 psi_plasma；评估时 `psi_total = psi_plasma_pred + Σ_k I_k·G_k`
（greens 解析加回，g2 恒等式 max diff 3.2e-8 Wb）。物理残差只定义在
等离子体场上。三桶 test：all=1000 / dn=500 / sn=500。

## 4. 训练设置

N=500（1000 池嵌套子集，perm seed 12345 → **256 DN + 244 SN**），seed 1，
AdamW lr 1e-3 wd 1e-4，ReduceLROnPlateau（patience 20，factor 0.5，
min_lr 1e-5），batch 16，800 epochs。模型 FNO2d2608：lift 18→64 +
4×FNOBlock（width 64, modes 16×16）+ proj 64→1，4.21M 参数。

**混合池统计（全 1000 训练样本）**：

| 统计量 | g2 混合（本实验） | v5 混合（exp103） |
|---|---|---|
| psi_plasma mean / std (Wb) | 0.0400 / 0.0372 | 0.0400 / 0.0373 |
| psi_total mean / std (Wb) | -0.0474 / 0.0442 | -0.0483 / 0.0448 |
| J mean / std (A/m²) | 4.12e5 / 4.90e5 | 4.05e5 / 4.85e5 |

best val rel L2 **1.3733%** @ epoch 736（lr 提前触底 1e-5 @ e~500）。

## 5. 结果（三桶 test）

| 指标 | 整体（n=1000） | DN 桶（n=500） | SN 桶（n=500） |
|---|---|---|---|
| rel_l2_total mean / median / p95 (%) | **1.20 / 0.95 / 2.93** | 1.06 / 0.85 / 2.59 | 1.33 / 1.09 / 3.12 |
| rel_l2_plasma mean / median / p95 (%) | 1.40 / 1.11 / 3.38 | 1.26 / 0.99 / 3.05 | 1.53 / 1.27 / 3.48 |
| rmse_phys (Wb) | 5.11e-4 | 4.29e-4 | 5.93e-4 |
| GS 残差 core：pred / truth | 0.0222 / 0.00082 | 0.0193 / 0.00082 | 0.0250 / 0.00083 |
| X 点定位 lo / up (cm) | 0.56 / 0.67（SN 仅 lo） | 0.69 / 0.67 | 0.44 / NaN |
| O 点 / 分离面 mean (cm) | 0.45 / 0.53 | 0.36 / 0.50 | 0.53 / 0.56 |
| n_xpt_fail | 0 | 0 | 0 |

对照 exp103：整体 0.70%（DN 0.69 / SN 0.72）——**g2 数据整体 +0.50pp**。
注意 g2 数据的真值 GS 残差（0.00082）比 v5（0.0044）小一个量级——gspack
约束解自洽性更好（X 点偏差 0.9 mm vs v5 16 mm），模型残差倍数（27× vs
3.5×）不是机制退化，是分母（truth 下限）更小所致；pred 残差本身 0.022
（v5 0.0156）与误差 1.2% 同趋势。

## 6. 诊断：为什么是 1.20% 而不是 0.70%（关键）

**结论：g2 数据的可达误差下限 ~1.1%，是数据侧固有；分布等价成立、误差等价
不成立。** 证据链：

1. **训练动态差异在 LR 分化前已存在**：e100 train loss g2 2.48e-3 vs v5
   1.07e-3（此时两实验 lr 均 5e-4）；e166 val 1.77% vs 1.41%（+0.36pp）——
   差异不是 lr 提前触底的产物（触底只是让差距维持，不是根源）
2. **交叉评估矩阵（决定性，18ch 输入同构 → 模型可互通评估）**：

   | 模型 | 测 v5 test | 测 g2 test | 测 g2 train |
   |---|---|---|---|
   | exp103（v5 训练） | **0.703%** | 1.116% | 1.107% |
   | exp201（g2 训练） | 1.135% | **1.197%** | — |

   - exp103 模型在 g2 的 **train/test 上同为 ~1.11%**——排除了"子集运气/
     泛化差距"，是 g2 目标函数对 FNO 的固有表示下限
   - exp201 模型在 g2 上 1.197% ≈ exp103 模型在 g2 上 1.116%（+0.08pp）——
     exp201 的训练本身基本到位（其 val 1.37% 与 test 1.20% 的差值 0.17pp
     与 exp103 的 val-test 差 0.08pp 同量级，无过拟合迹象）
3. **分布验证全过（下表），差异无法由任何可测输入统计解释**：

   | 检查 | v5 | g2 | 结论 |
   |---|---|---|---|
   | 输入参数/线圈电流范围 | 同 | 同 | 一致 |
   | psi_total mean / std | -0.0483/0.0448 | -0.0474/0.0442 | 差 <2.7%（归一化偏差可忽略） |
   | inwall 分数 | dn 0.203 / sn 0.144 | dn 0.209 / sn 0.148 | ±0.004-0.006 |
   | 场高频能量占比（>16 modes） | dn 0.0027 / sn 0.0018 | dn 0.0028 / sn 0.0018 | 一致 |
   | 等离子体内梯度 p95 | dn 0.065 / sn 0.071 | 同 | 一致 |
   | 收敛质量 psi_relchange | 3.3-4.5e-4 | 2.9-3.4e-4（略好） | 一致 |
   | X 点/锚点几何覆盖 | 0.640-0.760 / 1.2-1.6 | 同 | 一致 |
4. **不对称交叉提示微结构来源**：exp103 模型在 g2 上 +0.42pp，exp201 模型
   在 v5 上 -0.06pp（几乎不变）——g2 场包含 v5 场没有的微结构，FNO 在 v5
   上学不到、在 g2 上额外付出表示代价。最可能来源：gspack 约束解更紧
   （X 点偏差 0.9 mm vs freegs 16 mm → X 点/分离面附近局部结构更精确更
   "锐利"，虽总体频谱/梯度一致，但局部形态差异足以造成 FNO 模式截断
   表示代价 +0.4pp）

**判定**：按计划判据（±0.2pp）数据等价**未达成**；按机制判据（分布等价、
模型互通、无生成错误）**成立**。gspack2_TRAE 作为数据源是健康的（schema、
恒等式、收敛、约束全部合格），差异是求解器场微结构带来的表示难度增加，
量级 ~+0.4-0.5pp 且两实验一致（见 exp202）。

## 7. SN 桶几何口径说明

同 exp103 §6：SN 单分离面 X 点 → `x_up_cm` 恒 NaN（设计行为）；
n_xpt_fail=0（三桶）。

## 8. 复现

```bash
bash dn_fno_2608/scripts/run_exp201_202_pino.sh train   # exp201 + exp202
bash dn_fno_2608/scripts/run_exp201_202_pino.sh eval    # 三桶评估（exp201/202）
# 数据再生成/补数据见 data_gspack2_v1/README.md §9（topup）
```

产物：`best.pt`、`history.json`、`args.json`、`train.log`、`eval_all|dn|sn/`
（metrics.json）、`figures_all|dn|sn/`（fig1/2/3 + stats_per_sample.json）、
`eval_x_v5all/`（交叉评估：g2 模型测 v5 test）。
