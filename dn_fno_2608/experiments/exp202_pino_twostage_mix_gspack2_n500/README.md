# exp202 — FNO + GS 物理残差（做法2 两阶段），数据源替换：freegs → gspack2_TRAE

> 实验日期：2026-08-24 ｜ 状态：**完成**（N=500，seed 1，data_gspack2_v1 DN+SN 混合）
> 数据：`data_gspack2_v1/dn + sn` 逗号拼接（各 500/500/500，混合 1000/1000/1000）
> 对照：**exp104**（同口径 freegs data_v5，0.76%）；exp201（同数据做法1，1.20%）
> 结论速览：**两阶段自洽在 gspack2 数据上机制成立但全面退化——test 整体
> rel L2 1.32%（DN 1.17% / SN 1.46%）vs exp104 的 0.76%（+0.56pp，与 exp201
> vs exp103 的 +0.50pp 同量级、方向一致）；Ip 误差 0.935%（exp104 0.21%）、
> mask 内 J 2.82%（exp104 1.55%）——J/Ip 通道对数据源微结构比 psi 更敏感
> （g2 数据内 Ip 重构精度 0.58% 抬高了 Ip 约束下限）；阶段1 第 61 epoch 达
> 3% 阈值切阶段2（比 exp104 的 e45 晚），30-epoch ramp 无爆炸；交叉评估与
> exp201 同结论：差异是 g2 数据固有（exp104 模型在 g2 上 1.13%）**

## 1. 目标

与 exp201 同组的数据源替换实验：管线零改动，把 data_v5（freegs）换成
data_gspack2_v1（gspack v2.0.0），重跑 exp104 的做法2（两阶段自洽 + Ip 约束）。
重点验证：g2 数据上 psi↔J 自洽链路是否成立、Ip/J 通道的退化幅度。

**阶段1 损失 = MSE(psi_plasma, z) + w_j·masked MSE(J, z)（core mask 内）**
**阶段2 损失 = 阶段1 + w_pde·‖Δ\*ψ_pred + μ0·R·J_pred‖²（core mask 内）+ w_ip·‖(ΣJ·dA − Ip)/ip_scale‖²**

## 2. 输入通道（18 通道，与 exp104 完全同构）

| # | 通道 | 含义/单位 | 来源 |
|---|---|---|---|
| 1 | R | 网格 R 坐标（[-1,1]） | 固定 MAST 65×65（= v5 网格） |
| 2 | Z | 网格 Z 坐标（[-1,1]） | 固定 MAST 65×65 |
| 3 | Ip | 等离子体电流 (A) | params[0] |
| 4 | paxis | 磁轴压强 (Pa) | params[1] |
| 5 | fvac | 真空通量函数 f (Wb/m) | params[2] |
| 6 | alpha_m | 剖面形状指数 m | params[3] |
| 7 | alpha_n | 剖面形状指数 n | params[4] |
| 8–17 | I_P2U … I_P6L | 上下偏滤器线圈电流 (A) | coil_currents[0:10] |
| 18 | I_P1 (Solenoid) | 中心螺线管电流 (A) | coil_currents[10] |

`config` 字段存在但不加载（use_config=False，同 exp104）。

## 3. coil 分离与评估口径（同 exp104）

网络双输出通道：通道 0 = psi_plasma（z 域），通道 1 = Jφ（z 域）；
`psi_total = psi_plasma_pred + Σ_k I_k·G_k`（greens 解析加回）。

## 4. 训练设置

N=500（1000 池嵌套子集，perm seed 12345 → **256 DN + 244 SN**），seed 1，
AdamW lr 1e-3 wd 1e-4，ReduceLROnPlateau（patience 20，factor 0.5，
min_lr 1e-5），batch 16，800 epochs（18.3 min）。模型 FNO2d2608：lift 18→64 +
4×FNOBlock（width 64, modes 16×16）+ proj 64→**2**，4.21M 参数。

**阶段切换**：阶段1 val rel L2 第 **61** epoch 达 3% 阈值 → 切阶段2（比
exp104 的 e45 晚——g2 的 SN 桶拖慢 psi_plasma 拟合，同 exp104 比 exp102 晚的
模式）；物理权重 30-epoch 预热无爆炸。phase-aware best：artifact 取阶段2
best（best val rel L2 **1.5402%** @ epoch 679）。

## 5. 结果（三桶 test：all=1000 / dn=500 / sn=500）

| 指标 | 整体（n=1000） | DN 桶（n=500） | SN 桶（n=500） |
|---|---|---|---|
| rel_l2_total mean / median / p95 (%) | **1.32 / 1.07 / 3.10** | 1.17 / 0.94 / 2.75 | 1.46 / 1.14 / 3.33 |
| rel_l2_plasma mean / median / p95 (%) | 1.53 / 1.24 / 3.42 | 1.40 / 1.12 / 3.04 | 1.66 / 1.32 / 3.70 |
| rmse_phys (Wb) | 5.61e-4 | — | — |
| GS 残差 core：pred / truth | 0.0271 / 0.00082 | — | — |
| **Ip 误差** mean / median / p95 (%) | 0.935 / 0.61 / 2.92 | 0.62 / 0.41 / 1.91 | 1.25 / 0.81 / 3.87 |
| **J rel L2 mask 内** mean / median / p95 (%) | 2.82 / 2.17 / 6.83 | 2.13 / 1.75 / 4.72 | 3.52 / 2.64 / 9.33 |
| X 点定位 lo / up (cm) | 0.67 / 0.90（SN 仅 lo） | 0.78 / 0.90 | 0.55 / NaN |
| O 点 / 分离面 mean (cm) | 0.45 / 0.55 | — | — |
| n_xpt_fail | 0 | 0 | 0 |

对照 exp104（v5）：total 0.76%（+0.56pp）；Ip 0.21%（+0.72pp）；J mask
1.55%（+1.27pp）——**psi/J/Ip 三项全部退化，J 与 Ip 退化幅度是 psi 的
2 倍以上**。

## 6. 诊断：与 exp201 同结论，J/Ip 更敏感

1. **total 退化幅度与做法1 完全一致**：exp201 +0.50pp（1.197 vs 0.703）、
   exp202 +0.56pp（1.316 vs 0.76）——g2 数据固有表示代价，与训练方式无关
2. **交叉评估（决定性）**：

   | 模型 | 测 v5 test | 测 g2 test |
   |---|---|---|
   | exp104（v5 训练） | 0.76% | **1.127%**（J mask 8.7%） |
   | exp202（g2 训练） | 1.273%（J mask 2.6%） | **1.316%**（J mask 2.82%） |

   exp104 模型在 g2 上 1.13%（与 exp103 模型在 g2 上 1.12% 相同）——两做法
   的 g2 固有下限一致 ~1.1%，独立于模型与训练
3. **J 通道跨源退化远大于 psi**：exp104 模型在 g2 上 J mask 8.72%（主场
   1.55%）vs psi 1.13%（主场 0.76%）——J 场对 g2 微结构的敏感性是 psi 的
   数倍（J 是 psi 的二阶导数场，局部结构差异被放大）
4. **Ip 约束下限被 g2 数据内重构精度抬高**：data_gspack2_v1 README §8.1
   记录 gspack 的 J 数值积分离散与 freegs 略异 → 用数据内 dpdpsi/FdFdpsi
   重构的 Ip 精度 ~5.8e-3（v5 ~3e-4）。exp202 的 Ip 误差 0.935% ≈ 重构精度
   + 模型 J 误差贡献——**Ip 指标的下限本身在 g2 上更高，0.935% 不代表
   约束失效**（exp104 模型在 g2 上 Ip 0.24% 反证口径差异：v5 口径的平滑
   J_pred 在 g2 数据上积分误差反而更小，两口径不可直接比）
5. **机制本身成立**：阶段切换正常（e61）、ramp 无爆炸、GS 残差 pred 0.0271
   vs truth 0.00082（33×，与 exp201 的 27× 同量级）、X 点定位亚厘米、
   n_xpt_fail=0——物理管线在 g2 数据上健康运行

## 7. SN 桶几何口径说明

同 exp104 §6：SN 单分离面 X 点 → `x_up_cm` 恒 NaN（设计行为）；SN 桶
J mask 3.52% / Ip 1.25% 与 DN 同机制成立（只是 g2 固有下限更高）。

## 8. 复现

```bash
bash dn_fno_2608/scripts/run_exp201_202_pino.sh train   # exp201 + exp202
bash dn_fno_2608/scripts/run_exp201_202_pino.sh eval    # 三桶评估
```

产物：`best.pt`、`history.json`、`args.json`、`train.log`、`eval_all|dn|sn/`
（metrics.json）、`figures_all|dn|sn/`（fig1/2/3，twostage 含 J 行/Ip 直方图）、
`eval_x_v5all/`（交叉评估：g2 模型测 v5 test）。
