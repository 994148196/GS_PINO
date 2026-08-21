# exp105 — FNO + GS 物理残差（做法1 RHS，data_v6_clean 五配置混合）

> 实验日期：2026-08-21 ｜ 状态：**完成**（N=500，seed 1，data_v6_clean 五配置）
> 数据：`data_v6_clean/`（dn/sn/snow_single/snow_double/limiter 逗号拼接，
> train 池 5×500=2500，test 5×200=1000）
> 对照：exp013（同数据纯 MSE，整体 3.045%）；exp103（v5 混合做法1，0.70%）
> 结论速览：**做法1 物理残差推广到 data_v6_clean 五配置混合——test 整体
> rel L2 2.362%，全面优于 exp013 纯 MSE（每桶 −3% ~ −27%，整体 −22%）**；
> 物理正则在五配置混合上的增益与 v5 同机制（"PDE 项吃掉跨位形共享容量
> 代价"在五配置上更强成立）；X 点定位全面更好（all 3.09 vs 6.11 cm）；
> GS 残差 core pred 0.060 vs truth 0.003（20.8× FD 下限，v5 是 3.5×——
> 129² 差分噪声 + limiter 触壁 + SN 病态样本使正则拉紧程度下降）

## 1. 目标

exp103（v5 混合 DN+SN）证明做法1 物理残差在跨位形混合上"混合代价为零"。
exp013（纯 MSE）证明 data_v6_clean 五配置混合可行。本实验把两者结合：
**data_v6_clean 五配置（dn/sn/snowflake×2/limiter）+ 做法1 RHS 残差**——
验证物理约束在最复杂混合池（含触壁 limiter、双雪点、SN 病态）上是否
仍然成立、增益多大。

**损失 = MSE(psi_plasma) + w_pde · ‖Δ\*ψ_pred + μ0·R·J_data‖²（core mask 内）**

- RHS 来自数据集：`J_data = R·p′ + F·F′/(μ0·R)`（float64 重构；Ip 重构
  mean 3–16e-4，greens 恒等式 1e-7 量级——五配置均验证）
- 与 exp013 同口径：五配置逗号拼接、2500 池嵌套抽 500（perm seed 12345）、
  21ch coil 输入（R,Z + 5 params + 14 线圈电流）、无 config 通道
- coil 分离：网络只预测 psi_plasma，`psi_total = psi_plasma + Σ I_k·G_k`
  （greens 解析加回）

## 2. 输入通道（21 通道，同 exp013）

| # | 通道 | 含义/单位 | 来源 |
|---|---|---|---|
| 1 | R | 网格 R 坐标（[-1,1]） | 固定 MASTU_simple 129×129 |
| 2 | Z | 网格 Z 坐标（[-1,1]） | 固定 MASTU_simple 129×129 |
| 3–7 | Ip, paxis, fvac, alpha_m, alpha_n | 等离子体参数 | params[0:5] |
| 8–21 | I_1 … I_14 | 14 线圈电流 (A) | coil_currents[0:14] |

## 3. coil 分离（同 exp101/103）

网络**只预测 psi_plasma**（+J）；评估时 greens 加回 psi_total（恒等式
max diff 1e-7 Wb）。物理残差只定义在等离子体场上——线圈场占 |psi_total|
幅值 183%（v5 实测），残差落在 psi_total 上会被线圈导体奇性主导。

## 4. 训练设置

N=500（2500 池嵌套子集，perm seed 12345，五配置各 ~100），seed 1，
AdamW lr 1e-3 wd 1e-4，ReduceLROnPlateau（patience 20，factor 0.5，
min_lr 1e-5），batch 16，800 epochs（77.3 min）。模型 FNO2d2608：lift 21→64 +
4×FNOBlock（width 64, modes 16×16）+ proj 64→1，4.21M 参数。

**混合池统计（全 2500 训练样本）**：

| 统计量 | 五配置混合（本实验） | v5 混合（exp103） |
|---|---|---|
| psi_plasma mean / std (Wb) | 0.0806 / 0.0740 | 0.0400 / 0.0373 |
| J mean / std (A/m²) | 6.65e5 / 1.02e6 | 4.05e5 / 4.85e5 |
| pde_scale (Wb/m²) | **0.718** | 0.439 |
| ip_scale (A) | 1.11e6 | 5.50e5 |

v6 强等离子体（paxis 40–80 kPa vs v5 1–5 kPa）→ pde_scale 大 1.6×，
归一化后残差相对量级不变。GS 真值 FD 下限各配置 l_pde 1e-6~5e-5
（limiter 触壁样本最高）。

训练 l_pde 从 2.11e-2（e50）单调降到 3.02e-4（e797，70×）——物理项
全程拉紧。best val rel L2 **3.147%** @ epoch 797（psi_plasma 域；
exp013 纯 MSE val 3.368% @ e789 是 psi_total 域，口径不同仅参考）。

## 5. 结果（六桶 test：all=1000 / 各配置 200）

### 5.1 rel L2（psi_total 域，与 exp013 同口径）

| 桶 | exp105（本实验） | exp013（纯 MSE） | 变化 |
|---|---|---|---|
| **整体（n=1000）** | **2.362%** | 3.045% | **−22%** |
| dn | 2.426 | 3.193 | −24% |
| sn | 5.882 | 7.738 | **−24%** |
| snow_single | 1.387 | 1.433 | −3% |
| snow_double | 0.787 | 1.074 | −27% |
| limiter | 1.325 | 1.788 | **−26%** |

plasma 域：all 3.125% / dn 2.742% / sn 7.964% / snow_single 1.790% /
snow_double 0.859% / limiter 2.272%。

### 5.2 物理与几何

| 指标（all / dn / sn / snow_single / snow_double / limiter） | exp105 | exp013（参考） |
|---|---|---|
| GS 残差 core：pred | 0.060 / 0.044 / 0.157 / 0.038 / 0.016 / 0.047 | 3.75（全网格口径，不可比） |
| GS 残差 core：truth（FD 下限） | 0.0029 / 0.0019 / 0.0021 / 0.0027 / 0.0013 / 0.0067 | — |
| X 点 lo (cm) | **3.09 / 1.03 / 5.00 / 2.74 / 3.87 / NaN** | 6.11 / 4.13 / 7.25 / 5.44 / 7.55 / NaN |
| O 点 (cm) | 8.87 / 0.78 / 30.8 / 84.5* / 29.5* / NaN | 7.96 / 4.66 / 17.1 / 6.39 / 3.68 / NaN |
| 分离面 mean (cm) | 3.47 / 1.45 / 11.4 / 22.7* / 8.7* / NaN | 4.66 / 3.73 / 10.2 / 4.07 / 1.34 / NaN |

\* snowflake 桶 O 点/分离面指标受**检测算法局限**污染（见 §6），非预测
质量——这些样本 rel L2 仅 0.5–1.4%。

### 5.3 解读

1. **物理残差在五配置混合上全面增益**：整体 −22%，limiter −26%、sn −24%、
   snow_double −27%——"PDE 项吃掉跨位形共享容量代价"在五配置（含触壁、
   双雪点）上比 v5 双配置更强成立
2. **X 点定位全面更好**（3.09 vs 6.11 cm，每桶 −40%）：物理正则约束
   分离面附近场结构，直接收益几何指标
3. **GS 残差 20.8× FD 下限（v5 是 3.5×）**：129² 差分噪声 + limiter 触壁
   （mask 边界 J 不连续）+ SN 病态样本使正则拉紧程度下降；训练 l_pde
   仍单调下降 70×，sn 桶最差（0.157 = 74×）与 exp013 的 sn 病态已知
   问题同源
4. **limiter 桶健康**：rel L2 1.325%（−26%），GS 残差 7× 下限（触壁样本
   上物理项仍有效）——触壁位形不破坏物理残差管线
5. **无 config 通道自推断再验证**：五配置、14 线圈电流承载位形信息

## 6. 几何指标口径说明

- **limiter 桶**：无分离面 X 点 → x_lo/x_up/sep/O 点按设计 NaN（fig3
  显示 "no finite data"）
- **snowflake 桶 O 点/分离面**：find_critical 临界点分类对雪点（高阶
  零点）敏感——snow_single 139/200、snow_double 39/200 样本 O 点误差
  >10 cm（集中在 ~150 cm 固定错误点），但对应 rel L2 仅 0.5–1.4%，
  是**检测算法局限而非预测质量**；exp013 纯 MSE 场恰好未触发该分类
  混淆。snow 桶几何以 X 点误差与 rel L2 为准
- **sn 桶**：单 X 点（下），x_up 恒 NaN（设计行为）

## 7. 复现

```bash
bash dn_fno_2608/scripts/run_exp105_106_pino.sh train   # 训练 exp105+106
bash dn_fno_2608/scripts/run_exp105_106_pino.sh eval    # 六桶评估 + 可视化
```

产物：`best.pt`、`history.json`、`args.json`、`metrics.json`、
`train.log`/`eval_*.log`（日志落各自实验目录）、`eval_all|dn|sn|snow_*|limiter/`、
`figures_all|dn|sn|snow_*|limiter/`（fig1/2/3 + stats_per_sample.json，
exp011 风格，MASTU_simple 装置 14 线圈）。
