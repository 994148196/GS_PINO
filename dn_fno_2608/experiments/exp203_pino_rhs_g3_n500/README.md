# exp203 — FNO + GS 物理残差（做法1 RHS），数据源替换 + SN 质量修复

> 实验日期：2026-08-25 ｜ 状态：**完成**（N=500，seed 1，data_gspack2_v2 五配置混合）
> 数据：`data_gspack2_v2`（gspack2_TRAE 生成 MASTU_simple，129²、21ch；
>   train 各 500@123 / val 各 100@456 / test 各 200@789，逗号拼接）
> 对照：**exp105**（同口径 data_v6_clean，test 2.362%，sn 桶 5.882% 最差）
> 结论速览：**数据质量目标达成、误差目标未达成（诚实负面结果）**——SN
> 生成侧硬门全部生效（病态 0/200、gs_true>15 0/200 vs v6 raw 26/200、
> filter_v6 复核 15 split 移除 0 条），但六桶误差全面差于 exp105
> （all 2.963% vs 2.362% +0.60pp；**sn 7.719% vs 5.882% +1.84pp 最差**）。
> 双向交叉评估证明差异在**数据侧固有**（同一模型跨数据 +1.28~+1.73pp）：
> exp105 模型测 g3 test 3.642%，exp203 模型测 v6_clean test 4.689%——
> gspack 数值求解与 freegs_snow 的场差异在五配置下比 exp201（双配置）
> 更显著，SN 的 X 点约束对数值细节最敏感；"更干净"（无病态）≠
> "更容易学"（g3 模型学到 g3 特有数值特征，对 v6 不通用）

## 1. 目标

**从生成侧修复 data_v6 的 SN 数据质量差**（磁轴偏下、上瓣薄、中平面外翻，
exp012 §7.1 诊断，sn 病态率 18.4%）——用 gspack2_TRAE（freegs 风格求解包，
gspack v2.0.0）生成同构数据（34 键 npz schema、129² 网格、MASTU_simple 26
物理线圈 → 14 控制单元、21 输入通道），管线（train/evaluate/data/model/loss）
零改动，重跑 exp105 的做法1（RHS 单阶段物理残差）对照实验。
数据细节见 `dn_fno_2608/data_gspack2_v2/README.md`（SN 生成侧硬门：
midplane≥0.05 / zaxis≤0.5 / gs_true≤15）。

**损失 = MSE(psi_plasma) + w_pde · ‖Δ\*ψ_pred + μ0·R·J_data‖²（core mask 内）**，
w_pde=0.1，与 exp105 完全相同。

**成功判据**：SN 桶显著优于 exp105 的 7.738%；整体误差按 exp201 结论框架
解读（分布等价 vs 数据侧固有下限），必要时交叉评估定位差异归属。

## 2. 输入通道（21 通道，与 exp105 完全同构）

| # | 通道 | 含义/单位 | 来源 |
|---|---|---|---|
| 1 | R | 网格 R 坐标（[-1,1]） | 固定 129² 网格（= v6，array_equal） |
| 2 | Z | 网格 Z 坐标（[-1,1]） | 固定 129² 网格 |
| 3 | Ip | 等离子体电流 (A) | params[0] |
| 4 | paxis | 磁轴压强 (Pa) | params[1] |
| 5 | fvac | 真空通量函数 f (Wb/m) | params[2] |
| 6 | alpha_m | 剖面形状指数 m | params[3] |
| 7 | alpha_n | 剖面形状指数 n | params[4] |
| 8–21 | I_Solenoid, I_Pc, I_Px, I_D1, I_D2, I_D3, I_Dp, I_D5, I_D6, I_D7, I_P4, I_P5, I_P61, I_P62 | 14 单元电流 (A) | coil_currents[0:14] |

`config` 字段（dn=0/sn=1/snow_single=2/snow_double=3/limiter=4）存在但不
加载（use_config=False，同 exp105）；位形信息由 14 线圈电流自推断。

## 3. coil 分离与评估口径（同 exp105）

网络只预测 psi_plasma；评估时 `psi_total = psi_plasma_pred + Σ_k I_k·G_k`
（greens 解析加回，g3 恒等式 ~3e-8 Wb）。物理残差只定义在等离子体场上。
六桶 test：all=1000 + 五配置各 200（val/test 与 v6_clean 同规模同配置 →
逐桶 1:1 对比）。

## 4. 训练设置

N=500（2500 池嵌套子集，seed 1），AdamW lr 1e-3 wd 1e-4，
ReduceLROnPlateau（patience 20，factor 0.5，min_lr 1e-5），batch 16，
800 epochs（72.4 min）。模型 FNO2d2608：lift 21→64 + 4×FNOBlock（width 64,
modes 16×16）+ proj 64→1，4.21M 参数，与 exp105 相同。

**混合池统计（全 2500 训练样本）**：

| 统计量 | g3 混合（本实验） | v6 混合（exp105） |
|---|---|---|
| psi_plasma mean / std (Wb) | 0.0791 / 0.0722 | 0.0806 / 0.0740 |
| psi_total mean / std (Wb) | -0.0331 / 0.0964 | — |
| J mean / std (A/m²，mask 内) | 6.75e5 / 9.62e5 | 6.65e5 / 1.02e6 |
| pde_scale (Wb/m²) | 0.729 | 0.718 |
| ip_scale (A) | 1.09e6 | 1.11e6 |

**分布等价成立**（exp201 同款结论）：输入/场统计全部与 v6 一致（网格
逐点 array_equal、21ch、同参数范围）——误差差异不在分布层，而在
gspack 数值求解的可学习性层（§5.3/5.4）。

## 5. 结果（六桶 test：all=1000 / 各配置 200，对照 exp105 同口径）

### 5.1 rel L2（psi_total 域，与 exp105 同口径）

| 桶 | exp203（本实验） | exp105（v6_clean） | Δ |
|---|---|---|---|
| **整体（n=1000）** | **2.963%** | **2.362%** | **+0.60pp** |
| dn | 3.019 | 2.426 | +0.59pp |
| sn | **7.719** | 5.882 | **+1.84pp** |
| snow_single | 1.643 | 1.387 | +0.26pp |
| snow_double | 1.002 | 0.787 | +0.22pp |
| limiter | 1.430 | 1.325 | +0.11pp |

plasma 域：all 3.777% / dn 3.372% / sn 9.833% / snow_single 2.104% /
snow_double 1.093% / limiter 2.481%（exp105：3.125 / 2.742 / 7.964 /
1.790 / 0.859 / 2.272）。

val 侧：exp203 best val rel L2 **3.540%** @ e774 vs exp105 3.147% @ e797
——训练侧差距已存在，非 test 抽样运气。

### 5.2 物理与几何

| 指标（all / dn / sn / snow_single / snow_double / limiter） | exp203 | exp105（参考） |
|---|---|---|
| GS 残差 core：pred | 0.070 / 0.055 / 0.181 / 0.040 / 0.021 / 0.053 | 0.060 / 0.044 / 0.157 / 0.038 / 0.016 / 0.047 |
| GS 残差 core：truth（FD 下限） | 0.0023 / 0.0013 / 0.0043 / 0.0025 / 0.0010 / 0.0022 | 0.0029 / 0.0019 / 0.0021 / 0.0027 / 0.0013 / 0.0067 |
| X 点 lo (cm) | 3.23 / 1.35 / 4.61 / 3.14 / 3.95 / NaN | 3.09 / 1.03 / 5.00 / 2.74 / 3.87 / NaN |
| O 点 (cm) | 8.36 / 1.01 / 30.7 / 92.1\* / 26.3\* / NaN | 8.87 / 0.78 / 30.8 / 84.5\* / 29.5\* / NaN |
| n_xpt_fail | **0** / 0 / 0 / 0 / 0 / 0 | 0 |

\* snowflake 桶 O 点指标受检测算法局限污染（同 exp105 §6），非预测质量。

### 5.3 交叉评估（差异归属定位）

| 模型 | g3 test（exp203 数据） | v6_clean test（exp105 数据） |
|---|---|---|
| **exp105 ckpt**（v6 训练） | 3.642% | **2.362%** |
| **exp203 ckpt**（g3 训练） | **2.963%** | 4.689% |

1. 同一模型跨数据 +1.28pp（exp105 ckpt）/+1.73pp（exp203 ckpt）——
   差异是**数据侧固有**，不是训练失败或子集运气
2. 与 exp201 的双向互通不同：exp203 模型测 v6_clean 达 4.689%（远差于
   exp105 模型的 2.362%）——**g3 训练出的模型学到 g3 特有数值特征，对
   v6 不通用**；五配置（含 SN 硬门约束的 X 点、雪点、触壁）下 gspack 与
   freegs_snow 的场差异比 exp201 双配置更显著

### 5.4 解读（诚实负面结果）

1. **数据质量目标达成**：SN 病态 0/200、gs_true>15 0/200（v6 raw 26/200）、
   \|Z_axis\|/\|Z_lo\|>0.5 frac 0、filter_v6 独立复核 15 split 移除 0 条
   （详见 `data_gspack2_v2/README.md` §8）——**生成侧硬门全部生效**
2. **误差目标未达成**：六桶全面 +0.1~+1.8pp（SN 最差 +1.84pp），且 val
   侧（3.540 vs 3.147%）与交叉评估（双向 +1.3~+1.7pp）三重证据一致——
   **gspack 数据的固有误差下限高于 freegs_snow 数据**（exp201 已见模式，
   五配置下更显著；SN 的 X 点约束对数值细节最敏感）
3. **"更干净" ≠ "更容易学"**：g3 SN 无病态（分布更紧），但 gspack 数值
   求解的场高频特征与 freegs_snow 系统不同——g3 模型在 g3 数据上
   2.96% 自洽，但跨到 v6 4.69% 不通用。数据质量（病态率）与可学习性
   （误差下限）是两个正交维度
4. **物理残差机制在 g3 数据上依然成立**：GS pred/truth 同量级关系
   （all 0.070 vs exp105 0.060）、X 点 3.23 cm（exp105 3.09 cm）、
   n_xpt_fail=0——但绝对水平略高，与 rel L2 趋势一致

## 6. 复现

```bash
bash dn_fno_2608/scripts/run_generate_g3.sh gen   # 数据生成（15 split）
bash dn_fno_2608/scripts/run_exp203_pino.sh train # 训练
bash dn_fno_2608/scripts/run_exp203_pino.sh eval  # 六桶评估
```
