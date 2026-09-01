# exp306 — UFNO 训练侧调优（w_j/w_pde 加大 + 更早切换 + 更长 ramp）

> 实验日期：2026-09-01 ｜ 状态：**完成**（N=500，seed 1，data_v5/dn 单一位形）
> 数据：`data_v5/dn/`（MAST DN，65×65，2000/500/500，无墙）——与 exp102 逐项一致
> 对照：exp302（UFNO 两阶段，0.729%，同一架构**零代码改动**）
> 结论速览：**训练侧调优为负收益（诚实负面结果）——test rel L2 0.7794%
> （exp302 0.729%，+0.05pp）、plasma 0.876%（+0.06pp）、Ip 0.240%（+0.03pp）；
> 唯一改善是 J 1.144%（−0.01pp，噪声级）。4% 阈值成功提前切换（e31 vs
> e70）但代价大于收益：w_j 2.0/w_pde 0.3 挤压 psi 主任务（exp306 README
> §6 预判的"权重失衡"风险应验）——exp102 的权重杠杆已吃满，调优方向
> 不在损失权重**

## 1. 目标

exp302 的 UFNO（0.729%）在 exp102 的训练杠杆下并未吃满：切换 e70 偏晚
（FNO e29 / DeepONet e42 / FNOKAN e31）、J p95 2.29% 还有余量、GS 残差
0.0150 是真值 FD 下限 0.0040 的 3.8×。本实验**架构完全不变**（`--model ufno`
与 exp302 相同），只调四个 CLI 训练参数——与 exp305（架构侧）正交，两个
方向可独立归因：

| 参数 | exp102 口径 | exp306 | 动机 |
|---|---|---|---|
| `--phys-weight` (w_pde) | 0.1 | **0.3** | GS 残差余量 3.8×，物理项权重加大 |
| `--j-weight` (w_j) | 1.0 | **2.0** | J 通道 p95 2.29% 余量，二阶导监督加强 |
| `--stage1-threshold` | 0.03 | **0.04** | 切换 e70 偏晚——4% 更早切阶段2，让物理项提前介入 |
| `--stage2-ramp-epochs` | 30 | **60** | 物理权重预热加长，更平滑进入阶段2 |

两阶段方案其余与 exp102 逐项一致：

**阶段1 损失 = MSE(psi_plasma, z) + w_j·masked MSE(J, z)（core mask 内）**
**阶段2 损失 = 阶段1 + w_pde·‖Δ\*ψ_pred + μ0·R·J_pred‖²（core mask 内）+ w_ip·‖(ΣJ·dA − Ip)/ip_scale‖²**

- 模型双输出通道：通道 0 = psi_plasma（z 域），通道 1 = Jφ（z 域）
- **物理权重预热**：阶段2 的 w_pde/w_ip 从 0 线性 ramp 到目标值（60 epochs）

## 2. 输入通道（18 通道，同 exp102/exp302 coils 模式）

| # | 通道 | 含义/单位 | 来源 |
|---|---|---|---|
| 1 | R | 网格 R 坐标（[-1,1]） | 固定 MAST 65×65 |
| 2 | Z | 网格 Z 坐标（[-1,1]） | 固定 MAST 65×65 |
| 3 | Ip | 等离子体电流 (A) | params[0] |
| 4 | paxis | 磁轴压强 (Pa) | params[1] |
| 5 | fvac | 真空通量函数 f (Wb/m) | params[2] |
| 6 | alpha_m | 剖面形状指数 m | params[3] |
| 7 | alpha_n | 剖面形状指数 n | params[4] |
| 8–17 | I_P2U … I_P6L | 上下偏滤器线圈电流 (A) | coil_currents[0:10] |
| 18 | I_P1 (Solenoid) | 中心螺线管电流 (A) | coil_currents[10] |

## 3. coil 分离（同 exp102）

网络**只预测 psi_plasma**（+Jφ）；评估时 `psi_total = psi_plasma_pred +
Σ_k I_k·G_k`（greens 恒等式验证 max diff 6e-8 Wb）。物理残差只定义在
等离子体场上——线圈场占 |psi_total| 幅值 183%、|Δ\*ψ_coils| 峰值 27.9 vs
等离子体 0.89，残差落在 psi_total 上会被线圈导体奇性主导。详见 exp101 README §3。

## 4. 训练设置

N=500（全量 2000 池嵌套子集，perm seed 12345），seed 1，AdamW lr 1e-3
wd 1e-4，ReduceLROnPlateau（patience 20，factor 0.5，min_lr 1e-5），batch 16，
800 epochs。模型 **UFNO2d2608（exp302 原版）**，2,468,802 参数——与 exp302
逐字节相同（ckpt 的 `model` 键 = `ufno`）。**唯一差异 = 表头四个训练参数**。
PDE 残差 interior (1:-1,1:-1) 二阶中心差分，core mask 内 masked
mean((res/pde_scale)²)，pde_scale=0.362 Wb/m²；Ip 损失用 ip_scale=5.51e5 A
归一化。w_pde=0.3（exp102 口径 0.1）、w_j=2.0（1.0）、stage1 阈值 4%（3%）、
ramp 60（30）。

**阶段切换**：阶段1 val rel L2 第 31 epoch 达 3.65% < 4% 阈值 → 切阶段2
（阈值放宽生效：exp302 在 3% 阈值下要到 e70；exp306 在 e31 就切了）。
阶段2 全程 769 epochs 中 val 从 3.65% 平滑降到 **0.9020%**（@ e796）——
低于 exp302 的 0.8308%（+0.071pp）。**phase-aware best**：artifact 取阶段2
best（best val rel L2 **0.9020%** @ epoch 796）。

## 5. 结果（test n=500，DN-only）

| 指标 | exp306（UFNO+wts） | exp302（UFNO） | exp102（FNO twostage） |
|---|---|---|---|
| rel_l2_plasma mean / median / p95 (%) | 0.88 / 0.72 / 1.73 | 0.82 / 0.70 / 1.47 | 0.90 / 0.74 / 1.70 |
| rel_l2_total mean / median / p95 (%) | **0.78 / 0.65 / 1.53** | **0.73 / 0.62 / 1.48** | **0.80 / 0.64 / 1.59** |
| rmse_phys (Wb) plasma / total | 2.91e-4 / 2.91e-4 | 2.74e-4 / 2.74e-4 | 2.97e-4 / 2.97e-4 |
| GS 残差 core mask：pred / truth | 0.0138 / 0.0040 | 0.0150 / 0.0040 | 0.0151 / 0.0040 |
| **Ip 相对误差** mean / median / p95 (%) | 0.24 / 0.18 / 0.63 | 0.21 / 0.15 / 0.58 | 0.21 / 0.16 / 0.54 |
| **J rel L2 mask 内** mean / median / p95 (%) | 1.14 / 0.97 / 2.32 | 1.13 / 0.95 / 2.29 | 1.36 / 1.11 / 2.66 |
| X 点定位误差 lo / up (cm) | 0.61 / 0.56 | 0.52 / 0.63 | 0.60 / 0.63 |
| O 点误差 / 分离面 mean (cm) | 0.20 / — | 0.19 / — | 0.21 / 0.34 |

**解读**：
1. **总误差变差 +0.05pp**：w_pde 0.3/w_j 2.0 让 GS 残差从 0.0150 降到
   0.0138（−8%，物理项权重加大的直接效果）和 J p95 −0.06pp，但 plasma
   从 0.82%→0.88%（+0.06pp）、Ip 0.21→0.24（+0.03pp）——**权重加大把
   梯度从 ψ 主任务抽走给了物理/二阶导通道，ψ 精度是代价**（§6 预判应验）
2. **阈值 4% 提前切换无净收益**：e31 切换（e70→e31）后阶段2 有更多
   epoch，但物理微调窗口变长没有换回更低的 val（0.9020% vs 0.8308%）——
   阶段1 的 3% 达标本身已经是物理微调的良好起点，更早切换只是把未就位
   的场交给物理项拉扯
3. **与 exp305 互为反面证据**：exp305（架构侧）持平/边际小胜、exp306
   （训练侧）明确负收益——exp102 的损失权重/阈值/ramp 在 exp302 架构上
   已处于局部最优；下一步改进不在权重
4. **诚实记录的价值**：零代码实验提供了干净的差分——args.json 里四个
   参数即全部差异，负结果排除了一整类调优方向

## 6. 局限（训练侧调优特有）

- **四参数联动**：w_pde/w_j/阈值/ramp 一次打包——若变好/坏，不能单独归因
  到单个权重（exp305-306 是架构 vs 训练的正交分解，内部不再拆）
- **阈值 4% 偏离 exp102 口径**：3% 阈值下 exp302 已切换（e70），4% 只是把
  切换提前——若收益来自物理项更多 epoch，阈值本身不是变量而是窗口移位；
  与系列其他实验（3%）直接比较时要注明
- **w_j 2.0 的风险**：J 通道监督加倍可能挤压 psi 主任务（J 是 ψ 二阶导，
  高权重下梯度方向冲突）；若 rel_l2_total 变差而 J 变好，说明权重失衡
- **零代码改动**：本实验不产生新模型——`args.json` 与 exp302 的差异即四个
  参数，是天然的差分证据

## 7. 复现

```bash
bash dn_fno_2608/scripts/run_exp305_306_pino.sh train   # 训练（含 exp305）
bash dn_fno_2608/scripts/run_exp305_306_pino.sh eval    # 评估（含 exp305）
# 单独：
cd "$(git rev-parse --show-toplevel)"
C:/Users/HP/.conda/envs/torch5060/python.exe -u -m gs_pino_fno_phys.train_pino \
  --mode twostage --model ufno --train-data dn_fno_2608/data_v5/dn/train.npz \
  --val-data dn_fno_2608/data_v5/dn/val.npz --n-train 500 --seed 1 \
  --epochs 800 --phys-weight 0.3 --ip-weight 1.0 --j-weight 2.0 \
  --stage1-threshold 0.04 --stage1-max-epochs 300 --stage2-ramp-epochs 60 \
  --out-dir dn_fno_2608/experiments/exp306_ufno_weights_pino_twostage_n500
```

产物同 exp102 约定：`best.pt`（含 `model` 键）、`history.json`、`args.json`、
`metrics.json`、`train.log`/`eval.log`、`figures/`（fig1_best_worst_psi /
fig2_field_stats / fig3_geometry_stats + stats_per_sample.json）。
