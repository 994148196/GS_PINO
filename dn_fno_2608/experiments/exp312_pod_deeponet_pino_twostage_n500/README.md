# exp312 — POD-DeepONet：数据低秩基替代学习 trunk（SVD top-p 模态）

> 实验日期：2026-09-01 ｜ 状态：**完成（诚实负面结果）**——低秩先验过度
> 激进，3 模态 ψ 基封顶 ~1.5%（N=500，seed 1，data_v5/dn 单一位形）
> 数据：`data_v5/dn/`（MAST DN，65×65，2000/500/500，无墙）——与 exp102 逐项一致
> 对照：exp307（DeepONet wide，0.7575%，学习 trunk）；exp304（DeepONet，0.802%）
> 结论速览：**POD 基 p_psi=3（99.92% 能量）/ p_j=10（99.93%）——GS 解空间
> 的极低秩性本身是数据集结构诊断，但残余 0.08% 恰是 X 点尺度细节：test
> rel L2 1.508%（exp307 的 2×）、X 点 0.85/0.85 cm（+0.3 cm）、J 2.52%、
> Ip 0.253%（物理约束在低秩空间仍有效）。"数据低秩 ⇒ 线性基够用"被否定：
> 学习 trunk（exp307）学到的表达超越了 SVD 线性子空间——99.92% 能量
> ≠ 0.7% 精度**

## 1. 目标

POD-DeepONet（[Lu 等 2022](https://arxiv.org/abs/2304.00643) 系列）：把
DeepONet 的学习 trunk 换成**训练数据导出的 POD 基**（SVD 右奇异向量），
分支网络只学系数：

$$\psi_z(R,Z) = \psi_0(R,Z) + \textstyle\sum_{k=1}^{p} b_k\,\phi_k(R,Z)$$

本实验用**自动能量判据**（累计能量 ≥ 99.9%，模态数封顶 256）在 N=500 训练
子集（z 域）上做 SVD——p 不是调参而是数据结构的读数：

- **p_psi = 3**（能量 99.924%）——ψ 解空间惊人地低秩（椭圆光滑性 +
  coil 分离后 5 标量 + 11 电流只张出 3 维主导子空间）
- **p_j = 10**（能量 99.926%）——J 是 ψ 的二阶导场，低秩性弱 ~3 倍，但
  10 模态仍极省
- 分支 MLP（16 标量 → 256×3 层）输出 13 个系数（psi 3 + j 10）；基为
  固定 buffer（无梯度）——**阶段2 物理微调只能移动系数**，低秩先验即实验
  假说：若基的线性子空间足够，系数微调应能达学习 trunk 同档精度

两阶段方案与 exp102 逐项一致：

**阶段1 损失 = MSE(psi_plasma, z) + w_j·masked MSE(J, z)（core mask 内）**
**阶段2 损失 = 阶段1 + w_pde·‖Δ\*ψ_pred + μ0·R·J_pred‖²（core mask 内）+ w_ip·‖(ΣJ·dA − Ip)/ip_scale‖²**

- 模型双输出通道：通道 0 = psi_plasma（z 域），通道 1 = Jφ（z 域）
- 切换条件：阶段1 val rel L2 < 3% 自动切阶段2（`--stage1-max-epochs 300` 兜底）
- **物理权重预热**：阶段2 的 w_pde/w_ip 从 0 线性 ramp 到目标值（30 epochs）

## 2. 输入通道（18 通道，同 exp102/exp307 coils 模式）

| # | 通道 | 含义/单位 | 来源 |
|---|---|---|---|
| 1 | R | 网格 R 坐标（[-1,1]） | 固定 MAST 65×65（不用于 POD 基重建） |
| 2 | Z | 网格 Z 坐标（[-1,1]） | 固定 MAST 65×65（不用于 POD 基重建） |
| 3 | Ip | 等离子体电流 (A) | params[0]（branch 输入） |
| 4 | paxis | 磁轴压强 (Pa) | params[1]（branch 输入） |
| 5 | fvac | 真空通量函数 f (Wb/m) | params[2]（branch 输入） |
| 6 | alpha_m | 剖面形状指数 m | params[3]（branch 输入） |
| 7 | alpha_n | 剖面形状指数 n | params[4]（branch 输入） |
| 8–17 | I_P2U … I_P6L | 上下偏滤器线圈电流 (A) | coil_currents[0:10]（branch 输入） |
| 18 | I_P1 (Solenoid) | 中心螺线管电流 (A) | coil_currents[10]（branch 输入） |

（分支输入 = 16 个标量通道 `x[:,2:].mean((2,3))`，同 DeepONet 约定；R/Z
通道在物理基重建中无角色——基已含空间结构，这与 exp307 的学习 trunk
（吃 R/Z）是核心差异。）

## 3. coil 分离（同 exp102）

网络**只预测 psi_plasma**（+Jφ）；评估时 `psi_total = psi_plasma_pred +
Σ_k I_k·G_k`（greens 恒等式验证 max diff 6e-8 Wb）。物理残差只定义在
等离子体场上——线圈场占 |psi_total| 幅值 183%、|Δ\*ψ_coils| 峰值 27.9 vs
等离子体 0.89，残差落在 psi_total 上会被线圈导体奇性主导。详见 exp101 README §3。

## 4. 训练设置

N=500（全量 2000 池嵌套子集，perm seed 12345），seed 1，AdamW lr 1e-3
wd 1e-4，ReduceLROnPlateau（patience 20，factor 0.5，min_lr 1e-5），batch 16，
800 epochs。模型 **PODDeepONet2d(pod_deeponet)**：SVD 基（p_psi=3 / p_j=10，
能量 99.92%/99.93%，n=500）+ 分支 16→256×3 层 → 13 系数，**275,984 参数**
（exp307 1.34M 的 0.21×、exp304 0.60M 的 0.46×）。PDE 残差 interior
(1:-1,1:-1) 二阶中心差分，core mask 内 masked mean((res/pde_scale)²)，
pde_scale=0.362 Wb/m²；Ip 损失用 ip_scale=5.51e5 A 归一化。

**阶段切换**：阶段1 val rel L2 第 16 epoch 达阈值（系列最快——线性基给了
阶段1 巨大起步优势；FNO e29 / DeepONet e42 / TKNO e66）。阶段2 中 val 先
降后平：best **1.6894%** @ e255，此后 75 epochs 无改进 → **e330 早停**
（仅 3.7 min 总训练——分支 13 系数 + 800 epochs 上限远未用满）。最佳
1.69% vs 切换值 2.19%——物理项在低秩子空间内只能再修 ~0.5pp。

## 5. 结果（test n=500，DN-only）

| 指标 | exp312（POD） | exp307（DeepONet wide） | exp304（DeepONet） |
|---|---|---|---|
| rel_l2_plasma mean / median / p95 (%) | 1.69 / 1.44 / 3.61 | 0.87 / 0.64 / 1.99 | 0.92 / 0.69 / 2.06 |
| rel_l2_total mean / median / p95 (%) | **1.51 / 1.27 / 3.26** | **0.76 / 0.57 / 1.69** | **0.80 / 0.63 / 1.87** |
| rmse_phys (Wb) plasma / total | 5.70e-4 / 5.70e-4 | 2.88e-4 / 2.88e-4 | 3.04e-4 / 3.04e-4 |
| GS 残差 core mask：pred / truth | 0.0812 / 0.0040 | 0.0234 / 0.0040 | 0.0298 / 0.0040 |
| **Ip 相对误差** mean / median / p95 (%) | 0.25 / 0.17 / 0.72 | 0.21 / 0.15 / 0.58 | 0.38 / 0.34 / 0.84 |
| **J rel L2 mask 内** mean / median / p95 (%) | 2.52 / 2.13 / 4.98 | 2.13 / 1.92 / 3.67 | 2.61 / 2.36 / 4.87 |
| X 点定位误差 lo / up (cm) | 0.85 / 0.85 | 0.59 / 0.51 | 0.73 / 0.70 |
| O 点误差 (cm) | 0.65 | 0.20 | 0.24 |
| n_xpt_fail | 0 | 0 | 0 |

**解读**：
1. **1.508% = exp307 的 2×——低秩先验过度激进（核心结论）**：3 模态 ψ 基
   的线性子空间装不下磁面细节；99.92% 能量缺口看似 0.08%，但截断掉的正是
   X 点尺度的结构（X 点误差 0.85 vs 0.59 cm、O 点 0.65 vs 0.20 cm）。能量
   判据（0.999）与精度判据（0.7% 档）是不同标尺
2. **J 2.52% 与 DeepONet 0.6M 的 2.61% 同档**：p_j=10 的低秩空间对 J 够用
   （J 平滑、mask 外为零）——psi 是短板，不是 J
3. **Ip 0.253%：物理约束在低秩空间仍有效**（exp304 0.384% 的 -34%）——
   积分约束只要求 (J, mask, dA) 匹配 Ip，低秩 J 完全可以承载
4. **"数据低秩 ⇒ 线性基够用"被否定**：exp307 学习 trunk 的 0.7575% 说明
   非线性 trunk 学到的表达在 3 模态线性子空间之外仍有 0.7pp 精度——低秩是
   数据的统计性质，不是解的完整描述；学习 trunk 等价于"自适应非线性 POD"
5. **p_psi=3 本身是数据集诊断**：v5/dn 在 coil 分离后，ψ 的跨样本变化由
   极少数模式主导（椭圆光滑性 + 参数化采样结构）——对未来数据增强/参数
   采样设计有直接意义（冗余维度的采样不会增加网络需学的模式）

## 6. 局限（POD 特有）

- **能量判据是"自动"选择**：p 由 99.9% 能量阈值决定，不是扫描——p=50 或
  100 模态的折中（更低重建误差 vs 更宽子空间）未测；结论"低秩基封顶 1.5%"
  只在 p=3/10 成立
- **线性子空间 vs X 点移动**：X 点位置随参数连续移动，在固定基下需要模态
  越多越好（非线性移动 ≈ 需要"谱系"外的表达）——这正是实验要暴露的机制，
  已如实暴露
- **物理微调只动系数**：阶段2 无法改变基本身——GS 残差项只能把系数推到
  子空间内最优，不能修正子空间外的残差（0.0812 残差 = 基外成分的指纹）
- **快不是优势的错觉**：3.7 min 训练（0.46× 参数、0.21× 容量）——速度
  收益来自放弃表达力，不是更好的学习
- **branch 输入是标量均值**：R/Z 通道被忽略（基含空间结构）——若未来做
  "POD + 学习 trunk 混合"（系数 + 残差双通道），R/Z 才重新参与

## 7. 复现

```bash
bash dn_fno_2608/scripts/run_exp311_312_pino.sh train   # 训练（含 exp311）
bash dn_fno_2608/scripts/run_exp311_312_pino.sh eval    # 评估（含 exp311）
# 单独：
cd "$(git rev-parse --show-toplevel)"
C:/Users/HP/.conda/envs/torch5060/python.exe -u -m gs_pino_fno_phys.train_pino \
  --mode twostage --model pod_deeponet --train-data dn_fno_2608/data_v5/dn/train.npz \
  --val-data dn_fno_2608/data_v5/dn/val.npz --n-train 500 --seed 1 \
  --epochs 800 --phys-weight 0.1 --ip-weight 1.0 --j-weight 1.0 \
  --stage1-threshold 0.03 --stage1-max-epochs 300 --stage2-ramp-epochs 30 \
  --out-dir dn_fno_2608/experiments/exp312_pod_deeponet_pino_twostage_n500
```

POD 基自动计算（SVD 于 N=500 训练子集，能量 99.9% 封顶 256 模态）并存入
`best.pt` 的 `pod_basis` 键；评估从 checkpoint 恢复基，无需重算。产物同
exp102 约定：`best.pt`、`history.json`、`args.json`、`metrics.json`、
`train.log`/`eval.log`、`figures/`（fig1_best_worst_psi / fig2_field_stats /
fig3_geometry_stats + stats_per_sample.json）。
