# exp305 — UFNO 架构调优（解码器谱化 + bottleneck 加深 + 模态扩增）

> 实验日期：2026-09-01 ｜ 状态：**完成**（N=500，seed 1，data_v5/dn 单一位形）
> 数据：`data_v5/dn/`（MAST DN，65×65，2000/500/500，无墙）——与 exp102 逐项一致
> 对照：exp302（UFNO 两阶段，0.729%，本系列当前最优）
> 结论速览：**架构调优边际小胜（诚实小正结果）——test rel L2 0.7147%
> （exp302 0.729%，−2%）、Ip 0.182%（−14%）、GS 残差 0.0146（持平），但
> J 1.20%（+0.07pp 略差）、X 点 lo 0.63（+0.11cm）；最大变化在阶段1：
> 切换 e30（exp302 e70）——解码器谱化/模态扩增加速了阶段1 收敛但未改变
> 精度上限，exp302 架构已接近本数据的上限（差距 0.014pp 在单 seed 噪声
> 范围内）**

## 1. 目标

exp302 的 UFNO 已经以 2.47M 参数（FNO 的 0.59×）拿到系列最优 0.729%，其
README §6 记录了三个明确短板，本实验把三个架构杠杆一次打包验证（exp306
单独验证训练侧权重，二者正交、可独立归因）：

1. **解码器谱化**（§6 局限 1"上采样路径无谱核"）：`_UpBlock` 的第二个 3×3
   conv 换成 `FNOBlock`（`_UpFNOBlock`）——上采样路径也在傅里叶空间混合。
   第一个 3×3 conv 保留（up+skip 的通道融合），模态按上采样后分辨率收缩：
   8²:4×4 → 16²:8×8 → 32²:8×8 → 65²:8×8。J 是 ψ 的二阶导场，对高频恢复最
   敏感，预期受益最大
2. **bottleneck 加深 1→2 层**：底层 8² [32, 4×4] 是全局磁面拓扑的最终承载，
   双层 FNOBlock 只 +16k 参数
3. **65² 层模态 16×16→20×20**（+0.59M）：`SpectralConv2dR` 的 min-guard
   是 `min(modes, H/W//2+1)`——65² 层上限 33、32² 层上限 17、16² 层上限 9、
   8² 层上限 5，只有 65² 层有明显扩增空间（高频细节由高分辨率层谱核负责）

两阶段方案与 exp102 逐项一致：

**阶段1 损失 = MSE(psi_plasma, z) + w_j·masked MSE(J, z)（core mask 内）**
**阶段2 损失 = 阶段1 + w_pde·‖Δ\*ψ_pred + μ0·R·J_pred‖²（core mask 内）+ w_ip·‖(ΣJ·dA − Ip)/ip_scale‖²**

- 模型双输出通道：通道 0 = psi_plasma（z 域），通道 1 = Jφ（z 域）
- 切换条件：阶段1 val rel L2 < 3% 自动切阶段2（`--stage1-max-epochs 300` 兜底）
- **物理权重预热**：阶段2 的 w_pde/w_ip 从 0 线性 ramp 到目标值（30 epochs）

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
800 epochs。模型 **UFNO2d2608(ufno_tuned)**：`modes=((20,20),(16,16),(8,8),
(4,4))`、`bottleneck_layers=2`、`dec_modes=((4,4),(8,8),(8,8),(8,8))`
（dec_modes 顺序 = 解码器自底向上 8²→16²→32²→65²），**3,600,354 参数**
（FNO 4.21M 的 0.85×，exp302 的 1.46×）。PDE 残差 interior (1:-1,1:-1) 二阶
中心差分，core mask 内 masked mean((res/pde_scale)²)，pde_scale=0.362
Wb/m²；Ip 损失用 ip_scale=5.51e5 A 归一化。

**阶段切换**：阶段1 val rel L2 第 30 epoch 达 2.92% < 3% 阈值 → 切阶段2
（exp302 是 e70——解码器谱化 + 模态扩增让阶段1 收敛快了一倍多，这是本
实验最显著的行为变化）。阶段2 全程 770 epochs 中 val 从 2.92% 平滑降到
**0.8253%**（@ e799，exp302 为 0.8308% @ e782，基本持平）。**phase-aware
best**：artifact 取阶段2 best（best val rel L2 **0.8253%** @ epoch 799）。
训练 20.0 min（exp302 约 30 min，模态扩增后 65² 层谱核更大但低分辨率层
更早切阶段1，总时长略降）。

## 5. 结果（test n=500，DN-only）

| 指标 | exp305（UFNO tuned） | exp302（UFNO） | exp102（FNO twostage） |
|---|---|---|---|
| rel_l2_plasma mean / median / p95 (%) | 0.81 / 0.65 / 1.76 | 0.82 / 0.70 / 1.47 | 0.90 / 0.74 / 1.70 |
| rel_l2_total mean / median / p95 (%) | **0.71 / 0.58 / 1.53** | **0.73 / 0.62 / 1.48** | **0.80 / 0.64 / 1.59** |
| rmse_phys (Wb) plasma / total | 2.70e-4 / 2.70e-4 | 2.74e-4 / 2.74e-4 | 2.97e-4 / 2.97e-4 |
| GS 残差 core mask：pred / truth | 0.0146 / 0.0040 | 0.0150 / 0.0040 | 0.0151 / 0.0040 |
| **Ip 相对误差** mean / median / p95 (%) | 0.18 / 0.13 / 0.51 | 0.21 / 0.15 / 0.58 | 0.21 / 0.16 / 0.54 |
| **J rel L2 mask 内** mean / median / p95 (%) | 1.20 / 0.96 / 2.51 | 1.13 / 0.95 / 2.29 | 1.36 / 1.11 / 2.66 |
| X 点定位误差 lo / up (cm) | 0.63 / 0.61 | 0.52 / 0.63 | 0.60 / 0.63 |
| O 点误差 / 分离面 mean (cm) | 0.19 / 0.30 | 0.19 / — | 0.21 / 0.34 |

**解读**：
1. **边际小胜、方向正确**：rel_l2_total **0.7147% vs 0.729%（−2%）**、Ip −14%、
   GS 残差 0.0146（−3%）——但 J +0.07pp、X 点 lo +0.11cm 小退；全部在
   单 seed 噪声范围内（exp102 系 3-seed 方差 ~0.02-0.03pp 量级），不能
   视为实质改进，也不能说退化
2. **阶段1 收敛翻倍（最显著的行为收益）**：切换 e30 vs e70——解码器谱化
   让高分辨率细节提前就位，2× 更早进入物理微调阶段；若下游是快速评估
   场景（训练时长敏感），这个行为收益比 0.014pp 更有实用价值
3. **exp302 架构接近上限**：三处调优（解码器谱化 + bottleneck×2 + 模态
   20×20）把参数从 2.47M 提到 3.60M 只换来 ~0.01pp——多尺度谱骨架的
   容量/模态分配已经在本数据上接近饱和；进一步改进大概率不在骨架侧
   （与 exp306 的负结果相互印证：训练杠杆也吃满了）
4. **代价可控**：3.60M 仍 < FNO 4.21M，训练 20 min（exp302 30 min）——
   若只看精度，exp302 与 exp305 等价；若看阶段1 收敛速度，exp305 更优

## 6. 局限（ufno_tuned 特有）

- **三改动打包**：解码器谱化、bottleneck 加深、模态扩增一次验证——若显著
  变好/变坏，无法单独归因到某一项（需要时拆解做消融，exp306 已覆盖训练侧
  正交维度）
- **解码器模态是折中**：65² 级解码器只给 8×8（16×16 会 +2.1M 参数、超过
  FNO 基线）；若解码器谱化有效，扩大该层模态是下一步
- **参数从 2.47M→3.60M**：调优方向与 exp302 的"参数效率"卖点有张力；
  若收益小于模态扩增的边际成本，结论应偏向 exp302 原设计
- **32²/16²/8² 层模态已近 min-guard 上限**（17/9/5），本实验只扩了 65² 层——
  扩低分辨率层需要改 `SpectralConv2dR` 支持近 Nyquist 模态，未做

## 7. 复现

```bash
bash dn_fno_2608/scripts/run_exp305_306_pino.sh train   # 训练（含 exp306）
bash dn_fno_2608/scripts/run_exp305_306_pino.sh eval    # 评估（含 exp306）
# 单独：
cd "$(git rev-parse --show-toplevel)"
C:/Users/HP/.conda/envs/torch5060/python.exe -u -m gs_pino_fno_phys.train_pino \
  --mode twostage --model ufno_tuned --train-data dn_fno_2608/data_v5/dn/train.npz \
  --val-data dn_fno_2608/data_v5/dn/val.npz --n-train 500 --seed 1 \
  --epochs 800 --phys-weight 0.1 --ip-weight 1.0 --j-weight 1.0 \
  --stage1-threshold 0.03 --stage1-max-epochs 300 --stage2-ramp-epochs 30 \
  --out-dir dn_fno_2608/experiments/exp305_ufno_tuned_pino_twostage_n500
```

产物同 exp102 约定：`best.pt`（含 `model` 键）、`history.json`、`args.json`、
`metrics.json`、`train.log`/`eval.log`、`figures/`（fig1_best_worst_psi /
fig2_field_stats / fig3_geometry_stats + stats_per_sample.json）。
