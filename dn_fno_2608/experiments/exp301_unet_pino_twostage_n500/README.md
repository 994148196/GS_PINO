# exp301 — U-Net CNN + GS 物理残差（两阶段，自洽残差 + Ip 约束）

> 实验日期：2026-09-01 ｜ 状态：**完成**（N=500，seed 1，data_v5/dn 单一位形）
> 数据：`data_v5/dn/`（MAST DN，65×65，2000/500/500，无墙）——与 exp102 逐项一致
> 对照：exp102（FNO2d2608 两阶段，0.80%）；exp101（做法1 单阶段 RHS，0.72%）
> 结论速览：**UNet 纯卷积骨架（2.76M 参数）明显劣于 FNO——test rel L2 2.20%
> （FNO 0.80% 的 2.75×）、X 点定位 1.83/2.32 cm（FNO 0.60/0.63 cm）、GS 残差
> 4.3×。阶段1 卡 3.017%（差 0.017 点未达 3% 阈值）e300 兜底切换，阶段2
> best val 2.571%——纯卷积的局部感受野不足以承载磁面的全局拓扑（O 点/X 点
> 位置），§6 预期正中**。调研背景与选型证据见
> [ARCHS_SURVEY.md](../../ARCHS_SURVEY.md)（EXL-50U 基准：CNN 为精度/稳健性/
> 速度最佳平衡——本实验是该结论在"两阶段物理约束"方案上的反例）

## 1. 目标

验证 **U-Net 纯卷积骨架** 是否胜任 GS ψ 预测（exp102 的 FNO 纯谱混合骨架之外
的第一个对照）。文献证据（EXL-50U 基准 arXiv 2608.23217）：CNN 在平衡重建类
问题上精度/稳健性/速度综合最优，与 FNO 并列 OOD 外推最稳——但该基准没有
两阶段物理约束方案，本实验把 exp102 的方案原样换骨架验证。两阶段方案与
exp102 逐项一致：

**阶段1 损失 = MSE(psi_plasma, z) + w_j·masked MSE(J, z)（core mask 内）**
**阶段2 损失 = 阶段1 + w_pde·‖Δ\*ψ_pred + μ0·R·J_pred‖²（core mask 内）+ w_ip·‖(ΣJ·dA − Ip)/ip_scale‖²**

- 模型双输出通道：通道 0 = psi_plasma（z 域），通道 1 = Jφ（z 域）
- 切换条件：阶段1 val rel L2 < 3% 自动切阶段2（`--stage1-max-epochs 300` 兜底）
- **物理权重预热**：阶段2 的 w_pde/w_ip 从 0 线性 ramp 到目标值（30 epochs）——
  无 ramp 时阶段2 首 epoch 爆炸（exp102 修复记录，见记忆 twostage-physics-weight-ramp）

## 2. 输入通道（18 通道，同 exp102/exp101 coils 模式）

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
800 epochs（约 30 min）。模型 **UNet2d2608**：depth 4，base_width 24，
通道 [24,48,96,192]+bottleneck 192，块 = 2×Conv3×3+GELU，MaxPool 下采样，
双线性上采样（显式 `size=`，65 非 2 的幂）、skip 拼接，**无归一化**（与 FNO
基线对齐），proj 64→2，**2,760,218 参数**（FNO 的 0.66×）。PDE 残差 interior
(1:-1,1:-1) 二阶中心差分，core mask 内 masked mean((res/pde_scale)²)，
pde_scale=0.362 Wb/m²；Ip 损失用 ip_scale=5.51e5 A 归一化。

**阶段切换**：阶段1 未达 3% 阈值（best 3.017% @ e122，差 0.017pp）→
**e300 兜底切换**；阶段2 全程 500 epochs 中 val 从 3.02% 缓降到 2.57%。
**phase-aware best**：阶段1/阶段2 各自记录 best；artifact 取阶段2 best
（best val rel L2 **2.5710%** @ epoch 800，无早停触发）。

## 5. 结果（test n=500，DN-only）

| 指标 | exp301（UNet） | exp102（FNO twostage） | exp101（FNO rhs） |
|---|---|---|---|
| rel_l2_plasma mean / median / p95 (%) | 2.47 / 2.18 / 4.74 | 0.90 / 0.74 / 1.70 | 0.81 / 0.64 / 1.75 |
| rel_l2_total mean / median / p95 (%) | **2.20 / 1.92 / 4.19** | **0.80 / 0.64 / 1.59** | **0.72 / 0.59 / 1.58** |
| rmse_phys (Wb) plasma / total | 8.18e-4 / 8.18e-4 | 2.97e-4 / 2.97e-4 | 2.68e-4 / 2.68e-4 |
| GS 残差 core mask：pred / truth | 0.0643 / 0.0040 | 0.0151 / 0.0040 | 0.0133 / 0.0040 |
| **Ip 相对误差** mean / median / p95 (%) | 0.59 / 0.46 / 1.60 | 0.21 / 0.16 / 0.54 | —（无 Ip 约束） |
| **J rel L2 mask 内** mean / median / p95 (%) | 3.86 / 3.49 / 6.78 | 1.36 / 1.11 / 2.66 | —（不预测 J） |
| X 点定位误差 lo / up (cm) | 1.83 / 2.32 | 0.60 / 0.63 | 499/500 可定位 |
| O 点误差 / 分离面 mean (cm) | 0.60 / — | 0.21 / 0.34 | — |

**解读**：
1. **纯卷积在本问题明显不足**（负结果）：rel_l2_total 2.20% = FNO 的 2.75×，
   X 点定位 3-4×、GS 残差 4.3×、J 2.8×——所有"全局"指标同步退化，指向
   **感受野/全局拓扑承载不足**：U-Net 的 4 级下采样最深 16×16 bottleneck 与
   FNO 的 65² 全分辨率 16×16 模态相比，全局信息在池化链中丢失过多
2. **阶段1 阈值卡点**：3.017% vs 3% 差 0.017pp——不是"差一点"的运气问题，
   e122 后 val 长期在 3.0-4.3% 震荡（lr 衰减后未见突破），与测试 2.20%
   一致：模型容量/归纳偏置的上限就在 ~2.5-3%
3. **物理约束对弱骨架的兜底**：阶段2 的 PDE 项仍把 val 从 3.02% 压到
   2.57%（−15%）——物理残差的平滑正则机制与骨架无关（exp101 的机制复现），
   只是起点太低
4. **与文献的对照**：EXL-50U 基准的"CNN 最佳平衡"结论不迁移到本问题——
   该基准是单阶段监督、无物理约束、且网格/位形集不同；本实验证明
   "两阶段物理约束方案 + 65² 全局磁面"下，纯局部归纳偏置是劣势

## 6. 局限（UNet 特有）

- **无全局感受野的显式机制**：U-Net 靠 4 级下采样 + skip 拼接覆盖 65² 全网格；
  磁面的大尺度拓扑（O 点/X 点位置）由最深 bottleneck 特征承载，理论上不如
  谱卷积的全局混合直接——若性能差于 exp102，这是首要解释变量
- **双线性上采样的信息损失**：解码器上采样是固定插值（非可学习转置卷积），
  高分辨率细节依赖 skip 拼接补偿
- **容量**：2.76M 参数（FNO 的 0.66×），卷积共享权重对小样本 N=500 是优点
  （过拟合风险低），但表达力上限由 base_width 决定

## 7. 复现

```bash
bash dn_fno_2608/scripts/run_exp301_302_pino.sh train   # 训练（含 exp302）
bash dn_fno_2608/scripts/run_exp301_302_pino.sh eval    # 评估（含 exp302）
# 单独：
cd "$(git rev-parse --show-toplevel)"
C:/Users/HP/.conda/envs/torch5060/python.exe -u -m gs_pino_fno_phys.train_pino \
  --mode twostage --model unet --train-data dn_fno_2608/data_v5/dn/train.npz \
  --val-data dn_fno_2608/data_v5/dn/val.npz --n-train 500 --seed 1 \
  --epochs 800 --phys-weight 0.1 --ip-weight 1.0 --j-weight 1.0 \
  --stage1-threshold 0.03 --stage1-max-epochs 300 --stage2-ramp-epochs 30 \
  --out-dir dn_fno_2608/experiments/exp301_unet_pino_twostage_n500
```

产物同 exp102 约定：`best.pt`（含 `model` 键）、`history.json`、`args.json`、
`metrics.json`、`train.log`/`eval.log`、`figures/`（fig1_best_worst_psi /
fig2_field_stats / fig3_geometry_stats + stats_per_sample.json）。
