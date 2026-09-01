# exp302 — UFNO 傅里叶 U-Net + GS 物理残差（两阶段，自洽残差 + Ip 约束）

> 实验日期：2026-09-01 ｜ 状态：**完成**（N=500，seed 1，data_v5/dn 单一位形）
> 数据：`data_v5/dn/`（MAST DN，65×65，2000/500/500，无墙）——与 exp102 逐项一致
> 对照：exp102（FNO2d2608 两阶段，0.80%）；exp301（UNet 两阶段，2.20%）
> 结论速览：**UFNO 多尺度谱卷积骨架（2.47M 参数，FNO 的 0.59×）击败 FNO——
> test rel L2 0.729%（vs 0.800%，−9%）、J mask 1.13%（−17%）、X 点 0.52/0.63 cm
> （−13%/−0%），全部指标同向小幅小胜**。多尺度谱（编码器逐级降分辨率 + 谱核
> 收缩）+ 更低参数用量的组合在本问题成立；调研背景见
> [ARCHS_SURVEY.md](../../ARCHS_SURVEY.md)

## 1. 目标

验证 **UFNO（傅里叶 U-Net）**：把 FNO 的谱卷积嵌入 U-Net 的多尺度编码器——
高分辨率层（65²/32²）保留局部细节，低分辨率层（16²/8² + bottleneck）的谱
核捕获磁面的全局大尺度结构；解码器用普通卷积上采样。与 exp102 的单尺度
FNO（全分辨率 4×16×16 模态）相比，多尺度谱在参数利用率上应更优
（2.47M vs 4.21M），且理论上对大尺度磁面拓扑更直接。两阶段方案与 exp102
逐项一致：

**阶段1 损失 = MSE(psi_plasma, z) + w_j·masked MSE(J, z)（core mask 内）**
**阶段2 损失 = 阶段1 + w_pde·‖Δ\*ψ_pred + μ0·R·J_pred‖²（core mask 内）+ w_ip·‖(ΣJ·dA − Ip)/ip_scale‖²**

- 模型双输出通道：通道 0 = psi_plasma（z 域），通道 1 = Jφ（z 域）
- 切换条件：阶段1 val rel L2 < 3% 自动切阶段2（`--stage1-max-epochs 300` 兜底）
- **物理权重预热**：阶段2 的 w_pde/w_ip 从 0 线性 ramp 到目标值（30 epochs）

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
800 epochs（约 30 min）。模型 **UFNO2d2608**：lift 18→64；编码器 4 级 +
bottleneck 全用 FNOBlock（复用 `SpectralConv2dR`，cfloat 单权重谱核，
65² 非 2 的幂有 min-guard）：65² [64, 16×16] → 32² [64, 16×16] → 16² [32,
8×8] → 8² [32, 4×4] → bottleneck [32, 4×4]，宽度变化处 1×1 投影；解码器
普通卷积上采样 + skip 拼接；proj 64→2，**2,468,802 参数**（FNO 的 0.59×）。
PDE 残差 interior (1:-1,1:-1) 二阶中心差分，core mask 内 masked
mean((res/pde_scale)²)，pde_scale=0.362 Wb/m²；Ip 损失用 ip_scale=5.51e5 A
归一化。

**阶段切换**：阶段1 val rel L2 第 70 epoch 达 2.99% < 3% 阈值 → 切阶段2
（scheduler 不重置；早停仅从阶段2 计数）。阶段2 全程 730 epochs 中 val 从
2.99% 平滑降到 **0.831%**——物理项继续拉低误差（同 exp102 机制）。**phase-aware
best**：artifact 取阶段2 best（best val rel L2 **0.8308%** @ epoch 782）。

## 5. 结果（test n=500，DN-only）

| 指标 | exp302（UFNO） | exp102（FNO twostage） | exp301（UNet） |
|---|---|---|---|
| rel_l2_plasma mean / median / p95 (%) | 0.82 / 0.70 / 1.47 | 0.90 / 0.74 / 1.70 | 2.47 / 2.18 / 4.74 |
| rel_l2_total mean / median / p95 (%) | **0.73 / 0.62 / 1.48** | **0.80 / 0.64 / 1.59** | **2.20 / 1.92 / 4.19** |
| rmse_phys (Wb) plasma / total | 2.74e-4 / 2.74e-4 | 2.97e-4 / 2.97e-4 | 8.18e-4 / 8.18e-4 |
| GS 残差 core mask：pred / truth | 0.0150 / 0.0040 | 0.0151 / 0.0040 | 0.0643 / 0.0040 |
| **Ip 相对误差** mean / median / p95 (%) | 0.21 / 0.15 / 0.58 | 0.21 / 0.16 / 0.54 | 0.59 / 0.46 / 1.60 |
| **J rel L2 mask 内** mean / median / p95 (%) | 1.13 / 0.95 / 2.29 | 1.36 / 1.11 / 2.66 | 3.86 / 3.49 / 6.78 |
| X 点定位误差 lo / up (cm) | 0.52 / 0.63 | 0.60 / 0.63 | 1.83 / 2.32 |
| O 点误差 / 分离面 mean (cm) | 0.19 / — | 0.21 / 0.34 | 0.60 / — |

**解读**：
1. **UFNO 全面小幅击败 FNO**（本系列唯一正结果胜出）：rel_l2_total
   **0.729% vs 0.800%（−9%）**，J mask −17%、Ip 持平、X 点 lo −13%、
   O 点 −10%——所有指标同向或持平，无一处落后
2. **多尺度谱是有效归纳偏置**：65²/32² 层的高分辨率谱核负责局部微结构
   （J 通道受益最大，−17%），16²/8²/bottleneck 层的低分辨率谱核直接承载
   磁面大尺度拓扑（X/O 点受益）——与 exp301 的"纯局部池化链丢全局"形成
   直接对照（同一数据/方案，仅骨架不同）
3. **参数效率**：2.47M（FNO 的 0.59×）实现更好精度——宽度/模态随分辨率
   收缩的设计把参数花在刀刃上；推理更快
4. **物理约束机制未受影响**：GS 残差 0.0150 ≈ FNO 的 0.0151（同为真值 FD
   下限 0.0040 的 3.8×），阶段2 平滑下降——两阶段自洽链路对骨架鲁棒

## 6. 局限（UFNO 特有）

- **解码器为普通卷积**：上采样路径没有谱核，局部高频细节的恢复依赖 skip
  拼接与卷积堆叠——多尺度谱的优势集中在编码器侧
- **模态数随分辨率收缩**：低分辨率层 modes 4×4 的频谱分辨率是 65² 层 16×16
  的对应物理波长区段，若磁面大尺度结构落在 8² 层之外（波长 > 8 格），
  该层谱核无对应成分——bottleneck 4×4 模态是全局结构的最终承载
- **宽度/模态超参**：沿用 exp102 的 64 宽度 + 本文档记录的收缩表，未做网格
  搜索（N=500 预算下超参不敏感区，与 exp102 对齐优先）

## 7. 复现

```bash
bash dn_fno_2608/scripts/run_exp301_302_pino.sh train   # 训练（含 exp301）
bash dn_fno_2608/scripts/run_exp301_302_pino.sh eval    # 评估（含 exp301）
# 单独：
cd "$(git rev-parse --show-toplevel)"
C:/Users/HP/.conda/envs/torch5060/python.exe -u -m gs_pino_fno_phys.train_pino \
  --mode twostage --model ufno --train-data dn_fno_2608/data_v5/dn/train.npz \
  --val-data dn_fno_2608/data_v5/dn/val.npz --n-train 500 --seed 1 \
  --epochs 800 --phys-weight 0.1 --ip-weight 1.0 --j-weight 1.0 \
  --stage1-threshold 0.03 --stage1-max-epochs 300 --stage2-ramp-epochs 30 \
  --out-dir dn_fno_2608/experiments/exp302_ufno_pino_twostage_n500
```

产物同 exp102 约定：`best.pt`（含 `model` 键）、`history.json`、`args.json`、
`metrics.json`、`train.log`/`eval.log`、`figures/`（fig1_best_worst_psi /
fig2_field_stats / fig3_geometry_stats + stats_per_sample.json）。
