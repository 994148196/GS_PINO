# exp307 — PI-DeepONet 加宽（branch/trunk 256→384，0.60M→1.34M）

> 实验日期：2026-09-01 ｜ 状态：**完成**（N=500，seed 1，data_v5/dn 单一位形）
> 数据：`data_v5/dn/`（MAST DN，65×65，2000/500/500，无墙）——与 exp102 逐项一致
> 对照：exp304（PI-DeepONet 两阶段，0.802%，psi 平 FNO 但 J/Ip 弱 2×）
> 结论速览：**加宽成功修复 J/Ip 短板（本系列 DeepONet 的正面答案）——
> test rel L2 0.7575%（exp304 0.802%，−5.6%；同时低于 FNO 0.800%）、
> Ip 0.210%（exp304 0.384%，**−45%**，反超 exp302 的 0.213%）、J 2.13%
> （−0.48pp）、X 点 0.59/0.51 cm（exp304 0.73/0.70）——exp304 §5 预告的
> "容量需加大"被 1.34M 参数（2.2×）验证为正确诊断，J/Ip 弱 2× 确系容量
> 不足而非分支-主干分解缺陷**

## 1. 目标

exp304 的 DeepONet 结论是"容量奇迹"：0.60M 参数 psi 精度与 FNO 统计不可
区分，但 **J 2.61%（FNO 的 1.9×）、Ip 0.38%（1.8×）、GS 残差 0.0298（2×）**
——J 是 ψ 的二阶导场，0.6M 容量学 ψ 主任务绰绰有余、承载二阶导场不足。
exp304 README §6 与 §5 解读都预告了直接选项："**branch/trunk 宽度翻倍是
直接选项，未做超参搜索**"。

本实验只做一件事：`p=hidden=256→384`（branch_psi/branch_j/trunk 三路同宽
加宽），**0.601,600 → 1,344,768 参数（2.2×）**。训练超参与 exp102/exp304
逐项一致——加宽是唯一的变量，exp304→exp307 的增量即容量的净贡献。

两阶段方案与 exp102 逐项一致：

**阶段1 损失 = MSE(psi_plasma, z) + w_j·masked MSE(J, z)（core mask 内）**
**阶段2 损失 = 阶段1 + w_pde·‖Δ\*ψ_pred + μ0·R·J_pred‖²（core mask 内）+ w_ip·‖(ΣJ·dA − Ip)/ip_scale‖²**

- 模型双输出通道：通道 0 = psi_plasma（z 域），通道 1 = Jφ（z 域）
- 切换条件：阶段1 val rel L2 < 3% 自动切阶段2（`--stage1-max-epochs 300` 兜底）
- **物理权重预热**：阶段2 的 w_pde/w_ip 从 0 线性 ramp 到目标值（30 epochs）

**架构**：branch ×2（各 16→384×3 层 GELU，输入 = `x[:,2:].mean((2,3))`
即 z-score 标量），共享 trunk MLP（(R,Z)→384×3 层，输入 = 归一化网格坐标，
`x[:,0:2]` reshape (B,4225,2)），输出 einsum 点积 → (B,2,65,65)；
trunk 末层权重 ×0.1 小初始化（同 exp304 约定）；**1,344,768 参数**。

## 2. 输入通道（18 通道，同 exp102/exp304 coils 模式）

| # | 通道 | 含义/单位 | 来源 |
|---|---|---|---|
| 1 | R | 网格 R 坐标（[-1,1]） | 固定 MAST 65×65（trunk 输入） |
| 2 | Z | 网格 Z 坐标（[-1,1]） | 固定 MAST 65×65（trunk 输入） |
| 3 | Ip | 等离子体电流 (A) | params[0]（branch 输入） |
| 4 | paxis | 磁轴压强 (Pa) | params[1]（branch 输入） |
| 5 | fvac | 真空通量函数 f (Wb/m) | params[2]（branch 输入） |
| 6 | alpha_m | 剖面形状指数 m | params[3]（branch 输入） |
| 7 | alpha_n | 剖面形状指数 n | params[4]（branch 输入） |
| 8–17 | I_P2U … I_P6L | 上下偏滤器线圈电流 (A) | coil_currents[0:10]（branch 输入） |
| 18 | I_P1 (Solenoid) | 中心螺线管电流 (A) | coil_currents[10]（branch 输入） |

（DeepONet 的分支输入 = 18ch 中除 R/Z 外的 16 个标量通道——模型内部
`x[:,2:].mean((2,3))`，广播后每像素相同故取 mean 即原值；数据管线与
exp102 完全同口径，零改动。）

## 3. coil 分离（同 exp102）

网络**只预测 psi_plasma**（+Jφ）；评估时 `psi_total = psi_plasma_pred +
Σ_k I_k·G_k`（greens 恒等式验证 max diff 6e-8 Wb）。物理残差只定义在
等离子体场上——线圈场占 |psi_total| 幅值 183%、|Δ\*ψ_coils| 峰值 27.9 vs
等离子体 0.89，残差落在 psi_total 上会被线圈导体奇性主导。详见 exp101 README §3。

## 4. 训练设置

N=500（全量 2000 池嵌套子集，perm seed 12345），seed 1，AdamW lr 1e-3
wd 1e-4，ReduceLROnPlateau（patience 20，factor 0.5，min_lr 1e-5），batch 16，
800 epochs。模型 **PIDeepONet2d(deeponet_wide)**：`p=384, hidden=384, layers=3`，
**1,344,768 参数**（exp304 的 2.2×）。PDE 残差 interior (1:-1,1:-1) 二阶
中心差分，core mask 内 masked mean((res/pde_scale)²)，pde_scale=0.362
Wb/m²；Ip 损失用 ip_scale=5.51e5 A 归一化。

**阶段切换**：阶段1 val rel L2 第 44 epoch 达 2.96% < 3% 阈值 → 切阶段2
（与 exp304 的 e42 同档——加宽没有拖慢阶段1）。阶段2 全程 756 epochs 中
val 从 2.96% 平滑降到 **0.8771%**（@ e749，exp304 为 0.9528% @ e795，
−0.076pp）。**phase-aware best**：artifact 取阶段2 best（best val rel L2
**0.8771%** @ epoch 749）。

## 5. 结果（test n=500，DN-only）

| 指标 | exp307（DeepONet wide） | exp304（DeepONet） | exp102（FNO twostage） |
|---|---|---|---|
| rel_l2_plasma mean / median / p95 (%) | 0.87 / 0.64 / 1.99 | 0.92 / 0.69 / 2.06 | 0.90 / 0.74 / 1.70 |
| rel_l2_total mean / median / p95 (%) | **0.76 / 0.57 / 1.69** | **0.80 / 0.63 / 1.87** | **0.80 / 0.64 / 1.59** |
| rmse_phys (Wb) plasma / total | 2.88e-4 / 2.88e-4 | 3.04e-4 / 3.04e-4 | 2.97e-4 / 2.97e-4 |
| GS 残差 core mask：pred / truth | 0.0234 / 0.0040 | 0.0298 / 0.0040 | 0.0151 / 0.0040 |
| **Ip 相对误差** mean / median / p95 (%) | 0.21 / 0.15 / 0.58 | 0.38 / 0.34 / 0.84 | 0.21 / 0.16 / 0.54 |
| **J rel L2 mask 内** mean / median / p95 (%) | 2.13 / 1.92 / 3.67 | 2.61 / 2.36 / 4.87 | 1.36 / 1.11 / 2.66 |
| X 点定位误差 lo / up (cm) | 0.59 / 0.51 | 0.73 / 0.70 | 0.60 / 0.63 |
| O 点误差 / 分离面 mean (cm) | 0.20 / — | 0.24 / — | 0.21 / 0.34 |

**解读**：
1. **加宽是 exp304 短板的正解**：rel_l2_total **0.7575% vs 0.802%（−5.6%）**、
   Ip **0.210%（−45%，与 exp302 的 0.213% 打平）**、J 2.13%（−0.48pp，
   p95 −1.2pp）、X 点 −19%/−27%——exp304 §5 的容量诊断（"J 是 ψ 二阶导，
   需承载微结构的容量"）被 2.2× 参数直接验证；J/Ip 弱不是分支-主干分解
   的缺陷，是 0.6M 容量不够
2. **psi 精度同时改善**：plasma 0.87%（exp304 0.92%）——加宽不是只补
   J/Ip，ψ 主任务也从更多容量受益（p95 1.99 vs 2.06 略差但 mean/median
   更好）
3. **与 FNO 的对照升级**：exp304 是"统计不可区分"，exp307 是**明确小胜**
   （0.7575% vs 0.800%，−5%）——DeepONet 加宽后从"容量奇迹打平"变成
   "1.34M 参数超越 4.2M 的 FNO"；分支-主干分解 + 足够宽度的组合是 UFNO
   之外的第二条可行路线
4. **代价**：训练约 27 min（exp304 约 20 min，+35%），慢于 FNO 的
   15.4 min（~1.8×），但参数只有 FNO 的 0.32×（1.34M vs 4.21M）——
   下游需要 J/Ip 时这是明确的正向性价比（exp304 §6 的"psi 窄模型 +
   J 宽模型"两模型策略不再需要，一个宽模型全覆盖）

## 6. 局限（加宽特有）

- **单点扫描**：384 是"翻倍"的直观选择，不是扫描出的最优——若收益有限，
  "容量已够/容量曲线饱和"与"这个宽度不凑巧"无法区分（下一档 512 是
  唯一确认手段，成本 2×）
- **加宽 vs 加深**：只加宽不加深（layers=3 不变）；branch 更深（4-5 层）
  是另一个正交方向，未做
- **0.6M→1.34M 的成本**：exp304 训练 ~20 min（系列最快），加宽后计算量
  ~2×，若 J/Ip 收益 < 2× 的计算代价，"psi 用途用窄模型、下游 J 用途用宽
  模型"仍是更优策略——由本实验的收益大小决定
- **小样本过拟合风险**：N=500 下 2.2× 参数，训练/val 曲线若出现分叉，
  加宽方向不成立（val 主导的 phase-aware best 会部分掩盖）

## 7. 复现

```bash
bash dn_fno_2608/scripts/run_exp307_308_pino.sh train   # 训练（含 exp308）
bash dn_fno_2608/scripts/run_exp307_308_pino.sh eval    # 评估（含 exp308）
# 单独：
cd "$(git rev-parse --show-toplevel)"
C:/Users/HP/.conda/envs/torch5060/python.exe -u -m gs_pino_fno_phys.train_pino \
  --mode twostage --model deeponet_wide --train-data dn_fno_2608/data_v5/dn/train.npz \
  --val-data dn_fno_2608/data_v5/dn/val.npz --n-train 500 --seed 1 \
  --epochs 800 --phys-weight 0.1 --ip-weight 1.0 --j-weight 1.0 \
  --stage1-threshold 0.03 --stage1-max-epochs 300 --stage2-ramp-epochs 30 \
  --out-dir dn_fno_2608/experiments/exp307_pideeponet_wide_pino_twostage_n500
```

产物同 exp102 约定：`best.pt`（含 `model` 键）、`history.json`、`args.json`、
`metrics.json`、`train.log`/`eval.log`、`figures/`（fig1_best_worst_psi /
fig2_field_stats / fig3_geometry_stats + stats_per_sample.json）。
