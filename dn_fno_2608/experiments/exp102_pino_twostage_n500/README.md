# exp102 — FNO + GS 物理残差（做法2：两阶段，自洽残差 + Ip 约束）

> 实验日期：2026-08-19 ｜ 状态：**完成**（N=500，seed 1，data_v5/dn 单一位形）
> 数据：`data_v5/dn/`（MAST DN，65×65，2000/500/500，无墙）
> 对照：exp101（做法1 单阶段 RHS 残差）；exp011（纯 MSE，coil 18ch 混合 DN+SN）
> 结论速览：**两阶段成功——阶段1 监督 psi_plasma+J（第 29 epoch 达阈值切阶段2），
> 阶段2 加自洽 PDE 残差 + Ip 约束（物理权重 30-epoch 线性预热，修复了首版
> 阶段2 首 epoch 的爆炸）。最终 test rel L2 0.80%（≈ exp101 0.72%，+0.08 的
> 代价换来 J 自洽与 Ip 约束），mask 内 J rel L2 1.36%、Ip 误差 0.21%**——
> psi 与 J 自洽（J ≈ −Δ\*ψ/μ0R），且从纯数据管线拿不到的这个自洽性、Ip
> 约束，做法1 提供不了

## 1. 目标

做法2（两阶段学习）：**阶段1 先学好 psi_plasma 与 Jφ 的监督拟合，阶段2
再叠加两者之间的自洽物理残差与 Ip 积分约束**。与 exp101（做法1，残差 RHS
来自数据集冻结的 J_data）的区别：本实验的 PDE 残差用**网络自预测的 J**，
强制 psi↔J 自洽，并显式约束 Ip——这正是做法1 给不了的（做法1 不预测 J）。

**阶段1 损失 = MSE(psi_plasma, z) + w_j·masked MSE(J, z)（core mask 内）**
**阶段2 损失 = 阶段1 + w_pde·‖Δ\*ψ_pred + μ0·R·J_pred‖²（core mask 内）+ w_ip·‖(ΣJ·dA − Ip)/ip_scale‖²**

- 模型双输出通道：通道 0 = psi_plasma（z 域），通道 1 = Jφ（z 域）
- J 目标 = 数据重构值 `R·p′ + F·F′/(μ0·R)`（与 exp101 同源，float64 计算）
- **切换条件（用户要求"阶段1误差小才进阶段2"）**：阶段1 val rel L2
  （psi_plasma z 域）< 3% 自动切阶段2；`--stage1-max-epochs 300` 兜底
- **物理权重预热（关键修复）**：阶段2 的 w_pde/w_ip 从 0 线性 ramp 到目标值
  （30 epochs）。首版无 ramp 时阶段2 首 epoch pde 项 5.38（MSE 的 500 倍）、
  val rel L2 从 2.67% 跳到 54%——阶段1 结束 psi 与 J 各自好，但二者自洽关系
  未学（这正是阶段2 要学的），满权重 PDE 梯度与监督梯度剧烈冲突。ramp 后
  pde 项稳定在 ~5e-3，val 平滑下降（详见记忆 twostage-physics-weight-ramp）。

## 2. 输入通道（18 通道，同 exp101/exp011 coils 模式）

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

## 3. coil 分离（同 exp101）

网络**只预测 psi_plasma**（+Jφ）；评估时 `psi_total = psi_plasma_pred +
Σ_k I_k·G_k`（greens 恒等式验证 max diff 6e-8 Wb）。物理残差只定义在
等离子体场上——线圈场占 |psi_total| 幅值 183%、|Δ\*ψ_coils| 峰值 27.9 vs
等离子体 0.89，残差落在 psi_total 上会被线圈导体奇性主导。详见 exp101 README §3。

## 4. 训练设置

N=500（全量 2000 池嵌套子集，perm seed 12345），seed 1，AdamW lr 1e-3
wd 1e-4，ReduceLROnPlateau（patience 20，factor 0.5，min_lr 1e-5），batch 16，
800 epochs（15.4 min）。模型 FNO2d2608：lift 18→64 + 4×FNOBlock（width 64，
modes 16×16）+ proj 64→**2**，4.21M 参数。PDE 残差 interior (1:-1,1:-1)
二阶中心差分，core mask 内 masked mean((res/pde_scale)²)，pde_scale=0.362
Wb/m²；Ip 损失用 ip_scale=5.51e5 A 归一化。

**阶段切换**：阶段1 val rel L2 第 29 epoch 达 2.67% < 3% 阈值 → 切阶段2
（scheduler 不重置；早停仅从阶段2 计数）。**阶段2 全程 767 epochs 中 val
从 2.67% 平滑降到 0.90%**——物理项非但没有破坏拟合，反而继续拉低误差
（与 exp101 一致：PDE 监督等价于平滑正则）。

**phase-aware best**：阶段1/阶段2 各自记录 best；artifact 取阶段2 best
（best val rel L2 **0.9047%** @ epoch 796）。

## 5. 结果（test n=500，DN-only）

| 指标 | exp102（twostage） | exp101（rhs） | 参照 |
|---|---|---|---|
| rel_l2_plasma mean / median / p95 (%) | 0.90 / 0.74 / 1.70 | 0.81 / 0.64 / 1.75 | —（psi_plasma 域） |
| rel_l2_total mean / median / p95 (%) | **0.80 / 0.64 / 1.59** | **0.72 / 0.59 / 1.58** | exp011 DN 桶 0.84（psi_total 域） |
| rmse_phys (Wb) plasma / total | 2.97e-4 / 2.97e-4 | 2.68e-4 / 2.68e-4 | exp011 DN 3.41e-4 |
| GS 残差 core mask：pred / truth | **0.0151 / 0.0040** | **0.0133 / 0.0040** | truth = 有限差分下限 |
| GS 残差 geom mask {ψ≥ψ_bndry}：pred / truth | 0.249 / 0.166 | 0.180 / 0.166 | exp011 同口径 ~1.0（psi_total 域） |
| **Ip 相对误差** mean / median / p95 (%) | **0.21 / 0.16 / 0.54** | —（无 Ip 约束） | — |
| **J rel L2 mask 内** mean / median / p95 (%) | **1.36 / 1.11 / 2.66** | —（不预测 J） | — |
| X 点定位误差 lo / up (cm) | 0.60 / 0.63 | 499/500 可定位 | 与 exp101 同量级 |
| O 点误差 / 分离面 mean (cm) | 0.21 / 0.34 | 分离面面积相对误差 0.39% | — |
| J rel L2 全网格 (%) | 36.9 | — | mask 外振铃，无物理意义 |

**解读**：
1. **两阶段链路成立**：阶段2 加了自洽残差 + Ip 约束后 val 继续降到 0.90%
   （不是"加了物理就牺牲拟合"）——与 exp101 相同，PDE 项起平滑正则作用
2. **psi↔J 自洽**：mask 内 J rel L2 1.36% + GS 残差 1.51%（真值有限差分
   下限 0.40% 的 3.8 倍）同时成立——J ≈ −Δ\*ψ/(μ0R) 在预测对上是自洽的；
   这是做法1（J 冻结为数据）给不了的属性
3. **Ip 约束有效**：ΣJ·dA 相对误差 0.21%——能量/电流积分量级正确，下游
   （如平衡反演、Ip 监测）可用
4. **全网格 J rel L2 36.9% 是假指标**：J 目标在 mask 外为 0（硬截断），
   FNO 谱展开在截断处振铃；训练损失只算 mask 内，mask 内 1.36% 才是真实
   质量。评估时同口径报告两个数（j_rel_l2_pct / j_rel_l2_mask_pct）
5. **与 exp101 的取舍**：rel_l2_total 0.80% vs 0.72%（+0.08），GS 残差
   1.51% vs 1.33%（+0.18）——多约束竞争拟合自由度的小代价；换来 J 通道、
   Ip 约束与自洽性。做法1 更"省"，做法2 更"全"（见下）

## 6. 局限与定位（做法2 vs 做法1）

- **J 只保证 mask 内自洽**：mask 外 J 无定义（目标为 0 + 振铃），Ip 约束
  只在 mask 内积分——若下游需要 mask 外 J（如 edge 电流），需扩展 mask 或
  改目标为平滑延拓
- **阶段2 需要调 ramp**：ramp 期（30 epochs）物理权重从 0 升，切换瞬间
  仍是"纯监督"；ramp 过短会重现首版爆炸（见记忆
  twostage-physics-weight-ramp），过长浪费 epochs
- **切换阈值是经验值**：3% 是 N=500 DN 数据上的实测折中（该规模下阶段1
  可达 ~2.7%）；跨数据/位形需重新标定，兜底 `--stage1-max-epochs` 保证
  不卡死
- **两做法定位**：exp101（单阶段 RHS）＝把预测场 Δ\* 拉到数据 RHS 上的
  平滑正则，简单稳定、精度略优；exp102（两阶段自洽）＝多预测 J 通道 +
  psi↔J 自洽 + Ip 约束，适合需要场与电流同时输出的下游。二者共享同一
  数据/模型框架，只差损失配置（`--mode rhs|twostage`），可互相比对。

## 7. 复现

```bash
bash dn_fno_2608/scripts/run_exp101_102_pino.sh train   # 训练两实验
bash dn_fno_2608/scripts/run_exp101_102_pino.sh eval    # 评估两实验
# 单独：
cd "$(git rev-parse --show-toplevel)"
C:/Users/HP/.conda/envs/torch5060/python.exe -u -m gs_pino_fno_phys.train_pino \
  --mode twostage --train-data dn_fno_2608/data_v5/dn/train.npz \
  --val-data dn_fno_2608/data_v5/dn/val.npz --n-train 500 --seed 1 \
  --epochs 800 --phys-weight 0.1 --ip-weight 1.0 --j-weight 1.0 \
  --stage1-threshold 0.03 --stage1-max-epochs 300 --stage2-ramp-epochs 30 \
  --out-dir dn_fno_2608/experiments/exp102_pino_twostage_n500
```

产物：`best.pt`（model_state/stats/mode/switch_epoch/artifact_stage +
阶段1/2 各自 best）、`history.json`（逐 epoch stage/ramp/l_psi/l_j/l_pde/
l_ip/val_rel_l2）、`args.json`、`metrics.json`、
`train.log`/`eval.log`（脚本自动落本目录）、`figures/`（exp011 风格）：
`fig1_best_worst_psi.png`（best/worst 样本 psi_total 真值/预测/|diff|
3 行 + J 行，装置线圈 + 等高线 + 分离面 + X 点 + 磁轴）、
`fig2_field_stats.png`（rel L2/RMSE/累计分布/GS 残差 + Ip/J 直方图）、
`fig3_geometry_stats.png`（X 点/O 点/分离面误差）、
`stats_per_sample.json`（逐样本指标，exp011 schema 3 + J/Ip 列）。
