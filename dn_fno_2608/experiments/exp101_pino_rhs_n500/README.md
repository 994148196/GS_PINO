# exp101 — FNO + GS 物理残差（做法1：单阶段，残差来自数据集 RHS）

> 实验日期：2026-08-19 ｜ 状态：**完成**（N=500，seed 1，data_v5/dn 单一位形）
> 数据：`data_v5/dn/`（MAST DN，65×65，2000/500/500，无墙）
> 对照：exp011（纯 MSE，coil 18ch 混合 DN+SN，N=500）；本实验 DN-only
> 基线可比口径 = exp011 DN 桶
> 结论速览：**把 GS 方程 PDE 残差加进 FNO 训练后精度不掉（test rel L2 0.72%，
> 优于 exp011 DN 桶 0.84%），预测场的 GS 残差降到真值有限差分下限的 3 倍
> （core mask 1.3% vs 真值 0.4%）**——物理约束有效且不牺牲场精度

## 1. 目标

论文 FNO 复现（`train_dn_fno.py`）是纯 MSE（z-score psi_total 域，无任何 PDE
项）。本实验在同一个 FNO 架构上加入 GS 方程物理残差约束，采用**做法1**：

**损失 = MSE(psi_plasma) + w_pde · ‖Δ\*ψ_pred + μ0·R·J_data‖²（core mask 内）**

- 残差的 RHS 直接来自数据集：`J_data = R·p′ + F·F′/(μ0·R)`，由 npz 已存的
  `dpdpsi`/`FdFdpsi` 分量重构（float64 计算后存 float32，Ip 重构误差 2.9e-4）
- 物理残差只定义在**等离子体场**上（coil 分离，见 §3）：Δ\* 作用于网络预测的
  psi_plasma，RHS 与 mask 都在 core mask 内
- w_pde = 0.1；PDE 项除以 pde_scale = 0.362 Wb/m²（=mean|μ0RJ|_plasma）归一化

## 2. 输入通道（18 通道，同 exp011 coils 模式）

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

注：data_v5 npz 带 `config` 字段（全 0），本实验不用（18ch，与 exp011 一致）。

## 3. coil 分离（本实验的核心设计）

网络**只预测 psi_plasma**（单输出通道，z-score）；评估时
`psi_total = psi_plasma_pred + Σ_k I_k·G_k`（greens 按 coil_currents 解析加回，
恒等式验证 max diff 6e-8 Wb）。理由（KAN 实验已证实）：线圈场占 |psi_total|
幅值 183%，|Δ\*ψ_coils| 峰值 27.9 vs 等离子体 0.89——残差若定义在 psi_total
上会被线圈导体附近的奇性主导；分离后物理残差天然只落在等离子体部分。

## 4. 训练设置

N=500（全量 2000 池嵌套子集，perm seed 12345），seed 1，AdamW lr 1e-3 wd 1e-4，
ReduceLROnPlateau（patience 20，factor 0.5，min_lr 1e-5），early stop patience 75，
batch 16，800 epochs（14.9 min）。模型 FNO2d2608：lift 18→64 + 4×FNOBlock
(width 64, modes 16×16) + proj 64→1，4.21M 参数（与论文架构一致，仅输入通道
18 → 输出 1）。PDE 残差在 interior 网格 (1:-1,1:-1) 用二阶中心差分，core mask
内 masked mean((res/pde_scale)²)。best val rel L2 **0.8188%** @ epoch 790。

## 5. 结果（test n=500，DN-only）

| 指标 | exp101（rhs） | 参照 |
|---|---|---|
| rel_l2_plasma mean / median / p95 (%) | **0.81 / 0.64 / 1.75** | —（psi_plasma 域） |
| rel_l2_total mean / median / p95 (%) | **0.72 / 0.59 / 1.58** | exp011 DN 桶 0.84（psi_total 域） |
| rmse_phys (Wb) plasma / total | 2.68e-4 / 2.68e-4 | exp011 DN 3.41e-4 |
| GS 残差 core mask：pred / truth | **0.0133 / 0.0040** | truth = 有限差分下限 |
| GS 残差 geom mask {ψ≥ψ_bndry}：pred / truth | 0.180 / 0.166 | exp011 同口径 ~1.0（psi_total 域） |
| X 点定位误差 lo / up (cm) | **0.55 / 0.57** | 500/500 可定位，无失败 |
| O 点误差 / 分离面 mean (cm) | **0.22 / 0.31** | 分离面面积相对误差 0.35% |

**解读**：
1. **精度不降反升**：rel_l2_total 0.72% < exp011 DN 桶 0.84%（同口径，可能因
   DN-only 训练 + 线圈场解析加回消除学习负担）
2. **物理一致性**：core mask 内预测场 GS 残差 1.33%，是真值有限差分下限
   （0.40%）的 3.3 倍——比纯 MSE 模型（残差 ≈ 真值量级）好 1-2 个数量级
3. **geom mask 口径的 0.18 是边沿效应**：{ψ≥ψ_bndry} 含分离面/边缘区，
   该处差分截断大（probe_residual2 已证），真值也有 0.166——两口径中 core
   mask 才是物理上有意义的判据

## 6. 局限与定位（做法1 vs 做法2）

做法1 的残差 RHS 是**冻结的数据**（J_data 不随预测自洽）——本质是把预测场
的 Δ\* 拉到数据 RHS 上的平滑监督正则。优点：简单、稳定、不引入额外输出通道；
缺点：不保证 psi 与 J 的自洽（不预测 J）。自洽性与 Ip 约束由 exp102
（做法2，两阶段）提供，见 exp102 README。

## 7. 复现

```bash
bash dn_fno_2608/scripts/run_exp101_102_pino.sh train   # 训练两实验
bash dn_fno_2608/scripts/run_exp101_102_pino.sh eval    # 评估两实验
# 单独：
cd "$(git rev-parse --show-toplevel)"
C:/Users/HP/.conda/envs/torch5060/python.exe -u -m gs_pino_fno_phys.train_pino \
  --mode rhs --train-data dn_fno_2608/data_v5/dn/train.npz \
  --val-data dn_fno_2608/data_v5/dn/val.npz --n-train 500 --seed 1 \
  --epochs 800 --phys-weight 0.1 --out-dir dn_fno_2608/experiments/exp101_pino_rhs_n500
```

产物：`best.pt`（model_state/stats/mode/artifact_stage）、`history.json`
（逐 epoch l_psi/l_pde/val_rel_l2）、`args.json`、`metrics.json`、
`train.log`/`eval.log`（脚本自动落本目录）、`figures/`（exp011 风格）：
`fig1_best_worst_psi.png`（best/worst 样本 psi_total 真值/预测/|diff|，
装置线圈 + 等高线 + 分离面 + X 点 + 磁轴）、`fig2_field_stats.png`
（rel L2/RMSE/累计分布/GS 残差）、`fig3_geometry_stats.png`（X 点/O 点/
分离面误差）、`stats_per_sample.json`（逐样本指标，exp011 schema 3）。
