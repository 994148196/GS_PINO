# 双零自由边界 Grad-Shafranov 平衡的 FNO 神经算子代理 — 论文复现详解

> 复现目标：*Millisecond-Scale Neural Operator Surrogates for Double-Null Free-Boundary Grad-Shafranov Equilibria*（Plamen G. Krastev, 2026，arXiv:2608.05555v1）。
> 本目录为全部复现产物：数据、代码、训练记录、评估结果与报告。
> 结果速览见 [outputs/report/REPORT.md](outputs/report/REPORT.md)，残差与纯数据驱动分析见 [outputs/report/ANALYSIS.md](outputs/report/ANALYSIS.md)，后续改进实验约定见 [experiments/README.md](experiments/README.md)。

---

## 1. 背景与问题

托卡马克边界位形设计依赖求解自由边界 Grad–Shafranov（GS）方程。经典求解器（freegs）一次平衡求解约 1.8 s（本机实测），无法支撑设计空间的快速扫描（控制线圈参数、剖面、位形拓扑）。

本文目标是训练一个 **Fourier 神经算子（FNO）**，把"输入位形参数 → 平衡场 ψ(R,Z)"的映射学成毫秒级代理：输入 **9 通道 65×65 场**（R、Z 坐标 + 3 个运行参数 + 4 个 X 点坐标），输出 **单通道 ψ 场**。训练数据由 freegs 在 **TestTokamak** 双零（double-null, DN）位形参数族上生成。

## 2. 数据集

### 2.1 参数空间（论文 Eq. 4）

| 参数 | 范围 / 分布 |
|---|---|
| Paxis（轴处压强） | 均匀 U[200, 3000] Pa |
| Ip（等离子体电流） | 均匀 U[5e4, 4e5] A |
| fvac（真空区 FF′ 常数） | 均匀 U[0.5, 3.0] |
| 下 X 点 (R_lo, Z_lo) | (1.1, −0.6) + δ，δ 分量独立 U[−0.02, 0.02] m |
| 上 X 点 (R_up, Z_up) | (1.1, +0.6) + δ，δ 分量独立 U[−0.02, 0.02] m（上下对称） |

X 点抖动幅度小是论文原文设定（Eq. 4），数据流形由 7 个标量参数张成。

### 2.2 求解器配置

- freegs（TestTokamak 几何），域 R∈[0.1, 2.0] m、Z∈[−2, 2] m，**65×65 均匀网格**；
- 双 X 点约束 + isoflux 约束（锚定外中平面固定点 (1.5, 0.0)），γ=1e-12；
- Picard 迭代 rtol=1e-3、maxits=50；
- 剖面：`ConstrainPaxisIp`（alpha_m=1.0, alpha_n=2.0）。

### 2.3 采样与接受准则

- 训练 5000 样本（seed 123）/ 验证 500（seed 456）/ 测试 500（seed 789）；
- 接受准则：求解收敛 且 |Ip_sol − Ip_tgt|/Ip_tgt ≤ 10% 且 find_critical 找到 ≥2 个 X 点；
- **6000/6000 全部通过**（与论文一致）；单样本中位求解时间约 1.08 s（本机 24 核并行生成）。
- 缩放实验：N∈{500,1000,2000,5000} 用固定置换（seed 12345）取训练集**嵌套子集**，大集合严格包含小集合。

### 2.4 数据文件与字段

`dn_fno_2608/data/{train,val,test}.npz`（float32，另有分块目录 train/ val/ test/ 为生成中间产物，合并后冗余）。每个样本保存**全字段**，为后续 PINO（物理约束）阶段预留：

| 字段 | 含义 |
|---|---|
| psi_total | 总极向磁通（目标，z-score 输出） |
| psi_plasma / psi_plasma_norm / psi_coils | 等离子体 / 归一化 / 线圈磁通分量 |
| R, Z | 物理坐标 (65,65)（米制） |
| mask | freegs critical.core_mask |
| coil_currents (4) | 4 个控制线圈电流（PINO 预留） |
| greens (4×65×65) | 线圈 Green 函数（PINO 预留） |
| dpdpsi, FdFdpsi | GS 残差 RHS = −μ₀R²·dp/dψ − F·dF/dψ 所需字段 |
| params (3) | [Ip, paxis, fvac] |
| x_coords (4) | [R_lo, Z_lo, R_up, Z_up] X 点坐标 |
| psi_axis, psi_bndry, R_axis, Z_axis, L, Beta0 | 平衡标量量 |

### 2.5 输入归一化（论文 Sec. II.C / Eq. 5）

9 通道输入：R、Z 线性映射到 [−1,1]；Paxis、Ip、fvac 与 4 个 X 点坐标（共 7 个标量）用**训练集**均值/标准差 z-score 后广播到网格。目标 ψ 同样用训练集统计 z-score。训练损失与 rel L2 均在归一化域计算，物理量经逆变换恢复。

## 3. 模型结构

`src/gs_pino/model_dn_fno.py`，纯 PyTorch 手写实现（无 neuraloperator 依赖）：

```
输入 (B, 9, 65, 65)
  └─ lifting: 1×1 Conv  9 → 64
  └─ 4 × FNOBlock:
  │    ├─ SpectralConv2dR: rfft2 → 保留最低 modes(16,16) 模，乘以单复权重 R(k)
  │    │   （论文 Eq. 6: K v = F⁻¹(R(k)·F[v])，单复权张量，截断其余模）
  │    ├─ 1×1 Conv（逐点混合）
  │    └─ GELU（neuraloperator 默认；论文未指明）
  └─ projection: 1×1 Conv 64 → 1   → 输出 (B, 1, 65, 65)
```

- 无位置嵌入（R、Z 作为显式输入通道）；
- **可训练参数 4,211,649**（论文报告 4,770,241；论文模块组装细节不可恢复，按用户确认取近似，比率 0.88）；
- 仓库自带的经典双权 `SpectralConv2d`（`models.py`，~8.4M 参数）可替换 FNOBlock 作为对照变体。

## 4. 训练

`src/gs_pino/train_dn_fno.py`：

| 项 | 配置 |
|---|---|
| 损失 | 纯 MSE（归一化域，无 PDE/几何物理损失） |
| 优化器 | AdamW，lr=1e-3，weight decay=1e-4 |
| 调度 | ReduceLROnPlateau：验证 rel L2 连续 20 epoch 无改善 → lr×0.5，下限 1e-5 |
| 早停 | patience 75（保存 best 权重 + history.json + args.json） |
| batch / workers | 16 / 0（Windows spawn 稳定性；论文 4 workers，不影响结果） |
| 训练上限 | 800 epoch（论文未说明；本复现中早停未触发，全部跑满 800，论文最佳验证 epoch 310） |

**缩放实验**：4 次训练（N∈{500,1000,2000,5000} × seed 1，按用户决定 4 次足够；论文每档 3 seeds）。N=500/1000 另有历史遗留的 seed 2/3 checkpoint 可作多 seed 统计。

## 5. 结果

### Table I — 场级精度 vs 训练集大小（rel L2 %，均值±std）

| N_train | 复现 | 论文 |
|---|---|---|
| 500 | **0.222 ± 0.007** | 0.286 ± 0.034 |
| 1000 | **0.145 ± 0.010** | 0.182 ± 0.002 |
| 2000 | **0.080** | 0.109 ± 0.004 |
| 5000 | **0.056** | 0.061 ± 0.006 |

缩放律 ε ∝ N^−0.61（论文 N^−0.68）。物理 RMSE（Wb）：N=5000 时 1.503e-5 vs 论文 1.79e-5。**各档全面优于论文。**

### Table II — 几何指标（N=5000 seed=1，均值，距离 cm）

| 指标 | 复现 | 论文 |
|---|---|---|
| 分界面平均距离 | **0.028** | 0.072 |
| 分界面 Hausdorff | **0.477** | 2.128 |
| 分界面面积相对误差 | **0.224 %** | 0.465 % |
| 下 / 上 X 点误差 | 0.174 / 0.180 | 0.161 / 0.112 |
| O 点误差 | 0.042 | 0.031 |
| \|Δψ_bndry\| (Wb) | 1.03e-5 | 9.82e-6 |
| find_critical 失败 | **0/500** | 0 |

### GS 残差诊断（论文 Eq. 10–11，归一化）

预测场 2.318 / freegs 真值基线 2.320 / 比值 **0.999**（论文 0.998）。残差 O(1) 是真值场在该掩模（{ψ ≥ ψ_bndry}，含磁轴邻域 FD 截断误差）下的**地板**，不是模型误差——分析见 ANALYSIS.md。

### Table III — 推理延迟（batch 1，中位）

| 方法 | 复现 | 论文 (A100) | 加速比 |
|---|---|---|---|
| FNO GPU（RTX 5060 Laptop） | **1.625 ms** | 2.765 ms | 665×（论文 ~640×） |
| FNO CPU | 25.606 ms | 25.587 ms | 42× |
| freegs CPU | 1080 ms | 1768 ms | — |

> 论文延迟为 A100 实测，本机绝对数字不可直接比；方法论与加速比口径一致。GPU 延迟 1.6 ms 已达"毫秒级代理"目标。

## 6. 代码与脚本接口

### 6.1 模块（src/gs_pino/）

```bash
PY="C:/Users/HP/.conda/envs/torch5060/python.exe"   # 本项目环境

# ── 数据生成（分块 + 断点续跑 + 合并）──
"$PY" -m gs_pino.generate_dn_dataset --split train --n-samples 5000 --seed 123 \
      --out-dir dn_fno_2608/data --chunk-size 500 --n-jobs 24 --merge
#   参数: --split {train,val,test}  --n-samples  --seed 123/456/789
#         --out-dir  --chunk-size  --n-jobs     --merge（合并分块为 npz）

# ── 训练 ──
"$PY" -u -m gs_pino.train_dn_fno \
      --train-data dn_fno_2608/data/train.npz --val-data dn_fno_2608/data/val.npz \
      --n-train 5000 --seed 1 --out-dir dn_fno_2608/outputs/fno_n5000_s1
#   参数: --n-train 5000  --perm-seed 12345（嵌套子集置换）  --seed
#         --epochs 800  --batch-size 16  --lr 1e-3  --weight-decay 1e-4
#         --lr-patience 20  --lr-factor 0.5  --min-lr 1e-5  --patience 75
#         --workers 0  --device cuda
#   产物: best.pt（model_state + stats 归一化统计量 + args）、history.json、args.json

# ── 评估（场级 + 几何 + GS 残差）──
"$PY" -u -m gs_pino.evaluate_dn_fno \
      --test-data dn_fno_2608/data/test.npz \
      --checkpoint dn_fno_2608/outputs/fno_n5000_s1/best.pt \
      --out-dir dn_fno_2608/outputs/report/n5000_s1
#   参数: --max-samples 0（0=全部 500）  --device cuda
#   产物: metrics.json（rel L2 / RMSE / 几何 / GS 残差 / find_critical 统计）

# ── 可视化（fig1 好坏样本对比 / fig2 场统计 / fig3 几何统计）──
"$PY" -u -m gs_pino.visualize_dn_fno \
      --checkpoint dn_fno_2608/outputs/fno_n5000_s1/best.pt \
      --out-dir dn_fno_2608/outputs/report/figures
#   参数: --test-data  --max-samples  --device
#   产物: fig1_best_worst_psi.png、fig2_field_stats.png、fig3_geometry_stats.png、
#         stats_per_sample.json（逐样本指标缓存，重画图秒级复用）

# ── 延迟基准（GPU/CPU 前向 + freegs 对照）──
"$PY" -u -m gs_pino.latency_dn_fno \
      --test-data dn_fno_2608/data/test.npz \
      --checkpoint dn_fno_2608/outputs/fno_n5000_s1/best.pt \
      --out-dir dn_fno_2608/outputs/report
#   参数: --n 500  --skip-freegs（跳过 freegs 对照，只测 FNO）
#   产物: latency.json（中位/p95 + 加速比）

# ── 汇总报告（读 outputs/ 各 metrics.json → REPORT.md）──
"$PY" dn_fno_2608/scripts/make_tables.py
```

### 6.2 一键脚本（dn_fno_2608/scripts/，幂等可重跑）

```bash
bash dn_fno_2608/scripts/run_generate.sh    # 全量数据：val→test→train 分块生成 + 合并
bash dn_fno_2608/scripts/run_scaling.sh     # 4 次缩放训练（已存在 checkpoint 自动跳过）
bash dn_fno_2608/scripts/run_evaluate.sh    # 全部可用 checkpoint 评估 + latency
```

辅助脚本：`param_fingerprint.py`（模型参数量枚举核对）、`probe_residual*.py`（GS 残差口径探针）、`benchmark_timing.py`、`kill_old_training.ps1`（Windows 下清理孤儿训练进程）、`smoke_dn_solve.py`（单样本求解冒烟）。

### 6.3 目录约定

```
dn_fno_2608/
├── data/            # 共享数据集（冻结只读）；train/val/test 分块目录为中间产物
├── outputs/         # 论文复现基线（冻结）：fno_n{N}_s{S}/ + report/
│   ├── fno_n5000_s1/best.pt, history.json, args.json
│   └── report/      # REPORT.md、ANALYSIS.md、metrics JSON、latency.json、figures/
├── experiments/     # ★ 后续改进实验（见 experiments/README.md，数据通用）
├── logs/            # 所有运行日志
├── scripts/         # 一键复现与工具脚本
├── PLAN.md          # 复现计划（阶段 0–5 与验收标准）
└── README.md        # 本文档
```

## 7. 复现偏差记录（论文未指明处）

1. 参数量 4,211,649 vs 论文 4,770,241（近似匹配，用户确认）；
2. 激活 GELU（neuraloperator 默认）；
3. isoflux 参考点取 (1.5, 0.0) m（论文未给坐标，仅锚定 gauge）；
4. workers=0（Windows spawn 稳定性）；
5. epoch 上限 800、早停未触发（论文最佳 epoch 310）；
6. 缩放实验每档 1 seed（用户决定 4 次训练足够；N=500/1000 有额外 seed 可用）；
7. 延迟绝对值为本地 RTX 5060，与论文 A100 不可直接比。
