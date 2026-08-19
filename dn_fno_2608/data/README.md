# data/ — 论文基线数据集（TestTokamak 双零位形，9ch 输入）

> 生成日期：2026-07（复现第一阶段）
> 用途：论文 *Millisecond-Scale Neural Operator Surrogates for Double-Null
> Free-Boundary Grad-Shafranov Equilibria*（arXiv:2608.05555v1）复现基线；
> outputs/fno_n*_s* 训练数据
> 后续数据：data_v2（alpha 采样）/ data_v4（可行区+接受约束）/ data_v5（MAST
> 混合位形）/ data_v6（MASTU_simple 五配置）——见 §8

## 1. 数据集速览

| 项 | 值 |
|---|---|
| 机器 | TestTokamak（freegs 内置，4 控制线圈，有墙） |
| 位形 | 双零 DN（双 X 点） |
| 网格 | 65²，R∈[0.1,2.0] m、Z∈[−2,2] m |
| 规模 | train 5000（seed 123）/ val 500（seed 456）/ test 500（seed 789），6000/6000 全通过 |
| 输入通道 | **9ch**：R,Z + Paxis,Ip,fvac + 4 X 点坐标（§4） |
| 目标 | `psi_total`（65²，z-score 归一化） |

## 2. 目录与文件

```
dn_fno_2608/data/
├── train.npz / val.npz / test.npz   # 正式数据（float32）
└── train/ val/ test/                # 生成中间分块 chunk_*.npz（合并后冗余可删）
```

## 3. 字段说明（npz keys）

| 字段 | 含义 |
|---|---|
| `psi_total` | 总极向磁通（**训练目标**，z-score 输出） |
| `psi_plasma` / `psi_plasma_norm` / `psi_coils` | 等离子体 / 归一化 / 线圈磁通分量（PINO 预留） |
| `R`, `Z` | 物理坐标 (65,65)（R 沿行 axis 0、Z 沿列 axis 1） |
| `mask` | freegs critical.core_mask |
| `coil_currents` (4) | 4 个控制线圈电流（PINO 预留） |
| `greens` (4×65×65) | 线圈 Green 函数（PINO 预留） |
| `dpdpsi`, `FdFdpsi` | GS 残差 RHS = −μ₀R²·dp/dψ − F·dF/dψ 所需字段（PINO 预留） |
| `params` (3) | [Ip, paxis, fvac] |
| `x_coords` (4) | [R_lo, Z_lo, R_up, Z_up] X 点坐标 |
| `psi_axis`, `psi_bndry`, `R_axis`, `Z_axis`, `L`, `Beta0` | 平衡标量量 |
| `solve_time` | 单样本求解耗时 |

## 4. 输入通道明细（9 通道）

通道顺序 = 代码实际拼接顺序：2 几何 + 7 标量。R/Z 线性映射 [-1,1]；7 个标量
用**训练集** mean/std z-score 后广播到网格；目标 ψ 同样 z-score（训练损失与
rel L2 均在归一化域计算）。

| # | 通道 | 内容 | 单位 | 来源 |
|---|---|---|---|---|
| 1 | R | 网格 R 坐标 | m | 固定 65² 网格 |
| 2 | Z | 网格 Z 坐标 | m | 固定 65² 网格 |
| 3 | Paxis | 磁轴压强 | Pa | params[0] |
| 4 | Ip | 等离子体电流 | A | params[1] |
| 5 | fvac | 真空通量函数 f | Wb/m | params[2] |
| 6 | R_lo | 下 X 点 R | m | x_coords[0] |
| 7 | Z_lo | 下 X 点 Z | m | x_coords[1] |
| 8 | R_up | 上 X 点 R | m | x_coords[2] |
| 9 | Z_up | 上 X 点 Z | m | x_coords[3] |

## 5. 如何调用

### 5.1 直接加载

```python
import numpy as np
d = np.load("dn_fno_2608/data/train.npz")
psi = d["psi_total"]    # (5000, 65, 65)
params = d["params"]    # (5000, 3): [Ip, paxis, fvac]
x = d["x_coords"]       # (5000, 4): [R_lo, Z_lo, R_up, Z_up]
```

### 5.2 训练 / 评估（N=500 示例）

```bash
PY="C:/Users/HP/.conda/envs/torch5060/python.exe"

"$PY" -u -m gs_pino_dn_fno_2608.train_dn_fno \
  --train-data dn_fno_2608/data/train.npz --val-data dn_fno_2608/data/val.npz \
  --n-train 500 --seed 1 --out-dir dn_fno_2608/experiments/expXXX_xxx
"$PY" -u -m gs_pino_dn_fno_2608.evaluate_dn_fno \
  --test-data dn_fno_2608/data/test.npz \
  --checkpoint dn_fno_2608/experiments/expXXX_xxx/best.pt \
  --out-dir dn_fno_2608/experiments/expXXX_xxx
```

输入模式参数：`--input-mode {xpoints,xa}`（xpoints = 本表 9ch；xa = +2 锚点 11ch，
仅 data_v4 有 anchor 字段）。

## 6. 相关脚本（dn_fno_2608/scripts/）

| 脚本 | 用途 |
|---|---|
| `run_generate.sh` | 全量生成（val→test→train 分块 + 合并，可重入） |
| `run_scaling.sh` | 缩放实验 4 次训练（N=500/1000/2000/5000，已有 checkpoint 自动跳过） |
| `run_evaluate.sh` | 全部可用 checkpoint 评估 + latency |
| `make_tables.py` | 汇总 outputs/ 各 metrics.json → REPORT.md |
| `probe_residual*.py` / `benchmark_timing.py` / `smoke_dn_solve.py` / `param_fingerprint.py` | GS 残差探针 / 延迟基准 / 单样本求解冒烟 / 参数量核对 |

## 7. 生成设置

### 7.1 参数空间（论文 Eq. 4，7 个标量）

| 参数 | 范围 / 分布 |
|---|---|
| Paxis | 均匀 U[200, 3000] Pa |
| Ip | 均匀 U[5e4, 4e5] A |
| fvac | 均匀 U[0.5, 3.0] |
| 下 X 点 (R_lo, Z_lo) | (1.1, −0.6) + δ，δ 分量独立 U[−0.02, 0.02] m |
| 上 X 点 (R_up, Z_up) | (1.1, +0.6) + δ（上下共用 δ 镜像，严格对称） |

剖面形状固定：alpha_m=1.0、alpha_n=2.0（`ConstrainPaxisIp`）。

### 7.2 求解器与接受准则

- freegs 0.8.3.dev54，65×65 网格；双 X 点约束 + isoflux（锚定 (1.5, 0.0)），γ=1e-12；
  Picard rtol=1e-3、maxits=50
- 接受准则：收敛 且 |Ip_sol−Ip_tgt|/Ip_tgt ≤ 10% 且 find_critical ≥2 X 点
- 单样本中位求解 ~1.08 s（本机 24 核并行）
- 缩放实验：N∈{500,1000,2000,5000} 用固定置换（seed 12345）取**嵌套子集**，
  大集合严格包含小集合

## 8. 历史与偏差记录

1. 复现偏差（论文未指明处）：参数量 4,211,649 vs 论文 4,770,241（近似匹配，用户
   确认）；GELU 激活；isoflux 参考点 (1.5,0) 论文未给坐标；workers=0（Windows
   spawn）；epoch 上限 800（早停未触发，论文最佳 310）；缩放每档 1 seed；延迟
   本地 RTX 5060 与论文 A100 不可直接比。
2. **数据演进链**（各见其 README）：data/（本文档，基线 9ch）→ data_v2（+alpha_m/
   alpha_n 采样，11ch）→ data_v3（X 点 ±0.20 m + 锚点采样，**已删除**：isoflux
   约束不可达缺陷，被 data_v4 取代）→ data_v4（可行区采样 + 7 项接受约束，
   11/13ch）→ data_v5（MAST 真实装置，DN+SN 混合，18ch coil）→ data_v6
   （MASTU_simple 五配置，21ch coil）。
