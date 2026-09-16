# data/ — 论文基线数据集（TestTokamak 双零位形，9ch 输入）

> 生成日期：2026-07（论文复现第一阶段）｜ 代码：`gs_pino_dn_fno_2608.generate_dn_dataset`
> 用途：论文 *Millisecond-Scale Neural Operator Surrogates for Double-Null
> Free-Boundary Grad-Shafranov Equilibria*（arXiv:2608.05555v1）复现基线
> （outputs/fno_n*_s* 训练数据）+ exp001（coil 9ch 对照）
> 一键复现：`bash dn_fno_2608/scripts/run_generate.sh`（分块断点续跑，可重入）

## 1. 数据集速览

| 项 | 值 |
|---|---|
| 机器 | **TestTokamak**（freegs 内置，4 控制线圈 P1L/P1U/P2L/P2U，**有墙**） |
| 位形 | 双零 DN（上/下 X 点镜像对称） |
| 网格 | 65²，R∈[0.1,2.0] m、Z∈[−2,2] m |
| 规模 | train 5000（seed 123）/ val 500（seed 456）/ test 500（seed 789），**6000/6000 全通过** |
| 输入通道 | **9ch**（xpoints 主模式）：R,Z + 3 params + 4 X 点坐标；另有 9ch coil 模式（§4.2） |
| 目标 | `psi_total`（65²，z-score） |
| 求解器 | freegs 0.8.3.dev54：双 X 点约束 + isoflux（锚定 (1.5, 0.0)），γ=1e-12；Picard rtol=1e-3、maxits=50 |
| 基线成绩 | outputs/fno_n5000_s1 test rel L2 **0.0561%**（论文 0.061%，3-seed 均值） |

## 2. 目录与文件

```
dn_fno_2608/data/
├── train.npz / val.npz / test.npz   # 正式数据（float32；npz 不入 git）
├── train/ val/ test/                # 生成中间分块 chunk_*.npz（合并后冗余可删）
├── README.md
└── README.pdf
```

## 3. 字段说明（npz keys，17 个）

| 字段 | 形状 | 含义 | 是否模型输入 |
|---|---|---|---|
| `psi_total` | (N,65,65) | 总极向磁通（**训练目标**） | — |
| `psi_plasma` / `psi_plasma_norm` / `psi_coils` | (N,65,65) | 等离子体/归一化/线圈磁通分量 | 否（PINO 预留） |
| `R`, `Z` | (65,65) | 物理坐标网格（R 沿行 axis 0、Z 沿列 axis 1） | 通道 1–2 |
| `mask` | (N,65,65) | freegs critical.core_mask | 否 |
| `params` | (N,3) | **[Ip, paxis, fvac]**（Ip 在前，代码按 npz 原序拼接） | 通道 3–5 |
| `x_coords` | (N,4) | [R_lo, Z_lo, R_up, Z_up] X 点目标坐标（上下镜像对称） | 通道 6–9（xpoints 模式） |
| `coil_currents` | (N,4) | I_P1L/I_P1U/I_P2L/I_P2U（coil 模式用） | 通道 6–9（coil 模式） |
| `greens` | (N,4,65,65) | 线圈 Green 函数（逐样本存储；几何固定 → 各样本数值相同） | 否（PINO 预留） |
| `dpdpsi`, `FdFdpsi` | (N,65,65) | GS 残差 RHS = −μ₀R²·dp/dψ − F·dF/dψ 分量 | 否（PINO 预留） |
| `axes` | (N,4) | [R_axis, Z_axis, psi_bndry, psi_axis]（DN 对称 → Z_axis≡0） | 否（评估用） |
| `L`, `Beta0` | (N,1) | 等离子体电感 / 比压 | 否 |
| `solve_time` | (N,1) | 单样本求解耗时 | 否 |

## 4. 输入通道明细（9 通道，两种模式）

### 4.1 xpoints 9ch（主模式，`train_dn_fno` 默认）

通道顺序 = 代码实际拼接顺序（`data_dn_fno.py`）：R/Z 网格 + params（npz 原序）+
x_coords。R/Z 线性映射 [-1,1]；7 个标量用**训练集** mean/std z-score 后广播到
网格；目标 ψ 同样 z-score（训练损失与 rel L2 均在归一化域计算）。

| # | 通道 | 内容 | 单位 | 来源 |
|---|---|---|---|---|
| 1 | R | 网格 R 坐标 | m | 固定 65² 网格 |
| 2 | Z | 网格 Z 坐标 | m | 固定 65² 网格 |
| 3 | Ip | 等离子体电流 | A | params[0] |
| 4 | Paxis | 磁轴压强 | Pa | params[1] |
| 5 | fvac | 真空通量函数 f | Wb/m | params[2] |
| 6 | R_lo | 下 X 点 R | m | x_coords[0] |
| 7 | Z_lo | 下 X 点 Z | m | x_coords[1] |
| 8 | R_up | 上 X 点 R | m | x_coords[2] |
| 9 | Z_up | 上 X 点 Z | m | x_coords[3] |

### 4.2 coil 9ch（exp001，`train_dn_fno_coils`）

| # | 通道 | 内容 | 单位 | 来源 |
|---|---|---|---|---|
| 1–2 | R, Z | 网格坐标 | m | 固定 65² 网格 |
| 3–5 | Ip, Paxis, fvac | 3 个运行参数 | A / Pa / Wb·m⁻¹ | params[0:3] |
| 6–9 | I_P1L, I_P1U, I_P2L, I_P2U | 4 线圈电流 | A | coil_currents[0:4] |

线圈名/顺序 = freegs TestTokamak 定义顺序（`tokamak.coils`）。本数据集 X 点上下
镜像对称 → 解对称，实测 P1L=P1U（同正值）、P2L=P2U（同负值），4 个电流通道实际
只有 2 个独立自由度。

## 5. 如何调用

### 5.1 直接加载

```python
import numpy as np
d = np.load("dn_fno_2608/data/train.npz")
psi = d["psi_total"]    # (5000, 65, 65)
params = d["params"]    # (5000, 3): [Ip, paxis, fvac]
x = d["x_coords"]       # (5000, 4): [R_lo, Z_lo, R_up, Z_up]
coils = d["coil_currents"]  # (5000, 4): I_P1L, I_P1U, I_P2L, I_P2U
```

### 5.2 训练 / 评估

```bash
PY="C:/Users/HP/.conda/envs/torch5060/python.exe"

# xpoints 9ch（论文复现 / 缩放实验）
"$PY" -u -m gs_pino_dn_fno_2608.train_dn_fno \
  --train-data dn_fno_2608/data/train.npz --val-data dn_fno_2608/data/val.npz \
  --n-train 500 --seed 1 --out-dir dn_fno_2608/experiments/expXXX_xxx
"$PY" -u -m gs_pino_dn_fno_2608.evaluate_dn_fno \
  --test-data dn_fno_2608/data/test.npz \
  --checkpoint dn_fno_2608/experiments/expXXX_xxx/best.pt \
  --out-dir dn_fno_2608/experiments/expXXX_xxx

# coil 9ch（exp001 口径）
"$PY" -u -m gs_pino_dn_fno_2608.train_dn_fno_coils \
  --train-data dn_fno_2608/data/train.npz --val-data dn_fno_2608/data/val.npz \
  --n-train 500 --seed 1 --out-dir <out>
```

- 缩放实验 N∈{500,1000,2000,5000} 用固定置换（seed 12345）取**嵌套子集**
  （大集合严格包含小集合），`--n-train` 即可切换
- 通道数自动推断：`build_model(in_channels=2+len(scalar_mean))`

## 6. 相关脚本（dn_fno_2608/scripts/）

| 脚本 | 用途 |
|---|---|
| `run_generate.sh` | 全量生成（val→test→train 分块 + 合并，可重入） |
| `run_scaling.sh` | 缩放实验 4 次训练（N=500/1000/2000/5000，已有 checkpoint 自动跳过） |
| `run_evaluate.sh` | 全部可用 checkpoint 评估 + latency |
| `make_tables.py` | 汇总 outputs/ 各 metrics.json → REPORT.md |
| `probe_residual*.py` | GS 残差探针 |
| `benchmark_timing.py` / `smoke_dn_solve.py` / `param_fingerprint.py` | 延迟基准 / 单样本求解冒烟 / 参数量核对 |

## 7. 生成设置与复现

### 7.1 参数空间（论文 Eq. 4）

| 参数 | 范围 / 分布 |
|---|---|
| Ip | 均匀 U[5e4, 4e5] A |
| Paxis | 均匀 U[200, 3000] Pa |
| fvac | 均匀 U[0.5, 3.0] |
| 下 X 点 (R_lo, Z_lo) | (1.1, −0.6) + δ，δ 各分量独立 U[−0.02, 0.02] m |
| 上 X 点 (R_up, Z_up) | (1.1, +0.6) + δ（上下共用 δ 镜像，严格对称） |
| 剖面 | 固定 alpha_m=1.0、alpha_n=2.0（freegs `ConstrainPaxisIp`） |

### 7.2 求解器与接受准则

- freegs 0.8.3.dev54，65×65 网格；双 X 点约束 + isoflux（锚定 (1.5, 0.0)），
  γ=1e-12；Picard rtol=1e-3、maxits=50
- 接受准则：收敛 且 |Ip_sol−Ip_tgt|/Ip_tgt ≤ 10% 且 find_critical ≥2 X 点
- 单样本中位求解 ~1.08 s（本机 24 核并行均摊）

### 7.3 复现命令

```bash
bash dn_fno_2608/scripts/run_generate.sh   # 等价于下面两条（每个 split）+ merge

# 单条示例（train）
C:/Users/HP/.conda/envs/torch5060/python.exe -m gs_pino_dn_fno_2608.generate_dn_dataset \
  --split train --n-samples 5000 --seed 123 --out-dir dn_fno_2608/data \
  --chunk-size 500 --n-jobs 24
C:/Users/HP/.conda/envs/torch5060/python.exe -m gs_pino_dn_fno_2608.generate_dn_dataset \
  --split train --out-dir dn_fno_2608/data --merge
```

## 8. 历史与偏差记录

### 8.1 论文复现偏差（论文未指明处，用户确认）

1. 参数量 4,211,649 vs 论文 4,770,241（近似匹配）；GELU 激活；
2. isoflux 参考点 (1.5, 0)——论文未给坐标；
3. workers=0（Windows spawn）；epoch 上限 800（早停未触发，论文最佳 310）；
4. 缩放每档 1 seed；延迟本地 RTX 5060 与论文 A100 不可直接比；
5. 本文档 §3/§4 的 params 顺序与 greens 形状为 2026-09-16 按 npz 实测更正
   （旧版误写 params=[Paxis,Ip,fvac]、greens (4,65,65) 共用——实际
   [Ip,paxis,fvac]、(N,4,65,65) 逐样本存储）。

### 8.2 数据演进链（各版本均为自包含文档）

data/（本文档，基线 9ch）→ data_v2（+alpha_m/alpha_n 采样，11ch）→
data_v3（X 点 ±0.20 m + 锚点采样，**已删除**：isoflux 约束不可达缺陷，被
data_v4 取代）→ data_v4（可行区采样 + 7 项接受约束，11/13ch）→
data_v5（MAST 真实装置，DN+SN 混合，18ch coil）→ data_v6（MASTU_simple
五配置，21ch coil）→ data_v6_clean（v6 清洗版）→ data_gspack2_v1/v2
（gspack2_TRAE 求解包复刻，exp201–203）。
