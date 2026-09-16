# data_v2 — 剖面形状参数采样数据集（TestTokamak DN，11ch 输入）

> 生成日期：2026-08-14 ｜ 代码：`gs_pino_dn_fno_2608.generate_dn_dataset`（同 data/ 生成器）
> 用途：验证"原 7 维参数空间限制模型"假设——exp002（xpoints 11ch）/
> exp003（coil 11ch）数据
> 与 data/ 的唯一系统性差异：**alpha_m/alpha_n 从固定 (1.0, 2.0) 改为采样**
> （机器/网格/规模/seed/求解器全部一致）
> 一键复现：`bash dn_fno_2608/scripts/run_generate_v2.sh`（分块断点续跑，可重入）

## 1. 数据集速览

| 项 | 值 |
|---|---|
| 机器 | **TestTokamak**（freegs 内置，4 控制线圈 P1L/P1U/P2L/P2U，**有墙**） |
| 位形 | 双零 DN（X 点上下镜像对称） |
| 网格 | 65²，R∈[0.1,2.0] m、Z∈[−2,2] m |
| 规模 | train 5000（seed 123）/ val 500（seed 456）/ test 500（seed 789），**6000/6000 全通过** |
| 输入通道 | **11ch**：R,Z + 5 params + 4 X 点坐标（coil 模式 11ch = R,Z + 5 params + 4 线圈电流） |
| 目标 | `psi_total`（65²，z-score） |
| 求解器 | freegs 0.8.3.dev54：双 X 点 + isoflux（锚定 (1.5,0)），γ=1e-12；rtol=1e-3、maxits=50 |
| 参数空间 | **9 维**（data/ 的 7 维 + alpha_m + alpha_n） |
| 结果参照 | exp002 test 0.303%（xpoints，data/ 基线的 1.36×）、exp003 0.412%（coil） |

## 2. 目录与文件

```
dn_fno_2608/data_v2/
├── train.npz / val.npz / test.npz   # 正式数据（float32；npz 不入 git）
├── train/ val/ test/                # 生成中间分块 chunk_*.npz（合并后冗余可删）
├── README.md
└── README.pdf
```

## 3. 字段说明（npz keys，17 个）

与 data/ 同构（同 17 键同形状），仅 `params` 扩为 5 维：

| 字段 | 形状 | 含义 | 是否模型输入 |
|---|---|---|---|
| `psi_total` | (N,65,65) | 总极向磁通（**训练目标**） | — |
| `psi_plasma` / `psi_plasma_norm` / `psi_coils` | (N,65,65) | 磁通分量 | 否（PINO 预留） |
| `R`, `Z` | (65,65) | 物理坐标网格（R 沿行 axis 0、Z 沿列 axis 1） | 通道 1–2 |
| `mask` | (N,65,65) | freegs critical.core_mask | 否 |
| `params` | (N,**5**) | **[Ip, paxis, fvac, alpha_m, alpha_n]** | 通道 3–7 |
| `x_coords` | (N,4) | [R_lo, Z_lo, R_up, Z_up] X 点目标坐标（上下镜像对称） | 通道 8–11（xpoints 模式） |
| `coil_currents` | (N,4) | I_P1L/I_P1U/I_P2L/I_P2U（coil 模式用） | 通道 8–11（coil 模式） |
| `greens` | (N,4,65,65) | 线圈 Green 函数（逐样本存储；几何固定 → 数值相同） | 否（PINO 预留） |
| `dpdpsi`, `FdFdpsi` | (N,65,65) | GS 残差 RHS 分量 | 否（PINO 预留） |
| `axes` | (N,4) | [R_axis, Z_axis, psi_bndry, psi_axis] | 否（评估用） |
| `L`, `Beta0` | (N,1) | 电感 / 比压 | 否 |
| `solve_time` | (N,1) | 求解耗时 | 否 |

旧代码读 params 前三列语义不变（[Ip, paxis, fvac]）。

## 4. 输入通道明细（11 通道，两种模式）

通道顺序 = 代码实际拼接顺序（`data_dn_fno.py`）：R/Z + params（npz 原序）+
x_coords（coil 模式则接 coil_currents）。R/Z 线性映射 [-1,1]；9 个标量
z-score（训练集统计）；目标 ψ 同样 z-score。

### 4.1 xpoints 11ch（exp002）

| # | 通道 | 内容 | 单位 | 来源 |
|---|---|---|---|---|
| 1 | R | 网格 R 坐标 | m | 固定 65² 网格 |
| 2 | Z | 网格 Z 坐标 | m | 固定 65² 网格 |
| 3 | Ip | 等离子体电流 | A | params[0] |
| 4 | Paxis | 磁轴压强 | Pa | params[1] |
| 5 | fvac | 真空通量函数 f | Wb/m | params[2] |
| 6 | alpha_m | 剖面内指数（U[1.0,2.0]） | — | params[3] |
| 7 | alpha_n | 剖面外指数（U[1.5,2.5]） | — | params[4] |
| 8–11 | R_lo, Z_lo, R_up, Z_up | 下/上 X 点坐标 | m | x_coords[0:4] |

### 4.2 coil 11ch（exp003，`train_dn_fno_coils`）

通道 1–7 同上；通道 8–11 = I_P1L, I_P1U, I_P2L, I_P2U（coil_currents[0:4]，
X 点坐标不进模型；对称解 → P1L=P1U、P2L=P2U）。

**剖面 shape** = (1−ψn^αm)^αn（freegs `ConstrainPaxisIp`，p′ 与 FF′ 共用）：
αm 控制磁轴附近平坦度（越大轴心越平），αn 控制整体峰化程度（越大边缘梯度越陡）。
合法范围：αm > 0（=0 除零崩溃）、αn ≥ 0（=0 阶跃剖面）。

## 5. 如何调用

### 5.1 直接加载

```python
import numpy as np
d = np.load("dn_fno_2608/data_v2/train.npz")
psi = d["psi_total"]    # (5000, 65, 65)
params = d["params"]    # (5000, 5): [Ip, paxis, fvac, alpha_m, alpha_n]
x = d["x_coords"]       # (5000, 4)
```

### 5.2 训练 / 评估

```bash
PY="C:/Users/HP/.conda/envs/torch5060/python.exe"

# 训练（exp002 实际命令，xpoints 11ch）
"$PY" -u -m gs_pino_dn_fno_2608.train_dn_fno \
  --train-data dn_fno_2608/data_v2/train.npz --val-data dn_fno_2608/data_v2/val.npz \
  --n-train 500 --seed 1 --out-dir dn_fno_2608/experiments/exp002_profile_alphas
# 评估
"$PY" -u -m gs_pino_dn_fno_2608.evaluate_dn_fno \
  --test-data dn_fno_2608/data_v2/test.npz \
  --checkpoint dn_fno_2608/experiments/exp002_profile_alphas/best.pt \
  --out-dir dn_fno_2608/experiments/exp002_profile_alphas
# coil 11ch（exp003）：train_dn_fno_coils / evaluate_dn_fno_coils
```

通道数自动推断：`build_model(in_channels=2+len(scalar_mean))`（旧 9ch checkpoint
自动兼容为 9）。

## 6. 相关脚本（dn_fno_2608/scripts/）

| 脚本 | 用途 |
|---|---|
| `run_generate_v2.sh` | 全量生成（分块断点续跑 + 合并，val→test→train） |

## 7. 生成设置与统计

### 7.1 采样范围

- Ip U[5e4,4e5] A、Paxis U[200,3000] Pa、fvac U[0.5,3.0]、
  **alpha_m U[1.0,2.0]、alpha_n U[1.5,2.5]**（相对 data/ 新增的 2 维）
- X 点 (1.1, ±0.6) + δ 镜像：δ 各分量独立 U[−0.02, 0.02] m，上/下 X 点共用同一
  δ（严格上下对称）——与 data/ 口径一致

### 7.2 接受准则

- data/ 三项：收敛 且 |Ip_sol−Ip_tgt|/Ip_tgt ≤ 10% 且 ≥2 X 点
- **新增：L 有限且 0<Beta0<1**（剖面退化防护；data/ 未显式检查，实际样本均满足）
- 拒绝时**全参数向量重采样重试**（`--max-retries 20`）——仅换 X 点抖动救不回来
  （αm/αn 采样后半数组合确定性失败）

### 7.3 统计

- 6000/6000 全通过；单样本 0.17–0.41 s/solve（24 核，train 全程 14.3 min）
- merge sanity：alpha_m ∈[1.000,2.000]、alpha_n ∈[1.500,2.500]

## 8. 历史与偏差记录

1. 唯一系统性差异 vs data/：alpha_m/alpha_n 采样（§1）；其余（机器/网格/split/seed/求解器/接受准则主体）逐项一致；
2. 新增接受准则 L 有限且 0<Beta0<1（§7.2）；
3. params 扩为 5 维——旧代码读前三列语义不变；训练/评估/可视化/延迟脚本支持 in_channels=11（旧 checkpoint 不受影响）；
4. 生成过程：首轮（`--max-retries 5` 且重试不重采样）接受率仅 ~98% → 修复为全参数重采样 + 20 次后 6000/6000；train 曾混入首轮遗留分块，删除全部 train 分块后干净重生成（同 seed 123）；
5. 本文档 §3/§4 的 params 顺序为 2026-09-16 按 npz 实测更正（旧版通道表把 Paxis 排在 Ip 前，实际 params[0]=Ip）；
6. 结果参照：exp002（xpoints 11ch）0.303%、exp003（coil 11ch）0.412%——alpha 采样可学（相对损失恒定 ~1.4×）。
