# data_v2 — 剖面形状参数采样数据集（TestTokamak DN，11ch 输入）

> 生成日期：2026-08-14
> 用途：验证"原 7 维参数空间限制模型"假设（exp002/exp003 数据）
> 一键复现：`bash dn_fno_2608/scripts/run_generate_v2.sh`（分块断点续跑，可重入）

## 1. 数据集速览

| 项 | 值 |
|---|---|
| 机器 | TestTokamak（同 data/） |
| 位形 | 双零 DN |
| 网格 | 65²（同 data/） |
| 规模 | train 5000（seed 123）/ val 500（seed 456）/ test 500（seed 789），6000/6000 全通过 |
| 输入通道 | **11ch**：R,Z + 5 params + 4 X 点坐标（§3） |
| 目标 | `psi_total`（65²，z-score） |
| 与 data/ 唯一差异 | **alpha_m/alpha_n 从固定 (1.0, 2.0) 改为采样**（其余全部一致） |

参数空间由 data/ 的 7 维扩到 **9 维**（+2 剖面形状指数）。

## 2. 字段说明（npz keys）

同 data/ 全部字段，仅 `params` 扩为 5 维：

| 字段 | 形状 | 含义 |
|---|---|---|
| `params` | (N,**5**) | **[Ip, paxis, fvac, alpha_m, alpha_n]**（data/ 为 (3)，前三列语义不变） |
| 其余字段 | 同 data/ | `psi_total`/`psi_plasma*`/`R`/`Z`/`mask`/`coil_currents`(4)/`greens`(4×65×65)/`dpdpsi`/`FdFdpsi`/`x_coords`(4)/`axes` 标量/`solve_time`（见 [data/README.md §3](../data/README.md)） |

## 3. 输入通道明细（11 通道）

通道顺序 = 代码实际拼接顺序：2 几何 + 9 标量。R/Z 线性映射 [-1,1]；9 个标量
z-score（训练集统计）；目标 ψ 同样 z-score。

| # | 通道 | 内容 | 单位 | 来源 |
|---|---|---|---|---|
| 1 | R | 网格 R 坐标 | m | 固定 65² 网格 |
| 2 | Z | 网格 Z 坐标 | m | 固定 65² 网格 |
| 3 | Paxis | 磁轴压强 | Pa | params[0] |
| 4 | Ip | 等离子体电流 | A | params[1] |
| 5 | fvac | 真空通量函数 f | Wb/m | params[2] |
| 6 | alpha_m | 剖面内指数（U[1.0,2.0]） | — | params[3] |
| 7 | alpha_n | 剖面外指数（U[1.5,2.5]） | — | params[4] |
| 8–11 | R_lo, Z_lo, R_up, Z_up | 下/上 X 点坐标 | m | x_coords[0:4] |

剖面 shape = (1−ψn^αm)^αn（freegs `ConstrainPaxisIp`，p′ 与 FF′ 共用）：
αm 控制磁轴附近平坦度（越大轴心越平），αn 控制整体峰化程度（越大边缘梯度越陡）。
合法范围：αm > 0（=0 除零崩溃）、αn ≥ 0（=0 阶跃剖面）。

## 4. 如何调用

```python
import numpy as np
d = np.load("dn_fno_2608/data_v2/train.npz")
params = d["params"]    # (5000, 5): [Ip, paxis, fvac, alpha_m, alpha_n]
```

```bash
PY="C:/Users/HP/.conda/envs/torch5060/python.exe"

# 训练（exp002 实际命令）
"$PY" -u -m gs_pino_dn_fno_2608.train_dn_fno \
  --train-data dn_fno_2608/data_v2/train.npz --val-data dn_fno_2608/data_v2/val.npz \
  --n-train 500 --seed 1 --out-dir dn_fno_2608/experiments/exp002_profile_alphas
# 评估
"$PY" -u -m gs_pino_dn_fno_2608.evaluate_dn_fno \
  --test-data dn_fno_2608/data_v2/test.npz \
  --checkpoint dn_fno_2608/experiments/exp002_profile_alphas/best.pt \
  --out-dir dn_fno_2608/experiments/exp002_profile_alphas
```

通道数自动推断：`build_model(in_channels=2+len(scalar_mean))`（旧 9ch checkpoint
自动兼容为 9）。

## 5. 相关脚本（dn_fno_2608/scripts/）

| 脚本 | 用途 |
|---|---|
| `run_generate_v2.sh` | 全量生成（分块断点续跑 + 合并） |

## 6. 生成设置与统计

- 采样：Paxis U[200,3000] Pa、Ip U[5e4,4e5] A、fvac U[0.5,3.0]、alpha_m U[1.0,2.0]、
  alpha_n U[1.5,2.5]；X 点抖动同 data/（±0.02 m 镜像对称）
- 接受准则：同 data/（收敛 + |ΔIp|≤10% + ≥2 X 点）+ **新增 L 有限且 0<Beta0<1**
  （剖面退化防护）；拒绝时**全参数向量重采样重试**（`--max-retries 20`，仅换
  X 点抖动救不回来——αm/αn 采样后半数组合确定性失败）
- 统计：6000/6000；单样本 0.17–0.41 s/solve（24 核；train 全程 14.3 min）；
  merge sanity：alpha_m [1.000,2.000]、alpha_n [1.500,2.500]

## 7. 历史与偏差记录

1. 唯一系统性差异：alpha_m/alpha_n 采样（§1）；
2. 新增接受准则：L 有限且 0<Beta0<1（data/ 未显式检查，实际样本均满足）；
3. params 扩为 5 维——旧代码读前三列语义不变；
4. 训练/评估/可视化/延迟脚本支持 in_channels=11（旧 checkpoint 不受影响）；
5. 生成过程记录：首轮（`--max-retries 5` 且重试不重采样）接受率仅 ~98%；修复为
   全参数重采样 + 20 次后 6000/6000。train 曾混入首轮遗留分块，已删除全部
   train 分块后干净重生成（同 seed 123）。
