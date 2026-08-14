# data_v2 — 剖面形状参数采样的扩展数据集

> 生成日期：2026-08-14
> 数据规模：6000（train 5000 / val 500 / test 500）
> 生成脚本：`src/gs_pino_dn_fno_2608/generate_dn_dataset.py`（`--alpha-sampling`）
> 一键复现：`bash dn_fno_2608/scripts/run_generate_v2.sh`（分块断点续跑，可重入）

## 1. 与论文基线 data/ 的差异

> 基线数据集完整说明见 [data/README.md](../data/README.md)。

| 项 | data/（论文基线） | **data_v2/（本数据集）** |
|---|---|---|
| alpha_m（剖面内指数） | 固定 1.0 | **均匀 U[1.0, 2.0]** |
| alpha_n（剖面外指数） | 固定 2.0 | **均匀 U[1.5, 2.5]** |
| 其余全部参数/求解器/网格/字段 | — | 与基线完全一致 |

即：**唯一差异 = 剖面形状指数从固定值改为采样**（其余 X 点抖动、paxis/Ip/fvac 范围、
接受准则、网格、约束、字段布局均保持论文设定）。本数据集的动机：原参数空间只有 7 个
标量（3 运行参数 + 4 X 点坐标），剖面形状固定可能限制神经算子表达力；data_v2 把
参数空间扩到 9 维（+2 个剖面形状参数）。

## 2. 参数空间（采样分布）

| 参数 | 分布 | 含义 |
|---|---|---|
| Paxis | 均匀 U[200, 3000] Pa | 磁轴压强 |
| Ip | 均匀 U[5e4, 4e5] A | 等离子体电流 |
| fvac | 均匀 U[0.5, 3.0] | 真空区 F·F′ 常数 |
| **alpha_m** | **均匀 U[1.0, 2.0]** | 剖面内指数，控制**磁轴附近（ψn→0）平坦度**：越大轴心越平、梯度向边缘集中；越小轴心越尖 |
| **alpha_n** | **均匀 U[1.5, 2.5]** | 剖面外指数，控制**整体峰化程度/边缘梯度**：越大 shape 整体压低、边缘梯度越陡 |
| 下 X 点 (R_lo, Z_lo) | (1.1, −0.6) + δ，δ 分量独立 U[−0.02, 0.02] m | 论文 Eq.4 设定（不变） |
| 上 X 点 (R_up, Z_up) | (1.1, +0.6) + δ（上下共用 δ 镜像，严格对称） | 论文 Eq.4 设定（不变） |

**剖面形状公式**（freegs `jtor.ConstrainPaxisIp`，p′ 与 FF′ 共用同一 shape）：

$$J_\phi = L\left[\beta_0\frac{R}{R_{\mathrm{axis}}} + (1-\beta_0)\frac{R_{\mathrm{axis}}}{R}\right]
\left(1-\psi_n^{\alpha_m}\right)^{\alpha_n}$$

合法范围约束（freegs 源码）：αm > 0 必须严格（=0 时 shapeintegral=0 除零崩溃）；
αn ≥ 0（0 退化为阶跃剖面）；文献/示例从未超过 3。本数据集窄幅以默认 (1.0, 2.0)
为中心，保证求解器接受率与数值精度。

## 3. 求解器配置（同基线 data/）

- freegs 0.8.3.dev54（TestTokamak 几何），域 R∈[0.1, 2.0] m、Z∈[−2, 2] m，**65×65 均匀网格**；
- 双 X 点约束 + isoflux 约束（锚定外中平面固定点 (1.5, 0.0) m），γ=1e-12；
- Picard 迭代 rtol=1e-3、maxits=50；
- 剖面：`ConstrainPaxisIp(eq, paxis, Ip, fvac, alpha_m, alpha_n)`（alpha 按 §2 采样）。

## 4. 采样与接受准则

- 训练 5000 样本（seed 123）/ 验证 500（seed 456）/ 测试 500（seed 789）——与基线同 seeds；
- 接受准则：求解收敛 且 |Ip_sol − Ip_tgt|/Ip_tgt ≤ 10% 且 find_critical 找到 ≥2 个 X 点
  **且 L 有限 且 0 < Beta0 < 1**（新增剖面退化防护检查）；
- 拒绝时**全参数向量重采样重试**（同一参数族，alpha 一并重采样），v2 用
  `--max-retries 20`：αm/αn 采样后约半数 (paxis, Ip, fvac, alpha) 组合会确定性
  失败（如 Beta0 越界或位形脱离双零），仅换 X 点抖动永远救不回来，必须重采样。
  20 次独立尝试后终期接受率 ≈ 6000/6000。

## 5. 数据文件与字段

`dn_fno_2608/data_v2/{train,val,test}.npz`（float32；`{split}/chunk_XXX.npz` 为生成
中间产物，合并后冗余可删）。每个样本保存**全字段**（为后续 PINO 物理约束阶段预留）：

| 字段 | 含义 |
|---|---|
| psi_total | 总极向磁通（目标，z-score 输出） |
| psi_plasma / psi_plasma_norm / psi_coils | 等离子体 / 归一化 / 线圈磁通分量 |
| R, Z | 物理坐标 (65,65)（米制） |
| mask | freegs critical.core_mask |
| coil_currents (4) | 4 个控制线圈电流 |
| greens (4×65×65) | 线圈 Green 函数 |
| dpdpsi, FdFdpsi | GS 残差 RHS = −μ₀R²·dp/dψ − F·dF/dψ 所需字段 |
| **params (5)** | **[Ip, paxis, fvac, alpha_m, alpha_n]**（基线为 (3)） |
| x_coords (4) | [R_lo, Z_lo, R_up, Z_up] X 点坐标 |
| psi_axis, psi_bndry, R_axis, Z_axis, L, Beta0 | 平衡标量量 |
| solve_time | 单样本求解耗时 |

## 6. 输入结构：9 通道 → 11 通道

```
基线 data/（9 通道）:   R, Z | Paxis, Ip, fvac | R_lo, Z_lo, R_up, Z_up          (7 标量)
data_v2（11 通道）:      R, Z | Paxis, Ip, fvac, alpha_m, alpha_n | R_lo...Z_up  (9 标量)
```

归一化：R、Z 线性映射 [−1,1]；9 个标量用**训练集**均值/标准差 z-score 后广播；
目标 ψ 同样 z-score。`data_dn_fno.py` 数据类对通道数完全通用（params 从 3 扩到 5
后自动生成 11 通道），模型 `build_model(in_channels=...)` 已参数化——下游仅需
按 stats 推断 `in_channels = 2 + len(scalar_mean)`（train/evaluate/visualize/latency
已实现，旧 checkpoint 自动兼容为 9）。

## 7. 生成命令（复现）

```bash
PY="C:/Users/HP/.conda/envs/torch5060/python.exe"

# 一键全量（分块断点续跑 + 合并）
bash dn_fno_2608/scripts/run_generate_v2.sh

# 或分步（等价）
"$PY" -m gs_pino_dn_fno_2608.generate_dn_dataset --split val --n-samples 500 --seed 456 \
      --out-dir dn_fno_2608/data_v2 --chunk-size 500 --n-jobs 24 --alpha-sampling
"$PY" -m gs_pino_dn_fno_2608.generate_dn_dataset --split test --n-samples 500 --seed 789 \
      --out-dir dn_fno_2608/data_v2 --chunk-size 500 --n-jobs 24 --alpha-sampling
"$PY" -m gs_pino_dn_fno_2608.generate_dn_dataset --split train --n-samples 5000 --seed 123 \
      --out-dir dn_fno_2608/data_v2 --chunk-size 500 --n-jobs 24 --alpha-sampling
"$PY" -m gs_pino_dn_fno_2608.generate_dn_dataset --split train --out-dir dn_fno_2608/data_v2 --merge
"$PY" -m gs_pino_dn_fno_2608.generate_dn_dataset --split val   --out-dir dn_fno_2608/data_v2 --merge
"$PY" -m gs_pino_dn_fno_2608.generate_dn_dataset --split test  --out-dir dn_fno_2608/data_v2 --merge
```

日志：`dn_fno_2608/logs/gen_v2_{val,test,train}.log`。

## 8. 生成统计（2026-08-14 实测）

- **接受率：6000/6000**（val 500/500、test 500/500、train 5000/5000，`--max-retries 20`）；
- 单样本求解时间：0.17–0.41 s/solve（24 核并行；train 全程 14.3 min）；
- merge 输出参数范围 sanity：alpha_m [1.000, 2.000]、alpha_n [1.500, 2.500]，
  Ip/paxis/fvac/X 点与论文范围一致（见 logs/gen_v2_*.log）。

> 过程记录：首轮生成（`--max-retries 5` 且重试不重采样参数）接受率仅 ~98%；
> 修复为拒绝时全参数重采样 + 20 次尝试后达到 6000/6000。train 曾混入首轮遗留
> 分块（不同重试体制），已删除全部 train 分块后一次性干净重生成（同 seed 123）。

## 9. 与基线的偏差记录

1. 唯一系统性差异：alpha_m/alpha_n 采样（§1）；
2. 新增接受准则：L 有限且 0 < Beta0 < 1（基线未显式检查，但实际样本均满足）；
3. params 字段扩为 5 维（Ip, paxis, fvac, alpha_m, alpha_n）——旧代码若直接读
   data_v2 的 params 前三列语义不变；
4. 训练/评估/可视化/延迟脚本支持 `in_channels=11`（旧 checkpoint 不受影响）。
