# data/ — 论文基线数据集（arXiv:2608.05555）

> 生成日期：2026-07（复现第一阶段）
> 数据规模：6000（train 5000 seed 123 / val 500 seed 456 / test 500 seed 789），**6000/6000 全部通过**
> 生成脚本：`src/gs_pino_dn_fno_2608/generate_dn_dataset.py`（默认参数，无 `--alpha-sampling`）
> 一键复现：`bash dn_fno_2608/scripts/run_generate.sh`
> 扩展版（剖面形状参数采样）见 [data_v2/README.md](../data_v2/README.md)

## 1. 背景

论文 *Millisecond-Scale Neural Operator Surrogates for Double-Null Free-Boundary
Grad-Shafranov Equilibria*（arXiv:2608.05555v1）的复现数据集：freegs 在
TestTokamak 几何上求解**双零（DN）自由边界 GS 平衡**，训练 FNO 神经算子代理
（输入位形参数 → ψ(R,Z) 场）。本数据集同时为后续 PINO（物理约束）阶段预留全部字段。

## 2. 参数空间（论文 Eq. 4，共 7 个标量）

| 参数 | 范围 / 分布 |
|---|---|
| Paxis | 均匀 U[200, 3000] Pa |
| Ip | 均匀 U[5e4, 4e5] A |
| fvac | 均匀 U[0.5, 3.0] |
| 下 X 点 (R_lo, Z_lo) | (1.1, −0.6) + δ，δ 分量独立 U[−0.02, 0.02] m |
| 上 X 点 (R_up, Z_up) | (1.1, +0.6) + δ（上下共用 δ 镜像，严格对称） |

剖面形状参数**固定**为 freegs 默认：alpha_m=1.0、alpha_n=2.0
（`ConstrainPaxisIp`，shape = (1−ψn^αm)^αn）。

## 3. 求解器配置

- freegs 0.8.3.dev54（TestTokamak），域 R∈[0.1, 2.0] m、Z∈[−2, 2] m，**65×65 均匀网格**；
- 双 X 点约束 + isoflux 约束（锚定外中平面固定点 (1.5, 0.0) m），γ=1e-12；
- Picard 迭代 rtol=1e-3、maxits=50；
- 接受准则：求解收敛 且 |Ip_sol − Ip_tgt|/Ip_tgt ≤ 10% 且 find_critical 找到 ≥2 个 X 点；
- 单样本中位求解时间约 1.08 s（本机 24 核并行）。

## 4. 数据文件与字段

`dn_fno_2608/data/{train,val,test}.npz`（float32；`{split}/chunk_XXX.npz` 为生成
中间产物，合并后冗余可删）。每样本保存全字段（PINO 预留）：

| 字段 | 含义 |
|---|---|
| psi_total | 总极向磁通（目标，z-score 输出） |
| psi_plasma / psi_plasma_norm / psi_coils | 等离子体 / 归一化 / 线圈磁通分量 |
| R, Z | 物理坐标 (65,65)（米制） |
| mask | freegs critical.core_mask |
| coil_currents (4) | 4 个控制线圈电流 |
| greens (4×65×65) | 线圈 Green 函数 |
| dpdpsi, FdFdpsi | GS 残差 RHS = −μ₀R²·dp/dψ − F·dF/dψ 所需字段 |
| params (3) | [Ip, paxis, fvac] |
| x_coords (4) | [R_lo, Z_lo, R_up, Z_up] X 点坐标 |
| psi_axis, psi_bndry, R_axis, Z_axis, L, Beta0 | 平衡标量量 |
| solve_time | 单样本求解耗时 |

## 5. 输入结构（9 通道）

```
R, Z | Paxis, Ip, fvac | R_lo, Z_lo, R_up, Z_up   （7 标量 z-score 广播）
```

归一化：R、Z 线性映射 [−1,1]；7 个标量用**训练集**均值/标准差 z-score 后广播；
目标 ψ 同样 z-score。训练损失与 rel L2 均在归一化域计算。

## 6. 生成命令（复现）

```bash
PY="C:/Users/HP/.conda/envs/torch5060/python.exe"
bash dn_fno_2608/scripts/run_generate.sh    # val→test→train 分块生成 + 合并

# 或分步（默认参数，无 --alpha-sampling）
"$PY" -m gs_pino_dn_fno_2608.generate_dn_dataset --split train --n-samples 5000 --seed 123 \
      --out-dir dn_fno_2608/data --chunk-size 500 --n-jobs 24
```

日志：`dn_fno_2608/logs/gen_{val,test,train}.log`。

## 7. 与其他数据的关系

| 数据集 | 与 data/ 的关系 |
|---|---|
| data/（本文档） | 论文基线：7 标量参数空间，剖面形状固定 (1.0, 2.0)，9 通道输入 |
| data_v2/ | 唯一差异：alpha_m/alpha_n 采样（αm~U[1,2]、αn~U[1.5,2.5]），参数空间 9 维，11 通道输入；其余完全一致 |
