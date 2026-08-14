# data_v4 数据集说明（物理合理性接受约束 + 可行区采样）

> 生成日期：2026-08-14
> 代码：`generate_dn_dataset.py`（data_v4 全部新参数默认关，v3/基线路径字节级不变）
> 用途：修复 data_v3 的约束不可达缺陷（isoflux 残差 mean 0.50 / max 2.10），
> 验证"约束可达 + 可行区采样"下三模型复测（exp006/exp007）的恢复程度
> 取代：data_v3（2026-08-14 已删除；复现脚本 `scripts/run_generate_v3.sh` 保留，
> 其局限分析见 exp004 notes §5b 与本文件 §2）

## 1. 与 data_v3 的差异

| 项 | data_v3 | **data_v4** |
|---|---|---|
| X 点中心 | (1.1, ±0.6) | **(1.2, ±0.6)**（`--xpt-r0 1.2`） |
| X 点抖动 | R 与 Z 同 ±0.20 m | **R ±0.10、Z ±0.15 m**（`--xpt-jitter 0.10 --xpt-jitter-z 0.15`） |
| 锚点 | R~U[1.2,1.8] × Z~U[-0.3,0.3] | **中平面 (R, 0.0)，R~U[1.35,1.65]**（`--anchor-midplane`，外中平面分离面半径） |
| 接受约束 | 收敛 + \|ΔIp\|≤10% + ≥2 X点 + L/Beta0 | **+ 7 项物理合理性检查**（见 §3） |
| 诊断字段 | anchor | **+ 8 个约束诊断字段**（`--save-constraint-diag`，见 §4） |
| 规模 | 3000 | 3000（train 2000 / val 500 / test 500） |

采样范围选择依据（线圈四边形 = P1L/P1U (1.0,±1.1) + P2L/P2U (1.75,±0.6) 的凸包）：
- X 点 R∈[1.10,1.30]：中心 1.2 使最差点距左边缘 ≥0.10 m（v3 的 0.9 端大幅超出）；
- X 点 Z0∈[0.45,0.75]：保持离墙与四边形上下边缘足够远；
- 锚点 R∈[1.35,1.65]：四边形 Z=0 切片为 [1.0,1.75]（留 ≥0.1 余量），且恒 > X 点
  R 上限 1.30 → 磁轴必然夹在 X 点与锚点之间（排除"锚点内移"退化类）。

## 2. 阈值校准（关键步骤）

所有接受阈值先用**已知健康数据 data_v2**（500 val 样本）校准，要求 v2 几乎全过：

| 检查 | 初版阈值 | v2 通过率 | 校准后 | v2 通过率 |
|---|---|---|---|---|
| 四边形余量 | 0.10 m | **1.6%**（X 点 R0=1.1 距左边缘标称 0.1，抖动即跌破；且初版按 tokamak 线圈顺序组多边形，是自交叉蝴蝶结，距离算错） | **0.05 m** | 100% |
| isoflux 残差 | 0.25 | 94.0%（v2 p95=0.256） | **0.35** | 99.2%（仅 4 个 v2 固有离群） |
| 轴在三角形内 | — | 100% | 0.10 m（原值） | — |
| 墙内 / 锚点-X点距离 / core 深度 | — | 100% | 原值 | — |

校准结论：初版 0.10 m 余量对 v2 自身即 98% 拒绝 → 余量降到 0.05（v4 X 点 R∈[1.10,1.30]
最差 0.10 仍安全）；isoflux 0.25 对 v2 p95 边缘 → 0.35（v3 尾部 p95≈1.1 仍被砍大半）。

## 3. 7 项接受约束（全部在 `_solve_one` 内，每项一个 CLI 开关）

1. **磁轴（O 点）在三角形(lo, up, 锚点)内**：重心坐标法（strict + eps），物理含义
    = 等离子体主体被两个 X 点与锚点包围；
2. **三点在线圈四边形凸包内且距边 ≥ 余量**：`--coil-margin 0.05`（线圈中心按极角排
   凸包顺序；ShapedCoil 取形状点均值）；
3. **三点在墙内**：`--require-wall`（tokamak.wall 多边形，射线法）；
4. **isoflux 残差 ≤ 0.35 × core**：`--max-isoflux-residual 0.35`，core = psi_axis −
   0.5(psi(xpt[0])+psi(xpt[1]))（与保存的 psi_bndry 同定义）。4 线圈对 6 约束是
   过定系统，Tikhonov 定点只满足 Aᵀb=0，残差方向不可达——v3 缺陷根源，此处显式
   过滤；
5. **X 点偏差 ≤ 0.10 m**：`--max-xpt-deviation 0.10`，find_critical 实际 X 点与目标
   最近距离（网格步长 R≈0.03/Z≈0.06，0.1 合理）；
6. **锚点-两 X 点距离 ≥ 0.15 m**：`--min-anchor-xpt-dist 0.15`（防近退化区）；
7. **core 深度 ≥ 0.005 Wb**：`--min-core-depth 0.005`（防浅剖面）。

## 4. 字段说明

data_v3 全部字段（含 `anchor`）+ **8 个约束诊断字段**（`--save-constraint-diag`）：

| 字段 | 形状 | 内容 |
|---|---|---|
| `xpts_actual` | (N,2,3) | find_critical 实际 X 点 [R,Z,psi]，按目标 lo/up 贪婪配对排序 |
| `o_point` | (N,3) | 实际 O 点 [R,Z,psi]（axes 仅有 psi_bndry 与轴位置，无 psi） |
| `xpt_constraint_res` | (N,4) | 目标处 Br_lo, Bz_lo, Br_up, Bz_up（T）——X 点约束残差 |
| `isoflux_res` | (N,2) | psi(lo)−psi(锚点), psi(up)−psi(锚点)（Wb） |
| `psi_at_constraints` | (N,3) | psi(lo), psi(up), psi(锚点)（Wb） |
| `n_iter` | (N,1) | Picard 迭代数（`convergenceInfo=True` 历史长度） |
| `psi_relchange_final` | (N,1) | 末次 psi 相对变化（收敛样本 ≤ rtol=1e-3） |

意义：超定系统的约束残差/实际临界点信息**无法从 4 线圈电流恢复**，随样本落盘，
供下游按真实残差分桶分析、或做"残差修正模型"（把不可达信息作为输入/目标）。

## 5. 生成命令

```bash
PY="C:/Users/HP/.conda/envs/torch5060/python.exe"
# 一键：bash dn_fno_2608/scripts/run_generate_v4.sh（探针先验证，再 val/test/train）
"$PY" -u -m gs_pino_dn_fno_2608.generate_dn_dataset --split val --n-samples 500 --seed 456 \
    --out-dir dn_fno_2608/data_v4 --chunk-size 500 --n-jobs 24 \
    --alpha-sampling --xpt-r0 1.2 --xpt-jitter 0.10 --xpt-jitter-z 0.15 \
    --isoflux-sampling --anchor-midplane --max-isoflux-residual 0.35 \
    --max-xpt-deviation 0.10 --min-anchor-xpt-dist 0.15 --require-wall \
    --coil-margin 0.05 --min-core-depth 0.005 --save-constraint-diag --max-retries 20
# （test 500 seed 789、train 2000 seed 123 同理；--merge 后同命令）
```

## 6. 生成统计

| split | seed | 目标 | 接受 | 墙钟 | 速率 |
|---|---|---|---|---|---|
| 探针 | 123 | 80 | 80/80 | 2m10s | 1.63 s/solve |
| val | 456 | 500 | **496/500** | 11m20s | 1.36 s/solve |
| test | 789 | 500 | **494/500** | 11m00s | 1.32 s/solve |
| train | 123 | 2000 | **1982/2000** | 46m | 1.3–1.4 s/solve |

合计 **2972/3000（99.1%）**。被拒样本均为 20 次重试仍不满足 7 项检查的
参数极值组合（接受率与 v3 的 99.9% 相当，但 v4 的拒绝是**物理性**的——
不可达约束/越界几何——而非随机发散）。

**isoflux 残差分布 vs v3**（val 500，`--save-constraint-diag`）：

| 数据集 | mean | median | p95 | max | >0.35 占比 |
|---|---|---|---|---|---|
| data_v3 | 0.50 | 0.43 | 1.11 | 2.10 | 大量（>10% 占 93%） |
| **data_v4 (val)** | **0.172** | **0.168** | **0.332** | **0.350** | **0%**（阈值硬切） |

X 点偏差：mean 0.012 / p95 0.028 / max 0.037 m（阈值 0.10）。n_iter∈[10,43]、
psi_relchange_final ≤7.3e-4 < rtol。三点四边形覆盖 496/496。

## 7. 探针结果（80 样本，seed 123）

| 指标 | 探针 | 目标 |
|---|---|---|
| 接受率 | 80/80（100%） | ≥60% |
| isoflux 残差 mean / max | 0.177 / 0.337 | <0.25 / ≤0.35 |
| X 点偏差 max | 0.033 m | ≤0.10 m |
| 三点四边形覆盖 | 80/80 | 100% |
| n_iter | [11, 29] | ≤50 |

isoflux 残差 mean 0.177 ≈ data_v2 水平（0.133），v3 的 0.50 → 0.18，**约束不可达
缺陷消除**。注意 X 点 R 分布呈现接受截断（max 1.246 < 1.30）：锚点贴 X 点/残差
超阈的采样被滤掉，数据落在 4 线圈可达子集内——这正是 v4 的目的。

## 8. 复现偏差记录

1. 初版四边形余量 0.10 m + tokamak 线圈顺序多边形（自交叉蝴蝶结）→ 98% 误拒：
   修复为极角凸包排序 + 余量 0.05（用 data_v2 校准，见 §2）；
2. isoflux 残差阈值 0.25 → 0.35（v2 p95=0.256 校准）；
3. 所有新 flag 默认关，`--xpt-jitter` 语义不变（R 向），新增 `--xpt-jitter-z`
   （默认等于 R 向）→ v3 命令逐字节复现（回归验证通过）；
4. 诊断字段仅 `--save-constraint-diag` 时保存，merge 对缺字段容错；
5. 求解器在 `convergenceInfo=True` 时返回收敛历史，行为与默认调用一致。

## 9. 三模型复测结果（exp006/exp007，N=500 seed 1，test 494）

| 模型 | data_v3 test | **data_v4 test** | 恢复倍数 | 结论 |
|---|---|---|---|---|
| A（X点 11ch） | 31.09% | **3.38%** | 9.2× | 一对多退化仍在（7.7× vs A'），不再 100× |
| A'（X点+锚点 13ch） | 21.72% | **0.442%** | 49× | **约束不可达是 exp004 退化主因**；回到 data_v2 水平 |
| B（coil 11ch） | 12.47% | **0.499%** | 25× | 恢复到 data_v2 水平；A' ≥ B（exp005 优势为伪差） |

详情与分桶见 [exp006/README.md](../experiments/exp006_xpoints_anchor_v4/README.md) /
[exp007/README.md](../experiments/exp007_coil_input_v4/README.md)。
