# data_v4 — 可行区采样 + 物理合理性接受约束数据集（TestTokamak DN）

> 生成日期：2026-08-14
> 用途：exp006/exp007 数据；修复 data_v3 的 isoflux 约束不可达缺陷
> （残差 mean 0.50/max 2.10 → mean 0.172/max ≤0.35）
> 取代：data_v3（已删除；复现脚本 `scripts/run_generate_v3.sh` 保留）

## 1. 数据集速览

| 项 | 值 |
|---|---|
| 机器 | TestTokamak（同 data/） |
| 位形 | 双零 DN |
| 网格 | 65² |
| 规模 | **2972/3000（99.1%）**：train 1982（seed 123）/ val 496（seed 456）/ test 494（seed 789） |
| 输入通道 | **11ch**（X点）/ **13ch**（X点+锚点）/ **11ch**（coil 电流）——三种模式（§3） |
| 目标 | `psi_total`（65²，z-score） |
| 新增 | 8 个约束诊断字段（`--save-constraint-diag`，§2）+ 可行区采样 + 7 项接受约束（§4） |

## 2. 字段说明（npz keys）

data_v3 全部字段（含 `anchor`）+ 8 个约束诊断字段（仅 `--save-constraint-diag`
时生成，merge 对缺字段容错）：

| 字段 | 形状 | 内容 |
|---|---|---|
| `xpts_actual` | (N,2,3) | find_critical 实际 X 点 [R,Z,psi]，按目标 lo/up 贪婪配对排序 |
| `o_point` | (N,3) | 实际 O 点 [R,Z,psi]（axes 只有 psi_bndry 与轴位置，无 psi） |
| `xpt_constraint_res` | (N,4) | 目标处 Br_lo, Bz_lo, Br_up, Bz_up（T）——X 点约束残差 |
| `isoflux_res` | (N,2) | psi(lo)−psi(锚点), psi(up)−psi(锚点)（Wb） |
| `psi_at_constraints` | (N,3) | psi(lo), psi(up), psi(锚点)（Wb） |
| `n_iter` | (N,1) | Picard 迭代数 |
| `psi_relchange_final` | (N,1) | 末次 psi 相对变化（收敛样本 ≤ rtol=1e-3） |
| `anchor` | (N,2) | isoflux 锚点 [R,Z]，中平面 Z≡0、R~U[1.35,1.65] |

其余字段同 data_v2（`params` (5)、`coil_currents` (4)、`greens` (4×65×65)、
`x_coords`、`psi_total`/`psi_plasma*`/`R`/`Z`/`mask`/`dpdpsi`/`FdFdpsi`/`axes` 等）。

**诊断字段意义**：超定系统（4 线圈对 6 约束）的约束残差/实际临界点信息**无法从
4 线圈电流恢复**，随样本落盘供下游按真实残差分桶分析、或做"残差修正模型"。

## 3. 输入通道明细（三种模式，exp006/exp007 用）

| # | 11ch X点（exp006-A） | 13ch X点+锚点（exp006-A'） | 11ch coil（exp007-B） |
|---|---|---|---|
| 1–2 | R, Z 网格 | R, Z 网格 | R, Z 网格 |
| 3–7 | Ip, paxis, fvac, alpha_m, alpha_n | Ip, paxis, fvac, alpha_m, alpha_n | Ip, paxis, fvac, alpha_m, alpha_n |
| 8–11 | R_lo, Z_lo, R_up, Z_up | R_lo, Z_lo, R_up, Z_up | I_P1L, I_P1U, I_P2L, I_P2U |
| 12–13 | — | R_anc, Z_anc | — |

归一化：R/Z 线性 [-1,1]；标量 z-score（训练集统计）。A' 的锚点 Z≡0 通道 std=0，
数据集类分母 `np.maximum(std, 1e-8)` 防护 → 常量通道映射为 0。

## 4. 生成设置

### 4.1 采样范围（可行区，避开线圈四边形外/墙外）

- X 点中心 (1.2, ±0.6)（`--xpt-r0 1.2`），抖动 R ±0.10 / Z ±0.15 m
  （`--xpt-jitter 0.10 --xpt-jitter-z 0.15`）
- 锚点：中平面 (R, 0.0)，R~U[1.35,1.65]（`--anchor-midplane`；恒 > X 点 R 上限
  1.30 → 磁轴必然夹在 X 点与锚点之间，排除"锚点内移"退化类）
- 参数范围同 data_v2（paxis/Ip/fvac/alpha_m/alpha_n 采样）

### 4.2 7 项接受约束（全部在 `_solve_one` 内，每项一个 CLI 开关）

1. **磁轴（O 点）在三角形(lo, up, 锚点)内**：重心坐标法（strict + eps）；
2. **三点在线圈四边形凸包内且距边 ≥ 余量**：`--coil-margin 0.05`（极角凸包排序；
   ShapedCoil 取形状点均值）；
3. **三点在墙内**：`--require-wall`（射线法）；
4. **isoflux 残差 ≤ 0.35 × core**：`--max-isoflux-residual 0.35`，core = psi_axis
   − 0.5(psi(xpt[0])+psi(xpt[1]))（与 psi_bndry 同定义）——data_v3 缺陷根源，
   此处显式过滤；
5. **X 点偏差 ≤ 0.10 m**：`--max-xpt-deviation 0.10`（网格步长 R≈0.03/Z≈0.06）；
6. **锚点-两 X 点距离 ≥ 0.15 m**：`--min-anchor-xpt-dist 0.15`；
7. **core 深度 ≥ 0.005 Wb**：`--min-core-depth 0.005`。

### 4.3 阈值校准（用已知健康数据 data_v2 的 500 val 样本）

| 检查 | 初版 | v2 通过率 | 校准后 | 依据 |
|---|---|---|---|---|
| 四边形余量 | 0.10 m | **1.6%**（自交叉蝴蝶结 bug + 抖动即跌破） | **0.05 m** | 极角凸包排序修复 + 最差 0.10 仍安全 |
| isoflux 残差 | 0.25 | 94.0%（v2 p95=0.256） | **0.35** | 99.2%（仅 4 个 v2 固有离群） |

### 4.4 生成统计

| split | seed | 接受 | 速率 |
|---|---|---|---|
| 探针 | 123 | 80/80 | 1.63 s/solve |
| val | 456 | 496/500 | 1.36 s/solve |
| test | 789 | 494/500 | 1.32 s/solve |
| train | 123 | 1982/2000 | 1.3–1.4 s/solve |

isoflux 残差 val：mean 0.172 / median 0.168 / p95 0.332 / max 0.350（阈值硬切）；
X 点偏差 mean 0.012 / p95 0.028 / max 0.037；n_iter∈[10,43]。被拒样本均为 20 次
重试仍不满足检查的**参数极值组合**（物理性拒绝，非随机发散）。

## 5. 如何调用

```bash
PY="C:/Users/HP/.conda/envs/torch5060/python.exe"

# 训练（一键链：bash dn_fno_2608/scripts/run_exp006_007_train.sh）
"$PY" -u -m gs_pino_dn_fno_2608.train_dn_fno \
  --train-data dn_fno_2608/data_v4/train.npz --val-data dn_fno_2608/data_v4/val.npz \
  --n-train 500 --seed 1 --out-dir <out>                          # 11ch X点
"$PY" -u -m gs_pino_dn_fno_2608.train_dn_fno --input-mode xa \
  --train-data dn_fno_2608/data_v4/train.npz --val-data dn_fno_2608/data_v4/val.npz \
  --n-train 500 --seed 1 --out-dir <out>                          # 13ch X点+锚点
"$PY" -u -m gs_pino_dn_fno_2608.train_dn_fno_coils \
  --train-data dn_fno_2608/data_v4/train.npz --val-data dn_fno_2608/data_v4/val.npz \
  --n-train 500 --seed 1 --out-dir <out>                          # 11ch coil
# 评估（evaluate_dn_fno / evaluate_dn_fno_coils 对应）
# 锚点分桶分析：analyze_anchor_buckets（见 exp006 README §5）
```

## 6. 相关脚本（dn_fno_2608/scripts/）

| 脚本 | 用途 |
|---|---|
| `run_generate_v4.sh` | 全量生成（探针→val→test→train + 合并） |
| `run_generate_v3.sh` | data_v3 复现（**数据已删除**，仅脚本保留参考） |
| `run_exp004_005_train.sh` / `run_exp006_007_train.sh` | data_v3/data_v4 上三模型训练链 |

## 7. 历史与偏差记录

1. **与 data_v3 差异**：X 点中心 (1.1,±0.6)→(1.2,±0.6)、抖动 ±0.20→R±0.10/Z±0.15、
   锚点固定→中平面采样 R∈[1.35,1.65]、+7 项接受约束、+8 诊断字段；
2. 初版四边形余量 0.10 m + tokamak 线圈顺序多边形（自交叉蝴蝶结，距离算错）→
   98% 误拒：修复为极角凸包排序 + 余量 0.05（§4.3）；
3. isoflux 残差阈值 0.25 → 0.35（v2 p95 校准）；
4. 所有新 flag 默认关（v3 命令逐字节复现，回归验证通过）；
5. 诊断字段仅 `--save-constraint-diag` 时保存；
6. **三模型复测结果**（exp006/007，N=500 seed 1，test 494）：A（X点 11ch）
   31.09%→**3.38%**（9.2×）、A'（X点+锚点 13ch）21.72%→**0.442%**（49×）、
   B（coil 11ch）12.47%→**0.499%**（25×）——**约束不可达是 exp004/005 退化主因**；
7. A' 训练 NaN 修复：锚点 Z≡0 → z-score 0/0 → 分母 `np.maximum(std, 1e-8)`
   （data_dn_fno.py / data_dn_fno_coils.py，data_v2/v3 路径不受影响）。
