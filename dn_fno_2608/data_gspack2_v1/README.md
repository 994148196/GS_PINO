# data_gspack2_v1 — gspack2_TRAE 版 MAST 混合位形数据集（DN+SN，18ch coil 输入）

> 生成日期：2026-08-24 ｜ 代码：`gs_gspack2_dn_fno_2608.generate_g2_dataset`
> （`--config {dn,sn}` + `--machine mast_g2`）；求解包位于 `D:\D_F\Fusion\AI\PINN\gspack2_TRAE`
> （gspack v2.0.0）
> 用途：exp201（做法1 RHS 物理残差）/ exp202（做法2 两阶段）的数据——与 exp103/104
> 同口径的 **数据源替换实验**：管线（train/evaluate/data/model/loss）零改动，
> 仅把 freegs 生成的 data_v5 换成本数据集（同 26 键 npz schema、65² 网格、
> MAST 11 线圈 1:1 复刻、18 输入通道）
> 探针报告：`_probe_dn.json`/`_probe_sn.json`
> 结果：exp201（做法1）test **1.197%**、exp202（做法2）**1.316%**——与本数据源的
> 可达误差下限 ~1.1% 一致（v5 对应 0.703%/0.76%，分布等价、误差下限更高，
> 诊断链见 exp201 README §6）

## 1. 数据集速览

| 项 | 值 |
|---|---|
| 机器 | **MAST（gspack 版 1:1 复刻）**：11 控制线圈（P2U/P2L/P3U/P3L/P4U/P4L/P5U/P5L/P6U/P6L + P1 Solenoid），**无墙**；几何 = freegs `machine.py` MAST()（machine.py:1408-1428），turns=1、control=True |
| 求解 | gspack.Equilibrium（von Hagenow 自由边界，order=2/method=lu）+ ConstrainPaxisIp + constrain(xpoints/isoflux, gamma=1e-12) + picard.solve（rtol=1e-3, **maxits=80**） |
| 位形 | DN（双 X 点）+ SN（单 X 点），**按配置分开落盘**（`dn/`、`sn/`） |
| 网格 | 65²，R∈[0.1,2.0] m、Z∈[-2.0,2.0] m（与 data_v5 网格逐点相同） |
| 规模 | dn、sn 各 train 500（seed 123）/ val 500（seed 456）/ test 500（seed 789），6 split 全部满接受；**可补数据**（top-up，§9） |
| 输入通道 | **18ch**（coil 主用，exp201/202）——与 exp103/104 完全同构（2 网格 + 5 params + 11 线圈电流，**无 config 通道**） |
| 目标 | `psi_total`（65²，z-score） |
| 新增字段 | `config`（(N,1)，0=DN, 1=SN）——字段存在但不加载（use_config=False，同 exp103） |

## 2. 目录与文件

```
dn_fno_2608/data_gspack2_v1/
├── dn/  train.npz / val.npz / test.npz   # DN 配置（+ 分块目录 chunks，冗余可删）
├── sn/  train.npz / val.npz / test.npz   # SN 配置
├── _probe_dn.json / _probe_sn.json       # 探针结果（80 样本/配置）
└── README.md
```

每配置独立 out-dir（同 data_v5 布局）——chunk 断点续跑按 (config, split,
chunk_idx) 定位（共用 out-dir 会让 sn 的 chunk 与 dn 撞号被整块跳过，见 §8.2
教训）。npz 不入 git（.gitignore），`run_generate_g2.sh` 一键再生成。

## 3. 字段说明（npz keys，26 个，与 data_v5 逐键同构）

data_v4 全部 25 字段 + `config`。**x_coords 固定 4 通道**：SN 样本上 X 点对存
**(0.0, 0.0) 占位**（恒值通道 → z-score 后为 0；本实验无 config 通道时位形信息
由 11 线圈电流承载，up=(0,0) 占位通道恒 0 不参与）。SN 的诊断字段形状缩小：
xpt_constraint_res (N,2)、isoflux_res (N,1)、psi_at_constraints (N,2)（[p_lo,
p_anc]）；`xpts_actual` 行数 DN 2 行 / SN 1 行（混合拼接时 NaN-pad 到最大行数）。
与 data_v5 的逐键差异仅数值来源（freegs → gspack），键名/形状/语义完全一致。

| 字段 | 形状（DN） | 形状（SN） | 含义 | 是否模型输入 |
|---|---|---|---|---|
| `psi_total` | (N,65,65) | 同 | 总极向磁通（**训练目标**） | — |
| `psi_plasma` / `psi_plasma_norm` / `psi_coils` | (N,65,65) | 同 | 磁通分量 | 否（PINO 预留） |
| `R`, `Z` | (65,65) | 同 | 物理坐标网格 | 通道 1–2 |
| `mask` | (N,65,65) | 同 | core_mask | 否 |
| `params` | (N,5) | 同 | [Ip, paxis, fvac, alpha_m, alpha_n] | 通道 3–7 |
| `coil_currents` | (N,11) | 同 | 11 线圈电流（顺序见 §4） | 通道 8–18 |
| `x_coords` | (N,4) | 同 | [R_lo, Z_lo, R_up, Z_up]；SN 的 up=(0,0) 占位 | 否（exp201/202 不加载） |
| `anchor` | (N,2) | 同 | isoflux 锚点 [R,Z] | 否 |
| `config` | (N,1) | 同 | 0=DN / 1=SN（**不加载**，仅标注） | **否** |
| `greens` | (N,11,65,65) | 同 | 线圈 Green 函数（`eq._coil_psi_unit`，逐样本存储） | 否（PINO 预留） |
| `dpdpsi`, `FdFdpsi` | (N,65,65) | 同 | GS 残差 RHS 分量（**exp201 做法1 用**） | RHS 物理项 |
| `axes` | (N,4) | 同 | [R_axis, Z_axis, psi_bndry, psi_axis] | 否（评估用） |
| `L`, `Beta0` | (N,1) | 同 | 电感 / 比压 | 否 |
| `solve_time` | (N,1) | 同 | 求解耗时 | 否 |
| `xpts_actual` | (N,2,3) | **(N,1,3)** | 实际 X 点 [R,Z,psi] | 否（评估几何真值） |
| `o_point` | (N,3) | 同 | 实际 O 点 [R,Z,psi]（位置过滤后取 psi 最大） | 否 |
| `xpt_constraint_res` | (N,4) | **(N,2)** | 目标处 Br/Bz（T） | 否（诊断） |
| `isoflux_res` | (N,2) | **(N,1)** | psi(X 点)−psi(锚点)（Wb） | 否（诊断） |
| `psi_at_constraints` | (N,3) | **(N,2)** | psi(lo)[, psi(up)], psi(锚点)（Wb） | 否（诊断） |
| `n_iter` | (N,1) | 同 | Picard 迭代数（DN ~10 / SN ~44） | 否（诊断） |
| `psi_relchange_final` | (N,1) | 同 | 末次 psi 相对变化（全部 ≤1e-3） | 否（诊断） |

## 4. 输入通道明细（18ch，同 exp103）

| # | 通道 | 含义/单位 | 来源 |
|---|---|---|---|
| 1 | R | 网格 R 坐标（[-1,1]） | 固定 65² 网格（= v5） |
| 2 | Z | 网格 Z 坐标（[-1,1]） | 固定 65² 网格（= v5） |
| 3 | Ip | 等离子体电流 (A) | params[0] |
| 4 | paxis | 磁轴压强 (Pa) | params[1] |
| 5 | fvac | 真空通量函数 f (Wb/m) | params[2] |
| 6 | alpha_m | 剖面形状指数 m | params[3] |
| 7 | alpha_n | 剖面形状指数 n | params[4] |
| 8 | I_P2U | 上 P2 线圈电流 (A) | coil_currents[0] |
| 9 | I_P2L | 下 P2 线圈电流 (A) | coil_currents[1] |
| 10 | I_P3U | 上 P3 线圈电流 (A) | coil_currents[2] |
| 11 | I_P3L | 下 P3 线圈电流 (A) | coil_currents[3] |
| 12 | I_P4U | 上 P4 线圈电流 (A) | coil_currents[4] |
| 13 | I_P4L | 下 P4 线圈电流 (A) | coil_currents[5] |
| 14 | I_P5U | 上 P5 线圈电流 (A) | coil_currents[6] |
| 15 | I_P5L | 下 P5 线圈电流 (A) | coil_currents[7] |
| 16 | I_P6U | 上 P6 线圈电流 (A) | coil_currents[8] |
| 17 | I_P6L | 下 P6 线圈电流 (A) | coil_currents[9] |
| 18 | I_P1 (Solenoid) | 中心螺线管电流 (A) | coil_currents[10] |

线圈名/顺序 = freegs MAST() 定义顺序 = data_v5 线圈序（生成脚本按
`tokamak.coils` 收集，11 个全 control）。greens 单位响应 = gspack
`eq._coil_psi_unit`（构造时置 current=1.0 的缓存，Coil = 单位电流响应 ×
turns=1；Solenoid = Σ 子绕组）；**恒等式 Σ_k I_k·G_k ≡ psi_coils 实测
max diff 3.2e-8 Wb**（data_v5 同为 1e-8 量级）。

## 5. 如何调用

### 5.1 直接加载

```python
import numpy as np
d = np.load("dn_fno_2608/data_gspack2_v1/dn/train.npz")
cfg = d["config"]           # (500,): 0=DN / 1=SN
coils = d["coil_currents"]  # (500, 11): 顺序见 §4
rhs = d["dpdpsi"], d["FdFdpsi"]  # GS 残差 RHS 分量（做法1 物理）
```

### 5.2 训练 / 评估（exp201/202 实际命令）

```bash
PY="C:/Users/HP/.conda/envs/torch5060/python.exe"

# exp201 训练（做法1 rhs，coil 18ch，DN+SN 混合，逗号拼接；stats 全池 1000 计算）
"$PY" -u -m gs_pino_fno_phys.train_pino --mode rhs \
  --train-data dn_fno_2608/data_gspack2_v1/dn/train.npz,dn_fno_2608/data_gspack2_v1/sn/train.npz \
  --val-data dn_fno_2608/data_gspack2_v1/dn/val.npz,dn_fno_2608/data_gspack2_v1/sn/val.npz \
  --n-train 500 --seed 1 --epochs 800 --phys-weight 0.1 --out-dir <out>
# exp202 两阶段：--mode twostage --phys-weight 0.1 --ip-weight 1.0 --j-weight 1.0
# 评估（分桶：--test-data 单传 dn 或 sn 文件）
"$PY" -u -m gs_pino_fno_phys.evaluate_pino \
  --test-data dn_fno_2608/data_gspack2_v1/sn/test.npz \
  --checkpoint <out>/best.pt --out-dir <out>/eval_sn --machine mast
```

## 6. 相关脚本（dn_fno_2608/scripts/）

| 脚本 | 用途 |
|---|---|
| `run_generate_g2.sh` | 探针 + 全量生成一键（`probe`/`gen`/`topup`/`all`；首批 6 split ~20 min） |
| `probe_g2.py` | 探针：原始接受率 + SN midplane 门诊断（`--machine mast_g2 --config sn --n 80`） |
| `run_exp201_202_pino.sh` | exp201/202 训练 + 三桶评估（smoke/train/eval/all） |

## 7. 生成设置与统计

### 7.1 采样口径（= data_v5 MAST）

- 参数：paxis U[1e3,5e3] Pa / Ip U[3e5,8e5] A / fvac U[0.3,0.8] /
  alpha_m U[1,2] / alpha_n U[1.5,2.5]（**--alpha-sampling，5 params → 18ch**）
- X 点中心 (0.7, ±1.1)（MAST divertor 区域）、抖动 R ±0.06 / Z ±0.10 m；
  锚点中平面 R~U[1.2,1.6]（--isoflux-sampling --anchor-midplane）
- 重试：拒绝时整参重采样（max_retries=20，种子 `seed*100_000+i+attempt*1_000_000`
  逐位同 v5）；joblib `batch_size=1`（见 §8.3 教训）

### 7.2 探针校准（80 样本/配置，max_retries=1 = 原始接受率，seed 123）

| 配置 | 接受率 | SN midplane 门 | isoflux res 最大 | X 点偏差最大 | 收敛迭代 |
|---|---|---|---|---|---|
| mast_g2+DN | **75/80 (93.75%)** | — | 2.0e-4 | 0.0009 m | 10.4±（max 33） |
| mast_g2+SN | **80/80 (100%)** | 0 拒绝 | 5.4e-5 | 0.0006 m | 44.0±（max 75 < MAXITS 80） |

对比 data_v5（freegs）：DN 100%、SN 100%；gspack 的 DN 原始接受率略低
（93.75% vs 100%），生成时以重试补齐——**6 个 split 全部 500/500 接受**。

### 7.3 接受检查（= data_v5/v4 全部 7 项 + g2 新增两项）

1. **SN 分离面 X 点判据**：psi ≥ psi_bndry − 1e-6 的 X 点数 == 1（DN == 2）；
   psi_bndry 按 v5 口径重算（DN = 两 X 点 psi 均值；gspack 内置 = 轴下最大单
   X 点，差 ~1e-8，axes 字段用 v5 口径，normalization 用 eq 内置）
2. **磁轴**：O 点位置过滤（R_lo < R < R_anc 且 |Z| < |Z_lo|）取 max psi；
   gspack find_critical 的 O 点**按距网格中心排序**（非 psi）——本生成器过滤
   后重排 opt 使 opt[0]=轴，过滤空则拒绝（v5 DN 直接用 opt[0]）
3. **收敛门（g2 新增）**：picard 最后相对 psi 变化 ≤ 10×rtol（=1e-2）；
   gspack 不 raise、跑满 maxits=80 后返回——门在循环后判（成本见 §8.4）
4. **SN midplane 健康门（g2 新增，v5 无）**：分离面中平面交点
   R_hi ≥ R_anchor − 0.05 且 R_lo < R_axis，且 <2 次穿越的退化回退
   (Rmin,Rmax) 直接拒绝——防 SN 上瓣退化（v6 §8.1 同类病理）；
   探针实测门拒绝 0/80（名义解中平面交点恒在锚点附近，门是防病态的兜底）
5. 其余 = v5/v4 全部：Ip 误差 ≤10%、L 有限、0<Beta0<1、isoflux 残差 ≤0.35、
   X 点偏差 ≤0.10、锚距 ≥0.15、线圈凸包 margin 0.05、core 深度 ≥0.005、
   MAST 无墙 → --require-wall 忽略

### 7.4 生成实测（24 核，batch_size=1）

- DN split：~2–3 min（n_iter 7–31，均值 ~10）；SN split：~4–6 min
  （n_iter 29–72，均值 ~44）；首批 6 split 合计 ~20 min
- 收敛质量：全部样本 psi_relchange_final ≤ 1.0e-3（rtol）
- 约束质量：isoflux 残差 mean 1e-3 / max 3e-3（阈值 0.35）；X 点偏差
  mean 0.2 mm / max 7.7 mm（阈值 0.10 m）——gspack 约束解比 v5 更紧

## 8. 与 data_v5 的差异（freegs → gspack2_TRAE）

| 项 | data_v5 (freegs) | data_gspack2_v1 (gspack v2.0.0) |
|---|---|---|
| 求解包 | freegs（von Hagenow 自由边界） | gspack2_TRAE（同口径重实现） |
| 机器 | freegs MAST()（11 线圈，无墙） | gspack Coil/Solenoid 1:1 复刻（同几何/顺序/控制） |
| 网格 | 65² R[0.1,2] Z[-2,2] | 同（R/Z 数组逐点 array_equal） |
| 求解参数 | rtol 1e-3, maxits 50 | rtol 1e-3, **maxits 80**（SN 需 72） |
| 规模 | 各 3000（train 2000） | **首批各 1500（train 500，可 top-up）** |
| 接受率 | DN 100% / SN 100% | DN 93.75% 原始（重试补齐 100%）/ SN 100% |
| SN midplane 门 | 无 | **有**（§7.3.4，防上瓣退化） |
| 收敛门 | 无显式（Ip/X 点检查兜底） | **有**（§7.3.3，10×rtol） |
| 约束解精度 | isoflux ~2e-4 / xpt_dev ~0.016 m | isoflux ~2e-4 / xpt_dev ~0.0009 m |
| 磁轴判定 | freegs 按 psi 排序 opt[0] | 位置过滤取 max psi + opt 重排 |
| greens | `coil.createPsiGreens` | `eq._coil_psi_unit`（缓存，恒等式 3e-8） |

### 8.1 已知差异（不影响训练正确性）

- **Ip 重构精度**：gspack 的 Jtor 数值积分离散与 freegs 略异 → 用 dpdpsi/
  FdFdpsi 直算的 Ip 重构误差 ~5.8e-3（v5 ~3e-4）。训练 RHS 由数据内
  dpdpsi/FdFdpsi 直接计算，机制相同；exp201/202 的 Ip 约束（twostage）以
  数据重构口径为准（README 记录，不构成偏差）
- psi_bndry 口径 ~1e-8 差异（§7.3.1）

### 8.2 教训：共用 out-dir 导致整块跳过（已修复）

首版 run 脚本把 dn/sn 都写 `data_gspack2_v1/<split>/`，chunk 断点续跑
（`chunk_000 exists → skip`）不区分 config → sn 的 chunk_000 与 dn 撞号
被整块跳过、merge 出 dn-only 的 train.npz（500 行 config=0）→ **改为每配置
独立 out-dir（§2），并在生成器加 config 防呆校验**（跳过前检查 chunk 的
config 字段，不匹配直接 RuntimeError）

### 8.3 教训：joblib 默认分批被 80 迭代慢样本拖垮（已修复）

DN 有 ~3/500 样本到 maxits=80（收敛门 10×rtol 内接受）；joblib 默认
batch=n_jobs 整批等最慢成员 → 500 样本 ~7 min；**改 batch_size=1**（慢样本
只拖自己的 worker）→ 500 样本 ~2–3 min

### 8.4 教训：gspack 不 raise 于不收敛（成本提示）

freegs `solve` 在 maxits 未收敛时 raise（v5 快速拒绝）；gspack `picard.solve`
跑满 maxits 才返回 → 单个不收敛样本成本 80 次迭代（~1–3 s/次），拒绝后重试
链会放大。batch_size=1 已把影响限制在单 worker；无需改 gspack。

## 9. 补数据（top-up）

**机制**：chunk 断点续跑 + merge 幂等重建。

```bash
# 补 dn train 到 1500（同 seed 123）：只生成 chunk_001/002 → merge 重建
bash dn_fno_2608/scripts/run_generate_g2.sh topup dn train 1500 123
```

- 生成侧确定性：样本 i 的种子 = `seed*100_000 + i`（重试 `+attempt*1_000_000`，
  参数重采样 rng = `i_seed + 7_000_003`，与 v5 逐位一致）→ **行 0..N0-1 比特级
  不变**；merge 按 idx 重读全部 chunk 重建 `{split}.npz`，绝不就地追加（中断
  后重跑幂等）
- **诚实记录：训练子集不具嵌套性**。`nested_train_indices`
  （data_dn_fno.py:222-225，`permutation(n_full)[:n_train]`）在池增长后子集会
  变化：池 1000 → **256 DN + 244 SN**；池 3000（= top-up 1500×2）→ 258 DN +
  242 SN，与 500 子集重合仅 222/500。补数据**不改变已生成的样本与 test 集**，
  但 n_train=500 的嵌套子集随池大小变化 → **重训 = 新实验条目**（已训练
  artifact 不受影响，与 exp103 的可复现性也不受影响）
- 数据容量：首批 6×500；top-up 上限任意（chunk 500/块，`--n-samples` 给多大
  生成多大，磁盘 ~152 MB/500 样本）
