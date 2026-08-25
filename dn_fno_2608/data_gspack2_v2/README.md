# data_gspack2_v2 — gspack2_TRAE 版 MASTU_simple 五配置数据集（21ch，129²）

> 生成日期：2026-08-24
> 用途：exp203（做法1 RHS 物理残差）的数据——与 exp105/106 同口径的
> **数据源替换 + 数据质量改进实验**：管线（train/evaluate/data/model/loss）
> 零改动，仅把 freegs_snow 生成的 data_v6_clean 换成 **gspack2_TRAE
> （gspack v2.0.0）** 生成的同构数据（34 键 npz schema、129² 网格、
> MASTU_simple 26 物理线圈 → 14 控制单元 1:1 复刻、21 输入通道），
> 并从**生成侧修复 data_v6 的 SN 数据质量差**（磁轴偏下、上瓣薄、
> 中平面外翻，exp012 §7.1 诊断、sn 病态率 18.4%）
> 代码：`gs_gspack2_dn_fno_2608.generate_dn_g3_dataset`（`--config
> {dn,sn,snow_single,snow_double,limiter}` + `--machine mastu_g3`）；
> 求解包位于 `D:\D_F\Fusion\AI\PINN\gspack2_TRAE`
> 探针报告：`_probe/probe_{cfg}.json`；filter_v6 独立复核：`scores.json`

## 1. 数据集速览

| 项 | 值 |
|---|---|
| 机器 | **MASTU_simple（gspack 版 1:1 复刻）**：26 物理线圈（Solenoid(0.067,-0.6,0.6,142) control=False + Pc + 24 个 U/L Coil），壁 116 顶点；**14 控制单元**（Solenoid, Pc, Px, D1, D2, D3, Dp, D5, D6, D7, P4, P5, P61, P62——data_v6 通道序；P61/P62 下 = −上） |
| 求解 | gspack.Equilibrium（von Hagenow 自由边界，order=2/method=lu）+ ConstrainPaxisIp + constrain(xpoints/isoflux/雪点二阶, gamma) + picard.solve / **雪点松弛 Picard（blend=0.3）** |
| 位形 | **五配置**（DN/SN/雪花单/雪花双/限制器），按配置分开落盘（`dn/`、`sn/`、…） |
| 网格 | 129²，R∈[0.1,2.0] m、Z∈[-2.0,2.0] m（与 data_v6 逐点相同） |
| 规模 | **train 各 500（seed 123）/ val 各 100（seed 456）/ test 各 200（seed 789）**——val/test 对齐 data_v6_clean → exp203 与 exp105 逐桶 1:1 对比 |
| 输入通道 | **21ch**（R,Z + 5 params + 14 单元电流，**无 config 通道**）——与 exp105/106 完全同构 |
| 目标 | `psi_total`（129²，z-score） |
| 新增字段 | `gs_true`/`midplane_ratio`/`zaxis_ratio`（质量指标无条件写入，见 §3.1） |

## 2. 目录与文件

```
dn_fno_2608/data_gspack2_v2/
├── dn/  train.npz / val.npz / test.npz
├── sn/  ...        （五配置同构；+ 分块目录 chunks）
├── snow_single/ ...
├── snow_double/ ...
├── limiter/   ...
├── _probe/probe_{cfg}.json   # 探针（80 样本/配置，原始接受率 + 门分布）
├── scores.json               # filter_v6 独立判据复核（15 split，移除 0 条）
└── README.md
```

每配置独立 out-dir（同 data_v5/data_gspack2_v1 布局）——chunk 断点续跑按
(config, split, chunk_idx) 定位。npz 不入 git（`.gitignore`），
`run_generate_g3.sh` 一键再生成。

## 3. 字段说明（npz keys，34 个 = v6 全键 + 3 新质量指标）

data_v6 全部字段（含 `config`(N,1) 0-4、x_coords、xpts_actual、
o_point、anchor、snowflake_res 等）+ 新质量指标：

### 3.1 新质量指标（filter_v6.py 同款公式，生成侧无条件写入）

| 键 | 公式 | 阈值（生成侧硬门） |
|---|---|---|
| `gs_true` | gs_residual_ratio（mask=ψ≥ψ_bndry，evaluate_dn_fno 同款） | ≤ 15（全配置；limiter 报告） |
| `midplane_ratio` | (ψ(R_axis,Z≈0) − ψ_bndry)/core | ≥ 0.05（**sn 硬门**；其余报告） |
| `zaxis_ratio` | \|Z_axis\|/\|Z_lo\| | ≤ 0.5（**sn 硬门**） |

### 3.2 输入通道明细（21ch，同 exp105）

| # | 通道 | 含义/单位 | 来源 |
|---|---|---|---|
| 1 | R | 网格 R 坐标（[-1,1]） | 固定 129² 网格（= v6） |
| 2 | Z | 网格 Z 坐标（[-1,1]） | 固定 129² 网格（= v6） |
| 3-7 | Ip / paxis / fvac / alpha_m / alpha_n | 剖面参数（Pa/A/Wb·m⁻¹） | params |
| 8-21 | I_Solenoid, I_Pc, I_Px, I_D1, I_D2, I_D3, I_Dp, I_D5, I_D6, I_D7, I_P4, I_P5, I_P61, I_P62 | 14 单元电流 (A) | coil_currents |

线圈名/顺序 = data_v6 通道序（MASTU_simple 14 控制单元）。greens 单位响应
= `eq._coil_psi_unit`（构造时置 current=1.0 的缓存；每单元 = Σ sign·子线圈
单位响应）；**恒等式 Σ_k I_k·G_k ≡ psi_coils 实测 max diff ~3e-8 Wb**
（v6 同为 1e-8 量级；S2 复核 50 样本/配置 ≤1e-6）。

## 4. 如何调用

```python
import numpy as np
d = np.load("dn_fno_2608/data_gspack2_v2/dn/train.npz")
cfg = d["config"]           # (500,): 0-4
coils = d["coil_currents"]  # (500, 14): 顺序见 §3.2
```

```bash
PY="C:/Users/HP/.conda/envs/torch5060/python.exe"

# exp203 训练（rhs，五配置混合，逗号拼接；stats 全池 2500 计算）
"$PY" -u -m gs_pino_fno_phys.train_pino --mode rhs \
  --train-data dn_fno_2608/data_gspack2_v2/{dn,sn,snow_single,snow_double,limiter}/train.npz \
  --val-data dn_fno_2608/data_gspack2_v2/{dn,sn,snow_single,snow_double,limiter}/val.npz \
  --n-train 500 --seed 1 --epochs 800 --phys-weight 0.1 --out-dir <out>
# 评估（分桶：--test-data 单传各配置文件，--machine mastu_simple）
"$PY" -u -m gs_pino_fno_phys.evaluate_pino \
  --test-data dn_fno_2608/data_gspack2_v2/sn/test.npz \
  --checkpoint <out>/best.pt --out-dir <out>/eval_sn --machine mastu_simple
```

## 5. 相关脚本（dn_fno_2608/scripts/）

| 脚本 | 用途 |
|---|---|
| `run_generate_g3.sh` | 探针 + 全量生成一键（`probe`/`gen`/`topup`/`all`；首批 15 split **~5 h @ 16 核**，见 §6.3） |
| `probe_g3.py` | 探针：原始接受率 + 新门分布（`--config {dn,sn,snow_single,snow_double,limiter} --n 80`） |
| `validate_g3.py` | 数据校验（S2）：网格/通道/greens/Ip/filter_v6 复核/SN 质量对比 |
| `run_exp203_pino.sh` | exp203 训练 + 六桶评估 |

## 6. 生成设置与统计

### 6.1 采样口径（= data_v6 MASTU_simple）

- 参数：paxis U[4e4,8e4] Pa / Ip U[7e5,1.5e6] A / fvac U[0.4,0.9] /
  alpha_m U[1,2] / alpha_n U[1.5,2.5]（--alpha-sampling，5 params → 21ch）
- 五配置 specs = v6 CONFIG_SPECS 逐条照抄（dn xpt(0.80,±1.20)；sn
  (0.65,1.20)；snow_single 雪点(0.509,-1.291) jitter 0.02/0.04；
  snow_double (0.65,±1.20)；limiter 轴 R~U[0.65,0.95]）
- 锚点中平面 R~U[1.2,1.45]（--isoflux-sampling --anchor-midplane）
- 重试：拒绝时整参重采样（max_retries=20，种子 `seed*100_000+i+attempt*1_000_000`）；
  joblib `batch_size=1`

### 6.2 探针校准（80 样本/配置，max_retries=1 = 原始接受率，seed 123）

| 配置 | raw 接受率 | 门后 | solve 均/最 (s) | n_iter 均/最 | 关键质量 |
|---|---|---|---|---|---|
| dn | 72.5% | 71.25% | 27.3 / 60.6 | 9.2 / 14 | gs_true 均 8.7；门拒 1（gs_true） |
| sn | **28.75%** | **25%** | 64.1 / 100.4 | 67.7 / 80 | **病态 0**（midplane 全≥0.23、zaxis 全≤0.36）；门拒 3（gs_true） |
| snow_single | 52.5% | 51.25% | 70.9 / 169.2 | 96.5 / 200 | 雪点 X 点全命中；门拒 1（gs_true） |
| snow_double | 100% | 98.75% | 36.5 / 64.5 | 11.5 / 13 | 双 X 点全命中；门拒 1（gs_true） |
| limiter | 100% | 100% | 32.0 / 67.0 | 7.0 / 9 | gs_true 均 0.45（全 ≤5.2） |

**与 data_v6 的 SN 质量对照（生成侧修复的直接证据）**：
- 病态率（midplane_ratio<0）：v6 raw **18.4%** → g3 **0.0%**（探针 0/80）
- \|Z_axis\|/\|Z_lo\|：v6 病态特征"磁轴偏下" → g3 探针 max 0.36（门 0.5 全过）
- gs_true>15：g3 探针 5/368 样本（1.4%），生成侧硬门拒绝
- SN 门后接受率 25% < v6 raw 55%——但 v6 的 55% 含 18.4% 病态（有效
  ~45%）；g3 的 25% **全部高质量**。按计划"质量优先、接受率允许低于 v6"，
  生成侧用重试 20 补齐

**首批 15 split 实际生成耗时（@ 16 核，n_jobs=16，总 ~5 h）**：

| 配置 | train 500 | val 100 | test 200 | 小计 |
|---|---|---|---|---|
| dn | ~10 min（accept 72.5%） | 同链 | 同链 | ~10 min |
| sn | **80 min**（accept 28.75%，重试链） | 18 min | 31 min | ~2 h |
| snow_single | **91 min**（solve 71s 均值） | 18 min | 34 min | ~2.4 h |
| snow_double | 8 min | 2 min | 3 min | 13 min |
| limiter | 6 min | 1 min | 2 min | 9 min |

（生成按 dn → sn → snow_single → snow_double → limiter 顺序；耗时大头是
sn/snow_single 的接受率与单解耗时，非卡死——S1 修复的 cupy 后端问题后
全程无异常）

### 6.3 求解要点（与 v6/freegs_snow 的关键差异，已实证）

1. **雪点必须零电流初始 + 松弛 Picard**（详见 §7.2）——freegs_snow 的
   MASTU_INIT_CURRENTS 种子在 gspack 下进入上瓣假解并周期-3 极限环振荡；
   零电流 + blend=0.3（混合在 constrain 之后）稳定收敛到与 freegs_snow
   seed 分支**同一解**（X 点距雪点目标 6-13 mm）
2. **收敛门**：psi_relchange_final ≤ 10×rtol（=1e-2，g2 先例）；snow 用
   200 迭代、其余 80
3. **SN 磁轴过滤**：gspack find_critical 的 O 点按距网格中心排序（非 psi）
   → 按 v6 位置过滤（lo[0]<R<anchor[0] 且 |Z|<|Z_lo|）取 max psi，opt 重排
4. **limiter 两步法**：`Equilibrium(check_limited=True, limiter_mode="flood")`
   → step1 钉轴 xpoints → step2 `constrain=None`（rtol 5e-3 maxits 100）+
   触壁自洽检查

## 7. 与 data_v6 的差异（freegs_snow → gspack2_TRAE）

| 项 | data_v6 (freegs_snow) | data_gspack2_v2 (gspack v2.0.0) |
|---|---|---|
| 求解包 | freegs_snow fork（雪点二阶约束 fork） | gspack2_TRAE（同口径重实现） |
| 机器 | freegs MASTU_simple()（26 线圈 + 壁） | gspack MASTU_simple() 1:1 复刻（同几何/顺序） |
| 控制 | 26 物理线圈直接控制 | **14 控制单元（UnitMachine override）**，P61/P62 下=−上 |
| 网格 | 129² R[0.1,2] Z[-2,2] | 同（逐点 array_equal，S2 验证） |
| SN 质量门 | 生成后清洗（filter_v6 剔除病态） | **生成侧硬门**（midplane/zaxis/gs_true 三件套） |
| SN 病态率 | 18.4%（v6 raw，清洗后 test 27/200） | **0.0%**（探针 + S2 验证） |
| 雪点求解 | 种子电流 + 纯 Picard（blend=0） | **零电流 + 松弛 Picard（blend=0.3）** |
| 接受率 | SN raw 55% | SN raw 28.75%（全高质量，重试补齐） |
| greens | `coil.createPsiGreens` | `eq._coil_psi_unit`（Σ sign·子线圈，恒等式 3e-8） |

### 7.1 已知差异（不影响训练正确性）

- **Ip 重构精度**：gspack 的 Jtor 数值积分离散与 freegs 略异 → 用
  dpdpsi/FdFdpsi 直算的 Ip 重构误差 ~5e-3（v6 ~3e-4，g2 同 5.8e-3 已知）。
  训练 RHS 由数据内 dpdpsi/FdFdpsi 直接计算，机制相同；exp203 的 Ip 相关
  指标（如 twostage）以数据重构口径为准（README 记录，不构成偏差）
- psi_bndry 口径 ~1e-8 差异（同 g2 §8.1）

### 7.2 雪点求解问题与修复（2026-08-24 实证）

gspack 纯 Picard 在雪点问题上进入周期-3 极限环发散（rel 0.85→1.00→1.41
循环）；freegs_snow 同参数收敛（n=63-71）——差异在 GS 求解器/等离子场
数值细节（迭代动力学不同）。诊断链：

1. 种子电流版（v6 同款）：上瓣假解 + 振荡（轴 (1.013,1.173) 在上瓣，n=200
   rel=0.15 极限环）；noseed 版：收敛但上瓣空（疑似病态）
2. **对照实验**：freegs_snow 对同一参数/雪点/种子收敛到轴 (1.074,-0.503)、
   psi_axis=0.175、X 点 (0.501,-1.281)——与 gspack noseed 解**形态数值一致**
   （轴 (1.067,-0.435)、psi_axis=0.1702、X 点 (0.51,-1.28)）→ 该形态是
   MASTU 下瓣雪点的**正确解**（下偏滤器、上瓣真空）
3. **松弛修复**：混合放在 `constrain(eq)` 之后（picard.solve 里 constrain
   后 `psi = eq.psi()` 会把混合结果覆盖，松弛失效）；blend=0.3 实测
   snow_single/snow_double 8/8 收敛正确下瓣（blend=0.5 收敛但落上瓣假解、
   0.7 慢）

### 7.3 教训：接受率与质量

SN raw 28.75%（vs v6 55%）——初看倒退，但门分布揭示真相：v6 的 55% 含
18.4% 病态（生成后需 filter_v6 清洗 27/200 test）；g3 的 28.75% 无病态、
新门零拒绝（除 gs_true 3/80）。**接受率不是质量的替代品**——生成侧硬门
把 SN 质量门槛前移，下游无需再清洗（scores.json 15 split 移除 0 条）。

## 8. 数据校验（S2，2026-08-25 实测全通过）

`validate_g3.py` **ALL CHECKS PASS**：
- 网格 vs data_v6 逐点 array_equal（129² R[0.1,2] Z[-2,2]）
- 21ch 加载（DNPinoDataset x.shape[0]==21，params 通道序核对精确一致）
- greens 恒等式 50 样本/配置 max diff **1.2e-7**（≤1e-6）
- Ip 重构 mean rel：dn 2.2e-4 / sn 1.7e-3 / snow_single 4.0e-3 /
  snow_double 5.5e-4 / limiter 1.7e-3（均 <1e-2；gspack 固有，报告）
- filter_v6 独立复核（v6 阈值重算 15 split）：判据命中 **0 条**
  （scores.json 入库 `data_gspack2_v2/scores.json`）
- SN 质量 vs v6 raw（test 200 口径）：

| 指标 | g3 | v6 raw |
|---|---|---|
| midplane_ratio < 0（病态） | **0/200**（min 0.155） | 1/200（min -0.856） |
| gs_true > 15 | **0/200**（mean 9.57） | **26/200**（mean 11.86） |
| \|Z_axis\|/\|Z_lo\| > 0.5 | **0.000**（max 0.414） | 0.005（max 0.691） |

v6 的 27 条（test）正是 filter_v6 清洗/补足的判据命中——g3 从生成侧全部
消除，下游无需再清洗。

## 9. 补数据（top-up）

## 9. 补数据（top-up）

**机制**：chunk 断点续跑 + merge 幂等重建（同 data_gspack2_v1 §9）。

```bash
# 补 dn train 到 1500（同 seed 123）：只生成 chunk_001/002 → merge 重建
bash dn_fno_2608/scripts/run_generate_g3.sh topup dn train 1500 123
```

- 生成侧确定性：样本 i 的种子 = `seed*100_000 + i`（重试
  `+attempt*1_000_000`，参数重采样 rng = `i_seed + 7_000_003`，与 v6 逐位
  一致）→ 行 0..N0-1 比特级不变；merge 按 idx 重读全部 chunk 重建
  `{split}.npz`，绝不就地追加（中断后重跑幂等）
- **诚实记录：训练子集不具嵌套性**（同 g2 §9）：`nested_train_indices`
  在池增长后子集会变化 → 重训 = 新实验条目（已训练 artifact 不受影响）
- 数据容量：首批 15×（500/100/200）；top-up 上限任意（chunk 500/块，
  `--n-samples` 给多大生成多大）
