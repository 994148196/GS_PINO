# data_v6 数据集说明（MASTU_simple 五配置：DN/SN/snow_single/snow_double/limiter）

> 生成日期：2026-08-18
> 代码：`generate_dn_dataset.py`（`--config {dn,sn,snow_single,snow_double,limiter}`
> + `--machine mastu_simple`），freegs_snow fork 后端
> 方向（用户拍板）：**新建数据全用 MASTU_simple**（freegs 自带真实装置、真实真空室壁），
> 5 配置各生成 **train 500 / val 100 / test 200**，**保留扩展可能性**（同 seed 重跑更大
> n，chunk resume 追加，见 §7）
> 探针报告：[PLAN_v5_mixed_configs.md](../PLAN_v5_mixed_configs.md) 同结构；
> 探针 JSON：`dn_fno_2608/data_v6/_probe/probe_v6_*.json`
> 用途：exp012（混合训练，coil 电流输入端到端，21ch 输入）

## 1. 与 data_v5 的差异

| 项 | data_v5 (MAST) | **data_v6 (MASTU_simple)** |
|---|---|---|
| 机器 | MAST（11 控制线圈，**无墙**） | **MASTU_simple**（14 控制线圈，**真实真空室壁** rwall，R 0.244–2.0） |
| 网格 | 65² | **129²**（雪点二阶判据在 65² 精度不足，用户批准 65→129 切换） |
| 位形 | DN + SN | **DN + SN + snow_single + snow_double + limiter**（5 配置） |
| 输入通道 | 18ch（R,Z + 5 params + 11 coils） | **21ch**（R,Z + 5 params + 14 coils；separability 结论无 config 通道，§6） |
| 新字段 | `config`（0/1） | `config`（**0–4 编码**）、**标注字段** `wall_contact`/`wall_contact_excess`/`inwall_sep_frac`、limiter 专有字段（§5） |
| X 点中心 | DN (0.7,±1.1) / SN (0.7,1.1) | DN **(0.80,±1.20)**（物理修正，见 §3）/ SN (0.65,1.20) |
| 锚点 | R~U[1.2,1.6] | R~U**[1.20,1.45]**（例 18/19 参考 R=1.34） |
| 参数范围 | paxis (1e3,5e3) / Ip (3e5,8e5) / fvac (0.3,0.8) | **paxis (4e4,8e4) / Ip (7e5,1.5e6) / fvac (0.4,0.9)**（例 18/19 强等离子体；Ip 下限 7e5 防高 beta 自由边界振荡） |
| 求解 | — | snowflake：maxits 200/**rtol 1e-3**（rtol 5e-3 会停在错分叉的上瓣假解）+ 初始电流种子（选下瓣分支）；limiter：两步法（例 23） |

## 2. 机器：MASTU_simple 14 控制线圈（freegs `machine.py` 定义顺序）

| # | 线圈 | 位置 (R, Z) | 说明 |
|---|---|---|---|
| 0 | Solenoid | (0.195, ±1.581) | 中心螺线管，324 匝 |
| 1 | Pc | (0.067, ±0.6) | 中心柱 |
| 2 | Px | (0.2405, ±1.2285) | |
| 3–5 | D1/D2/D3 | 下偏滤器 | |
| 6 | Dp | | |
| 7–9 | D5/D6/D7 | 外壁 | D5 外壁 R≈1.9 |
| 10 | P4 | | |
| 11 | P5 | R≈1.65 | 外壁 |
| 12 | P61 | 反接 | 垂直场（与 P62 成对反接） |
| 13 | P62 | 反接 | 垂直场 |

上下成对线圈在 freegs 中以单个 Circuit 表示（14 控制元素 → 14 通道）。全部
`coil.control = True`（例 18/19 做法）。**壁内结构判据**（§3）的壁内区域以
`wall.R.min() = 0.244` 为界（Solenoid 所在 R<0.244 柱区）。

## 3. 物理修正与接受检查（data_v6 新增）

**X 点中心修正（用户确认"没问题"）**：dn X 点中心 R 由 0.65 修正为 **0.80**
（0.65 时 20 样本中 12 例出现 Solenoid 柱区与分离线交叉的病态结构；0.80 保持
18/20 干净）。sn 保持 0.65。

**壁内结构判据 `INWALL_SEP_FRAC_TOL = 0.02`**（纯 numpy 穿越边计数）：
psi=psi_bndry 等高线在壁内（R<0.244）穿越网格边的比例，阈值标定自干净样本
（max 1.84%）与 0.65 病态（~5%）。全部 5 配置的生成样本均 ≤2%，snowflake
恒定小结构（0.82–1.31%）不被误杀。limiter 的 psi_bndry 是限制面（非分离面），
该指标无物理意义——数据保持同一算法计算（max 5.78%），分析时跳过。

**触壁策略（用户拍板）**：深触壁（wall_contact_excess > 20% core depth）必须排除；
浅触壁保留 + 每样本标注 `wall_contact`/`wall_contact_excess` 供筛选。

data_v5 的 isoflux/X 点/锚点/线圈裕度/core 深度检查全部保留（`--max-isoflux-residual
0.35 --max-xpt-deviation 0.10 --min-anchor-xpt-dist 0.15 --coil-margin 0.05
--min-core-depth 0.005`）。

## 4. 探针校准（probe_v6.py，seed 123，80 样本/配置，max_retries=1 = 原始接受率）

| 配置 | 接受率 | 触壁（n_touch / max excess） | 壁内结构 max | 单解耗时 mean |
|---|---|---|---|---|
| dn | **61/80 (76.3%)** | 27/80 / 7.4% | 1.68% | 13.4 s |
| sn | **44/80 (55.0%)** | 6/80 / 19.1% | 1.84% | 103 s |
| snow_single | **52/80 (65.0%)** | 10/80 / 15.3% | 0.82% | 71.8 s |
| snow_double | **80/80 (100%)** | 1/80 / 7.6% | 1.31% | 14.4 s |
| limiter | **80/80 (100%)** | 80/80 / 0.0%（触壁是定义） | 0.0% | 11.6 s |

- dn/sn 失败多为 No O-points / Picard 不收敛（物理失败），snow 失败多为一/二阶
  判据超阈；limiter 100%（两步法 + 磁轴 R~U[0.65,0.95]）；
- 触壁全部浅触（max 19.1% < 20% 阈值）→ 保留；
- **全量 500/配置 train 统计**：触壁 dn 53%（max 10.1%）、sn 9%（19.6%）、
  snow_single 14%（18.4%）、snow_double 1%（14.9%）、limiter 100%（0.1%）；
  max inwall dn/sn 1.97%、snow_single 0.82%、snow_double 1.47%（limiter 见 §3）。

**电流可分性 → exp012 输入 21ch（无 config 通道）**：
14 通道 pooled 分离度**全部 >2.0 std**，5 配置两两配对最优通道阈值判别**全部
≥90.75%** → `decision_no_config_channel = true`（与 exp011 同逻辑）。
config 字段仍保留在数据中（标注/分桶用），不作为模型输入。

## 5. 字段说明（train.npz / val.npz / test.npz）

- 物理场：`psi_total`（(N,129,129)）、`psi_plasma`/`psi_plasma_norm`/`psi_coils`、
  `mask`、`dpdpsi`、`FdFdpsi`、`greens`（14,129,129）
- 标量：`params`（[Ip, paxis, fvac, alpha_m, alpha_n]）、`coil_currents`（14）、
  `config`（0=dn 1=sn 2=snow_single 3=snow_double 4=limiter）、`axes`
  （[R_axis, Z_axis, psi_bndry, psi_axis]）、`L`、`Beta0`、`solve_time`、`n_iter`
- 约束诊断：`x_coords`、`isoflux_res`、`xpt_constraint_res`、`psi_at_constraints`
- 几何真值：`xpts_actual`（DN 2 行 / SN·snow_single 1 行 / limiter **0 行**）、
  `o_point`、`anchor`（**limiter 无**——无 isoflux 约束，混合拼接时 NaN-pad）
- **标注字段（数据_v6 新增）**：`wall_contact`（0/1）、`wall_contact_excess`
  （超 psi_bndry 深度 / core 深度）、`inwall_sep_frac`（壁内穿越边比例）
- limiter 专有：`is_limited`（恒 1）、`Rlim`/`Zlim`（触壁点，落在真实壁点上）、
  `psi_limit`（限制面 psi = psi_bndry，自洽检查 |psi(Rlim,Zlim)-psi_bndry| ≤ 0.01）
- `R`/`Z`（129,129 网格），limiter 的 `xpts_actual` 为 (N,0,3) 空数组

## 6. exp012 输入通道（21ch 明细）

| 通道 | 内容 |
|---|---|
| 1–2 | R, Z 网格通道（[-1,1] 归一化） |
| 3–7 | Ip, paxis, fvac, alpha_m, alpha_n（z-score） |
| 8–21 | 14 线圈电流：I_Solenoid, I_Pc, I_Px, I_D1, I_D2, I_D3, I_Dp, I_D5, I_D6, I_D7, I_P4, I_P5, I_P61, I_P62（z-score） |

不用 config 通道（§4 separability）。数据集类 `DNFnoDatasetCoils(use_config=False)`
（训练 `--no-config-channel`）；ckpt 记录 `no_config_channel` 标志，评估自动匹配。

## 7. 扩展方式（用户要求"以后直接加在后面"）

同 seed 重跑更大 `--n-samples`：已有 `chunk_*.npz` 自动跳过（resume），新样本追加，
`--merge` 重新合成 split.npz。同 seed + 确定性采样 → 前段样本与现数据**完全一致
不重复**。例（追加到 2000）：

```bash
export PYTHONPATH="D:/D_F/Fusion/AI/PINN/freegs_snow;D:/D_F/Fusion/AI/PINN/GitHub-tests/GS_PINO/src"
python -m gs_pino_dn_fno_2608.generate_dn_dataset --split train --n-samples 2000 \
  --seed 456 --out-dir dn_fno_2608/data_v6/dn --config dn \
  --machine mastu_simple --alpha-sampling --xpt-jitter 0.06 --xpt-jitter-z 0.10 \
  --isoflux-sampling --anchor-midplane --max-isoflux-residual 0.35 \
  --max-xpt-deviation 0.10 --min-anchor-xpt-dist 0.15 --coil-margin 0.05 \
  --min-core-depth 0.005 --save-constraint-diag --max-retries 20 \
  --chunk-size 500 --n-jobs 24
python -m gs_pino_dn_fno_2608.generate_dn_dataset --split train \
  --out-dir dn_fno_2608/data_v6/dn --merge
```

## 8. 复现命令

全量生成（5 配置 × 3 split）：`bash dn_fno_2608/scripts/run_generate_v6.sh`
（val seed 789 / test seed 1011 / train seed 456；标注字段补齐工具
`_backfill_v6_labels.py` 在 STACKED_KEYS 升级前的 chunk 上重算
wall_contact/wall_contact_excess/inwall_sep_frac，逻辑与生成判据逐位一致）。

## 9. 偏差记录

0. **SN 磁轴系统性偏下 / 上瓣薄（用户发现，2026-08-19）**：sn 200 test 样本
   Z_axis mean **-0.339 m**（仅 2/200 在 |Z|<0.15），76/200 中平面 psi 距分离面
   <0.5 core 深度；极端样本（如 test idx158）中平面 psi < psi_bndry（等离子体
   不跨中平面，分离面外翻到壁区）。SN 磁轴检查（R_lo<R_axis<R_anc 且
   |Z_axis|<|Z_lo|）放行了这些形态——上瓣薄不算"不合格"。**建议加"上瓣健康度"
   判据**（中平面 psi > psi_bndry + k·core）或收紧 Z_axis 范围；exp012 SN 桶
   8.99% 部分归因于此（详见 exp012 README 偏差记录 5）。
1. **limiter 无 anchor / xpts_actual 0 行**：设计如此（无 isoflux 约束、无分离面
   X 点）；混合数据集的评估几何指标对这些行 NaN（evaluate 的
   match_xpoints_and_axis 全 NaN 防护）。
2. **limiter inwall_sep_frac 非零（max 5.78%）**：psi_bndry 是限制面不是分离面，
   指标无语义，仅记录。
3. **snowflake 需初始电流种子**（MASTU_INIT_CURRENTS）：Tikhonov 系统多固定点，
   种子选择下瓣分支（无种子收敛到上瓣假解，探针验证）。
4. **snowflake rtol=1e-3 而非 5e-3**：5e-3 会在错分叉的上瓣假解停下（探针验证）。
5. **data_v6 与 data_v5 跨机器对比仅参考**（机器/网格/线圈数不同）。
