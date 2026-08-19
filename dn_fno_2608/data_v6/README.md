# data_v6 数据集（MASTU_simple 五配置，21ch 输入）

> 生成日期：2026-08-18 ｜ 代码：`gs_pino_dn_fno_2608.generate_dn_dataset`
> 上游实验：exp012（五配置混合训练，端到端 psi 生成）
> 探针报告：`dn_fno_2608/data_v6/_probe/probe_v6_*.json`、`PLAN_v5_mixed_configs.md`（同结构）
> ⚠️ **使用 sn 数据前必读 §8.1**（SN 位形存在已记录的数据质量问题）

## 1. 数据集速览

| 项 | 值 |
|---|---|
| 机器 | **MASTU_simple**（freegs 自带，14 控制线圈 + 真实真空室壁 rwall，R 0.244–2.0） |
| 位形 | 5 种：dn / sn / snow_single / snow_double / limiter（config 编码 0/1/2/3/4） |
| 网格 | 129²，R∈[0.1,2.0] m、Z∈[−2,2] m |
| 规模 | 每配置 train **500** / val **100** / test **200**（共 2500/500/1000） |
| 输入通道 | **21ch**：R,Z + 5 params + 14 线圈电流（§4） |
| 目标 | `psi_total`（129²，均值/标准差归一化） |
| 求解器 | freegs_snow fork（`export PYTHONPATH="D:/D_F/Fusion/AI/PINN/freegs_snow;..."`） |

## 2. 目录与文件

```
dn_fno_2608/data_v6/
├── {dn,sn,snow_single,snow_double,limiter}/
│   ├── train.npz / val.npz / test.npz   # 正式数据（float32）
│   └── train/ val/ test/                # 生成中间分块 chunk_*.npz（合并后冗余可删）
└── _probe/                              # 探针结果 JSON（接受率/separability）+ figs/
```

## 3. 字段说明（npz keys）

所有 split/配置文件字段同构；**limiter 例外**（`xpts_actual` 为 (N,0,3) 空数组、
**无 anchor 字段**——混合拼接时由数据集类 NaN-pad）。

| 字段 | 形状 | 含义 | 是否模型输入 |
|---|---|---|---|
| `psi_total` | (N,129,129) | 总极向磁通（**训练目标**） | — |
| `psi_plasma` / `psi_plasma_norm` / `psi_coils` | (N,129,129) | 等离子体/归一化/线圈磁通分量 | 否（PINO 预留） |
| `R`, `Z` | (129,129) | 物理坐标网格（R 沿行 axis 0、Z 沿列 axis 1） | 通道 1–2 |
| `mask` | (N,129,129) | freegs critical.core_mask | 否 |
| `params` | (N,5) | [Ip, paxis, fvac, alpha_m, alpha_n] | 通道 3–7 |
| `coil_currents` | (N,14) | 14 线圈电流（顺序见 §4） | 通道 8–21 |
| `config` | (N,) | 位形编码 0=dn 1=sn 2=snow_single 3=snow_double 4=limiter | **否**（仅标注/分桶） |
| `axes` | (N,4) | [R_axis, Z_axis, psi_bndry, psi_axis] | 否（评估用） |
| `L`, `Beta0` | (N,) | 等离子体电感/比压 | 否 |
| `solve_time`, `n_iter` | (N,) | 求解耗时/迭代数 | 否 |
| `x_coords` | (N,4) | X 点目标坐标（limiter 为占位） | 否 |
| `xpts_actual` | (N,2,3) 或 (N,1,3) 或 **(N,0,3)** | find_critical 实际 X 点 [R,Z,psi]（DN 2 行 / SN·snow_single 1 行 / **limiter 0 行**） | 否（评估几何真值） |
| `o_point` | (N,3) | 实际 O 点 [R,Z,psi] | 否 |
| `anchor` | (N,2) | isoflux 锚点（**limiter 无**） | 否 |
| `isoflux_res`, `xpt_constraint_res`, `psi_at_constraints` | — | 约束残差诊断 | 否 |
| `dpdpsi`, `FdFdpsi` | (N,129,129) | GS 残差 RHS 分量 | 否（PINO 预留） |
| `greens` | (14,129,129) | 线圈 Green 函数（全样本共用一份） | 否（PINO 预留） |
| `wall_contact` | (N,) | 是否触壁（0/1） | 否（标注） |
| `wall_contact_excess` | (N,) | 超 psi_bndry 深度 / core 深度（>0.2 = 深触壁，生成时已排除） | 否（标注） |
| `inwall_sep_frac` | (N,) | 分离面在壁内（R<0.244）穿越网格边比例（limiter 无语义） | 否（标注） |
| `is_limited` | (N,) | limiter 专有：是否限制位形（恒 1） | 否 |
| `Rlim`, `Zlim` | (N,) | limiter 专有：触壁点（落在真实壁点上） | 否 |
| `psi_limit` | (N,) | limiter 专有：限制面 psi（= psi_bndry，自洽 |Δpsi|≤0.01） | 否 |

## 4. 输入通道明细（21 通道）

通道顺序 = 代码实际拼接顺序（`data_dn_fno_coils.py` `__getitem__`）：2 几何 +
5 标量 + 14 线圈电流。R/Z 线性映射 [-1,1]；19 个标量用**训练 pool 的 mean/std**
z-score；目标 psi 同样 z-score（训练损失与 rel L2 均在归一化域计算）。

| # | 通道 | 内容 | 单位 | 来源 |
|---|---|---|---|---|
| 1 | R | 网格 R 坐标 | m | 固定 129² 网格 |
| 2 | Z | 网格 Z 坐标 | m | 固定 129² 网格 |
| 3 | Ip | 等离子体电流 | A | params[0] |
| 4 | paxis | 磁轴压强 | Pa | params[1] |
| 5 | fvac | 真空通量函数 f | Wb/m | params[2] |
| 6 | alpha_m | 剖面内指数 | — | params[3] |
| 7 | alpha_n | 剖面外指数 | — | params[4] |
| 8 | I_Solenoid | 中心螺线管电流（R 0.195，±1.581，324 匝） | A | coil_currents[0] |
| 9 | I_Pc | 中心柱 | A | coil_currents[1] |
| 10 | I_Px | | A | coil_currents[2] |
| 11 | I_D1 | 下偏滤器 | A | coil_currents[3] |
| 12 | I_D2 | 下偏滤器 | A | coil_currents[4] |
| 13 | I_D3 | 下偏滤器 | A | coil_currents[5] |
| 14 | I_Dp | | A | coil_currents[6] |
| 15 | I_D5 | 外壁（R≈1.9） | A | coil_currents[7] |
| 16 | I_D6 | 外壁 | A | coil_currents[8] |
| 17 | I_D7 | 外壁 | A | coil_currents[9] |
| 18 | I_P4 | | A | coil_currents[10] |
| 19 | I_P5 | 外壁（R≈1.65） | A | coil_currents[11] |
| 20 | I_P61 | 垂直场（与 P62 成对反接） | A | coil_currents[12] |
| 21 | I_P62 | 垂直场（与 P61 成对反接） | A | coil_currents[13] |

线圈名/顺序 = freegs `machine.py` MASTU_simple() 定义顺序（0-1 号是 Solenoid/Circuit
等成对封装，生成脚本按 `tokamak.coils` 顺序收集）。**无 config 通道**：14 通道电流
pooled 分离度全部 >2.0 std、5 配置两两配对阈值判别 ≥90.75% → 电流本身携带位形信息
（探针 `_probe/probe_v6_separability.json`）。config 字段仅作标注/分桶。

## 5. 如何调用

### 5.1 直接加载（numpy）

```python
import numpy as np
d = np.load("dn_fno_2608/data_v6/dn/train.npz")        # 或 sn/snow_single/snow_double/limiter
psi = d["psi_total"]    # (500, 129, 129)
params = d["params"]    # (500, 5): [Ip, paxis, fvac, alpha_m, alpha_n]
coils = d["coil_currents"]  # (500, 14): 顺序见 §4
cfg = d["config"]       # (500,): 0/1/2/3/4
```

### 5.2 训练（exp012 实际命令）

```bash
PY="C:/Users/HP/.conda/envs/torch5060/python.exe"
"$PY" -u -m gs_pino_dn_fno_2608.train_dn_fno \
  --train-data dn_fno_2608/data_v6/dn/train.npz,dn_fno_2608/data_v6/sn/train.npz,dn_fno_2608/data_v6/snow_single/train.npz,dn_fno_2608/data_v6/snow_double/train.npz,dn_fno_2608/data_v6/limiter/train.npz \
  --val-data dn_fno_2608/data_v6/dn/val.npz,dn_fno_2608/data_v6/sn/val.npz,dn_fno_2608/data_v6/snow_single/val.npz,dn_fno_2608/data_v6/snow_double/val.npz,dn_fno_2608/data_v6/limiter/val.npz \
  --input-mode coils --no-config-channel --n-train 500 --seed 1 \
  --epochs 800 --batch-size 16 --lr 1e-3 --weight-decay 1e-4 \
  --lr-patience 20 --lr-factor 0.5 --min-lr 1e-5 --patience 75 \
  --out-dir dn_fno_2608/experiments/expXXX_xxx
```

- 多文件逗号拼接 = 五配置混合 pool（2500 样本），`--n-train 500` 从 pool 按
  `--perm-seed 12345` nested 抽取；单配置训练只传对应文件即可
- ckpt 顶层存 `no_config_channel` 标志 → evaluate/visualize 自动匹配输入通道

### 5.3 评估 / 可视化

```bash
"$PY" -u -m gs_pino_dn_fno_2608.evaluate_dn_fno \
  --test-data dn_fno_2608/data_v6/sn/test.npz \
  --checkpoint dn_fno_2608/experiments/exp012_coil_input_v6/best.pt \
  --out-dir dn_fno_2608/experiments/exp012_coil_input_v6/eval_sn
"$PY" -u -m gs_pino_dn_fno_2608.visualize_dn_fno \
  --checkpoint dn_fno_2608/experiments/exp012_coil_input_v6/best.pt \
  --machine mastu_simple --out-dir <figures 目录>
```

## 6. 相关脚本（dn_fno_2608/scripts/）

| 脚本 | 用途 | 示例 |
|---|---|---|
| `run_generate_v6.sh` | 全量生成 5 配置 × 3 split（探针→val→test→train） | `bash dn_fno_2608/scripts/run_generate_v6.sh` |
| `probe_v6.py` | 探针：接受率/判据分布/位形电流可分性/计时 | `python dn_fno_2608/scripts/probe_v6.py --config sn --n 80` |
| `plot_v6_examples.py` | 示例平衡图（psi 等高线 + 雪点/接触点/壁标记），物理确认用 | 参数见脚本头注释 |
| `_backfill_v6_labels.py` | 旧 chunk 补写标注字段（wall_contact 等），生成脚本升级后不需要 | — |
| `run_exp012_eval_vis.sh` | exp012 六桶评估 + 5 配置可视化一键 | `bash dn_fno_2608/scripts/run_exp012_eval_vis.sh` |
| `_dbg_*.py` | 各次调试探针（未跟踪，不入 git；仓库清理时按需删除） | — |

## 7. 生成设置与复现

### 7.1 采样与接受检查（全部 5 配置共用）

- 参数范围：paxis U[4e4,8e4] Pa、Ip U[7e5,1.5e6] A、fvac U[0.4,0.9]（例 18/19
  强等离子体；Ip 下限 7e5 防高 beta 自由边界振荡）；alpha_m/alpha_n 采样；
  X 点抖动 R ±0.06 / Z ±0.10 m；锚点 R~U[1.20,1.45]
- X 点中心：dn (0.80,±1.20)、sn (0.65,1.20)；snow_single 雪点 (0.509,−1.291)、
  snow_double (0.65,±1.20)；limiter 磁轴 R~U[0.65,0.95]
- 检查（同 data_v4 7 项 + v6 新增）：`--max-isoflux-residual 0.35
  --max-xpt-deviation 0.10 --min-anchor-xpt-dist 0.15 --coil-margin 0.05
  --min-core-depth 0.005` + 深触壁排除（wall_contact_excess ≤ 0.2）+
  壁内结构判据 INWALL_SEP_FRAC_TOL=0.02
- snowflake 求解：maxits 200 / **rtol 1e-3**（5e-3 会停在错分叉的上瓣假解）+
  初始电流种子（选下瓣分支）；limiter 两步法（Step1 磁轴约束 γ=1e-12 →
  Step2 无约束，均 check_limited=True/limit_it=0）
- 触壁策略（用户拍板）：深触壁排除、浅触壁保留 + 标注

### 7.2 探针统计（80 样本/配置，max_retries=1 = 原始接受率）

| 配置 | 接受率 | 触壁（n_touch / max excess） | 单解耗时 mean |
|---|---|---|---|
| dn | 61/80 (76.3%) | 27/80 / 7.4% | 13.4 s |
| sn | 44/80 (55.0%) | 6/80 / 19.1% | 103 s |
| snow_single | 52/80 (65.0%) | 10/80 / 15.3% | 71.8 s |
| snow_double | 80/80 (100%) | 1/80 / 7.6% | 14.4 s |
| limiter | 80/80 (100%) | 80/80 / 0.0%（触壁是定义） | 11.6 s |

全量 train 500/配置：触壁 dn 53%（max 10.1%）、sn 9%（19.6%）、snow_single 14%
（18.4%）、snow_double 1%（14.9%）、limiter 100%（0.1%）。

### 7.3 复现与扩展

```bash
# 全量生成（val seed 789 / test seed 1011 / train seed 456）
bash dn_fno_2608/scripts/run_generate_v6.sh

# 扩展（同 seed 重跑更大 n：chunk_*.npz 自动跳过，--merge 合成新 split.npz；
# 同 seed 确定性采样 → 前段样本与现数据完全一致不重复）例：train 追加到 2000
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

## 8. 历史与偏差记录

### 8.1 ⚠️ SN 数据质量问题（2026-08-19，用户发现）

sn 200 test 样本的 freegs 真值形态存在系统性偏差：
- **磁轴系统性偏下**：Z_axis mean **−0.339 m**（仅 2/200 在 |Z|<0.15，min −0.848）
- 76/200 上瓣薄：中平面 psi 距分离面 <0.5 core 深度；极端样本（test idx158）
  中平面 psi < psi_bndry（等离子体不跨中平面，分离面外翻到壁区，物理不可信）
- 根因：SN 磁轴检查（R_lo<R_axis<R_anc 且 |Z_axis|<|Z_lo|）放行了上瓣薄的形态
- 影响：exp012 sn 桶 rel L2 8.99%（其他配置 1.3–3.6%）部分归因于此
- **建议（待拍板）**：① 生成加"上瓣健康度"判据（中平面 psi > psi_bndry + k·core）
  ② 重生成病态样本 ③ 专职 SN 对照模型

### 8.2 关键物理修正（与 data_v5 差异）

| 项 | data_v5 (MAST) | data_v6 (MASTU_simple) |
|---|---|---|
| 机器/壁 | MAST 11 线圈、**无墙** | MASTU_simple 14 线圈、**真实壁** |
| 网格 | 65² | **129²**（雪点二阶判据在 65² 精度不足） |
| 位形 | DN+SN | DN+SN+snow_single+snow_double+limiter |
| 输入 | 18ch（无 config 字段） | 21ch（14 线圈；config 字段存在但不作为输入） |
| dn X 点中心 | (0.7,±1.1) | **(0.80,±1.20)**（0.65 时 12/20 样本 Solenoid 柱区与分离线交叉病态） |
| 参数范围 | paxis 1e3-5e3 / Ip 3e5-8e5 / fvac 0.3-0.8 | paxis 4e4-8e4 / Ip 7e5-1.5e6 / fvac 0.4-0.9 |

### 8.3 其余偏差记录

1. limiter 无 anchor / xpts_actual 0 行：设计如此（无 isoflux 约束、无分离面 X 点）；
   混合数据集的评估几何指标对这些行 NaN。
2. limiter inwall_sep_frac 非零（max 5.78%）：psi_bndry 是限制面不是分离面，
   指标无语义，仅记录。
3. snowflake 需初始电流种子（MASTU_INIT_CURRENTS）：Tikhonov 系统多固定点，
   无种子收敛到上瓣假解（探针验证）。
4. snowflake rtol=1e-3 而非 5e-3：5e-3 在错分叉的上瓣假解停下（探针验证）。
5. data_v6 与 data_v5/v4 跨机器对比仅参考（机器/网格/线圈数不同）。
6. 数据 npz 不入 git（.gitignore `data_v*/**`，仅 README 白名单）。
