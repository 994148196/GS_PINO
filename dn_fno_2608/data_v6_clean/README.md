# data_v6_clean — data_v6 清洗版（剔除病态样本 + 候选池补足）

> 生成日期：2026-08-21 ｜ 用途：exp013（清洗数据重训，与 exp012 同设置对照）
> 代码：`dn_fno_2608/scripts/filter_v6.py`（判据打分与构建）+
> `dn_fno_2608/scripts/run_generate_v6_topup.sh`（补足候选池生成）
> 数据本体 = data_v6 的健康子集 + 新 seed（20260821 系）补足样本
> （**原 data_v6 不动，只增不改**）
> 结果：exp013 eval_all **3.045%**（对照 exp012 同设置 3.435%）

## 1. 数据集速览

| 项 | 值 |
|---|---|
| 机器 | **MASTU_simple**（freegs 自带，14 控制线圈 + 真实真空室壁 rwall，R 0.244–2.0） |
| 位形 | 5 种：dn / sn / snow_single / snow_double / limiter（config 编码 0/1/2/3/4） |
| 网格 | 129²，R∈[0.1,2.0] m、Z∈[−2,2] m |
| 规模 | 每配置 train **500** / val **100** / test **200**（与 data_v6 相同；15 split 实测全部满额） |
| 输入通道 | **21ch**：R,Z + 5 params + 14 线圈电流（**无 config 通道**）（§4） |
| 目标 | `psi_total`（129²，均值/标准差归一化） |
| 求解器 | freegs_snow fork（同 data_v6：雪点二阶约束，maxits 200、rtol 1e-3） |
| 与 data_v6 差异 | **仅样本构成**：健康子集逐样本一致（相同 chunk 序）；补足样本来自新 seed，参数不重叠 |

## 2. 目录与文件

```
dn_fno_2608/data_v6_clean/
├── {dn,sn,snow_single,snow_double,limiter}/
│   └── train.npz / val.npz / test.npz   # 与 data_v6 同结构同 keys（npz 不入 git）
├── manifest.json   # 逐 (cfg,split)：剔除 idx、判据值明细、topup 来源与行号、最终规模
├── scores.json     # 逐 (cfg,split) 剔除数量与 idx（data_v6 原始打分）
├── README.md
└── README.pdf
```

## 3. 字段说明（npz keys）

与 data_v6 **完全相同**：dn/sn/snow_single/snow_double 为 29 键，limiter 为 32 键
（多 `Rlim`/`Zlim`/`is_limited`/`psi_limit`；**无 `anchor`**、`xpts_actual` 为
(N,0,3) 空数组——混合拼接时由数据集类 NaN-pad）。

| 字段 | 形状 | 含义 | 是否模型输入 |
|---|---|---|---|
| `psi_total` | (N,129,129) | 总极向磁通（**训练目标**） | — |
| `psi_plasma` / `psi_plasma_norm` / `psi_coils` | (N,129,129) | 磁通分量 | 否（PINO 预留） |
| `R`, `Z` | (129,129) | 物理坐标网格（R 沿行 axis 0、Z 沿列 axis 1） | 通道 1–2 |
| `mask` | (N,129,129) | freegs critical.core_mask | 否 |
| `params` | (N,5) | [Ip, paxis, fvac, alpha_m, alpha_n] | 通道 3–7 |
| `coil_currents` | (N,14) | 14 线圈电流（顺序见 §4） | 通道 8–21 |
| `config` | (N,1) | 位形编码 0=dn 1=sn 2=snow_single 3=snow_double 4=limiter | **否**（仅标注/分桶） |
| `axes` | (N,4) | [R_axis, Z_axis, psi_bndry, psi_axis] | 否（评估用） |
| `L`, `Beta0` | (N,1) | 电感 / 比压 | 否 |
| `solve_time` / `n_iter` / `psi_relchange_final` | (N,1) | 求解耗时 / 迭代数 / 末次相对变化 | 否 |
| `x_coords` | (N,4) | X 点（雪点）目标坐标（limiter 为占位） | 否 |
| `xpts_actual` | (N,2,3) / (N,1,3) / **(N,0,3)** | 实际 X 点 [R,Z,psi]（DN 2 行 / sn·snow_single 1 行 / **limiter 0 行**） | 否（评估几何真值） |
| `o_point` | (N,3) | 实际 O 点 [R,Z,psi] | 否 |
| `anchor` | (N,2) | isoflux 锚点（**limiter 无**） | 否 |
| `isoflux_res` / `xpt_constraint_res` / `psi_at_constraints` | (N,2)/(N,4)/(N,3) | 约束残差诊断（limiter 缩形 (N,1)/(N,2)/(N,1)） | 否（诊断） |
| `dpdpsi`, `FdFdpsi` | (N,129,129) | GS 残差 RHS 分量 | 否（PINO 预留） |
| `greens` | (N,14,129,129) | 线圈 Green 函数（逐样本存储；几何固定 → 数值相同） | 否（PINO 预留） |
| `wall_contact` / `wall_contact_excess` / `inwall_sep_frac` | (N,) | 触壁标注（生成时深触壁已排除、浅触壁保留+标注） | 否（标注） |
| limiter 专有：`is_limited` / `Rlim`, `Zlim` / `psi_limit` | (N,1) | 限制位形标注（恒 1）/ 触壁点 / 限制面 psi（=psi_bndry，自洽 \|Δψ\|≤0.01） | 否 |

## 4. 输入通道明细（21 通道）

通道顺序 = 代码实际拼接顺序（`data_dn_fno_coils.py`）：2 几何 + 5 标量 + 14 线圈
电流（**无 config 通道**）。R/Z 线性映射 [-1,1]；19 个标量用**训练 pool** 的
mean/std z-score；目标 psi 同样 z-score（训练损失与 rel L2 均在归一化域计算）。

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

线圈名/顺序 = freegs `machine.py` MASTU_simple() 定义顺序（0-1 号是 Solenoid/
Circuit 等成对封装，生成脚本按 `tokamak.coils` 顺序收集）。**无 config 通道**：
14 通道电流 pooled 分离度全部 >2.0 std、5 配置两两配对阈值判别 ≥90.75% → 电流
本身携带位形信息（data_v6 探针 `_probe/probe_v6_separability.json`）。

## 5. 剔除判据与补足（本数据集的核心内容）

### 5.1 剔除判据（2026-08-21 研究结论，exp012 stats 关联验证）

| 配置 | 判据 | test 剔除 | 依据 |
|---|---|---|---|
| **sn** | `gs_true > 15` ∪ `midplane_ratio < 0` | 27/200 (13.5%) | 磁轴上瓣薄/GS 未收敛病态；被剔样本 rel_l2 mean 18.9%，桶 8.99→7.44%（−17%）；worst10 图抓 8 个 |
| **dn** | `gs_true > 15` | 4/200 (2%) | 仅清未收敛尾部（idx 36/49/124/172） |
| **snow_double** | X 点记录偏差 > 0.15 m | 4/200 (2%) | idx 30/158/189/197：雪点约束在目标处 B≈0 但 find_critical 配对到内柱区（触壁+结构破坏），含桶内最差样本 |
| **snow_single** | 无 | 0 | 2026-08-21 用户拍板：数据健康（gs 4.2 < 其余 7.4、rel_l2 1.70%），仅标注问题不筛（§8.3） |
| **limiter** | 无 | 0 | gs_true 0.32 最健康；wall_contact 100% 是有限位形设计行为 |

指标口径（与 exp012 evaluate 一致）：
- `gs_true`：GS 残差比 = ‖Δ*ψ−RHS‖/‖RHS‖（mask = ψ≥ψ_bndry，
  `evaluate_dn_fno.gs_residual_ratio`）
- `midplane_ratio`：(ψ(R_axis,Z≈0)−ψ_bndry)/(ψ_axis−ψ_bndry)，<0 = 上瓣塌陷
- `xpt_dev`：xpts_actual 与 x_coords 按配对顺序的最大偏差

### 5.2 补足（topup）

- 生成：`run_generate_v6_topup.sh`，**新 seed 独立池**（train 20260821/val
  20260822/test 20260823 起，dn/snow_double 用 20260831+/20260911+ 系列）——
  **不用同 seed 扩展**（chunk resume 不推进 rng，新 chunk 参数序列会与前段重复）
- 候选池 → filter_v6 判据 → 健康候选按序取前 `need = target − 原健康数` 个
- 候选池目录 `dn_fno_2608/data_v6_topup/` 为一次性中间产物（已删除，无残留）
- 补足量与来源行号见 `manifest.json` 的 `topup_n`/`topup_rows`；最终 15 split
  全部恢复满额 500/100/200

## 6. 如何调用

### 6.1 直接加载

```python
import numpy as np
d = np.load("dn_fno_2608/data_v6_clean/sn/train.npz")
psi = d["psi_total"]        # (500, 129, 129)
coils = d["coil_currents"]  # (500, 14): 顺序见 §4
cfg = d["config"]           # (500,): 0-4
```

### 6.2 训练 / 评估

```bash
PY="C:/Users/HP/.conda/envs/torch5060/python.exe"
CLEAN=dn_fno_2608/data_v6_clean

# exp013 训练（= exp012 命令仅数据源换 clean）
bash dn_fno_2608/scripts/run_exp013_train.sh
# 或手写等价命令（21ch coil，五配置混合 pool 2500）：
"$PY" -u -m gs_pino_dn_fno_2608.train_dn_fno \
  --train-data $CLEAN/dn/train.npz,$CLEAN/sn/train.npz,$CLEAN/snow_single/train.npz,$CLEAN/snow_double/train.npz,$CLEAN/limiter/train.npz \
  --val-data $CLEAN/dn/val.npz,$CLEAN/sn/val.npz,$CLEAN/snow_single/val.npz,$CLEAN/snow_double/val.npz,$CLEAN/limiter/val.npz \
  --input-mode coils --no-config-channel --n-train 500 --seed 1 \
  --epochs 800 --batch-size 16 --lr 1e-3 --weight-decay 1e-4 \
  --lr-patience 20 --lr-factor 0.5 --min-lr 1e-5 --patience 75 --out-dir <out>

# 6 桶评估 + 5 配置可视化
bash dn_fno_2608/scripts/run_exp013_eval_vis.sh
```

- 多文件逗号拼接 = 五配置混合 pool（2500 样本），`--n-train 500` 按
  `--perm-seed 12345` nested 抽取；单配置训练只传对应文件即可
- ckpt 顶层存 `no_config_channel` 标志 → evaluate/visualize 自动匹配输入通道

## 7. 相关脚本（dn_fno_2608/scripts/）

| 脚本 | 用途 |
|---|---|
| `filter_v6.py` | 判据打分（score，写 scores.json）+ 构建 clean（compose，写 manifest.json） |
| `run_generate_v6_topup.sh` | 补足候选池生成（新 seed 独立池 + merge） |
| `run_exp013_train.sh` / `run_exp013_eval_vis.sh` | exp013 训练 / 6 桶评估+可视化一键 |

## 8. 历史与偏差记录

1. **候选池新 seed 而非同 seed 扩展**：run_generate_v6.sh 注释的"同 seed 更大 n"
   方式在 chunk resume 下（跳过不推进 rng）新 chunk 参数序列重复旧段前段 → 弃用；
2. **snow_double x_coords 顺序与 dn 相反**（[上雪点, 下雪点] vs [下, 上]）——
   filter 按 xpts_actual 生成器配对顺序对齐，不依赖 lo/up 语义；
3. **snow_single 标注问题不筛**：xpts_actual 对 31/200 样本记录到上偏滤器 X 点
   （雪点不在最终 psi_bndry 分离面），数据本身健康——待生成侧修复（雪点分离面
   判据），供 future data_v7；
4. **结果参照（exp013 vs exp012 同设置）**：eval_all 3.045% vs 3.435%；sn 7.738%
   vs 8.988%；snow_double 1.074% vs 1.307%。清洗收益分解：dn −10.7% /
   snow_double −9.6% 为模型真实收益（同 test 集），**sn −13.9% 主要来自 test
   剔除病态**（模型原 test +3.1% 未变强，剔除过量警示）——生成侧修复优于事后
   筛选，后续 data_gspack2_v2 即按此思路做生成侧硬门。
