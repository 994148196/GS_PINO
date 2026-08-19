# data_v5 — MAST 混合位形数据集（DN+SN，18ch coil 输入）

> 生成日期：2026-08-17
> 用途：exp008（混合 config 输入）/ exp009（专职对照）/ exp010（无 config 消融）/
> exp011（coil 电流输入端到端）的数据
> 代码：`gs_pino_dn_fno_2608.generate_dn_dataset`（`--config {dn,sn}` + `--machine mast`）
> 探针报告：`PLAN_v5_mixed_configs.md`；探针 JSON：`_probe_mast_dn.json`/`_probe_mast_sn.json`

## 1. 数据集速览

| 项 | 值 |
|---|---|
| 机器 | **MAST**（freegs 真实装置，11 控制线圈：P2U/P2L/P3U/P3L/P4U/P4L/P5U/P5L/P6U/P6L + P1 Solenoid，**无墙**） |
| 位形 | DN（双 X 点）+ SN（单 X 点），**按配置分开落盘**（`dn/`、`sn/`） |
| 网格 | 65² |
| 规模 | dn、sn 各 train 2000（seed 123）/ val 500（seed 456）/ test 500（seed 789），接受率 100% |
| 输入通道 | **18ch**（coil 主用，exp011）/ **13ch**（xa，exp008-010）/ **14ch**（xa+config，exp008） |
| 目标 | `psi_total`（65²，z-score） |
| 新增字段 | `config`（0=DN, 1=SN） |

## 2. 目录与文件

```
dn_fno_2608/data_v5/
├── dn/  train.npz / val.npz / test.npz   # DN 配置（+ 分块目录，冗余可删）
├── sn/  train.npz / val.npz / test.npz   # SN 配置
├── _probe_mast_dn.json / _probe_mast_sn.json   # 探针结果
└── _smoke_mix/                            # 冒烟产物
```

后续可只用一种（专职）或逗号拼接混合（`dn/train.npz,sn/train.npz`）训练。

## 3. 字段说明（npz keys）

data_v4 全部字段 + 新增 `config`（(N,1)，0=DN / 1=SN）。**x_coords 固定 4 通道**：
SN 样本上 X 点对存 **(0.0, 0.0) 占位**（恒值通道 → z-score 后为 0，由 config 通道
区分；模型侧 `np.maximum(std, 1e-8)` 防护）。SN 的诊断字段形状缩小：
xpt_constraint_res (2,)、isoflux_res (1,)、psi_at_constraints [p_lo, p_anc]；
o_point = 位置过滤后的磁轴。`xpts_actual` 行数：DN 2 行 / SN 1 行（混合拼接时
NaN-pad 到最大行数）。

## 4. 输入通道明细

### 4.1 coil 输入 18ch（exp011，主用）

| # | 通道 | 含义/单位 | 来源 |
|---|---|---|---|
| 1 | R | 网格 R 坐标（[-1,1]） | 固定 65² 网格 |
| 2 | Z | 网格 Z 坐标（[-1,1]） | 固定 65² 网格 |
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

线圈名/顺序 = freegs `machine.py` MAST() 定义顺序，生成脚本按 `tokamak.coils`
顺序收集。R/Z 对全部样本相同 → 16 个标量通道承载全部样本信息；其中 5 个 params
在 DN/SN 间统计不可区分，**位形信息完全由 11 个线圈电流通道承载**（单通道阈值
判别 95.95%，最强通道 P3U 分离度 3.29 std）。

### 4.2 xa 输入 13ch（exp008-010）/ 14ch（+config）

| # | 13ch | 14ch（+config） |
|---|---|---|
| 1–2 | R, Z | R, Z |
| 3–7 | Ip, paxis, fvac, alpha_m, alpha_n | 同左 |
| 8–11 | R_lo, Z_lo, R_up, Z_up | 同左（SN 的 up 恒为 (0,0) 占位） |
| 12–13 | R_anc, Z_anc | 同左 |
| 14 | — | config（0/1，z-score） |

## 5. 如何调用

```python
import numpy as np
d = np.load("dn_fno_2608/data_v5/dn/train.npz")
cfg = d["config"]           # (2000,): 0=DN / 1=SN
coils = d["coil_currents"]  # (2000, 11): 顺序见 §4.1
```

```bash
PY="C:/Users/HP/.conda/envs/torch5060/python.exe"

# exp011 训练（coil 18ch，DN+SN 混合，逗号拼接）
"$PY" -u -m gs_pino_dn_fno_2608.train_dn_fno \
  --train-data dn_fno_2608/data_v5/dn/train.npz,dn_fno_2608/data_v5/sn/train.npz \
  --val-data dn_fno_2608/data_v5/dn/val.npz,dn_fno_2608/data_v5/sn/val.npz \
  --input-mode coils --n-train 500 --seed 1 --out-dir <out>
# 评估（分桶：--test-data 单传 dn 或 sn 文件）
"$PY" -u -m gs_pino_dn_fno_2608.evaluate_dn_fno \
  --test-data dn_fno_2608/data_v5/sn/test.npz \
  --checkpoint <out>/best.pt --out-dir <out>/eval_sn
# 可视化：--machine mast
"$PY" -u -m gs_pino_dn_fno_2608.visualize_dn_fno \
  --checkpoint <out>/best.pt --machine mast --out-dir <out>/figures_sn
```

xa 系（exp008-010）：`--input-mode xa`（+ `--config-input` 为 14ch）。

## 6. 相关脚本（dn_fno_2608/scripts/）

| 脚本 | 用途 |
|---|---|
| `run_generate_v5.sh` | 全量生成（探针 + 6 个 split 一键；DN ~6 min + SN ~31 min） |
| `probe_v5.py` | 探针：原始接受率统计，输出 JSON（`--machine mast --config sn --n 80`） |
| `run_exp008_009_train.sh` / `run_exp008_009_eval.sh` | exp008/009 训练 / 评估+可视化链 |
| `rerun_exp008_010_geom_v2.sh` | 几何指标 v2 重评估（exp008/009/010） |
| `tabulate_exp008_009.py` | exp008/009 对照表生成 |
| `run_exp011_eval_vis.sh` | exp011 评估 + 可视化一键 |

## 7. 生成设置与统计

### 7.1 探针校准（80 样本/配置，max_retries=1 = 原始接受率）

| 配置 | 接受率 | isoflux res 最大 | X 点偏差最大 | core 深度 (mean/min) | 收敛迭代 |
|---|---|---|---|---|---|
| MAST+DN | **80/80 (100%)** | 2.0e-4 | 0.016 m | 0.077 / 0.039 | 8.7± |
| MAST+SN | **80/80 (100%)** | 2.0e-5 | 0.0015 m | 0.085 / 0.044 | 40.6±（max 51 < MAXITS） |

- X 点中心 (0.7, ±1.1)（MAST divertor 区域）、抖动 R ±0.06 / Z ±0.10 m；
  锚点中平面 R~U[1.2,1.6]；参数 paxis (1e3,5e3) / Ip (3e5,8e5) / fvac (0.3,0.8)
- SN 收敛迭代显著高于 DN（40.6 vs 8.7）→ 求解 ~0.63 s/solve（24 核均摊；
  单核 ~15 s），全量 3000 SN 约 31 min

### 7.2 接受检查（data_v4 7 项 + SN 适配）

1. **SN 分离面 X 点判据**：`find_critical` 会报告真空区假鞍点（MAST 5–6 个/样本），
   SN 判定改为 **psi ≥ psi_bndry − 1e-6 的 X 点数 == 1**（分离面恰一个 X 点）；
   psi_bndry = 该 X 点 psi（DN 仍为两 X 点 psi 均值）；
2. **SN 磁轴**：改为 **O 点位于 X 点与中平面锚点之间**（R_lo < R_axis < R_anc 且
   |Z_axis| < |Z_lo|）。注意 psi_total 含线圈贡献，`find_critical` 会报告线圈场
   真空假极值（墙外 R>1.8），磁轴**按位置过滤后再取 psi 最大**；
3. **无墙适配**：MAST `tokamak.wall` 为 None → `--require-wall` 忽略（打印
   WARNING），墙检查跳过；
4. **Solenoid 线圈**：`_control_coil_centers` 对 Solenoid（Rs/Zsmin/Zsmax）取
   中点 (Rs, Z 中点)；凸包 sanity 按线圈数跳过。

## 8. 历史与偏差记录

1. **SN 磁轴取错（0/8 全拒）**：初版 `max(opt, key=psi)` 选到线圈场真空假极值
   （TestTokamak 固定 (1.002,−1.102)、MAST 墙外 (1.87,−0.20)）→ 位置过滤后再取
   psi 最大。修复后 TestTokamak+SN 冒烟 8/8、MAST+SN 探针 80/80。
2. **SN 轴检查负 Z 比较 bug**：`abs(za) < lo[1]`（lo[1] 为负）恒假 → `abs(za) < abs(lo[1])`。
3. **MAST Solenoid 无 `.R`**：`_control_coil_centers` AttributeError → 按
   `hasattr(coil, "Rs")` 分支取 (Rs, Zsmin/Zsmax 中点)。
4. **MAST 无墙崩溃**：`tokamak.wall.R`（NoneType）→ wall_verts 仅在有墙时构造。
5. **SN 配对统计 bug（不影响数据）**：x_coords (0,0) 占位被当成配对目标 →
   探针与 merge 打印按 xpts_actual 实际行数分支配对（数据本身正确，SN 偏差 ≤0.002 m）。
6. **SN 求解耗时显著高于 DN**（单核 ~15 s vs ~4.4 s/solve，n_iter 40 vs 9）：
   不影响正确性。
7. 与 data_v4 差异：机器 MAST 11 线圈无墙 vs TestTokamak 4 线圈有墙；X 点中心
   (0.7,±1.1) vs (1.2,±0.6)；位形 DN+SN vs 单 DN；参数范围收窄；新增 config 字段。
