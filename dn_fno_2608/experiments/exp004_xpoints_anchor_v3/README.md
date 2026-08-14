# exp004 — X 点输入在锚点大变化下的可靠性（data_v3，X点 vs X点+锚点）

> 实验日期：2026-08-14
> 状态：**完成**（N=500，seed 1，三模型对比）
> 代码：`src/gs_pino_dn_fno_2608/`（`train_dn_fno --input-mode` + data_dn_fno `use_anchor`，默认行为不变）
> 数据：`dn_fno_2608/data_v3/`（X 点 ±0.20 m + 锚点采样；**数据集已删除**（2026-08-14，
> 被 data_v4 取代），复现脚本 `scripts/run_generate_v3.sh` 保留）
> 结论速览：X 点输入在锚点大变化下**不可靠**（test rel L2 31.1%，一对多映射）；
> 即使输入含锚点（A' 21.7%）或线圈电流（B 12.5%），N=500 也远不足以学习
> 展宽后的解空间——**主要瓶颈是数据量而非输入信息**。coil 输入最好且各桶平坦。

---

## 1. 动机与科学问题

data_v3 把两个此前固定的几何自由度放开采样：

1. **X 点位置** ±0.02 → ±0.20 m；
2. **isoflux 锚点**从固定 (1.5, 0.0) 改为 R∈[1.2,1.8]×Z∈[-0.3,0.3] 采样。

isoflux 约束 `psi(X点)=psi(锚点)` 使分离面必须同时穿过两个 X 点和锚点——锚点
位置变化会改变**分离面形状**（不是磁通 gauge）。于是：

- **模型 A（仅 X 点坐标，11 通道）** 的输入不含锚点信息 → 同一输入对应多个
  不同 psi 场（一对多映射）→ 只能学条件均值 → 预期 rel L2 大幅退化；
- **模型 A'（X 点 + 锚点，13 通道）** 包含全部约束信息 → 上界对照。

问题："如果 X 点、锚点都大范围变动，程序还可靠吗？" 本实验用 A（退化组）与
A'（上界组）在同一 test 上对比，量化"缺锚点信息"的代价。

## 2. 方法

```
模型 A  （11 通道）:  R, Z | Paxis, Ip, fvac, alpha_m, alpha_n | R_lo, Z_lo, R_up, Z_up       (9 标量)
模型 A' （13 通道）:  R, Z | Paxis, Ip, fvac, alpha_m, alpha_n | R_lo...Z_up | R_anc, Z_anc   (11 标量)
```

- 两个模型同数据（data_v3）、同规模（N=500 seed 1）、同训练超参（与 exp002/003 完全一致）；
- A 复用既有 train_dn_fno（零代码改动）；A' 用 `--input-mode xa`（新增路径，
  锚点坐标与 X 点一样 z-scored）；
- 评估：test 500 全量 + 按锚点属性分桶的 rel L2（`analyze_anchor_buckets.py`）。

### 代码改动（向后兼容）

| 文件 | 改动 |
|---|---|
| `generate_dn_dataset.py` | `--xpt-jitter` / `--isoflux-sampling`（默认关），anchor 字段，重采样含锚点 |
| `data_dn_fno.py` | `DNFnoDataset(use_anchor=...)` + `compute_stats(use_anchor=...)`（默认 False 行为不变） |
| `{train,evaluate,visualize,latency}_dn_fno.py` | `--input-mode {xpoints,xa}`（默认 xpoints）+ stats 转换过滤 `input_mode` 字符串 |
| `model_dn_fno.py` | 零改动 |
| coils 变体 | 零改动 |

验证：默认 flag 强回归——同 seed 下 params/x_coords/coil_currents/psi_total 与
data_v2 前 8 行**逐字节相等**；冒烟 8/8 + 40/40 接受。

## 3. 结果（test 499，N=500，seed 1）

| 指标 | A X点 11ch | A' X点+锚点 13ch | B coil 11ch (exp005) | exp002 (data_v2) 参考 |
|---|---|---|---|---|
| rel L2 mean % | 31.09 | 21.72 | **12.47** | 0.303 ± 0.227 |
| rel L2 median % | 25.88 | 12.70 | **8.37** | 0.242 |
| rel L2 <0.12% | 0.0% | 0.0% | 0.0% | (paper >95%) |
| RMSE phys (Wb) | 1.04e-2 | 7.88e-3 | 3.85e-3 | 8.23e-5 |
| GS 残差比值 | 1.034 | 1.219 | 1.245 | 0.996 |
| find_critical 失败 | 21/499 | 48/499 | 35/499 | 1/500 |

训练（val，均早停）：A 29.57% @e18（1.3 min）｜A' 20.68% @e356（6.0 min）｜
B 12.99% @e220（4.1 min）。通道数已核对（11/13/11），无静默错误。

### 锚点分桶（rel L2 mean %，[analysis/anchor_buckets.json](analysis/anchor_buckets.json)）

| 锚点距默认位 (1.5,0.0) | n | A | A' | B |
|---|---|---|---|---|
| <0.2 m | 207 | 25.2 | 16.5 | 11.2 |
| 0.2–0.4 m | 291 | 35.3 | 25.4 | 13.4 |

| 锚点距最近 X 点 | n | A | A' | B |
|---|---|---|---|---|
| <0.3 m | 13 | 52.5 | 54.2 | **8.6** |
| 0.3–0.6 m | 179 | 39.5 | 31.5 | 12.9 |
| ≥0.6 m | 307 | 25.3 | 14.6 | 12.4 |

## 4. 结论

1. **仅 X 点输入在锚点大变化下不可靠（31.1% vs 0.3% 基线，退化 100×）**——
   一对多映射（同一输入 → 多种分离面）被分桶证实：误差随锚点偏移增大
   （25→35%），锚点贴近 X 点时最高（52.5%）。
2. **信息完整也救不了 N=500**：A'（13ch 含锚点）21.7%、B（coil，隐式含锚点）
   12.5%——都是确定性映射，退化来自 data_v3 解空间复杂度爆炸，属数据量瓶颈。
3. **coil 输入相对最可靠**：整体最好（12.5%）、全桶平坦、近退化区最优（8.6%）。
   freegs 的 Tikhonov 电流解是锚点的确定性函数，电流直接决定线圈场，映射
   更"因果"、更好学。
4. 对用户问题的直接回答：若 X 点与锚点都 ±0.2 m 量级变动，现 N=500 策略下
   程序**不可靠**（任何输入模式）；要恢复可靠性需放大数据量（~5000+ 或按
   d_min 分层采样），coil 输入是当前信息架构下的最佳选择。

## 5. 复现命令

```bash
PY="C:/Users/HP/.conda/envs/torch5060/python.exe"

# 数据（已生成，data_v3 已删除；复现：bash dn_fno_2608/scripts/run_generate_v3.sh）
# 模型 A（X点 11ch，退化对照）
"$PY" -u -m gs_pino_dn_fno_2608.train_dn_fno \
  --train-data dn_fno_2608/data_v3/train.npz --val-data dn_fno_2608/data_v3/val.npz \
  --n-train 500 --seed 1 --out-dir dn_fno_2608/experiments/exp004_xpoints_anchor_v3/model_a11ch_xpt
"$PY" -u -m gs_pino_dn_fno_2608.evaluate_dn_fno \
  --test-data dn_fno_2608/data_v3/test.npz \
  --checkpoint dn_fno_2608/experiments/exp004_xpoints_anchor_v3/model_a11ch_xpt/best.pt \
  --out-dir dn_fno_2608/experiments/exp004_xpoints_anchor_v3/model_a11ch_xpt

# 模型 A'（X点+锚点 13ch，上界）
"$PY" -u -m gs_pino_dn_fno_2608.train_dn_fno --input-mode xa \
  --train-data dn_fno_2608/data_v3/train.npz --val-data dn_fno_2608/data_v3/val.npz \
  --n-train 500 --seed 1 --out-dir dn_fno_2608/experiments/exp004_xpoints_anchor_v3/model_a13ch_xa
"$PY" -u -m gs_pino_dn_fno_2608.evaluate_dn_fno \
  --test-data dn_fno_2608/data_v3/test.npz \
  --checkpoint dn_fno_2608/experiments/exp004_xpoints_anchor_v3/model_a13ch_xa/best.pt \
  --out-dir dn_fno_2608/experiments/exp004_xpoints_anchor_v3/model_a13ch_xa

# 锚点分桶分析（三模型对比：A、A'、exp005 的 B）
"$PY" -u -m gs_pino_dn_fno_2608.analyze_anchor_buckets \
  --test-data dn_fno_2608/data_v3/test.npz \
  --checkpoint a11ch=dn_fno_2608/experiments/exp004_xpoints_anchor_v3/model_a11ch_xpt/best.pt \
  --checkpoint a13ch=dn_fno_2608/experiments/exp004_xpoints_anchor_v3/model_a13ch_xa/best.pt \
  --checkpoint b11ch=dn_fno_2608/experiments/exp005_coil_input_v3/best.pt \
  --out-dir dn_fno_2608/experiments/exp004_xpoints_anchor_v3/analysis
```

## 6. 产物

```
exp004_xpoints_anchor_v3/
├── README.md
├── notes.md
├── model_a11ch_xpt/      # A：best.pt / history.json / args.json / metrics.json
├── model_a13ch_xa/       # A'：同上
└── analysis/             # anchor_buckets.json（三模型分桶）
```

训练日志：`dn_fno_2608/logs/exp004_*.log`
无 figures/：可视化需要 data_v3/test.npz 真值，数据集已删除，按用户决定跳过
（exp005 保留旧 worst_best 图）。

## 7. 后续可做（供参考）

- 若 A 退化显著而 A' 恢复正常，则证实"X 点输入不含锚点信息"是根本限制；
  实际控制场景中锚点通常固定（gauge 由控制系统定义），data/ 与 data_v2/
  的设定仍是贴近实际的一档。
