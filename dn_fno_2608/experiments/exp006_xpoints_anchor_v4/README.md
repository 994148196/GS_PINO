# exp006 — data_v4 上 X 点输入复测（X点 vs X点+锚点，约束可达）

> 实验日期：2026-08-14
> 状态：**完成**（N=500，seed 1，三模型对比）
> 代码：`src/gs_pino_dn_fno_2608/`（`train_dn_fno --input-mode` + data_dn_fno `use_anchor`，与 exp004 同路径零改动）
> 数据：`dn_fno_2608/data_v4/`（可行区采样 + 7 项物理合理性接受约束，见 [data_v4/README.md](../../data_v4/README.md)）
> 结论速览：**约束不可达（而非数据量）是 exp004 退化的主因**——data_v4 上
> A'（X点+锚点 13ch）test rel L2 21.72% → **0.442%**（49× 恢复，达到 data_v2 水平），
> B（coil 11ch）12.47% → 0.499%，A（X点 11ch）31.09% → 3.38%。A 仍落后 A' 7.7×
> （锚点信息缺失的一对多映射仍在），但 v3 的 100× 级退化已消除。

---

## 1. 动机与科学问题

exp004/exp005 在 data_v3 上观察到三模型全部大幅退化（A 31.1% / A' 21.7% /
B 12.5% vs data_v2 的 ~0.3%），当时归因于"数据量瓶颈"。但 data_v3 有两个
未被检查的缺陷（data_v3 已删除；残差统计见 [exp004 notes §5b](../exp004_xpoints_anchor_v3/notes.md)）：

1. **isoflux 约束不可达**：4 控制线圈对 6 约束是过定系统，Tikhonov 定点解保留
   不可达方向残差，data_v3 中 |psi(X点)−psi(锚点)|/core 均值 0.50、最大 2.10——
   多数样本的"真值"分离面并不通过锚点；
2. **采样越界**：X 点/锚点范围大量落在线圈四边形凸包之外。

data_v4（exp006/exp007 的数据）收紧到可行区采样，并增加 7 项接受检查把
约束残差、几何越界显式过滤（isoflux 残差降至 mean 0.172 / max ≤0.35）。
**本实验验证：修复数据质量问题后，三模型各自恢复到什么水平？**

三条预期（data_v4/README.md §1 计划）：
- **A'（X点+锚点，信息完整）应显著恢复**（v3 残差污染被过滤）；
- **B（coil，隐式含锚点信息）应接近 data_v2 水平**；
- **A（仅 X 点，缺锚点信息）仍应退化**，但退化应远小于 v3（v3 的 100×
  里混入了约束不可达污染）。

## 2. 方法

与 exp004/exp005 完全同配置，仅换数据（data_v4，3000 样本）：
- A（11ch，X点）、A'（13ch，X点+锚点）→ [exp006/model_a11ch_xpt](model_a11ch_xpt) /
  [model_a13ch_xa](model_a13ch_xa)
- B（11ch，coil）→ [exp007](../exp007_coil_input_v4/)
- 训练：N=500 seed 1，MSE/AdamW/ReduceLROnPlateau，同 exp004/005；
  评估 test 494（data_v4 test 接受 494/500）+ 锚点分桶。

**代码修复（训练侧）**：data_v4 锚点 Z≡0 → z-score 分母 std=0 → 0/0 NaN
→ A' 首训全 NaN（best_state 为 None 报错）。已修：
`data_dn_fno.py:129` 与 `data_dn_fno_coils.py:99` 用
`np.maximum(self.scalar_std, 1e-8)` 防护（常量通道映射为 0），非常量通道
std >> 1e-8 不受影响。data_v2/v3 训练路径不受影响（无常量通道）。

## 3. 结果（test，N=500，seed 1）

### 3a. v3 ↔ v4 同表对比（核心）

| 指标 | A X点 11ch v3→**v4** | A' X点+锚点 13ch v3→**v4** | B coil 11ch v3→**v4** | exp002 (data_v2) 参考 |
|---|---|---|---|---|
| rel L2 mean % | 31.09 → **3.38** | 21.72 → **0.442** | 12.47 → **0.499** | 0.303 ± 0.227 |
| rel L2 median % | 25.88 → 2.84 | 12.70 → 0.374 | 8.37 → 0.392 | 0.242 |
| rel L2 p95 % | 70.55 → 8.41 | 73.08 → 0.844 | 34.83 → 1.13 | — |
| RMSE phys (Wb) | 1.04e-2 → 1.28e-3 | 7.88e-3 → 1.59e-4 | 3.85e-3 → 1.76e-4 | 8.23e-5 |
| GS 残差比值 | 1.034 → 0.991 | 1.219 → 0.994 | 1.245 → 0.997 | 0.996 |
| find_critical 失败 | 21/499 → 7/494 | 48/499 → **0/494** | 35/499 → **0/494** | 1/500 |
| sep_mean 误差 (cm) | 36.2 → 1.71 | 32.4 → **0.23** | 34.6 → 0.30 | — |
| X 点误差 (cm) | x_lo 35→2.3 / x_up 31→1.9 | x_lo **0.50** / x_up **0.73** | x_lo 0.76 / x_up 1.36 | — |

（v3 列 = exp004/exp005 test 499；v4 列 = 本次评估 test 494）

**恢复倍数（rel L2 mean）**：A 9.2× ｜ A' **49×** ｜ B 25×。

### 3b. 训练（val，data_v4）

| 模型 | best val | epoch | 早停 |
|---|---|---|---|
| A | 3.1642% | 231 | e306 |
| A' | 0.4548% | 793 | e800（满 800） |
| B | 0.5101% | 792 | e800（满 800） |

### 3c. 锚点分桶（rel L2 mean %，[analysis/anchor_buckets.json](analysis/anchor_buckets.json)）

data_v4 锚点固定 Z=0、R∈[1.35,1.65]，因此 dist_ref（距基准锚点 (1.5,0)）
分桶全部落在 <0.2 m（R 向 ≤0.15 m），不再有 v3 的桶间区分度；quadrant 的
Z<0 桶为空（Z=0 对称）。有效维度只剩锚点-X 点距离：

| 锚点距最近 X 点 | n | A | A' | B |
|---|---|---|---|---|
| 0.3–0.6 m | 8 | 6.01 | **0.44** | 0.51 |
| ≥0.6 m | 486 | 3.34 | **0.44** | 0.50 |

三模型全部桶内平坦（误差不再随锚点偏移增大）——与 v3 的分桶梯度（25→35%）
形成对比：v4 的残差过滤消除了"锚点越远越难"的虚假趋势。

## 4. 结论

1. **约束不可达是 exp004 退化的主因**：A' 在 data_v4 上恢复到 0.442%——
   与 B（0.499%）同一水平、与 data_v2 基线（0.303%）同量级。v3 的 21.7%
   不是"N=500 学不会"的纯数据量问题，而是"真值场本身不符合输入约束"
   的污染问题（含锚点输入的模型在被污染的数据上无法学到一致映射）。
2. **A 的一对多退化仍在但大幅缩小**：3.38% vs A' 0.442% = 7.7×（v3 上是
   31.1 vs 21.7 = 1.4× 的"假对比"——当时两模型都被污染）。缺锚点信息的
   代价在干净数据上是 ~7.7×，不再是 100×。
3. **A' ≥ B（0.442 vs 0.499）**：v4 上直接输入锚点与隐含锚点（coil 电流）
   精度相当，A' 略优。exp005 "B 优于 A' 1.7×"的结论同样被污染数据反转。
4. 分离面/临界点几何全部恢复到 cm 量级以下（A' 0.23 cm、B 0.30 cm），
   find_critical 失败 0/494——v4 数据下模型输出物理上更干净。
5. 对 [exp004 §7](../exp004_xpoints_anchor_v3/README.md) 遗留问题的更新：
   **不是"信息完整也救不了 N=500"，而是"信息完整 + 数据物理合理 →
   N=500 就够回到 v2 水平"**。data_v2 与 data_v4 的残余差距（0.303 vs
   0.442%）来自 v4 更大的 X 点/锚点变化范围，属正常扩展代价。

## 5. 复现命令

```bash
PY="C:/Users/HP/.conda/envs/torch5060/python.exe"

# 训练（一键链：bash dn_fno_2608/scripts/run_exp006_007_train.sh）
"$PY" -u -m gs_pino_dn_fno_2608.train_dn_fno \
  --train-data dn_fno_2608/data_v4/train.npz --val-data dn_fno_2608/data_v4/val.npz \
  --n-train 500 --seed 1 --out-dir dn_fno_2608/experiments/exp006_xpoints_anchor_v4/model_a11ch_xpt
"$PY" -u -m gs_pino_dn_fno_2608.train_dn_fno --input-mode xa \
  --train-data dn_fno_2608/data_v4/train.npz --val-data dn_fno_2608/data_v4/val.npz \
  --n-train 500 --seed 1 --out-dir dn_fno_2608/experiments/exp006_xpoints_anchor_v4/model_a13ch_xa

# 评估
"$PY" -u -m gs_pino_dn_fno_2608.evaluate_dn_fno \
  --test-data dn_fno_2608/data_v4/test.npz \
  --checkpoint dn_fno_2608/experiments/exp006_xpoints_anchor_v4/model_a11ch_xpt/best.pt \
  --out-dir dn_fno_2608/experiments/exp006_xpoints_anchor_v4/model_a11ch_xpt
# （A' 同，--checkpoint model_a13ch_xa/best.pt）

# 锚点分桶（三模型：A、A'、exp007 的 B）
"$PY" -u -m gs_pino_dn_fno_2608.analyze_anchor_buckets \
  --test-data dn_fno_2608/data_v4/test.npz \
  --checkpoint a11ch=dn_fno_2608/experiments/exp006_xpoints_anchor_v4/model_a11ch_xpt/best.pt \
  --checkpoint a13ch=dn_fno_2608/experiments/exp006_xpoints_anchor_v4/model_a13ch_xa/best.pt \
  --checkpoint b11ch=dn_fno_2608/experiments/exp007_coil_input_v4/best.pt \
  --out-dir dn_fno_2608/experiments/exp006_xpoints_anchor_v4/analysis
```

## 6. 产物

```
exp006_xpoints_anchor_v4/
├── README.md
├── notes.md
├── model_a11ch_xpt/      # A：best.pt / history.json / args.json / metrics.json / figures/
├── model_a13ch_xa/       # A'：同上
└── analysis/             # anchor_buckets.json（三模型分桶）
```

训练日志：`dn_fno_2608/logs/exp006_*.log`

## 7. 偏差记录

1. **A' 训练 NaN（z-score 0/0）**：data_v4 锚点 Z≡0 → 该通道 std=0 →
   `(x−mean)/std = 0/0 = NaN` → loss/val NaN → best_state 恒为 None → 首训
   报错。修复：data_dn_fno.py / data_dn_fno_coils.py 分母加
   `np.maximum(std, 1e-8)`（常量通道 → 0）。data_v2/v3 无常量通道，
   结果不变；修复后 A' 重训正常（0.455%）。
2. 分桶退化（预期内）：锚点 Z=0 → quadrant Z<0 桶空；dist_ref 全 <0.2 m。
   脚本未改（兼容），README 标注桶语义。
