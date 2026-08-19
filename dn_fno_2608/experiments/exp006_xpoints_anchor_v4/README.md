# exp006 — data_v4 上 X 点输入复测（X点 vs X点+锚点，约束可达）

> 实验日期：2026-08-14 ｜ 状态：**完成**（N=500，seed 1，三模型对比）
> 数据：`data_v4/`（可行区采样 + 7 项接受约束，test 494）
> 对照：exp004/exp005（data_v3 上的同一批模型）
> 结论速览：**约束不可达（而非数据量）是 exp004 退化的主因**——data_v4 上
> A'（X点+锚点 13ch）21.72%→**0.442%**（49× 恢复）、B（coil 11ch）12.47%→
> 0.499%（exp007）、A（X点 11ch）31.09%→3.38%。A 仍落后 A' 7.7×（锚点缺失的
> 一对多映射仍在），但 100× 级退化已消除。

## 1. 目标

exp004/005 在 data_v3 上观察到三模型全部大幅退化，当时归因数据量瓶颈。但
data_v3 有两个未被检查的缺陷：**isoflux 约束不可达**（残差 mean 0.50/max 2.10，
真值分离面大多不过锚点）+ **采样越界**。data_v4 修复后本实验验证：
**三模型各自恢复到什么水平？**

预期：A'（信息完整）显著恢复；B（coil 隐式含锚点）接近 data_v2 水平；
A（仅 X 点，缺锚点信息）仍退化但远小于 v3。

## 2. 输入通道

| 模型 | 通道 | 目录 |
|---|---|---|
| A（11ch） | R, Z + 5 params + R_lo, Z_lo, R_up, Z_up | `model_a11ch_xpt/` |
| A'（13ch） | A + R_anc, Z_anc | `model_a13ch_xa/` |
| B（11ch coil） | R, Z + 5 params + I_P1L..I_P2U | exp007 |

同 exp004/005 配置，仅换数据（data_v4）；N=500 seed 1，评估 test 494。

## 3. 结果（v3 ↔ v4 同表对比，核心）

| 指标 | A X点 11ch v3→**v4** | A' X点+锚点 13ch v3→**v4** | B coil 11ch v3→**v4** |
|---|---|---|---|
| rel L2 mean % | 31.09 → **3.38** | 21.72 → **0.442** | 12.47 → **0.499** |
| rel L2 median % | 25.88 → 2.84 | 12.70 → 0.374 | 8.37 → 0.392 |
| rel L2 p95 % | 70.55 → 8.41 | 73.08 → 0.844 | 34.83 → 1.13 |
| RMSE phys (Wb) | 1.04e-2 → 1.28e-3 | 7.88e-3 → 1.59e-4 | 3.85e-3 → 1.76e-4 |
| GS 残差比 | 1.034 → 0.991 | 1.219 → 0.994 | 1.245 → 0.997 |
| find_critical 失败 | 21/499 → 7/494 | 48/499 → **0/494** | 35/499 → **0/494** |
| sep_mean (cm) | 36.2 → 1.71 | 32.4 → **0.23** | 34.6 → 0.30 |
| X 点误差 (cm) | 35→2.3 / 31→1.9 | **0.50** / **0.73** | 0.76 / 1.36 |

**恢复倍数（rel L2 mean）**：A 9.2× ｜ A' **49×** ｜ B 25×（exp007）。

训练（val）：A 3.1642% @e231（早停 e306）｜ A' 0.4548% @e793（跑满）｜
B 0.5101% @e792（跑满）。

**锚点分桶**（data_v4 锚点 Z≡0，只剩"锚点-X 点距离"维度）：0.3–0.6 m（n=8）
A 6.01 / A' 0.44 / B 0.51；≥0.6 m（n=486）A 3.34 / A' 0.44 / B 0.50——三模型
全部桶内平坦（v3 的"锚点越远越难"趋势是污染伪差）。

## 4. 结论

1. **约束不可达是 exp004 退化主因**：A' 恢复到 0.442%（与 B 0.499% 同水平、
   与 data_v2 0.303% 同量级）——v3 的 21.7% 是"真值场本身不符合输入约束"
   的污染问题，不是纯数据量问题；
2. **A 的一对多退化仍在但缩小**：3.38% vs A' 0.442% = 7.7×（v3 上是假对比）——
   缺锚点信息的真实代价 ~7.7×；
3. **A' ≥ B（0.442 vs 0.499）**：显式锚点与隐含锚点（coil 电流）精度相当，
   exp005 "B 优于 A' 1.7×"被污染数据制造、在干净数据上不成立；
4. 几何全部恢复到 cm 以下（A' 0.23 cm）、find_critical 0/494；
5. 更新 exp004 §7 遗留问题：**信息完整 + 数据物理合理 → N=500 就够回到 v2 水平**。

## 5. 复现命令

```bash
PY="C:/Users/HP/.conda/envs/torch5060/python.exe"
# 一键链：bash dn_fno_2608/scripts/run_exp006_007_train.sh

"$PY" -u -m gs_pino_dn_fno_2608.train_dn_fno \
  --train-data dn_fno_2608/data_v4/train.npz --val-data dn_fno_2608/data_v4/val.npz \
  --n-train 500 --seed 1 --out-dir dn_fno_2608/experiments/exp006_xpoints_anchor_v4/model_a11ch_xpt
"$PY" -u -m gs_pino_dn_fno_2608.train_dn_fno --input-mode xa \
  --train-data dn_fno_2608/data_v4/train.npz --val-data dn_fno_2608/data_v4/val.npz \
  --n-train 500 --seed 1 --out-dir dn_fno_2608/experiments/exp006_xpoints_anchor_v4/model_a13ch_xa
# 评估（evaluate_dn_fno，--checkpoint 各 best.pt）
# 分桶：analyze_anchor_buckets --checkpoint a11ch=... a13ch=... b11ch=<exp007>/best.pt
```

## 6. 产物

`model_a11ch_xpt/`、`model_a13ch_xa/`（各含 best.pt/history/args/metrics/figures）+
`analysis/anchor_buckets.json`。训练日志：`logs/exp006_*.log`。

## 7. 偏差记录

1. **A' 训练 NaN（z-score 0/0）**：data_v4 锚点 Z≡0 → 该通道 std=0 → 0/0=NaN →
   best_state 恒 None 首训报错。修复：data_dn_fno.py / data_dn_fno_coils.py 分母
   `np.maximum(std, 1e-8)`（常量通道→0；data_v2/v3 无常量通道不受影响）。
   修复后 A' 重训正常（0.455%）。
2. 分桶退化（预期内）：锚点 Z=0 → quadrant Z<0 桶空、dist_ref 全 <0.2 m。
