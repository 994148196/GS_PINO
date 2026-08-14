# exp007 — data_v4 上 coil 电流输入复测（coil 11ch，约束可达）

> 实验日期：2026-08-14
> 状态：**完成**（N=500，seed 1）
> 代码：`src/gs_pino_dn_fno_2608/train_dn_fno_coils.py`（与 exp003/exp005 同路径零改动）
> 数据：`dn_fno_2608/data_v4/`（可行区采样 + 7 项接受约束，见 [data_v4/README.md](../../data_v4/README.md)）
> 结论速览：coil 输入在 data_v4 上恢复到 **test rel L2 0.499%**（v3 的 12.47% →
> 25× 恢复，达到 data_v2 水平 0.412%）；与 exp006 的 A'（X点+锚点 0.442%）
> 同水平、略逊（1.13×）。exp005 "B 优于 A' 1.7×"在干净数据上不成立。

---

## 1. 动机

exp005 在 data_v3 上结论：coil 输入隐含锚点信息、相对最可靠（12.5% vs A'
21.7%）。但 data_v3 的 isoflux 约束不可达缺陷（残差 mean 0.50 / max 2.10）
意味着**真值分离面大多不过锚点**——所有输入模式都学到了被污染的目标。
exp006/exp007 在 data_v4（7 项接受约束，残差 mean 0.172 / max ≤0.35）上
复测，检验两个问题：

1. coil 输入能恢复到什么水平？
2. exp005 的"B 优于 A' 1.7×"是否仍成立？

## 2. 方法

与 exp005 完全同配置，仅换数据（data_v4）：
- B（11ch：R, Z | 5 params | 4 coil 电流）→ [exp007/best.pt](best.pt)
- 训练：N=500 seed 1，与 exp005 同超参；评估 test 494 + 分桶（与 exp006 同表）。
- 代码修复：data_dn_fno_coils.py 的 z-score 分母加 `np.maximum(std, 1e-8)`
  防护（data_v4 无常量通道，本模型本不受影响，与 exp006 同步修复）。

## 3. 结果（test 494，N=500，seed 1）

| 指标 | B coil 11ch v3→**v4** | exp003 (data_v2) 参考 | exp002 (data_v2) 参考 |
|---|---|---|---|
| rel L2 mean % | 12.47 → **0.499** | 0.412 ± 0.328 | 0.303 ± 0.227 |
| rel L2 median % | 8.37 → 0.392 | 0.324 | 0.242 |
| rel L2 p95 % | 34.83 → 1.13 | — | — |
| RMSE phys (Wb) | 3.85e-3 → 1.76e-4 | 1.12e-4 | 8.23e-5 |
| GS 残差比值 | 1.245 → 0.997 | 0.995 | 0.996 |
| find_critical 失败 | 35/499 → **0/494** | 0/500 | 1/500 |
| sep_mean 误差 (cm) | 34.6 → 0.30 | — | — |
| X 点误差 (cm) | — | x_lo 0.76 / x_up 1.36 | — |

训练：best val 0.5101% @ e792（满 800 epoch 未早停）。

三方对比与分桶（A X点 / A' X点+锚点 / B coil）见
[exp006/README.md §3-4](../exp006_xpoints_anchor_v4/README.md)。

## 4. 结论

1. **coil 输入恢复到 data_v2 水平**（0.499% vs exp003 的 0.412%）——v3 的
   12.47% 退化几乎全部来自数据污染，而非输入信息不足。data_v4 上
   find_critical 失败 0/494、GS 残差比值 0.997，几何全在 cm 以下。
2. **A' ≥ B（0.442 vs 0.499）**：直接输入锚点坐标与隐含锚点（Tikhonov
   电流解）精度相当，A' 略优 1.13×。exp005 的 1.7× 优势被污染数据制造
   （v3 中 A' 受残差污染更深），在干净数据上不成立。
3. 与 exp006 合观：**约束不可达（数据物理合理性）是 exp004/005 退化的
   决定性因素**；修复后三种输入模式全部回到 ~0.5% 量级，差异缩小到
   信息论层面（缺锚点信息 7.7×、coil vs 显式锚点 ~1.1×）。

## 5. 复现命令

```bash
PY="C:/Users/HP/.conda/envs/torch5060/python.exe"

"$PY" -u -m gs_pino_dn_fno_2608.train_dn_fno_coils \
  --train-data dn_fno_2608/data_v4/train.npz --val-data dn_fno_2608/data_v4/val.npz \
  --n-train 500 --seed 1 --out-dir dn_fno_2608/experiments/exp007_coil_input_v4
"$PY" -u -m gs_pino_dn_fno_2608.evaluate_dn_fno_coils \
  --test-data dn_fno_2608/data_v4/test.npz \
  --checkpoint dn_fno_2608/experiments/exp007_coil_input_v4/best.pt \
  --out-dir dn_fno_2608/experiments/exp007_coil_input_v4
```

分桶对比见 exp006 README §5（analyze_anchor_buckets，三模型同表）。

## 6. 产物

```
exp007_coil_input_v4/
├── README.md
├── notes.md
├── best.pt / history.json / args.json / metrics.json
└── figures/              # fig1_best_worst_psi / fig2_field_stats / fig3_geometry_stats
```

训练日志：`dn_fno_2608/logs/exp007_*.log`

## 7. 偏差记录

- 无独立偏差（与 exp006 同一数据修复；训练/评估命令与 exp005 完全一致）。
