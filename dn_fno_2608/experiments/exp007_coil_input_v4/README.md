# exp007 — data_v4 上 coil 电流输入复测（coil 11ch，约束可达）

> 实验日期：2026-08-14 ｜ 状态：**完成**（N=500，seed 1）
> 数据：`data_v4/`（test 494）
> 对照：exp005（coil on data_v3，12.47%）、exp006 的 A'（X点+锚点 0.442%）
> 结论速览：coil 输入在 data_v4 上恢复到 **test rel L2 0.499%**（v3 的 12.47% →
> 25× 恢复，达 data_v2 水平）；与 A'（0.442%）同水平、略逊（1.13×）——
> **exp005 "B 优于 A' 1.7×"在干净数据上不成立**

## 1. 目标

exp005 在 data_v3 上结论"coil 隐含锚点信息、相对最可靠"，但 v3 的 isoflux
约束不可达意味着所有模型都学到了被污染的目标。在 data_v4（约束可达）上复测：
① coil 能恢复到什么水平；② "B 优于 A' 1.7×"是否仍成立。

## 2. 输入通道（11 通道）

R, Z + Ip, paxis, fvac, alpha_m, alpha_n + **I_P1L, I_P1U, I_P2L, I_P2U**（coil 电流）。

与 exp005 完全同配置仅换数据（N=500 seed 1，评估 test 494 + 分桶）。

## 3. 结果（test 494）

| 指标 | B coil 11ch v3→**v4** | exp003 (data_v2) 参考 |
|---|---|---|
| rel L2 mean % | 12.47 → **0.499** | 0.412 ± 0.328 |
| rel L2 median % | 8.37 → 0.392 | 0.324 |
| rel L2 p95 % | 34.83 → 1.13 | — |
| RMSE phys (Wb) | 3.85e-3 → 1.76e-4 | 1.12e-4 |
| GS 残差比 | 1.245 → 0.997 | 0.995 |
| find_critical 失败 | 35/499 → **0/494** | 0/500 |
| sep_mean (cm) | 34.6 → 0.30 | — |
| X 点误差 (cm) | — | x_lo 0.76 / x_up 1.36 |

训练：best val 0.5101% @ e792（满 800 未早停）。
三方对比与分桶见 [exp006/README.md §3-4](../exp006_xpoints_anchor_v4/README.md)。

## 4. 结论

1. **coil 恢复到 data_v2 水平**（0.499% vs exp003 0.412%）——v3 的 12.47%
   退化几乎全部来自数据污染，而非输入信息不足；
2. **A' ≥ B（0.442 vs 0.499）**：显式锚点与隐含锚点（Tikhonov 电流解）精度
   相当；exp005 的 1.7× 优势被污染数据制造（v3 中 A' 受残差污染更深）；
3. 与 exp006 合观：**约束不可达是 exp004/005 退化的决定性因素**；修复后三种
   输入模式全部回到 ~0.5% 量级（缺锚点信息 7.7×、coil vs 显式锚点 ~1.1×）。

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

## 6. 产物

`best.pt` / `history.json` / `args.json` / `metrics.json` / `figures/`。
训练日志：`logs/exp007_*.log`。

## 7. 偏差记录

无独立偏差（与 exp006 同一数据修复；训练/评估命令与 exp005 完全一致）。
