# exp002 — data_v2 剖面形状参数采样数据集探针

> 实验日期：2026-08-14 ｜ 状态：**完成**（N=500，seed 1）
> 数据：`data_v2/`（alpha_m/alpha_n 采样，6000/6000）
> 对照：基线 data/ N=500（0.222%）
> 结论速览：**data_v2 完全可学**——rel L2 0.303% vs 基线 0.222%（差 1.36×，同量级），
> GS 残差比 0.996；扩参数空间是低成本的数据增强方向

## 1. 目标

假设：基线 7 维参数空间（剖面形状固定）可能限制模型表达力。data_v2 把剖面
形状指数 alpha_m/alpha_n 纳入采样（参数空间 9 维），本实验用 N=500 探针验证
模型在新数据上的精度（用户小数据集策略）。

## 2. 输入通道（11 通道）

| # | 通道 | 内容 |
|---|---|---|
| 1–2 | R, Z | 网格坐标（[-1,1]） |
| 3–7 | Paxis, Ip, fvac, **alpha_m, alpha_n** | 运行参数 + 剖面指数（z-score；基线 5 个标量） |
| 8–11 | R_lo, Z_lo, R_up, Z_up | X 点坐标 |

模型唯一变化：in_channels 9→11（lifting +128 参数，共 4,211,777）；训练超参
与基线完全一致。代码改动全部向后兼容（`--alpha-sampling` 默认关；通道数由
`2+len(scalar_mean)` 推断，旧 checkpoint 自动仍为 9）。

## 3. 训练设置

同基线（MSE / AdamW / ReduceLROnPlateau / 早停 75 / batch 16 / 800 epoch /
嵌套子集 seed 12345）。best val 0.2809% @ epoch 788，11.7 min。

## 4. 结果（test 500）

| 指标 | exp002 (data_v2) | 基线 N=500 (data) |
|---|---|---|
| rel L2 mean % | **0.303 ± 0.227** | 0.222 ± 0.007 |
| rel L2 median % | 0.242 | 0.167 |
| RMSE phys (Wb) | 8.23e-5 | 5.48e-5 |
| sep_mean (cm) | 0.333 | 0.150 |
| x_lo / x_up (cm) | 0.711 / 0.468 | 0.494 / 0.477 |
| o_point (cm) | 0.261 | 0.142 |
| find_critical 失败 | 1/500 | 0/500 |
| GS 残差比 | 0.996 | 0.999 |

## 5. 结论

- **data_v2 完全可学**：数据复杂度 ↑（9D 参数空间）的精度代价很小（1.36×），
  扩参数空间是低成本方向；
- find_critical 1/500 + 几何 P95 略升：边界拓扑更复杂的尾部样本略差（可接受）；
- 按缩放律外推 data_v2 N=5000 预计 ~0.08–0.10%（基线 0.056%）。

## 6. 复现命令

```bash
PY="C:/Users/HP/.conda/envs/torch5060/python.exe"

"$PY" -u -m gs_pino_dn_fno_2608.train_dn_fno \
  --train-data dn_fno_2608/data_v2/train.npz --val-data dn_fno_2608/data_v2/val.npz \
  --n-train 500 --seed 1 --out-dir dn_fno_2608/experiments/exp002_profile_alphas
"$PY" -u -m gs_pino_dn_fno_2608.evaluate_dn_fno \
  --test-data dn_fno_2608/data_v2/test.npz \
  --checkpoint dn_fno_2608/experiments/exp002_profile_alphas/best.pt \
  --out-dir dn_fno_2608/experiments/exp002_profile_alphas
```

## 7. 产物

`best.pt`（in_channels=11）/ `history.json` / `args.json` / `metrics.json` /
`figures/`。训练日志：`logs/exp002_profile_alphas_n500_s1.log`。

## 8. 偏差记录

代码改动（全部向后兼容）：generate_dn_dataset `--alpha-sampling` + `--max-retries`
（拒绝时全参数重采样）；data_dn_fno 标量广播 `(7,)`→`(len(scalars),)`；训练/
评估/可视化/延迟脚本 `build_model(in_channels=2+len(scalar_mean))`；model_dn_fno
零改动。
