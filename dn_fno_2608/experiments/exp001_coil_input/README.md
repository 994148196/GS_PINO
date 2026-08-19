# exp001 — coil 电流作为输入（替代 X 点坐标）

> 实验日期：2026-08-14 ｜ 状态：**完成**（N=500，seed 1）
> 数据：`data/`（论文基线） ｜ 对照：基线 X 点输入（0.222%）
> 结论速览：**coil 电流输入可行**——rel L2 0.326% vs 基线 0.222%（差 1.5×，同量级），
> 几何健康（0/500 find_critical 失败）、GS 残差比 0.995

## 1. 目标

论文基线用 4 个 X 点坐标作输入标量，但工程中 X 点是**设计目标**、线圈电流才是
**执行量**（freegs 控制环反解的物理输入）。本实验把 4 个 X 点坐标通道替换为
4 个自由边界控制线圈电流（P1L/P1U/P2L/P2U），验证精度代价。

## 2. 输入通道（9 通道，与基线同构仅换 4 个标量）

| # | 通道 | 内容 |
|---|---|---|
| 1–2 | R, Z | 网格坐标（[-1,1]） |
| 3–5 | Paxis, Ip, fvac | 运行参数（z-score） |
| 6–9 | **I_P1L, I_P1U, I_P2L, I_P2U** | 4 控制线圈电流（z-score；基线为 R_lo, Z_lo, R_up, Z_up） |

通道数不变 → 模型架构完全不变（FNO2d2608，in_channels=9，4.21M 参数）。
coil 电流来自数据已有字段 `coil_currents`（生成时预留），无需重新生成数据。

## 3. 训练设置

MSE / AdamW(1e-3, 1e-4) / ReduceLROnPlateau(20, ×0.5) / 早停 75 / batch 16 /
800 epoch / 嵌套子集 seed 12345。best val 0.327% @ epoch 783（基线 0.225% @ 756），
11.0 min。

## 4. 结果（test 500）

| 指标 | coil 输入 | 基线 (X 点) | 论文 N=500 |
|---|---|---|---|
| rel L2 mean % | **0.326** | 0.222 ± 0.007 | 0.286 ± 0.034 |
| rel L2 median % | 0.257 | 0.167 | — |
| RMSE phys (Wb) | 8.31e-5 | 5.48e-5 | 8.53e-5 |
| sep_mean (cm) | 0.267 | 0.150 | — |
| x_lo / x_up (cm) | 0.386 / 1.007 | 0.494 / 0.477 | — |
| o_point (cm) | 0.254 | 0.142 | — |
| find_critical 失败 | **0/500** | 0/500 | 0 |
| GS 残差比 | 0.995 | 0.995 | — |

## 5. 结论

1. **coil 电流输入可行**：N=500 即达 0.326%，毫秒级代理可用；
2. **比 X 点坐标差约 1.5×**：X 点坐标是平衡边界的直接拓扑锚点（易学）；coil
   电流是控制环反解结果 + 4 电流间存在控制环冗余，信息密度更低、映射非线性更强；
3. 按用户策略不跑更大数据集；若追平基线按缩放律 ε ∝ N^-0.61 估需 N≈1500–2000。

## 6. 复现命令

```bash
PY="C:/Users/HP/.conda/envs/torch5060/python.exe"

"$PY" -u -m gs_pino_dn_fno_2608.train_dn_fno_coils \
  --train-data dn_fno_2608/data/train.npz --val-data dn_fno_2608/data/val.npz \
  --n-train 500 --seed 1 --out-dir dn_fno_2608/experiments/exp001_coil_input
"$PY" -u -m gs_pino_dn_fno_2608.evaluate_dn_fno_coils \
  --test-data dn_fno_2608/data/test.npz \
  --checkpoint dn_fno_2608/experiments/exp001_coil_input/best.pt \
  --out-dir dn_fno_2608/experiments/exp001_coil_input
```

## 7. 产物

`best.pt`（含 stats + input_mode="coils"）/ `history.json` / `args.json` /
`metrics.json` / `figures/`（fig1-3 + stats_per_sample.json）。
训练日志：`logs/exp001_coil_n500_s1.log`。

## 8. 偏差记录

coil 变体使用独立文件（data/train/evaluate_dn_fno_coils.py），基线代码零改动
（本实验的 `*_coils` 镜像脚本为 **v1 几何**（find_critical 配对 bug），仅用于
exp001/003/005/007 复现；后续实验一律用主脚本 `--input-mode coils`，v2 几何）。
