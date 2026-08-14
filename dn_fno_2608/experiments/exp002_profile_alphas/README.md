# exp002 — 剖面形状参数采样数据集（data_v2）探针

> 实验日期：2026-08-14
> 状态：**完成**（N=500，seed 1）
> 代码：`src/gs_pino_dn_fno_2608/`（generate_dn_dataset `--alpha-sampling` + 共享脚本通道数推断，基线行为不变）
> 数据：`dn_fno_2608/data_v2/`（6000/6000，见 [data_v2/README.md](../../data_v2/README.md)）
> 结论速览：data_v2 完全可学——rel L2 0.303% vs 基线 0.222%（差 1.36×，同量级），GS 残差比值 0.996

---

## 1. 动机

用户假设：论文基线数据集参数空间只有 7 个标量（Paxis/Ip/fvac + 4 X 点坐标），
剖面形状参数固定（alpha_m=1.0, alpha_n=2.0），可能限制了模型表达力。本实验把
剖面形状指数也纳入采样生成更复杂的数据集 data_v2（参数空间 9 维），先按
**小数据集策略**用 N=500 探针验证模型在新数据上的精度。

## 2. 方法

### 2.1 数据集差异（data/ → data_v2/）

```
基线 data/（9 通道）:   R, Z | Paxis, Ip, fvac | R_lo, Z_lo, R_up, Z_up          (7 标量)
data_v2（11 通道）:      R, Z | Paxis, Ip, fvac, alpha_m, alpha_n | R_lo...Z_up  (9 标量)
```

- alpha_m ~ U[1.0, 2.0]、alpha_n ~ U[1.5, 2.5]（剖面 shape = (1−ψn^αm)^αn）；
- 其余（paxis/Ip/fvac 范围、X 点抖动 ±0.02 m、求解器、接受准则、seeds）与基线完全一致；
- **模型唯一变化：in_channels 9 → 11**（lifting 层 +128 参数，共 4,211,777）；
- 训练超参与基线完全一致（MSE、AdamW、ReduceLROnPlateau、早停 75、batch 16、800 epoch、嵌套子集 seed 12345）。

### 2.2 代码改动（向后兼容）

| 文件 | 改动 |
|---|---|
| `generate_dn_dataset.py` | `--alpha-sampling`（默认关）+ `--max-retries`；修复拒绝时全参数重采样 |
| `{train,evaluate,visualize,latency}_dn_fno.py` | `build_model(in_channels=2+len(scalar_mean))`（旧 checkpoint 自动仍为 9） |
| `data_dn_fno.py` | 标量广播形状硬编码 `(7,)` → `(len(scalars),)`（compute_stats/拼接本就通用；旧数据仍得 7） |
| `model_dn_fno.py` | **零改动**（build_model 已参数化） |

## 3. 结果（test 500，N=500，seed 1）

| 指标 | exp002（data_v2） | 基线 N=500 (data) | 论文 N=500 |
|---|---|---|---|
| rel L2 mean % | **0.303 ± 0.227** | 0.222 ± 0.007 | 0.286 ± 0.034 |
| rel L2 median % | 0.242 | 0.167 | — |
| RMSE phys (Wb) | 8.23e-5 | 5.48e-5 | 8.53e-5 |
| sep_mean_cm | 0.333 | 0.150 | — |
| sep_hausdorff_cm | 1.709 | 1.305 | — |
| x_lo / x_up_cm | 0.711 / 0.468 | 0.494 / 0.477 | — |
| o_point_cm | 0.261 | 0.142 | — |
| find_critical 失败 | 1/500 | 0/500 | 0 |
| GS 残差比值 | 0.996 | 0.999 | 0.998 |

训练：best val rel L2 0.2809% @ epoch 788（基线 0.225% @ 756），11.7 min。

## 4. 结论

- **data_v2 完全可学**：N=500 达 0.303%，与基线 0.222% 差 1.36×、同量级；
  物理一致性 0.996、几何指标合理。数据复杂度 ↑（9D 参数空间）的精度代价很小，
  "原 7 维参数空间限制模型"的假设未被证伪——扩参数空间是低成本的。
- find_critical 1/500 失败 + 几何 P95 略升：新数据边界拓扑更复杂，尾部样本
  边界提取略差（轻微回退，可接受）。
- 按缩放律 ε ∝ N^-0.61 外推，data_v2 N=5000 预计 ~0.08–0.10%（基线 0.056%）。

## 5. 复现命令

```bash
PY="C:/Users/HP/.conda/envs/torch5060/python.exe"

# 数据（已生成，见 data_v2/README.md §7）
# 训练
"$PY" -u -m gs_pino_dn_fno_2608.train_dn_fno \
  --train-data dn_fno_2608/data_v2/train.npz --val-data dn_fno_2608/data_v2/val.npz \
  --n-train 500 --seed 1 --out-dir dn_fno_2608/experiments/exp002_profile_alphas

# 评估
"$PY" -u -m gs_pino_dn_fno_2608.evaluate_dn_fno \
  --test-data dn_fno_2608/data_v2/test.npz \
  --checkpoint dn_fno_2608/experiments/exp002_profile_alphas/best.pt \
  --out-dir dn_fno_2608/experiments/exp002_profile_alphas
```

## 6. 产物

```
exp002_profile_alphas/
├── README.md       # 本文档
├── notes.md        # 简短结论记录
├── best.pt         # 最优权重（in_channels=11）
├── history.json    # 训练曲线
├── args.json       # 训练参数
├── metrics.json    # test 评估指标（全 500 样本）
└── figures/        # fig1_best_worst_psi / fig2_field_stats / fig3_geometry_stats
                    #   + stats_per_sample.json（逐样本指标缓存）
```

训练日志：`dn_fno_2608/logs/exp002_profile_alphas_n500_s1.log`

## 7. 后续可做（供参考）

- **N 缩放**（1000/2000/5000）看 data_v2 上缩放律是否与基线一致；
- **多 seed 统计**（基线 N=500 有 3 seeds）；
- 若 N=500 精度可接受，可对比同 N 下两数据的精度差异定量评估"数据复杂度 ↑ 模型难度 ↑"。
