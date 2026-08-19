# exp009 — 专职单一位形训练（DN-only / SN-only 对照）

> 实验日期：2026-08-17 ｜ 状态：**完成**（N=500，seed 1）
> 数据：`data_v5/`（dn/、sn/ 分开，各 3000）
> 对照：exp008（混合位形训练，config 输入）
> 结论速览：**专职模型在自身配置上最优**（DN 0.492%、SN 0.953%），但**跨配置
> 外推崩溃**（DN→SN 253%、SN→DN 4.0e8%）——单一位形训练没有跨位形泛化；
> 混合训练（exp008）以 +26~35% 代价换来两配置都可用

## 1. 目标

exp008 混合训练的核心对照：专职单一位形（DN/SN 各自训练）的性能基线。量化和
横跨两个问题：① 专职 vs 混合的代价/收益；② 配置外推（跨配置交叉评估）。

## 2. 输入通道（13 通道，xa 无 config）

R, Z + Ip, paxis, fvac, alpha_m, alpha_n + R_lo, Z_lo, R_up, Z_up + R_anc, Z_anc。

- 专职 DN 模型：只见过 DN 样本（上 X 点通道非零）；
- 专职 SN 模型：只见过 SN 样本（上 X 点通道恒 0 占位——std 防护下自动映射为
  0 通道，模型学习忽略它）；
- 训练：N=500 seed 1，同 exp006 惯例；`model_dn_13ch` 用 dn/*.npz、
  `model_sn_13ch` 用 sn/*.npz。

## 3. 结果

### 3a. 专职基线（自身 test）

| 指标 | 专职 DN | 专职 SN |
|---|---|---|
| rel L2 mean % | **0.492** | **0.953** |
| rel L2 median % | 0.411 | 0.818 |
| rel L2 p95 % | 1.010 | 1.908 |
| RMSE phys (Wb) | 1.85e-4 | 4.31e-4 |
| GS 残差比 | 0.9876 | 0.9906 |
| find_critical 失败 | 0/500 | 0/500 |
| X 点误差 lo/up (cm) | 2.14 / 1.91 | 0.97 / — |
| sep_mean (cm) | 0.40 | 0.34 |
| sep 面积相对误差 (%) | 0.47 | 0.33 |
| O 点误差 (cm) | 0.34 | 0.42 |

训练：DN best val 0.4903%（e795）、SN best val 0.8834%（e797），各 11.1 min。

### 3b. 交叉评估（配置外推，rel L2 mean %）

| 模型 | DN test | SN test |
|---|---|---|
| 专职 DN | 0.492 | **253.4**（外推崩溃） |
| 专职 SN | **4.0e8**（完全失效） | 0.953 |
| 混合（exp008） | 0.667 | 1.203 |

交叉详情：DN 模型喂 SN 样本 rel L2 253%（RMSE 1.0e-1 Wb、GS 残差比 2.27）；
SN 模型喂 DN 样本 rel L2 4.0e8%（RMSE 1.8e5 Wb、GS 残差比 1024，输出无物理
意义）。交叉评估仅量化外推失败，不参与任何指标汇总。

## 4. 结论

1. **专职基线成立**：自身 test 上优于混合模型（0.492 vs 0.667%、0.953 vs
   1.203%）——单一位形内专职学习更充分；
2. **单一位形训练没有跨位形泛化**：模型学到的是位形内的映射，而非物理一致的
   "任意位形 → GS 平衡"映射；
3. **混合训练是跨位形泛化的必要条件**（与 exp008 §4 互证）；
4. 生产决策：单一固定位形 → 专职更优（省 +26~35% 误差）；多位形/切换 →
   混合是唯一可行解。

## 5. 复现命令

```bash
PY="C:/Users/HP/.conda/envs/torch5060/python.exe"
# 训练链：bash dn_fno_2608/scripts/run_exp008_009_train.sh
"$PY" -u -m gs_pino_dn_fno_2608.train_dn_fno --input-mode xa \
  --train-data dn_fno_2608/data_v5/dn/train.npz --val-data dn_fno_2608/data_v5/dn/val.npz \
  --n-train 500 --seed 1 --out-dir dn_fno_2608/experiments/exp009_split_configs/model_dn_13ch
# （model_sn_13ch 同理换 sn 文件；评估分别用 dn/test.npz、sn/test.npz）
```

## 6. 产物

`model_dn_13ch/`、`model_sn_13ch/`（各含 best.pt/history/args/metrics + 分桶评估
eval_dn/、eval_sn/ + figures）。日志：`logs/exp009_*.log`。

## 7. 偏差记录

无（训练/评估/可视化全程无异常；exp008 的 visualize config 修复对 exp009 无
影响——exp009 checkpoint 无 config_input 字段，条件分支关闭）。
