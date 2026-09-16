# exp008 — MAST 混合位形训练（DN+SN，config 输入）

> 实验日期：2026-08-17 ｜ 状态：**完成**（N=500，seed 1）
> 数据：`data_v5/`（MAST DN+SN 分开落盘，各 3000，逗号拼接混合）
> 对照：exp009（专职单一位形分开训练）
> 结论速览：**混合训练在两配置上都工作**（整体 test rel L2 0.935%），代价相对
> 专职模型 +26~35%（DN 桶 0.667% vs 0.492%、SN 桶 1.203% vs 0.953%）；而专职
> 模型**跨配置外推崩溃**（DN→SN 253%、SN→DN 4.0e8%，exp009）——config 通道
> 被模型有效利用，混合训练换来的是**跨位形泛化**

## 1. 目标

data_v5 是位形泛化的第一步：单一位形训练的模型能否用混合位形数据学到
跨位形（DN↔SN）的一致映射？设计决策（用户拍板）：混合位形先做 DN+SN；
真实装置用 MAST；数据按配置分开生成。

三个问题：① 混合模型整体水平 vs 专职；② 分桶是否接近专职（跨配置泛化代价）；
③ config 通道是否被利用（SN 的 up=(0,0) 占位由 config 区分）。

## 2. 输入通道（14 通道，xa+config）

| # | 通道 |
|---|---|
| 1–2 | R, Z |
| 3–7 | Ip, paxis, fvac, alpha_m, alpha_n |
| 8–11 | R_lo, Z_lo, R_up, Z_up（SN 的 up 恒 (0,0) 占位） |
| 12–13 | R_anc, Z_anc |
| 14 | **config**（0/1，z-score；`--config-input`） |

训练：N=500（拼接 4000 样本上嵌套取 500）seed 1；stats 来自拼接后全量 train
pool（DN/SN 共享 z-score）；test 分桶 = dn / sn 单独 + 拼接整体。

## 3. 结果

### 3a. 混合 vs 专职（核心对比，exp009）

| 指标 | 混合 DN 桶 | 专职 DN | 混合 SN 桶 | 专职 SN |
|---|---|---|---|---|
| rel L2 mean % | **0.667** | **0.492** | **1.203** | **0.953** |
| rel L2 median % | 0.560 | 0.411 | 1.033 | 0.818 |
| rel L2 p95 % | 1.289 | 1.010 | 2.518 | 1.908 |
| RMSE phys (Wb) | 2.74e-4 | 1.85e-4 | 5.24e-4 | 4.31e-4 |
| GS 残差比 | 0.9888 | 0.9876 | 0.9836 | 0.9906 |
| find_critical 失败 | 0/500 | 0/500 | 0/500 | 0/500 |
| X 点误差 lo/up (cm) | 3.19 / 2.37 | 2.14 / 1.91 | 1.06 / — | 0.97 / — |
| sep_mean (cm) | 0.56 | 0.40 | 0.43 | 0.34 |
| sep 面积相对误差 (%) | 0.68 | 0.47 | 0.40 | 0.33 |
| O 点误差 (cm) | 0.48 | 0.34 | 0.61 | 0.42 |

（SN 无上 X 点 → x_up NaN；几何指标 v2：真值基准配对 + 射线法分离面，§5.1）

### 3b. 训练

混合 xa+config 14ch：best val 0.8789% @ epoch 797，14.3 min。
整体 test（n=1000）：rel L2 0.935%、RMSE 3.99e-4 Wb、GS 残差比 0.986。
分桶明细 `model_a14ch_xa_mix/analysis/config_buckets.json`；figures_dn/、figures_sn/。

## 4. 结论

1. **混合训练可行**：单一模型在 DN+SN 拼接 test 上 0.935%、find_critical 0 失败；
2. **混合 vs 专职代价 +26~35%**；SN 桶绝对误差整体高于 DN 桶（两种训练一致，
   SN 分离面约束自由度少、X 点位置对场更敏感）；
3. **config 通道被有效利用**：专职模型跨配置外推完全失败而混合模型两桶健康
   ——up=(0,0) 占位 + config 的组合让模型区分位形并切换映射；
4. 几何与 rel L2 同量级、两桶同趋势（X 点 1–3 cm、sep_mean <0.6 cm）；
   SN 桶 X 点略优于 DN（单 X 点无配对歧义），与 rel L2 呈弱反相关。

## 5. 复现命令

```bash
PY="C:/Users/HP/.conda/envs/torch5060/python.exe"
# 训练链：bash dn_fno_2608/scripts/run_exp008_009_train.sh
# 评估+可视化：bash dn_fno_2608/scripts/run_exp008_009_eval.sh
"$PY" -u -m gs_pino_dn_fno_2608.train_dn_fno --input-mode xa --config-input \
  --train-data dn_fno_2608/data_v5/dn/train.npz,dn_fno_2608/data_v5/sn/train.npz \
  --val-data dn_fno_2608/data_v5/dn/val.npz,dn_fno_2608/data_v5/sn/val.npz \
  --n-train 500 --seed 1 --out-dir dn_fno_2608/experiments/exp008_mixed_configs/model_a14ch_xa_mix
```

## 6. 产物

`model_a14ch_xa_mix/`（best.pt/history/args/metrics + analysis/config_buckets.json
+ figures_dn/、figures_sn/）。日志：`logs/exp008_*.log`。

## 7. 偏差与修复记录

1. **visualize config 广播 bug**：exp008 ckpt stats 含 config 通道（12 维标量），
   visualize 重建 dataset 未传 use_config → 广播失败。修复：stats 过滤（排除
   input_mode/config_input）+ `use_config=bool(ckpt["config_input"])`（exp009
   无 config 路径不受影响）。
2. **visualize 不支持逗号多文件**：figures 按配置分开生成。
3. **几何指标 v2（2026-08-17，用户报告 fig3 大误差后修复）**：旧版
   geometry_metrics 从 find_critical 全候选按 Z 符号取第一个点配对（MAST 无墙
   → 混入 5–6 个真空假鞍点，配对失败时报数十 cm 假误差）、SN 的 psi_bndry 错取
   "前两个 xpt 均值"、分离面追踪依赖 matplotlib 闭合环（pred 场等高线穿出网格
   → fallback 抓全域散点 → sep_mean 虚报 25–30 cm）。v2：X 点按数据集真值
   （xpts_actual/o_point/axes[2]）配对（psi≥bnd−2e-3 + 距离 ≤0.2 m，漏检走
   ±0.35 m 局部鞍点找回，仍无则 NaN）；psi_bndry = 配对 X 点 psi 均值（DN）/
   唯一 X 点 psi（SN）；分离面 = 磁轴射线法（首穿 level，天然闭合）。旧版备份
   eval*/metrics_v1.json、figures*/stats_per_sample_v1.json。X 点仍不可定位率：
   DN ~10%、SN ~16%（FNO 平滑场中鞍点消失，非配对错误）。
