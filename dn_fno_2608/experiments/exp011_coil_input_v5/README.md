# exp011 — coil 电流输入跨位形（DN+SN）混合训练：端到端 psi 生成

> 实验日期：2026-08-17 ｜ 状态：**完成**（N=500，seed 1）
> 数据：`data_v5/`（MAST DN+SN，逗号拼接混合）
> 对照：exp008（xa+config 14ch）、exp010（xa 13ch 无 config）、exp009（专职
> xa 13ch）、exp007（coil 11ch data_v4 单一位形）
> 结论速览：**不显式告诉模型位形/拓扑，只给 (R,Z + 5 物理参数 + 11 线圈电流)，
> 模型直接生成物理正确的 psi 场**——混合训练下 DN/SN 两桶均健康，与 xa 输入
> 同量级；位形识别是隐含能力（信息完全由线圈电流承载），不是显式输入

## 1. 目标

真实装置上可测量的量是**线圈电流与工程参数**；X 点坐标、锚点、分离面都是后验
推导量（freegs 也是从线圈电流反解平衡）。前面实验（exp006-010）输入含 X 点/
锚点坐标或 config 标签，本质是"把解的一部分告诉模型"。本实验问：

**只输入物理参数 + coil 电流 + 坐标，模型能否直接生成 psi 分布？**

前提已实证：DN/SN 的 5 个物理参数分布几乎重合（不可区分）；11 个线圈电流
系统性分离（最强通道 P3U 分离度 3.29 std、单通道阈值判别 95.95%）→
**位形信息完全由线圈电流承载**。

## 2. 输入通道（18 通道，`--input-mode coils`）

| # | 通道 | 含义/单位 | 来源 |
|---|---|---|---|
| 1 | R | 网格 R 坐标（[-1,1]） | 固定 MAST 65×65 |
| 2 | Z | 网格 Z 坐标（[-1,1]） | 固定 MAST 65×65 |
| 3 | Ip | 等离子体电流 (A) | params[0] |
| 4 | paxis | 磁轴压强 (Pa) | params[1] |
| 5 | fvac | 真空通量函数 f (Wb/m) | params[2] |
| 6 | alpha_m | 剖面形状指数 m | params[3] |
| 7 | alpha_n | 剖面形状指数 n | params[4] |
| 8–17 | I_P2U, I_P2L, I_P3U, I_P3L, I_P4U, I_P4L, I_P5U, I_P5L, I_P6U, I_P6L | 上下偏滤器线圈电流 (A) | coil_currents[0:10] |
| 18 | I_P1 (Solenoid) | 中心螺线管电流 (A) | coil_currents[10] |

无 X 点/锚点坐标、无 config 标签。R/Z 全样本相同 → 16 个标量通道承载全部样本
信息，其中位形信息由 11 个线圈电流通道承载（隐含使用）。

## 3. 训练设置

N=500（拼接 4000 样本上嵌套取 500）seed 1，MSE/AdamW/ReduceLROnPlateau，
800 epochs（同 exp008/010 惯例）；stats 来自拼接后全量 train pool；混合拼接下
xpts_actual 行数不同（DN 2 / SN 1）→ NaN-pad 到最大行数（v2 配对对 NaN 行诚实
输出 NaN）。best val **0.8813%** @ epoch 795，14.5 min。

## 4. 结果（test 分桶，rel L2 mean %）

| test | **exp011 coil 18ch** | exp010 xa 13ch 无 config | exp008 xa+config 14ch | exp009 专职 xa 13ch |
|---|---|---|---|---|
| DN | **0.841** | 0.611 | 0.667 | 0.492 |
| SN | **0.948** | 1.148 | 1.203 | 0.953 |
| 整体（n=1000） | **0.894** | — | 0.935 | — |
| RMSE (Wb) DN/SN | 3.41e-4 / 3.98e-4 | 2.51e-4 / 5.02e-4 | 2.74e-4 / 5.24e-4 | 1.85e-4 / 4.31e-4 |
| GS 残差比 DN/SN | 1.013 / 1.008 | 0.995 / 0.977 | 0.989 / 0.984 | 0.988 / 0.991 |
| find_critical 失败 | 1/500 / 0/500 | 1/500 / 0/500 | 0/0 | 0/0 |
| X 点误差 lo/up (cm) DN/SN | 3.16/2.23；1.25/— | 2.65/2.59；0.94/— | 3.19/2.37；1.06/— | 2.14/1.91；0.97/— |
| sep_mean (cm) DN/SN | 0.62 / 0.73 | 0.55 / 0.40 | 0.56 / 0.43 | 0.40 / 0.34 |
| sep 面积相对误差 (%) DN/SN | 0.69 / 0.75 | 0.70 / 0.44 | 0.68 / 0.40 | 0.47 / 0.33 |
| O 点误差 (cm) DN/SN | 0.53 / 0.84 | 0.50 / 0.50 | 0.48 / 0.61 | 0.34 / 0.42 |

eval_all（n=1000）：rel L2 0.894%、RMSE 3.69e-4 Wb、GS 残差比 1.010、
find_critical 失败 1/1000。

## 5. 结论

1. **端到端 psi 生成成立（核心结果）**：无任何显式位形/拓扑信息，两桶均健康
   （0.841%/0.948%，GS ≈1.01）；
2. **coil 与 xa 同量级，且 SN 桶反超**：SN 桶 0.948% < xa 无 config 1.148%
   （−17%）；整体 0.894% vs exp008 0.935%（coil 反而略优）——X 点/锚点坐标是
   后验推导量，原始可测量的线圈电流信息无损；
3. **位形识别是隐含能力**：模型从电流模式中隐含区分 DN（双 X 点）与 SN
   （单 X 点），无需任何显式信号；
4. **几何保真同健康**：X 点 1–3 cm、sep_mean <0.8 cm、O 点 <0.9 cm；
5. **部署含义**：真实装置可测输入（电流+工程参数）直接可用，无需位形判定/
   磁面重构中间环节；coil 输入是"因果"输入（exp001/007 结论在混合位形下延续）。

## 6. 复现命令

```bash
PY="C:/Users/HP/.conda/envs/torch5060/python.exe"

"$PY" -u -m gs_pino_dn_fno_2608.train_dn_fno \
  --train-data dn_fno_2608/data_v5/dn/train.npz,dn_fno_2608/data_v5/sn/train.npz \
  --val-data dn_fno_2608/data_v5/dn/val.npz,dn_fno_2608/data_v5/sn/val.npz \
  --input-mode coils --n-train 500 --seed 1 \
  --out-dir dn_fno_2608/experiments/exp011_coil_input_v5/model_b18ch_coils_mix
# 六桶评估 + 可视化一键：
bash dn_fno_2608/scripts/run_exp011_eval_vis.sh
```

## 7. 产物

`model_b18ch_coils_mix/`（best.pt/history/args/metrics + eval_all/、eval_dn/、
eval_sn/ + figures_dn/、figures_sn/）。日志：`logs/exp011_*.log`。

## 8. 偏差记录

1. **visualize 不支持逗号多文件**：figures 按配置分开生成（figures_dn/、
   figures_sn/）；
2. **`*_dn_fno_coils.py` 镜像脚本仍是 v1 几何**（find_critical 假鞍点配对 bug）
   ——本实验一律用主脚本（`--input-mode coils`，v2 几何）；镜像脚本仅供
   exp001/003/005/007 复现；
3. **DNFnoDatasetCoils 补丁**（2026-08-17）：多文件拼接（逗号分隔）、
   xpts_actual/o_point/anchor 真值属性、xpts_actual 变长行 NaN-pad；
   match_xpoints_and_axis 对 NaN 行诚实输出（SN 无上 X 点 → x_up NaN）；
4. DN 桶 find_critical 失败 1/500：单样本定位失败（FNO 平滑场鞍点消失），
   同 exp010 个案量级；
5. coil 镜像文件家族保留（单文件路径行为不变，向后兼容）。
