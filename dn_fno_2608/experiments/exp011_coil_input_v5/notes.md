# exp011 notes — coil 电流输入混合训练（DN+SN）端到端解读

> 2026-08-17 训练与评估。配套 README.md（设计/结论/18 通道明细）。
> 对照 exp008（xa+config）、exp010（xa 无 config）、exp009（专职）、
> exp007（coil 单一位形）。N=500 seed 1，延续惯例。

## 1. 数据（data_v5，零新增生成）

- 直接用 data_v5 dn/sn 各 3000（coil_currents (N,11)、greens (N,11,65,65)
  已在 npz 中，生成时落盘）；
- 混合拼接：DNFnoDatasetCoils 接入多文件（逗号分隔），stats 来自拼接后
  全量 train pool（DN/SN 共享 z-score）；
- 前提实证：5 params 在 DN/SN 间几乎重合（不可区分），11 线圈电流系统性
  分离（单通道阈值判别 96%）→ 位形信息完全由线圈电流承载。

## 2. 训练

| 模型 | 输入 | best val rel L2 | epoch | 时间 |
|---|---|---|---|---|
| coil 18ch 混合 | R,Z+5 params+11 coil 电流（无 X点/锚点/config） | 0.8813% | 795 | 14.5 min |

## 3. test 评估（分桶 DN/SN + 混合整体）

| 指标 | 整体（n=1000） | DN 桶 | SN 桶 |
|---|---|---|---|
| rel L2 mean % | 0.894 | 0.841 | 0.948 |
| RMSE phys (Wb) | 3.69e-4 | 3.41e-4 | 3.98e-4 |
| GS 残差比 | 1.010 | 1.013 | 1.008 |
| find_critical 失败 | 1/1000 | 1/500 | 0/500 |
| X 点误差 lo/up (cm) | 2.21 / 2.23 | 3.16 / 2.23 | 1.25 / NaN |
| sep_mean (cm) | 0.67 | 0.62 | 0.73 |
| sep 面积相对误差 (%) | 0.72 | 0.69 | 0.75 |
| O 点误差 (cm) | 0.69 | 0.53 | 0.84 |

对照（rel L2 mean %）：DN 专职 0.492 < coil 0.841 ≈ xa+config 0.667 < xa
无 config 0.611（DN 桶 coil 差 0.23pp）；SN 专职 0.953 ≈ **coil 0.948 <
xa 无 config 1.148**（SN 桶 coil 反超 −0.20pp）；整体 coil 0.894 < exp008
0.935。

## 4. 解读

1. **端到端可行性成立**：不喂任何显式位形/拓扑信息，模型直接出 psi，
   两桶健康（0.84/0.95%）+ GS 残差比 ≈1.01 + 几何 cm 级。
2. **coil vs xa 几乎打平、SN 反超**：X 点/锚点坐标是后验推导量，线圈电流
   是原始可测量——信息无损。SN 桶反超的可能机制：SN 的 xa 输入有 up=(0,0)
   占位恒值通道（z-score 后恒常数），对模型是"死通道"还占维度；coil 输入
   无占位，16 个标量全是活跃信息。
3. **隐含位形识别**：5 params 不可区分位形（实测均值几乎重合），位形信息
   完全在 11 线圈电流里（单通道阈值判别 96%）——模型从电流模式隐含区分
   DN/SN 并输出正确场，无任何显式信号。
4. **与 exp007 的关系**：单一位形 coil（data_v4 0.499%）→ 混合位形 coil
   （data_v5 0.841/0.948%），位形范围扩大 +26~90%（DN 桶 +68% vs 单一位形
   DN；与 exp008 xa 的 +36% 混合代价同一量级）。
5. **DN 桶 coil 比 xa 差 0.23pp 的可能因素**：DN 双 X 点拓扑对线圈电流
   更"远"（电流是磁场的源，X 点结构是磁场的后验特征），xa 输入直接把
   解的一部分告诉模型；SN 单 X 点拓扑差异小，此差距不显著——方向与
   exp001/007 的"coil 可行但略差"结论一致。

## 5. 偏差记录

- visualize 不支持逗号分隔多文件 → figures 按配置分开生成（figures_dn/、
  figures_sn/），exp008 先例；
- `train/evaluate_dn_fno_coils.py` 镜像脚本仍是 v1 几何（find_critical 假鞍点
  bug），**本实验一律用主脚本**（--input-mode coils，v2 几何）；镜像脚本仅供
  exp001/003/005/007 复现；
- DNFnoDatasetCoils 补丁：多文件拼接 + xpts_actual/o_point/anchor 真值属性 +
  xpts_actual 变长行 NaN-pad（DN 2 行 vs SN 1 行）；match_xpoints_and_axis
  对 NaN 行诚实输出 NaN（SN 无上 X 点 → x_up NaN）。
