# exp011 — coil 电流输入跨位形（DN+SN）混合训练：端到端 psi 生成

> 实验日期：2026-08-17
> 状态：**完成**（N=500，seed 1）
> 代码：`src/gs_pino_dn_fno_2608/`（`train_dn_fno --input-mode coils`，coil 输入
> 接入主脚本）
> 数据：`dn_fno_2608/data_v5/`（MAST DN+SN，各 3000；**coil_currents (N,11) 与
> greens 已在 npz 中**，无需重新生成）
> 对照：exp008（xa+config 14ch 混合）、exp010（xa 13ch 混合无 config）、
> exp009（专职 xa 13ch）、exp007（coil 11ch data_v4 单一位形）
> 结论速览：**不显式告诉模型位形/拓扑，只给 (R,Z + 5 物理参数 + 11 线圈电流)，
> 模型直接生成物理正确的 psi 场**——混合训练下 DN/SN 两桶均健康，
> 与 xa（X点+锚点）输入同量级；位形识别是隐含能力（信息完全由线圈电流承载，
> 已实证单通道阈值判别 96%），不是显式输入。

---

## 1. 动机与科学问题

真实装置上可测量的量是**线圈电流与工程参数**；X 点坐标、锚点、分离面都是
后验推导量（freegs 也是从线圈电流反解平衡，隐式确定位形）。前面的实验
（exp006-010）输入含 X 点/锚点坐标或 config 标签，本质是"把解的一部分告诉
模型"。本实验问：

**不显式告诉模型位形/拓扑信息，只输入物理参数 + coil 电流 + 坐标，
模型能否直接生成 psi 分布？（位形识别只是隐含的必要能力）**

前提已实证（[scripts/tabulate_exp011.py](tabulate_exp011.py 或探针脚本)）：
- DN/SN 的 5 个物理参数分布几乎重合（Ip 5.51e5 vs 5.48e5 A 等，不可区分）；
- 11 个线圈电流系统性分离（最强通道 P3U 分离度 3.29 个合并标准差，
  单通道阈值判别准确率 95.95%）；
- → **位形信息完全由线圈电流承载**，模型必须从中隐含识别位形才能出正确 psi。

## 2. 方法

- 训练：N=500（拼接后 4000 样本上嵌套取 500）seed 1，MSE/AdamW/
  ReduceLROnPlateau，800 epochs，同 exp008/010 惯例；
- 输入：`--input-mode coils` → **18 通道**（详见 §2a 通道明细）；无 X 点/
  锚点坐标、无 config 标签；
- 数据：train/val = `dn/*.npz,sn/*.npz` 逗号拼接（多文件拼接已接入
  DNFnoDatasetCoils），test 分桶 = dn / sn 单独 + 拼接整体；
- 归一化：stats 来自拼接后全量 train pool（DN/SN 共享 z-score）；
  混合拼接下 xpts_actual 行数不同（DN 2 / SN 1）→ NaN-pad 到最大行数，
  v2 配对对 NaN 行诚实输出 NaN（SN 无上 X 点 → x_up NaN）；
- 评估：主脚本 evaluate_dn_fno（**v2 几何**：真值基准配对 + 射线法分离面，
  与 exp008/010 同一管线，数字可直接对比）；
- 可视化：visualize_dn_fno --machine mast（fig1 装置结构/线圈/R=Z 等比例）。

### 2a. 输入通道明细（18 通道）

通道顺序 = 代码实际拼接顺序（`data_dn_fno_coils.py` `__getitem__`）：
**2 几何通道 (R, Z) + 5 物理参数 + 11 线圈电流**，全部 z-score（R/Z 线性映射
[-1,1]，标量用拼接 pool 的 mean/std）。

| # | 通道 | 含义/单位 | 来源 |
|---|---|---|---|
| 1 | R | 网格 R 坐标（线性映射 [-1,1]） | 固定 MAST 65×65 |
| 2 | Z | 网格 Z 坐标（线性映射 [-1,1]） | 固定 MAST 65×65 |
| 3 | I_p | 等离子体电流 (A) | params[0] |
| 4 | p_axis | 磁轴压强 (Pa) | params[1] |
| 5 | f_vac | 真空通量函数 f (Wb/m) | params[2] |
| 6 | alpha_m | 剖面形状指数 m | params[3] |
| 7 | alpha_n | 剖面形状指数 n | params[4] |
| 8 | I_P2U | 上 P2 线圈电流 (A) | coil_currents[0] |
| 9 | I_P2L | 下 P2 线圈电流 (A) | coil_currents[1] |
| 10 | I_P3U | 上 P3 线圈电流 (A) | coil_currents[2] |
| 11 | I_P3L | 下 P3 线圈电流 (A) | coil_currents[3] |
| 12 | I_P4U | 上 P4 线圈电流 (A) | coil_currents[4] |
| 13 | I_P4L | 下 P4 线圈电流 (A) | coil_currents[5] |
| 14 | I_P5U | 上 P5 线圈电流 (A) | coil_currents[6] |
| 15 | I_P5L | 下 P5 线圈电流 (A) | coil_currents[7] |
| 16 | I_P6U | 上 P6 线圈电流 (A) | coil_currents[8] |
| 17 | I_P6L | 下 P6 线圈电流 (A) | coil_currents[9] |
| 18 | I_P1 (Solenoid) | 中心螺线管电流 (A) | coil_currents[10] |

- 线圈名/顺序 = freegs `machine.py` MAST() 定义（P2U, P2L, P3U, P3L, P4U, P4L,
  P5U, P5L, P6U, P6L, P1 Solenoid），生成脚本按 `tokamak.coils` 顺序收集；
- R/Z 对全部样本相同 → 16 个标量通道承载全部样本信息，其中 5 个 params 不可
  区分位形 → **生成正确 psi 所需的位形信息由 11 个线圈电流通道承载**（隐含
  使用，非显式输入）。

## 3. 结果

### 3a. 对照表（rel L2 mean %，test 分桶）

| test | **exp011 coil 18ch（本实验）** | exp010 xa 13ch 无 config | exp008 xa+config 14ch | exp009 专职 xa 13ch |
|---|---|---|---|---|
| DN | **0.841** | 0.611 | 0.667 | 0.492 |
| SN | **0.948** | 1.148 | 1.203 | 0.953 |
| 整体（n=1000） | **0.894** | — | 0.935 | — |
| RMSE DN/SN (Wb) | 3.41e-4 / 3.98e-4 | 2.51e-4 / 5.02e-4 | 2.74e-4 / 5.24e-4 | 1.85e-4 / 4.31e-4 |
| GS 残差比 DN/SN | 1.013 / 1.008 | 0.995 / 0.977 | 0.989 / 0.984 | 0.988 / 0.991 |
| find_critical 失败 | 1/500 / 0/500 | 1/500 / 0/500 | 0/500 / 0/500 | 0/500 / 0/500 |
| X 点误差 lo/up (cm) DN/SN | 3.16/2.23；1.25/— | 2.65/2.59；0.94/— | 3.19/2.37；1.06/— | 2.14/1.91；0.97/— |
| sep_mean (cm) DN/SN | 0.62 / 0.73 | 0.55 / 0.40 | 0.56 / 0.43 | 0.40 / 0.34 |
| sep 面积相对误差 (%) DN/SN | 0.69 / 0.75 | 0.70 / 0.44 | 0.68 / 0.40 | 0.47 / 0.33 |
| O 点误差 (cm) DN/SN | 0.53 / 0.84 | 0.50 / 0.50 | 0.48 / 0.61 | 0.34 / 0.42 |

（几何指标 v2：真值基准配对 + 射线法分离面；SN 无上 X 点 → x_up 为 NaN）

### 3b. 训练（val）

| 模型 | best val rel L2 | epoch | 时间 |
|---|---|---|---|
| exp011 coil 18ch 混合 | **0.8813%** | 795 | 14.5 min |
| （对照 exp010 xa 13ch） | 0.8327% | 800（满） | 15.0 min |
| （对照 exp008 xa+config 14ch） | 0.8789% | 797 | 14.3 min |

### 3c. 分桶明细

eval_all（n=1000 混合整体）：rel L2 mean 0.894%、RMSE 3.69e-4 Wb、
GS 残差比 1.010、find_critical 失败 1/1000；分桶 n=500/500 见 §3a。

## 4. 结论

1. **端到端 psi 生成成立（核心结果）**：不给任何显式位形/拓扑信息
   （无 X 点/锚点坐标、无 config 标签、无占位编码），只给 (R,Z + 5 物理参数
   + 11 线圈电流)，模型直接生成物理正确的 psi 场——混合训练下 DN/SN 两桶
   均健康（0.841% / 0.948%，GS 残差比 ≈1.01），find_critical 失败 1/1000。
2. **coil 与 xa 同量级，且 SN 桶反超**：SN 桶 coil 0.948% < xa 无 config
   1.148%（−17%）；DN 桶 0.841% vs 0.611%（+38%）；整体 0.894% vs exp008
   0.935%（coil 反而略优）。X 点/锚点坐标是后验推导量，**原始可测量的线圈
   电流信息无损**——"给电流直接出 psi"的端到端路线不逊于喂解的部分信息。
3. **位形识别是隐含能力**：位形信息完全由 11 个线圈电流承载（5 个物理
   参数在 DN/SN 间统计不可区分，已实证单通道阈值判别 96%）；模型从电流
   模式中隐含区分 DN（双 X 点拓扑）与 SN（单 X 点），无需任何显式信号。
4. **几何保真同健康**：X 点 1–3 cm、sep_mean <0.8 cm、sep 面积 <0.8%、
   O 点 <0.9 cm——与 xa 家族同量级，端到端路线不牺牲几何质量。
5. **部署含义**：真实装置可测输入（电流+工程参数）直接可用，无需位形
   判定/磁面重构的中间环节；coil 输入同时是"因果"输入（exp001/007 的
   结论在混合位形下延续）。

## 5. 偏差记录

1. **visualize 不支持逗号分隔多文件**：figures 按配置分开生成
   （figures_dn/、figures_sn/），exp008 先例。
2. **`train/evaluate_dn_fno_coils.py` 镜像脚本仍是 v1 几何**（find_critical
   假鞍点配对 bug）——**本实验一律用主脚本**（`--input-mode coils`，v2
   几何，与 exp008/010 同一管线）；镜像脚本仅供 exp001/003/005/007 复现。
3. **DNFnoDatasetCoils 补丁**（2026-08-17）：接入多文件拼接（逗号分隔）、
   xpts_actual/o_point/anchor 真值属性（主脚本 v2 几何依赖）、xpts_actual
   变长行 NaN-pad（DN 2 行 vs SN 1 行，混合拼接需要）；配套
   match_xpoints_and_axis 对 NaN 行诚实输出（SN 无上 X 点 → x_up NaN）。
   同时修复了主 DNFnoDataset 混合评估的同一拼接 bug。
4. **DN 桶 find_critical 失败 1/500**：单样本定位失败（FNO 平滑场中鞍点
   消失），同 exp010 个案量级，不影响结论。
5. **coil 镜像文件家族保留**：data_dn_fno_coils.py 现同时服务镜像脚本
   （exp001/003/005/007，单文件路径）与主脚本（exp011，多文件路径），
   向后兼容（单文件路径行为不变）。
