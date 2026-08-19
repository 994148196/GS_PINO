# exp010: 只预测 psi_plasma——线圈场线性先验分解（MAST data_v5）

## 动机（2026-08-19 数据验证）

exp002-008 全部停在 11-15% rel L2 的直接根因：**线圈场占主导且 B 样条表达不了**。

| 量 | 值 | 含义 |
|---|---|---|
| |psi_coils|/|psi_total| | **183%** | 线圈场淹没等离子体场（Biot-Savart 奇点） |
| max|Δ*psi_coils| | **27.9** | 线圈奇点（网格中心差分） |
| max|Δ*psi_plasma| | **0.89** | 等离子体场光滑 31 倍 |
| greens·I vs psi_coils | **1e-7 相对误差** | 线圈场 = 电流的精确线性函数 |
| ||psi_plasma||_F/||psi_total||_F | 0.99 | 去掉线圈场后目标不缩水 |

grid-2 二次样条（每区间 ~1.1m，网格 3.4cm）在参数空间表达不了 11 线圈
Biot-Savart 场 → 网络把容量浪费在线圈场上。**分解**：psi_total = psi_plasma + psi_coils，
网络只学光滑的 psi_plasma（自由边界问题的本征输出），评估时线圈场精确加回。
这也与 LPDE plasma 语义一致（"PDE 残差不包含 coil"）。

## 输入通道（19ch 明细，同 exp007）

| # | 通道 | 含义 | 单位 | 来源 |
|---|---|---|---|---|
| 1 | R | 归一化径向坐标 [-1,1] | — | 网格 65×65 |
| 2 | Z | 归一化轴向坐标 [-1,1] | — | 网格 65×65 |
| 3 | Ip | 等离子体电流 | A | params[:,0] |
| 4 | paxis | 磁轴压 | Pa | params[:,1] |
| 5 | fvac | 真空磁通函数 | T·m | params[:,2] |
| 6 | alpha_m | 压力剖面形状 | — | params[:,3] |
| 7 | alpha_n | 电流剖面形状 | — | params[:,4] |
| 8-18 | I_P2U..I_P1 | 11 线圈电流 | A | coil_currents[:,0:11] |
| 19 | config | 位型码（DN=-1/SN=+1） | — | config |

**目标**：psi_plasma（z-score 用 train 池 psi_plasma 的 mean/std）。
评估重建：psi_total = psi_pred + psi_coils（数据存储），rel L2 在总场上度量
（`rel_l2_total_pct`，与 exp001-009 可比）。

## 设置

与 exp007 完全一致，唯一差异 `--target plasma`：DN-only、N=500 seed 1、
论文架构 BSplineKAN(19→10→2) grid2、1000 ep、milestones 300,500,700,900、
256 点/样本 × 32 × 100 step、j=0 pde=0 reg=0。

## 对照与结果（2026-08-19 完成）

| 实验 | 目标 | 架构 | 100ep | 250ep | 1000ep 评估（总场） |
|---|---|---|---|---|---|
| exp007 | total | grid2 | 14.0% | 11.2% | —（早停） |
| exp008 | total | grid8 | 7.2% | 6.5% | —（早停） |
| **exp010** | **plasma** | grid2 | **5.1%** | — | **2.30%**（pre-prune @590ep） |

exp010 最终评估（DN test 500 行，pre-prune best @ ep 590）：
- 总场 rel L2（线圈加回）：**mean 2.30%**（std 1.22%）
- RMSE 1.12e-3 Wb；**R² = 0.9987**（论文 0.9953，已超）
- GS 残差比 1.022；X 点 4.0cm、o_point 1.4cm、sep_mean 1.42cm
- 0/500 critical fail（X 点计数全对）
- 剪枝观察：无 reg 训练下 600ep 剪枝有害（31% 剪枝 → val 3.16%→4.59%）——
  印证论文顺序：先 reg（KAN-3）后剪枝；监督阶段看 pre-prune

对照里程碑 M1：DN rel L2 ≤ 1.3% 尚未达到（2.30%）→ exp011 叠加 grid8。

训练全程见 `train_exp010.log`（33 min / 1000ep）。

## 运行

```bash
bash dn_fno_2608/kan/scripts/run_kan_exp010_plasma_target.sh
```
