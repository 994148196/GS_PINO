# exp009: TestTokamak（data_v4）简单数据——验证数据复杂度假说

## 动机（用户反馈 2026-08-19："如果用 data 这种简单的数据，会不会结果好一些？"）

MAST（data_v5）数据上 B 样条 KAN 始终停在 ~11-15% rel L2（exp002-008），
已排除采样/信息量/细节淹没等假说。最新根因：**数据复杂度**——
线圈场占比 |psi_coils|/|psi_total|：

| 数据集 | 线圈数 | 线圈场占比 | 网格 | 说明 |
|---|---|---|---|---|
| MAST data_v5 | 11 | 183-185% | 65² | 线圈场淹没等离子体场；grid-2 样条（每区间 1.1m）表达不了 11 线圈 Biot-Savart 场 |
| TestTokamak data_v4 | **4** | **62%** | 65² | 简单位形，线圈场不占主导，更接近论文 EXL-50U（5 独立对称线圈） |

## 输入通道（11ch 明细）

| # | 通道 | 含义 | 单位 | 来源 |
|---|---|---|---|---|
| 1 | R | 归一化径向坐标 [-1,1] | — | 网格 (65×65) |
| 2 | Z | 归一化轴向坐标 [-1,1] | — | 网格 (65×65) |
| 3 | Ip | 等离子体电流 | A | params[:,0] |
| 4 | paxis | 磁轴压 | Pa | params[:,1] |
| 5 | fvac | 真空磁通函数 | T·m | params[:,2] |
| 6 | alpha_m | 压力剖面形状 | — | params[:,3] |
| 7 | alpha_n | 电流剖面形状 | — | params[:,4] |
| 8 | I_PF1 | 极向场线圈 1 电流 | A | coil_currents[:,0] |
| 9 | I_PF2 | 极向场线圈 2 电流 | A | coil_currents[:,1] |
| 10 | I_PF3 | 极向场线圈 3 电流 | A | coil_currents[:,2] |
| 11 | I_PF4 | 极向场线圈 4 电流 | A | coil_currents[:,3] |

（vs MAST 19ch = 上述 + 6 个额外线圈 + 1 个 config 通道；TestTokamak 无 config 离散通道。）

## 设置

与 exp007 完全一致，唯一差异 = 数据源 + `--device-config testtokamak`：

- data_v4 train 1982 行 / val / test 494 行；N=500 seed 1（nested 子集）
- 论文架构：BSplineKAN(11→10→2)，grid 2、degree 2、silu 基；1260-ish 参数
- 1000 ep；Milestones 300,500,700,900 γ=0.5；LR 1e-3
- **纯监督只预测 psi**：j-weight 0、pde 0、reg 0；256 点/样本 × 32 样本 × 100 step

## 对照

- exp007（grid2 psi-only, MAST 19ch）：200ep 12.6% / 250ep 11.2%（运行中）
- exp008（grid8 psi-only, MAST 19ch）：100ep 7.2%（运行中，空间分辨率假说已获支持）
- exp009（grid2 psi-only, TestTokamak 11ch）：本实验 → 数据复杂度净增益
- 论文 EXL-50U 参考：KAN-1 200ep 0.63%

## 运行

```bash
bash dn_fno_2608/kan/scripts/run_kan_exp009_testtokamak.sh
```

## 结论（2026-08-19，250ep 早停）

**简单数据假说被证伪**：TestTokamak 反而更难。同期对照（全部 psi-only 监督）：

| 实验 | 数据 | 目标 | 50ep | 100ep | 250ep |
|---|---|---|---|---|---|
| exp007 | MAST grid2 | total | 14.8% | 14.0% | 11.2% |
| exp009 | TestTokamak grid2 | total | 23.6% | 22.1% | **19.7%** |

原因（数据验证）：TestTokamak 等离子体只占 **6.7%**（MAST 20.3%）——等离子体
通道更窄，同样的 grid-2 样条空间分辨率问题（1.1m/区间 vs 3.4cm 网格）更严重；
且线圈场占比 62% 仍须表达。**数据"简单"≠对全局坐标样条简单**。

后续方向转为 exp010：线圈场线性先验分解（--target plasma），见
`../exp010_plasma_target_kan_v5/README.md`。

产物在 `model_psionly_tt_kan/`（best.pt / history.json / args.json）。
