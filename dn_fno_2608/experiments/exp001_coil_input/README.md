# exp001 — coil 电流作为输入（替代 X 点坐标）

> 实验日期：2026-08-14
> 状态：**完成**（N=500，seed 1）
> 代码：`src/gs_pino_dn_fno_2608/`（data_dn_fno_coils / train_dn_fno_coils / evaluate_dn_fno_coils，**基线代码零改动**）
> 结论速览：coil 电流输入可行，rel L2 0.326% vs 基线 X 点 0.222%（差 1.5×，同量级）

---

## 1. 动机

论文基线（arXiv:2608.05555）用 **4 个 X 点坐标**作为输入标量，把"位形参数 → 平衡场"映射学成神经算子。但实际工程中，X 点位置是**设计目标**，控制线圈电流才是**执行量**（freegs 控制环根据约束反解出来的物理输入）。本实验验证一个更贴近执行器的输入方案：

**把 4 个 X 点坐标通道替换为 4 个自由边界控制线圈电流**（P1L/P1U/P2L/P2U），模型精度如何。

用户策略：**小数据集（N=500）结果不错就不用更大数据集**。

## 2. 方法

### 2.1 输入结构（与基线同构，仅换 4 个标量）

```
基线（9 通道）:  R, Z | Paxis, Ip, fvac | R_lo, Z_lo, R_up, Z_up  (X 点坐标)
本实验（9 通道）: R, Z | Paxis, Ip, fvac | I_P1L, I_P1U, I_P2L, I_P2U (线圈电流)
```

- 通道数不变 → **模型架构完全不变**（FNO2d2608，in_channels=9，4.21M 参数）；
- coil 电流 z-score 归一化（统计量取全训练集，与基线 X 点同处理）；
- 数据集无需重新生成：`coil_currents` 字段在数据生成时已预留（4 个控制线圈电流）；
- 其余全部一致：MSE 损失、AdamW(1e-3,1e-4)、ReduceLROnPlateau、早停 75、batch 16、800 epoch、嵌套子集（seed 12345）。

### 2.2 代码结构（独立文件，不污染基线）

| 文件 | 说明 |
|---|---|
| `src/gs_pino_dn_fno_2608/data_dn_fno_coils.py` | 数据集：9 通道构造（coil 电流版），物理字段属性与基线一致 |
| `src/gs_pino_dn_fno_2608/train_dn_fno_coils.py` | 训练 CLI（复用基线的 set_seed/evaluate，训练超参同基线） |
| `src/gs_pino_dn_fno_2608/evaluate_dn_fno_coils.py` | 评估 CLI（复用基线的 gs_residual_ratio/geometry_metrics，口径一致） |

基线模块（data_dn_fno.py / train_dn_fno.py / evaluate_dn_fno.py）**未做任何修改**。

## 3. 结果（test 500，N=500，seed 1）

### 3.1 场级精度

| 指标 | coil 输入 | 基线 (X 点) | 论文 N=500 |
|---|---|---|---|
| rel L2 mean | **0.326%** | 0.222% ± 0.007 | 0.286% ± 0.034 |
| rel L2 median | 0.257% | 0.167% | — |
| rel L2 P95 | 0.762% | — | — |
| RMSE phys (Wb) | 8.31e-5 | 5.48e-5 | 8.53e-5 |
| <0.12% 样本占比 | 2.6% | — | 论文 N=5000 >95% |

### 3.2 几何与物理一致性

| 指标 | coil 输入 | 基线 (X 点) |
|---|---|---|
| sep_mean_cm | 0.267 | 0.150 |
| sep_hausdorff_cm | 1.872 | 1.305 |
| x_lo / x_up_cm | 0.386 / 1.007 | 0.494 / 0.477 |
| o_point_cm | 0.254 | 0.142 |
| find_critical 失败 | **0/500** | 0/500 |
| GS 残差比值（pred/truth） | **0.995** | 0.995 |

### 3.3 训练

- best val rel L2 0.327% @ epoch 783（基线 N=500 s1：0.225% @ 756）
- 训练时长 11.0 min（GPU RTX 5060 Laptop）

## 4. 结论

1. **coil 电流输入可行**：N=500 即达 0.326%，几何指标合理、物理一致性 0.995、0/500 失败——作为毫秒级代理完全可用。
2. **但比 X 点坐标输入差约 1.5×**（0.326% vs 0.222%），原因是信息量差异：
   - X 点坐标是平衡边界的**直接拓扑锚点**，与分界面形态强相关，模型容易学习；
   - coil 电流是自由边界控制环反解的结果，与 ψ 场是非线性隐式映射，且 4 个电流间存在控制环冗余，等效信息密度更低。
3. 按用户策略**不跑更大数据集**（"小数据集结果不错就不用更大"）。若未来要追平基线，按缩放律 ε ∝ N^-0.61 估需 N≈1500–2000（一次约 40–55 min 训练）。

## 5. 复现命令

```bash
PY="C:/Users/HP/.conda/envs/torch5060/python.exe"

# 训练
"$PY" -u -m gs_pino_dn_fno_2608.train_dn_fno_coils \
  --train-data dn_fno_2608/data/train.npz --val-data dn_fno_2608/data/val.npz \
  --n-train 500 --seed 1 --out-dir dn_fno_2608/experiments/exp001_coil_input

# 评估
"$PY" -u -m gs_pino_dn_fno_2608.evaluate_dn_fno_coils \
  --test-data dn_fno_2608/data/test.npz \
  --checkpoint dn_fno_2608/experiments/exp001_coil_input/best.pt \
  --out-dir dn_fno_2608/experiments/exp001_coil_input
```

## 6. 产物

```
exp001_coil_input/
├── README.md       # 本文档（实验介绍）
├── notes.md        # 简短结论记录
├── best.pt         # 最优权重（含 stats + input_mode="coils"）
├── history.json    # 训练曲线
├── args.json       # 训练参数
└── metrics.json    # test 评估指标（全 500 样本）
```

训练日志：`dn_fno_2608/logs/exp001_coil_n500_s1.log`

## 7. 后续可做（供参考）

- **coil + X 点混合输入**（13 通道）：信息互补，可能优于两者任一；
- **coil 输入 + 较大 N**（1500–2000）：追平基线精度；
- **用 coil 电流 + Green 函数物理约束**（数据集 greens 字段已预留）：coil 电流与 ψ_coils = G·I 直接相关，PINO 阶段天然适配。
