# 研究计划：基于物理信息神经算子的Grad-Shafranov方程快速求解方法（修订版 v3）

> 2026-08-20 修订 v3：篇幅 4000-6000 字；**全文聚焦自由边界**（传统方法只写 freegs，
> 不写固定边界）；核心为 exp101/102（PINO 物理残差），exp011 作纯数据基线对比。

---

## 论文主线

GS方程背景 → 传统自由边界求解器(freegs) → PINN → **自由边界 PINO（核心）**

**标签**: 科学计算和数值仿真 / 机器学习算法与模型算法

**核心故事**:
1. 自由边界 GS 方程求解慢（freegs Picard 迭代秒级/次）
2. 神经算子（FNO）端到端替代：只给可测量量（R,Z + 5 参数 + 11 线圈电流），毫秒级出 psi 场
3. **关键提升：加 GS 方程物理残差损失**
   - 做法1（exp101 单阶段）：预测场 Δ\* 拉到数据 RHS → 精度不降反升（0.72% vs 0.84%）
   - 做法2（exp102 两阶段自洽）：先监督 psi+J，再自洽残差 + Ip 约束 → 纯数据拿不到的自洽性与 Ip 满足度
4. 关键物理设计：**coil 分离**（网络只预测 psi_plasma，psi_total 由 Green 函数解析加回）

---

## 论文结构（4000-6000 字，5 章）

### 1. 引言（~600字）
- 托卡马克等离子体平衡依赖 GS 方程，实时平衡重建需求
- 传统自由边界求解器（freegs，Picard 迭代 + von Hagenow Green 函数法）精确但慢（秒级）
- **PINN 作为背景介绍**（一段）：ML for PDE 的开端，autograd 将 PDE 残差嵌入损失；局限——单参数单解、逐次重训，难覆盖自由边界大参数空间
- 神经算子（FNO）学习参数→场的映射，毫秒级推理
- **本文**: 自由边界端到端 PINO + GS 物理残差约束，两阶段自洽训练
- 贡献三点：端到端可测量输入 / coil 分离设计 / 物理残差两方案

### 2. GS 方程与自由边界传统求解（~700字）—— 简要
- 2.1 GS 方程: Δ\*ψ = -μ₀RJ_φ，Jeon 2015 剖面参数化
- 2.2 自由边界问题: PF 线圈电流为外部条件，LCFS 由平衡自洽确定（X 点/分离面）
- 2.3 **freegs 求解器**: 有限差分 + Picard 迭代 + von Hagenow Green 函数法处理边界
- 2.4 物理量分解: psi_total = psi_plasma + psi_coils = ψ_plasma + Σ I_k G_k（Green 函数线性叠加）
- 2.5 展示: freegs 求解实例（ψ 场等值线 + 分离面 + X 点图）

### 3. 自由边界 PINO 方法（~1800字）—— **核心**
- 3.1 **FNO2d2608 架构**: 1×1 升维(width=64) + 4× FNOBlock(SpectralConv2d 频域 16×16 模态 + 1×1 混合 + GELU) + 1×1 投影。~4.2M 参数
- 3.2 **端到端输入（18ch）**: R,Z 坐标 + 5 等离子体参数(Ip, paxis, fvac, αm, αn) + 11 线圈电流。只给可测量量，无 X 点/锚点/config 标签
- 3.3 **coil 分离设计（关键物理洞察）**: 线圈场占 psi_total 幅度 183%，|Δ\*ψ_coils| 峰值 27.9 vs 等离子体 0.89——残差被线圈奇异点主导。网络只预测 psi_plasma，psi_total = psi_plasma_pred + Σ I_k G_k 解析加回
- 3.4 **物理残差损失（做法1 单阶段，exp101）**: L = MSE(psi_plasma) + w_pde·‖Δ\*ψ_pred + μ₀RJ_data‖²（core mask，pde_scale=0.362 Wb/m²）。RHS 冻结数据 J。效果：平滑正则，精度 0.72% < 纯数据 0.84%
- 3.5 **两阶段自洽（做法2，exp102）**: 阶段1 监督 psi_plasma+J 两通道（val rel L2<3% 自动切换，e29）；阶段2 加自洽残差（RHS 用网络自己的 J_pred）+ Ip 积分约束（w_ip=1.0, ip_scale=5.51e5 A）。**30-epoch 线性预热**修复阶段2爆炸（无预热时阶段2首 epoch pde=5.38，val 2.67%→54%）
- 3.6 物理量纲归一化: PDE 项 /pde_scale，Ip 项 /ip_scale

### 4. 实验与结果（~1300字）
- 4.1 数据: data_v5/dn（MAST 真实装置 DN 位形，freegs 生成，65²，2000/500/500），N=500 嵌套子集
- 4.2 训练: AdamW lr=1e-3, batch=16, 800 epochs, ReduceLROnPlateau + 早停（~15min/实验）
- 4.3 **三实验对比（核心表格 Table 1）**:

| 指标 | exp011 纯MSE | exp101 单阶段RHS | exp102 两阶段自洽 |
|---|---|---|---|
| rel L2 psi_total | 0.84% | **0.72%** | 0.80% |
| GS 残差 (core, vs 真值 FD 下限 0.40%) | ~1× | 3.3× | 3.8× |
| Ip 误差 | — | — | **0.21%** |
| J mask 内 rel L2 | — | — | **1.36%** |
| find_critical 失败 | 1/500 | 0/500 | 0/500 |
| X 点误差 (cm) | ~3.2/2.2 | 0.55/0.57 | 0.60/0.63 |

- 4.4 结果分析:
  - 物理残差不降精度反升（0.72 < 0.84）——输出侧平滑正则
  - 两阶段自洽: J ≈ -Δ\*ψ/μ₀R（J 1.36% + GS 1.51% 同时成立），Ip 积分约束 0.21%
  - 全网格 J 36.9% 是 mask 外谱振铃假指标，mask 内 1.36% 为真实质量
  - 几何指标: 分离面 ~0.3cm、O 点 ~0.2cm、X 点 ~0.6cm，find_critical 全通过
- 4.5 预测实例图: psi_total 真值/预测/|diff|（best/worst，exp102 含 J 行）
- 4.6 推理速度: GPU 前向 ~1.6ms/次 vs freegs 秒级（665×）

### 5. 结论（~300字）

---

## 图表（6 图 + 2 表）

> PINN 仅作引言背景介绍（一段），不再单独成章、不再配图。gs_pinn_torch_2.py
> 可在引言中一句话提及（验证 PINN 原理可行），不占篇幅。

### 图

| 编号 | 内容 | 状态 |
|------|------|------|
| Fig 1 | 自由边界平衡磁通分解 3 子图（ψ_total / ψ_plasma / ψ_coils，白色平滑分离面，X点/磁轴仅在总通量子图）| ✅ paper/figures/fig1_freeboundary_problem.png |
| Fig 2 | 方法演进图: 传统 freegs → PINN → PINO | ✅ paper/figures/fig2_evolution.png |
| Fig 3 | FNO2d2608 架构 + 18ch 输入 + coil 分离示意（psi_plasma → +ΣI_kG_k → psi_total）| ✅ paper/figures/fig3_architecture.png |
| Fig 4 | 两种方案训练曲线（train loss + val rel L2，两阶段方案标阶段切换线，无 loss 设计示意图）| ✅ paper/figures/fig4_loss_design.png |
| Fig 5 | 两种方案各自的最优样本预测实例（KAN 论文图3风格 2×3：ψ 真值/预测对比 + 全域相对误差 + 全域 PDE 残差；单阶段 J 取数据、两阶段 J 取网络自洽）| ✅ paper/figures/fig5_predictions.png |
| Fig 6 | 三方法对比柱状图（rel L2 + Ip 误差）+ 验证曲线（标注为 纯数据基线/单阶段方案/两阶段自洽方案）| ✅ paper/figures/fig6_comparison.png |

> 图由 `paper/figures/make_figures.py`（Fig 1-4、6）与 `paper/figures/make_fig5_prediction.py`
> （Fig 5）生成（英文标签，待精修中文标签）；数据全部来自
> data_v5/dn/test.npz 与各实验 metrics.json/history.json。

### 表

| 编号 | 内容 |
|------|------|
| Table 1 | 三实验定量对比（rel L2, RMSE, GS 残差, Ip, J, 几何误差, find_critical）|
| Table 2 | 18ch 输入通道明细（5 参数 + 11 线圈电流 + 坐标）|

---

## 核心看点（3 点）

1. **端到端 + 物理残差**: 只给可测量量（线圈电流+工程参数）直接出物理正确的场；加 GS 残差损失精度不降反升（0.72% vs 0.84%）
2. **coil 分离设计**: 网络只预测 psi_plasma，线圈场解析加回——避免残差被线圈奇异点主导（物理洞察驱动的架构设计）
3. **两阶段自洽 + 权重预热**: psi↔J 自洽（J 1.36% + Ip 0.21%）是纯数据管线拿不到的能力；30-epoch 预热修复阶段切换爆炸

---

## 关键文件

| 文件 | 用途 |
|------|------|
| `src/gs_pino_dn_fno_2608/model_dn_fno.py` | FNO2d2608 架构 |
| `src/gs_pino_fno_phys/train_pino.py` | exp101/102 训练（rhs/twostage 模式）|
| `src/gs_pino_fno_phys/losses_pino.py` | 物理损失（lap_star, Ip 约束, 自洽残差）|
| `src/gs_pino_fno_phys/data_pino.py` | 18ch 数据加载 + coil 分离 |
| `src/gs_pino_fno_phys/evaluate_pino.py` | 评估（rel L2, GS 残差, Ip, J）|
| `src/gs_pino_fno_phys/verify_pde.py` | 物理项验证（FD 一致性, Ip 重建）|
| `dn_fno_2608/scripts/run_exp101_102_pino.sh` | 复现脚本 |
| `dn_fno_2608/experiments/exp101_pino_rhs_n500/` | exp101 结果（metrics.json, figures/）|
| `dn_fno_2608/experiments/exp102_pino_twostage_n500/` | exp102 结果（metrics.json, figures/）|
| `dn_fno_2608/experiments/exp011_coil_input_v5/` | exp011 纯数据基线 |
| `GS-PINN/gs_pinn_torch_2.py` | §3 PINN baseline |
| freegs（data_v5 生成器） | §2 自由边界求解器 |

---

## 验证标准

1. 结果全部来自现成 metrics.json，无需重跑实验
2. 核心数字: exp011 0.84% / exp101 0.72% / exp102 0.80%, Ip 0.21%, J 1.36%
3. 图表数据可追溯到各实验的 metrics.json / figures/
4. 篇幅控制在 4000-6000 字（正文，不含图表）
