# 复现 arXiv:2608.05555v1 — Double-Null 自由边界 GS 平衡的 FNO 神经算子代理

## Context（背景）

目标：**100% 复现**论文 *"Millisecond-Scale Neural Operator Surrogates for Double-Null Free-Boundary Grad-Shafranov Equilibria"*（Plamen G. Krastev, 2026），覆盖数据生成、模型、损失、训练、评估全套，并对照论文 Table I/II/III 逐项数字。数据集同时为下一阶段 PINO（物理约束训练）预留全部所需字段。

依据与决策（已与用户确认）：
- 论文代码未公开（"upon reasonable request"），**不联系作者**，按论文描述 + 合理默认假设实现，用硬指纹校验：模型参数量精确 = 4,770,241、Table I 的 rel L2 = 0.061%±0.006、数据接受率 6000/6000、freegs 中位求解时间 ≈1.77 s。
- **本地 GPU 训练**（conda `torch5060`，CUDA 可用，freegs 0.8.3.dev54 已装）；延迟数字按本地硬件报告（论文 A100 的 2.77 ms 绝对值不可比，方法论一致）。
- **先冒烟后全量**：小规模冒烟验证 pipeline，跑通后全量生成 6000 样本 + 12 次缩放训练 + 完整评估。
- 所有零碎产物（计划副本、数据、结果、日志、脚本）放入**新建文件夹 `dn_fno_2608/`**（仓库根目录）；代码模块按仓库惯例放 `src/gs_pino/`。
- 数据生成**先基准计时再并行分阶段生成**，支持断点续跑。

## 论文规格（复现目标，已从 PDF 全文提取）

| 项 | 规格 |
|---|---|
| 问题 | 双零（DN）自由边界 GS 平衡，TestTokamak 几何；域 R∈[0.1,2.0], Z∈[-2.0,2.0] m，65×65 均匀网格 |
| 参考示例 | freegs `06-xpoints.py`：TestTokamak + 双 X 点 `control.constrain(xpoints=[(1.1,-0.8),(1.1,0.8)])`（与论文同构，非 MAST 的 03 示例） |
| 输入（9 通道 65×65） | R、Z（线性映射到 [-1,1]）；Paxis、Ip、fvac（z-score 标准化广播）；下/上 X 点坐标 (R_lo^X,Z_lo^X),(R_up^X,Z_up^X)（z-score 广播） |
| 输出 | 单通道 ψ(R,Z)（总极向磁通，z-score 标准化） |
| 采样 | Paxis∈[200,3000] Pa、Ip∈[5e4,4e5] A、fvac∈[0.5,3.0]（均匀）；X 点参考 (1.1,±0.6) m + 上下对称抖动 \|δR\|,\|δZ\|≤0.02 m（**原文 Eq. 4 如此**，范围小是论文设定） |
| 数据 | 训练 5000 (seed 123) / 验证 500 (seed 456) / 测试 500 (seed 789)；接受准则：收敛 + \|Ip,sol−Ip,tgt\|/Ip,tgt≤10% + find_critical 找到 ≥2 个 X 点（论文 6000/6000 全通过） |
| 求解器 | FREEGS Picard 迭代，γ=1e-12，isoflux 约束到固定外中平面参考，maxits=50，rtol=1e-3 |
| 模型 | FNO（neuraloperator 库）：lifting 9→64；4 个 Fourier 层；每层截断谱卷积（保留 nmodes=(16,16) 最低模 + 逐点混合）；隐宽 64；无位置嵌入；最终 1×1 投影 → 1 通道；**总参数 4,770,241** |
| 训练 | 纯 MSE（无 PDE/几何损失）；AdamW lr=1e-3, wd=1e-4；ReduceLROnPlateau（验证 rel L2 连续 20 epoch 无改善 → lr 减半，下限 1e-5）；早停 patience 75；batch 16；4 workers；每档 3 个初始化 seed |
| 缩放实验 | N∈{500,1000,2000,5000} 固定嵌套子集（置换 seed 12345），每档 3 seeds |
| 目标数字 | N=5000：rel L2 0.061%±0.006，物理 RMSE 1.79e-5 Wb；最佳模型（seed 3）测试 rel L2 0.052%、RMSE 1.54e-5 Wb、最佳验证 epoch 310；GS 归一化残差 2.29（与 freegs 基线一致）；freegs 中位 1.77 s |

## 新建文件夹结构（阶段 0 创建，复制本计划为 PLAN.md）

```
dn_fno_2608/
├── PLAN.md            # 本计划副本
├── data/              # train_123/ val_456/ test_789/（分块 npz，生成完合并）
├── outputs/           # fno_n{500,1000,2000,5000}_s{1,2,3}/ (best.pt, history.json, args.json)
│   └── report/        # 评估指标 JSON、Table I/II/III 对照、图表
├── logs/              # 数据生成日志（含每样本耗时）
└── scripts/           # run_generate.sh / run_scaling.sh / run_evaluate.sh / run_latency.sh / make_tables.py
```

## 实施阶段

### 阶段 0：冒烟验证（验收：全部通过才进全量）

1. 创建 `dn_fno_2608/` 结构，复制计划为 `PLAN.md`。
2. **单样本 DN 求解冒烟**（临时脚本，以 `06-xpoints.py` 为模板）：TestTokamak + `Equilibrium(Rmin=0.1, Rmax=2.0, Zmin=-2.0, Zmax=2.0, nx=65, ny=65, boundary=freeBoundaryHagenow)` + `ConstrainPaxisIp(paxis, Ip, fvac)`（默认 alpha_m=1.0, alpha_n=2.0, Raxis=1.0）+ `control.constrain(xpoints=[lo, up], gamma=1e-12)` + 论文所述 isoflux 约束（两 X 点各连固定外中平面参考点 (1.5, 0.0)）+ `freegs.solve(..., rtol=1e-3, maxits=50)`。验证：`find_critical` 返回 2 个 X 点、ψ_bndry=½(ψ_lo^X+ψ_up^X) 可行、`profiles.pprime/ffprime` 可生成 dpdpsi/FdFdpsi 场、**若 isoflux 导致不收敛/接受率受损则回退纯 xpoints 模式（06-xpoints.py 原样）并记录偏差**。
3. **耗时基准**：连续求解 10 个样本，测单次求解均值/中位数（论文对照 1.77 s），据此估算 6000 样本 × 并行核心数的总时长，写入日志。
4. **模型冒烟**：FNO 前向跑通 + 参数计数脚本（见阶段 2 指纹匹配）。

### 阶段 1：数据生成 — 新文件 `src/gs_pino/generate_dn_dataset.py`

复用 [generate_freegs_dataset.py](src/gs_pino/generate_freegs_dataset.py) 骨架（joblib 并行、`_solve_with_retry`、npz 保存模式），改动：
- Z 域 [-1,1] → **[-2,2]**；参数采样只保留 `paxis/Ip/fvac`（论文范围），alpha_m=1.0、alpha_n=2.0 固定（论文："剖面形状由 FREEGS 设置固定"）。
- X 点采样：`δR, δZ ~ U[-0.02, 0.02]`（独立，**论文 Eq. 4 原文**），下点 `(1.1+δR, -(0.6+δZ))`、上点 `(1.1+δR, +(0.6+δZ))` —— 严格上下对称。
- 约束：`control.constrain(xpoints=[lo, up], isoflux=[(lo_R, lo_Z, 1.5, 0.0), (up_R, up_Z, 1.5, 0.0)], gamma=1e-12)`（冒烟阶段已验证）。
- 接受准则：solve 收敛（try/except）且 `|eq.plasmaCurrent()−Ip|/Ip ≤ 10%` 且 `find_critical(eq.R, eq.Z, eq.psi())` 的 xpt 列表长度 ≥2；拒绝则重采样参数重试。
- **每样本保存全字段（为下一阶段 PINO 预留）**：`psi_total`、`psi_plasma`、`psi_plasma_norm`、`psi_coils`、`R`、`Z`、`mask`（critical.core_mask）、**`coil_currents`（4 个控制线圈电流）**、**`greens`（4×65×65 Green 函数，coil.createPsiGreens）**、**`dpdpsi`、`FdFdpsi`**（`profiles.pprime(psi_norm_field)` / `profiles.ffprime(...)`，GS 残差 RHS = −μ₀R²·dpdpsi − FdFdpsi 的全部所需）、`params`=[Ip,paxis,fvac]、`x_coords`=[R_lo,Z_lo,R_up,Z_up]、`psi_axis/psi_bndry`、`R_axis/Z_axis`、`L/Beta0`。全部 float32。估算约 220 KB/样本 × 6000 ≈ 1.3 GB。
- **分阶段生成（断点续跑）**：按 train/val/test 分别生成；每个 split 再按 **500 样本一块** 输出到 `dn_fno_2608/data/{split}/chunk_{i:03d}.npz`（块文件即天然断点——重跑时跳过已存在的块），全部完成后用现有 [merge_datasets.py](src/gs_pino/merge_datasets.py) 合并成 `train_123.npz / val_456.npz / test_789.npz`。
- 并行：joblib `n_jobs=-1`（现有代码同款，Windows spawn 下 `_solve_one` 已是模块级函数，兼容）；`--n-jobs` 可显式指定。总时长 = 6000 × 单次耗时 ÷ 核心数（阶段 0 已基准）。
- 记录每样本求解耗时与接受率（目标 100%）进 `dn_fno_2608/logs/`。
- 验收：6000/6000 接受、参数范围统计与论文一致、合并后的 npz 字段完整。

### 阶段 2：模型 — `src/gs_pino/model_dn_fno.py` + 参数指纹匹配

- **neuraloperator 完整本地安装**：`pip install neuraloperator tensorly`（论文原库，torch5060 环境内完整可用，不依赖远程）；同时保留仓库已验证的 [models.py:278-305 SpectralConv2d](src/gs_pino/models.py#L278-L305) 手写版本作为对照与回退。
- 按论文描述实现 FNO2d：lifting(9→64) → 4×[谱卷积(截断 modes=(16,16) 复数权重) + 1×1 逐点混合] → 1×1 投影(→1)。激活 **GELU**（neuraloperator 默认，论文未指明的最合理假设）。
- **参数指纹**：写枚举脚本（宽度/模数/rfft 半模 (16,9)/复权计数惯例/skip 形式组合，库版与手写版对照）搜索 trainable params **精确 == 4,770,241** 的配置。已知：标准配置（width 64, modes(16,16), 复权×2）≈8.41M 对不上；接近值 width 64 + modes(12,12)（或 rfft 半模 16×9）≈4.74M。以精确匹配为硬验收，匹配不到则取最接近配置并在 PLAN.md 记录偏差。
- 冒烟：前向输出形状 (B,1,65,65)，打印参数量。

### 阶段 3：数据类 + 训练 — `src/gs_pino/data_dn_fno.py`、`src/gs_pino/train_dn_fno.py`

- **数据类**（新写，不用仓库现有的 65→64 插值预处理——论文直接用 65×65）：构造 9 通道输入（R,Z→[-1,1]；Paxis/Ip/fvac/X 点 4 坐标用**训练集**均值/标准差 z-score 后广播）；目标 ψ z-score；归一化统计量随 checkpoint 保存（评估逆变换用）。
- **训练**（新写，不用现有 cosine 退火+课程学习）：纯 MSE；AdamW(1e-3, 1e-4)；`ReduceLROnPlateau(patience=20, factor=0.5, min_lr=1e-5)` 监控验证 rel L2；早停 patience 75（保存 best 权重 + history.json + args.json）；batch 16、workers 4；AMP 可用；max_epochs 设 800（论文未说明，最佳 epoch 310 + patience 75 通常 <400 内收敛）。
- **冒烟**：N=500 子集、1 seed、小 epoch，验证收敛趋势与保存/恢复逻辑。
- **全量**：`dn_fno_2608/scripts/run_scaling.sh` 跑 12 次（N∈{500,1000,2000,5000} × seed∈{1,2,3}）；嵌套子集用固定置换（seed 12345）从 train_123 前 N 个索引取，保证大集合严格包含小集合；输出到 `dn_fno_2608/outputs/fno_n{N}_s{s}/`。

### 阶段 4：评估 — `src/gs_pino/evaluate_dn_fno.py`（+ 延迟脚本）

- **场级**：归一化 rel L2（均值/标准差/中位/P95）、逆归一化物理 RMSE(Wb)——对照 Table I。
- **几何指标**（对照 Table II）：复用/改造 [screening.py:88-133 find_plasma_geometry](src/gs_pino/screening.py#L88-L133) 的 find_critical 调用：定位 O/X 点（500 样本全成功，论文零失败）；ψ_bndry=½(ψ_lo^X+ψ_up^X) 处提取**最大闭合等值线**为分界面（matplotlib contour），Q3+3×IQR 距离滤波剔除偏滤器腿伪影（真值/预测同等处理）；报告分界面平均最近点距离/Hausdorff/面积相对误差/上下 X 点与 O 点误差/|Δψ_bndry|。
- **GS 残差诊断**：对预测场逆归一化后二阶中心差分算 Δ⋆ψ（复用 [screening.py:209-263 compute_gs_residual](src/gs_pino/screening.py#L209-L263) 思路），RHS = −μ₀R²·dpdpsi − FdFdpsi（数据集预存字段），等离子体掩模内归一化；对照 freegs 真值场同格式基线（目标两者一致，≈2.29）。
- **延迟**（对照 Table III）：batch 1、500 测试样本；GPU 用 `torch.cuda.Event` 同步计时、CPU 单线程 `perf_counter`；freegs 基线用与生成完全相同的求解配置跑 500 样本；报告中位/p95/加速比（本地硬件，注明与论文 A100 不可直接比）。
- 产物：`dn_fno_2608/outputs/report/` 的 metrics JSON + 可视化图（复用 [visualize_freegs.py](src/gs_pino/visualize_freegs.py) 风格）。

### 阶段 5：汇总对照 — `dn_fno_2608/scripts/make_tables.py`

生成 Table I/II/III 对照表（复现值 vs 论文值 + 偏差），输出复现报告 markdown，明确列出：哪些数字复现成功、哪些因硬件差异（延迟）或未指明细节（激活函数等）有偏差。

## 假设与决策记录（论文未指明处）

1. **参考示例**：以 freegs `06-xpoints.py` 为模板（TestTokamak + 双 X 点 constrain），而非 MAST 的 03 示例。
2. **isoflux 约束**：按论文原文保留"isoflux 约束到固定外中平面参考"——两 X 点各连固定点 (1.5, 0.0) m（仿 MAST DN 示例 (1.45,0.0) 惯例；该点仅锚定 gauge）。冒烟阶段验证；若不收敛/接受率受损则回退 06-xpoints.py 纯 xpoints 模式并记录偏差。
3. **激活函数**：GELU（neuraloperator 库默认）。
4. **剖面形状**：ConstrainPaxisIp 默认 alpha_m=1.0, alpha_n=2.0, Raxis=1.0。
5. **X 点抖动**：±0.02 m 是论文 Eq. 4 原文（范围小为论文设定）；δR、δZ 独立均匀采样，上/下点共用同值并镜像 Z —— 严格上下对称。
6. **epoch 上限**：论文未说明，取 800（记录实际触发早停的 epoch 与论文 310 对照）。
7. **参数指纹**：枚举配置至精确 4,770,241；不匹配则记录最接近值。
8. **数据集布局**：分块 500/块生成后合并为 3 个独立 npz（train/val/test），float32，不压缩。
9. **目标 ψ**：总极向磁通 `eq.psi()`（论文输出"完整极向磁通场"）。
10. **输出通道归一化**：ψ 用训练集均值/标准差（论文原文）；训练与 rel L2 均在归一化域计算，物理量逆变换后计算。

## 验证（端到端验收）

1. **冒烟**：阶段 0 四项全过（DN 求解 2 X 点、isoflux 验证、耗时基准、FNO 前向）。
2. **数据**：6000/6000 接受率；train/val/test 参数分布统计与论文范围一致；全字段（含 coil_currents/greens/dpdpsi/FdFdpsi）完整。
3. **参数指纹**：`sum(p.numel()) == 4,770,241`。
4. **Table I**：N=5000 三 seed 平均 rel L2 ≈ 0.061%±0.006、物理 RMSE ≈1.79e-5 Wb（容差内视为成功）；N 档缩放律 ≈ N^-0.68。
5. **最佳模型**（N=5000, seed 3）：测试 rel L2 ≈0.052%、RMSE ≈1.54e-5 Wb、>95% 样本 <0.12%、最佳验证 epoch ≈310。
6. **Table II**：分界面均值误差 ~0.07 cm、X 点 0.11–0.16 cm、O 点 ~0.03 cm 量级；find_critical 500/500 找到 2 X 点。
7. **残差**：FNO 归一化 GS 残差 ≈ freegs 基线（比值 ≈1.0）。
8. **延迟**：freegs 中位 ≈1.77 s（论文对照）；本地 GPU/CPU 数字与加速比记录。

## 关键文件

| 文件 | 动作 |
|---|---|
| `dn_fno_2608/` | 新建（PLAN.md 副本、data/、outputs/、logs/、scripts/） |
| [src/gs_pino/generate_dn_dataset.py](src/gs_pino/generate_dn_dataset.py) | 新建，复用 generate_freegs_dataset.py 骨架 + 分块断点续跑 |
| [src/gs_pino/data_dn_fno.py](src/gs_pino/data_dn_fno.py) | 新建，9 通道输入数据集（65×65 直接使用） |
| [src/gs_pino/model_dn_fno.py](src/gs_pino/model_dn_fno.py) | 新建，neuraloperator 库版 + 手写 SpectralConv2d 对照 |
| [src/gs_pino/train_dn_fno.py](src/gs_pino/train_dn_fno.py) | 新建，纯 MSE + ReduceLROnPlateau + 早停 |
| [src/gs_pino/evaluate_dn_fno.py](src/gs_pino/evaluate_dn_fno.py) | 新建，复用 screening.py 几何/残差诊断 |
| [src/gs_pino/merge_datasets.py](src/gs_pino/merge_datasets.py) | 复用（分块合并） |
| `dn_fno_2608/scripts/*.sh` + `make_tables.py` | 新建，一键复现与对照表 |

**环境**：conda `torch5060`（torch 2.12+cu128）；`pip install neuraloperator tensorly`（阶段 2，本地完整安装）；freegs/joblib 已装。
