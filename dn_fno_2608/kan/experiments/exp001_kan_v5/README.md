# exp001 — KAN 19ch 混合训练（DN+SN）：多阶段 KAN 端到端 psi 生成

> 实验日期：2026-08-19 ｜ 状态：**M1 完成——未达验收（rel L2 20.9%，
> 判定容量受限），见 §3 与 §7；容量增强变体见 `../exp002_kan_v5/`**
> M2/M3 工具链完成；OOD 数据 `kan/data_ood/`（DN 520 / SN 487 行）
> 数据：`data_v5/`（MAST DN+SN，逗号拼接混合，N=500，seed 1）
> 对照：FNO exp011（coil 18ch 无 config，rel L2 0.894%）
> 方法：论文 aps.75.20260331 四阶段（监督 → 半监督 → 剪枝 → 符号回归
> → 线性外推+位型分类），本文档覆盖 M1（训练+评估+可视化）与 M3 方案；
> M2 见 `sr/`、M3 见 `ext/`

## 1. 输入通道（19 通道，`data_kan.py` 常量一致）

| # | 通道 | 含义/单位 | 来源 |
|---|---|---|---|
| 1 | R | 网格 R 坐标（[-1,1]） | 固定 MAST 65×65 |
| 2 | Z | 网格 Z 坐标（[-1,1]） | 固定 MAST 65×65 |
| 3 | Ip | 等离子体电流 (A) | params[0] |
| 4 | paxis | 磁轴压强 (Pa) | params[1] |
| 5 | fvac | 真空通量函数 f (Wb/m) | params[2] |
| 6 | alpha_m | 剖面形状指数 m | params[3] |
| 7 | alpha_n | 剖面形状指数 n | params[4] |
| 8 | config | 位形标签（min-max 后 DN=1 / SN=−1） | data_v5 每样本 |
| 9 | I_P2U | 上内偏滤器线圈电流 (A) | coil_currents[0] |
| 10 | I_P2L | 下内偏滤器线圈电流 (A) | coil_currents[1] |
| 11 | I_P3U | 上中偏滤器线圈电流 (A) | coil_currents[2] |
| 12 | I_P3L | 下中偏滤器线圈电流 (A) | coil_currents[3] |
| 13 | I_P4U | 上外偏滤器线圈电流 (A) | coil_currents[4] |
| 14 | I_P4L | 下外偏滤器线圈电流 (A) | coil_currents[5] |
| 15 | I_P5U | 上闭合线圈电流 (A) | coil_currents[6] |
| 16 | I_P5L | 下闭合线圈电流 (A) | coil_currents[7] |
| 17 | I_P6U | 上开放线圈电流 (A) | coil_currents[8] |
| 18 | I_P6L | 下开放线圈电流 (A) | coil_currents[9] |
| 19 | I_P1 | 中心螺线管电流 (A) | coil_currents[10] |

标量通道 17 个（5 params + config + 11 线圈）min-max 到 [−1,1]；ψ/J 输出 z-score
（J 仅在等离子体单元上统计）。config 通道为显式位形标签（区别于 exp011 的
18ch 无 config——FNO 侧 exp008 已验证 config 有帮助）。

## 2. 训练设置

- N=500（拼接 4000 样本上嵌套取 500）seed 1，MSE/Adam(1e-3)/MultiStepLR
  [300,500,700,900]×0.5，1000 epochs，每 epoch 32 样本 × 256 点
- 四阶段同一次训练：t<120 监督；120≤t<240 λPDE/λIp 生效；
  240≤t<600 +λ(t)Lreg 斜坡至 0.001；t=600 剪枝（阈值 1e-2·max 连接强度）；
  之后微调（式 8）至 1000
- λ(t) 按论文式 16：0 → 线性(120–240) → 恒 0.001(240–600) → 0(≥600)
- LPDE 用解析导数（B 样条一/二阶导链式法则），排除线圈单元+边界；
  LIp 用无偏等离子体地层估计器；best 相位感知（剪枝前后分开跟踪）
- stats 来自拼接后全量 train pool；验收指标见 §4

## 3. 结果（test 分桶，rel L2 mean %）

_全部指标未达验收 → M1 判定：**容量受限**（点式 1260 参 KAN 不足以刻画
19ch MAST 混合位形；exp011 FNO 同输入 0.894%）。原因与对策见 §7。_

| test | **KAN 19ch (exp001)** | FNO exp011 coil 18ch | 验收 | 判定 |
|---|---|---|---|---|
| DN | **20.96** | 0.841 | ≤1.3 | ✗ |
| SN | **20.79** | 0.948 | ≤1.5 | ✗ |
| 整体（n=1000） | **20.87** | 0.894 | ≤1.4 | ✗ |
| RMSE (Wb) DN/SN | 8.72e-3 / 9.47e-3 | 3.41e-4 / 3.98e-4 | — | ✗ |
| R² DN/SN | 0.942 / 0.943 | — | 论文 0.9953 参考 | ✗ |
| GS 残差比 DN/SN | 1.165 / 2.125 | 1.013 / 1.008 | ∈[0.9,1.1] | ✗ |
| X 点误差 (cm) | 11.5/13.1；4.4/— | 3.16/2.23；1.25/— | ≤2×exp011 | ✗ |
| sep_mean (cm) DN/SN | 12.3 / 11.1 | 0.62 / 0.73 | ≤2 | ✗ |
| Jψ 自检 (%) DN/SN | 16.9 / 29.9 | — | ≤5 | ✗ |

训练轨迹：val rel L2 98.8%→21.9% (ep100)→20.0% (ep450)→平台；剪枝 @600
去除 67/210 连接 (31.9%，接近论文 27.5%)；best = 剪枝后 ep830 val 20.011%
（best_pre_prune ep580 19.093%）；find_critical 失败 54/1000（几何指标在
其余样本上统计）。

## 4. 偏差记录（相对论文 aps.75.20260331）

- **LPDE 无 L_coils 项（已修正，2026-08-19）**：初版用总场残差
  Δ*ψ_total + μ0RJ_plasma（"排除线圈单元后 Δ*ψ_coils≈0"假设）——**该假设
  在粗网格上不成立**：中心差分把线圈奇性涂抹出排除区（近线圈区
  |Δ*ψ_coils|≈1.19），真空区旧残差在完美预测下仍有 0.406（自检实测）→
  Lψ（拟合 ψ_total≈线圈场，幅值 183%）与 LPDE（真空 Δ*ψ_total=0）互相矛盾，
  是 exp001/002/003 欠拟合平台的元凶之一。**修正**：`--pde-coils-model plasma`
  （默认）残差 = Δ*ψ_total − Δ*ψ_coils + μ0RJ，Δ*ψ_coils 由数据预计算
  （data_kan `psi_coils_lap`，中心差分）；真值场自检：等离子体区 0.0015、
  真空区 0.0023（旧 0.0024/0.406）。exp001/002/003 用旧语义，记录数字不变
- **LIp 无偏化**：50/50 分层采样下直接缩放有偏（~22%）；只对等离子体层抽中点
  加权 n_p/k_p·dA（无偏，单次抽取噪声 ~15%）
- **剪枝相位**：best.pt 存剪枝后最优（phase="pruned"），best_pre_prune.pt 存剪枝前
- **延迟基准**：M2 起补充（论文表 5：KAN-2 推理 6.54ms）
- **M3 包络用 axis-aligned box 而非凸包**：16D（5 params+11 线圈）训练点云
  因线圈电流近似由等离子体参数决定而近退化，scipy qhull 在原始 16D 与 PCA
  k=12 子空间均不收敛（2026-08-19 实测 >2min 无返回）→ 采用计划的 box 兜底
  （Mahalanobis 椭球包络也已实现 `--hull-mode ellipsoid`，但 thr=0.41 太紧，
  OOD 保留率 <10%）。最近边界点 = 逐维 clip；外推距离按训练每维范围归一
- **M3 分类**：分离面 X 点计数（psi ≥ max_saddle − tol），≥2→DN、==1→SN；
  tol 由 in-dist 校准（max(1e-3, 10×DN 上两个分离面 X 点 psi 差中位数)，
  且不超过 0.5×SN 最小间隙）；平滑场漏检用 4× 三次上采样重扫兜底
- **M3 外推梯度**：归一化域中心差分（h_norm=5e-3，16 连续维 × 2 次前向），
  换算回物理梯度；config 通道离散不参与梯度

## 5. 产物

```
smoke/              M0 冒烟（N=100/50ep）
model_b19ch_kan_mix/   best.pt/history.json/args.json（M1）
eval_all/ eval_dn/ eval_sn/   metrics.json
figures_dn/ figures_sn/      fig1-3
sr/ ext/            M2/M3
kan/data_ood/       M3 OOD 数据（dn/sn test.npz + hull.json + ood_box_{cfg}.json）
```

## 6. M3 方案与当前数据（结果待 M1 ckpt 回填）

- OOD 采样范围锚定训练 500 行 min/max：Ip/paxis ×[0.8,1.25]、fvac±0.15、
  alpha±0.3、X 点抖动 ×1.5（R 0.09/Z 0.15）；`gen_ood_kan.py` 复用 FNO
  generator（registry 覆盖，FNO 代码零触碰），probe 接受率 DN/SN 均 100%
- 当前数据集：DN 600 候选 → box 过滤保留 520；SN 600 候选 → 保留 487
  （≥200/配置 验收阈值满足；分块生成可续 --skip-probe/--skip-generate，
  探针隔离在 probe/ 子目录避免污染生成块）
- 外推验收：in-dist 校准分类 ~100%；OOD Accuracy ≥85%（论文 89.1）+
  Precision/Recall/Specificity（论文 72.1/98.9/85.3）+ 距离-误判分析图
  （镜像论文图 10/11）→ `ext/classify_metrics.json`

## 7. M1 结论：容量受限（未达验收）→ exp002

- **M2 管道验证中发现并修复三个 bug**（2026-08-19，model_kan/symbolic_kan）：
  1. **torch B 样条边界 bug**：`_bspline_bases` 右端钳制 eps=1e-10·span 转
     float32 舍入为 0 → x=±1 未被钳制，样条贡献在 |R| 或 |Z|=1 的边界环
     （65×65 网格 6% 点）恰好消失。修复：eps=1e-6·span（partition of
     unity 恢复，导数仍解析，模型自测通过）。M1 权重按旧语义训练——
     本实验记录数字均为 as-trained 语义（用 `kan/_eval_m1_astrained.py`
     重评恢复）；修复后语义重评 24.8% 属"未训练过的函数"，仅作参考。
     **exp002 起使用修复后语义**
  2. **符号拟合范围**：symbolic_kan 的 L1 连接按 [-1,1] 拟合，但隐藏激活
     实际范围可达 [-12,1] → 多项式外推爆炸（实测 KAN-SR 误差 610878%）；
     改为按测试数据观测范围逐节点拟合（输入通道 + 隐藏激活）
  3. **生成模块逐层 clip**：测试归一化通道可能越出 [-1,1]（训练/测试
     min-max 差异），torch 样条钳制扩展而多项式表达式外推爆炸；生成的
     symbolic_gs_kan.py 按拟合范围逐层 clip。另：hidden 数硬编码 10 →
     从 arch 取（exp002 hidden 24 兼容）
- **M2 在 M1 ckpt 的结果**（`sr/verify.json`）：SR 附加误差 48pp——受边界
  bug 影响（SR 用正确样条语义无法复刻 float32 归零怪癖），仅作管道验证；
  验收数字以 exp002 为准
- **M3 在 M1 ckpt 的结果**（`ext/classify_metrics.json`）：OOD 分类 accuracy
  46%、DN recall 2.1%、475/1007 未分类——20% 场误差摧毁分离面拓扑，符合
  失败模型的预期（管道端到端跑通）；验收（≥85%）以 exp002 为准

- **现象**：val rel L2 在 ~19-20% 平台（1000 轮全程），train≈val（psi MSE
  0.026 ≈ 全场 rel L2 16%）——欠拟合而非过拟合；J 自检/GS 残差比/X 点/分离面
  全项随 psi 误差放大失守
- **诊断**：论文架构（1 隐含层 ×10 节点、B 样条 2 区间）在其 14ch 较简单
  问题上报 0.631%；本实验 19ch（5 params + 11 线圈 + config）MAST 混合位形
  的点式映射难度显著更高——每连接每维只有 G+degree=4 个二次 B 样条基，
  11 个线圈通道的平衡响应自由度不足以被 1260 参数表达（FNO 以卷积空间结构
  取胜，同输入 0.894%）
- **对策**：`../exp002_kan_v5/` —— 容量受控放大（hidden 24 / grid 4 /
  2000 轮，λ/剪枝窗口按比例平移），方法学不变；若仍不足则进一步加大
  网格区间数（grid 6-8，空间样条分辨率是 X 点/分离面精度的关键杠杆）
