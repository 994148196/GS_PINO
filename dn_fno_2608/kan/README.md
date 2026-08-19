# KAN 版 Grad-Shafranov 自由边界求解（Kolmogorov-Arnold Networks）

> 论文参考：*基于 Kolmogorov-Arnold Networks 的 Grad-Shafranov 方程自由边界问题
> 求解方法*，物理学报 75(2026)150504，DOI 10.7498/aps.75.20260331（见 `REFS/`）
> 代码：`src/gs_pino_kan_2608/`（全手写 B 样条 KAN，纯 PyTorch，禁 pykan/efficient_kan）
> 数据：`dn_fno_2608/data_v5/`（MAST DN+SN 混合，与 FNO exp011 同源）
> 对照：FNO exp011（coil 18ch，rel L2 0.894%）— 基线脚本/产物零触碰

## 1. 方法（论文四阶段 + 外推分类）

| 阶段 | 论文 | 本实现 | 状态 |
|---|---|---|---|
| KAN-1 | 监督训练 Lψ+LJψ | train_kan.py（--reg-weight 0 或 t<120） | M1 |
| KAN-2 | +λPDE·LPDE（式9）+λIp·LIp（式10） | 同一次训练内 t≥120 生效 | M1 |
| KAN-3 | +λ(t)·Lreg（式11-16）→ t=600 剪枝 | 同一次训练内 t≥240 生效，600 轮剪枝+冻结+微调 | M1 |
| KAN-SR | 逐连接符号化（最小二乘候选族） | symbolic_kan.py（M2） | pending |
| KAN-EXT | 凸包外推+一阶泰勒+位型分类 | extrapolate_kan.py + gen_ood_kan.py（M3） | pending |

网络结构按论文：点式 `(19 标量输入, R, Z) → (ψ, Jψ)`，1 隐含层 × 10 节点，
B 样条（2 区间、3 阶、输入域 [−1,1]、基函数 silu），式 3 的
`φ(x) = ωb·b(x) + ωs·Σ c_b·B_b(x)`。参数量 1260（19ch；论文 14ch 为 960）。

## 2. 输入通道（19 通道）

| # | 通道 | 含义/单位 | 来源 |
|---|---|---|---|
| 1 | R | 网格 R 坐标（[-1,1]） | 固定 MAST 65×65 |
| 2 | Z | 网格 Z 坐标（[-1,1]） | 固定 MAST 65×65 |
| 3 | Ip | 等离子体电流 (A) | params[0] |
| 4 | paxis | 磁轴压强 (Pa) | params[1] |
| 5 | fvac | 真空通量函数 f (Wb/m) | params[2] |
| 6 | alpha_m | 剖面形状指数 m | params[3] |
| 7 | alpha_n | 剖面形状指数 n | params[4] |
| 8 | config | 位形标签（DN=1/SN=−1，min-max 后） | data_v5 每样本 config |
| 9–18 | I_P2U…I_P6U/L | 10 个偏滤器线圈电流 (A) | coil_currents[0:10] |
| 19 | I_P1 (Solenoid) | 中心螺线管电流 (A) | coil_currents[10] |

标量通道 17 个（5 params + config + 11 线圈）min-max 到 [−1,1]；ψ/J 输出 z-score；
R/Z 网格归一化到 [−1,1]。config 通道是显式位形标签（FNO exp008 同款思路）——
混合训练下它让 KAN 不必从线圈电流隐式反推位形，专注学习映射。

## 3. 损失与训练（train_kan.py，对齐论文式 5–16）

- Lψ / LJψ：MSE（式 5-6，λ=1）
- LPDE：总场 GS 残差 `(Δ*ψ_phys + μ0RJ_phys)/pde_scale` ²（式 9，λPDE=0.1；
  有效单元 = 排除线圈单元+边界；**不含 L_coils 项**——论文式 9 用总场，
  Δ*ψ_coils≈0 时减掉反而重复计数，见 exp001 README §偏差）
- LIp：`|Σ J·weight − Ip|/Ip_scale` ²（式 10，λIp=1）；无偏等离子体地层估计器
  （只对抽到的等离子体单元 ×n_p/k_p·dA 加权，避免 50/50 分层采样的比例失真）
- Lreg：B 样条系数 L1 + 逐连接熵（式 11-15，λ(t) 按式 16 四段：0→线性→恒→0）
- 剪枝：t=600，阈值 = max(connection_magnitudes)·1e-2，置零+梯度冻结+微调至 1000
- 相位感知 best：剪枝前后分开跟踪；best.pt = 剪枝后最优（phase="pruned"），
  best_pre_prune.pt 为剪枝前最优

## 4. 目录

```
kan/
├── README.md / EXPERIMENTS.md
├── scripts/            # run_kan_smoke.sh / run_kan_m1_train_eval_vis.sh / ...
├── logs/
├── data_ood/           # (M3) OOD 外推样本
└── experiments/exp001_kan_v5/
    ├── README.md       # 19ch 通道明细 + 训练设置 + vs exp011 对比表
    ├── model_b19ch_kan_mix/   # best.pt / history.json / args.json
    ├── eval_all/ eval_dn/ eval_sn/   # metrics.json
    ├── figures_dn/ figures_sn/
    └── sr/ ext/        # (M2) (M3)
```

## 5. 一键复现

```bash
# M0 冒烟（单测 + N=100/50ep 四阶段 + round-trip）
bash dn_fno_2608/kan/scripts/run_kan_smoke.sh
# M1 训练 + 六桶评估 + 可视化
bash dn_fno_2608/kan/scripts/run_kan_m1_train_eval_vis.sh
# M2 符号回归 / M3 外推+分类（待里程碑就绪）
bash dn_fno_2608/kan/scripts/run_kan_m2_symbolic.sh
bash dn_fno_2608/kan/scripts/run_kan_m3_ood_gen_extrap.sh
```

环境：`C:/Users/HP/.conda/envs/torch5060/python.exe`；在仓库根目录运行。

## 6. 里程碑状态

- [x] M0 冒烟（N=100/50ep，四阶段走完，管道 round-trip OK）
- [x] M1 N=500 训练 + 六桶评估 + 可视化——**完成但未达验收**：rel L2 20.87%
  （DN 20.96 / SN 20.79，验收 ≤1.4%），R² 0.942，J 自检 16.9/29.9%，
  GS 残差比 1.17/2.13，X 点 4-13cm，sep 11-12cm；判定容量受限
  （1260 参点式 KAN < 19ch 混合位形复杂度），诊断与对策见 exp001 README §7
- [x] exp002 容量增强变体（hidden 24 / grid 4 / 2000 ep，4536 参）——
  **半程停止**（ep 1700，val 15.0%）：监督 psi MSE_z 0.016 ≈ 15% 误差，
  按用户判定"第一阶段学不好，后面无意义"，于 2026-08-19 停止
- [x] exp003 DN-only × 论文架构（total 语义 LPDE）——**半程停止**
  （ep 700，剪枝 67.1% 后 val 21%）：监督同样不达标，停止
- [~] exp004 纯监督（DN-only × 论文架构 × pde=0 reg=0，256 点/样本）——
  运行中：验证监督阶段上限（对照 exp002/003 的监督期）
- [~] exp006 纯监督 + 采样密度 4×（1024 点/样本，其余同 exp004）——
  运行中：验证"每 epoch 空间覆盖率"是否是监督学不好的根因；
  仍不足则 2048 点/全网格（4225）或 grid 6-8 提容量
- [~] M2 符号回归 KAN-SR（sr_stats.json 镜像论文表 3）：管道验证通过
  （210 连接全部符号化 R²≥0.9999，SR 9.5ms/4225 点 <20ms）；M1 ckpt 上
  附加误差 48pp 系边界 bug 与 scipy 语义不一致（详见 exp001 README §7）；
  验收数字以 exp002 为准
- [~] M3 工具链完成：`gen_ood_kan.py`（DN/SN 各 600 候选 → box 过滤保留
  520/487，`kan/data_ood/`）+ `extrapolate_kan.py`（式 4 一阶泰勒 + 分离面
  X 点分类 + 距离-误判分析）真实 ckpt 管道验证通过（accuracy 46%，失败
  模型预期内）；分类验收 Accuracy ≥85%（论文 89.1）以 exp002 为准

## 7. 已知偏差（相对论文）

- Jψ 分离面跳变 vs B 样条光滑 → 边界误差环（论文同款现象），stratified 采样缓解
- 线圈单元 δ 型奇性 → LPDE/LIp 排除 coil_mask（论文用均匀截面线圈，网格实现差异）
- **LPDE 残差定义（修正，2026-08-19）**：默认 `--pde-coils-model plasma`，
  残差 = Δ*ψ_total − Δ*ψ_coils + μ0RJ（Δ*ψ_coils 由数据预计算）。旧 total
  语义把线圈奇性的中心差分涂抹（近线圈 |Δ*ψ_coils|≈1.19）留在残差里，
  真值场自检真空区残差 0.406 vs 修正后 0.0023——Lψ 与 LPDE 在真空区互相
  矛盾是欠拟合平台元凶（详见 exp001 README §4）。exp001-003 为旧语义
- config 离散通道在 {0,1} 间插值无物理意义，但不会被求值（min-max 后 ∈{−1,1}）
- 剪枝率不硬性对齐论文 27.5%（参考上限）
- M3 包络：16D 凸包在近退化点云上 qhull 不收敛 → axis-aligned box 兜底
  （ellipsoid 已实现但过紧；详见 exp001 README §4）
