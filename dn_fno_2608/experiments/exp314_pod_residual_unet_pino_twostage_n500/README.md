# exp314 — POD 固定基 + 并联 UNet 残差：基外细节需要全局混合吗？

> 实验日期：2026-09-02 ｜ 状态：**完成（部分正面）**——exp313 的残差通道
> 从小 FNO 换成纯卷积 UNet，测"基外细节需要多少全局性"（N=500，seed 1，
> data_v5/dn 单一位形）
> 数据：`data_v5/dn/`（MAST DN，65×65，2000/500/500，无墙）——与 exp102 逐项一致
> 对照：exp313（POD + FNO 残差，0.883%）；exp301（UNet 完整任务，2.202%
> 失败）；exp312（POD 单独，1.508%）；exp102（FNO 完整任务，0.800%）
> 结论速览：**基外细节"部分需要"全局混合——纯卷积 UNet 残差把 POD 从
> 1.508% 拉到 1.099%（−27%，X 点 0.85→0.64 cm、O 点 0.65→0.36 cm、GS
> 残差 −62%），但明显弱于 FNO 残差（1.099% vs 0.883%，+0.22pp）。对比
> exp301（完整任务下纯卷积是 FNO 的 2.75×）：POD 基承担全局形状后，残差
> 通道的全局性需求大幅下降（差距 2.75× → 1.25×），但谱混合仍提供 ~0.2pp
> 的边际增益——X 点 0.64/0.64 cm 与 FNO 全任务（0.60/0.63）相当、弱于
> FNO 残差（0.54/0.56）**

## 1. 目标

exp313 用**小 FNO（谱卷积）**做残差通道，把 POD 基外细节（X 点尺度结构）
学回来了（1.508% → 0.883%）。本实验把残差通道**换成纯卷积 UNet**（瘦身版
base 16 / depth 4），其余一切不变，回答系列缺环问题：

> **基外细节是"局部的"（POD 基已提供全局形状，残差只需补局部结构），
> 还是"在残差通道上也需要全局混合"？**

- exp301 已证：**纯卷积完整任务失败**（2.202% = FNO 的 2.75×，局部感受野
  不足承载磁面全局拓扑）——但那是有"从零学全局形状"的负担
- exp313 已证：**谱卷积残差成功**（0.883%）——但谱卷积自带全局混合，
  不能区分"残差任务本来只需要局部"与"残差任务也需要全局"
- 本实验（exp314）：UNet 残差若**成功**（≈0.883% 档）→ 基外细节本质是
  局部的，POD 基把全局形状提供好后纯卷积足够补细节；若**失败**（接近
  exp312 1.508% 或明显高于 0.883%）→ 全局性在任意通道上都必要，连
  0.08% 能量的基外成分都不能靠局部感受野表达

$$\psi_z(R,Z) = \underbrace{\psi_0 + \sum_{k=1}^{p} b_k(x)\,\phi_k(R,Z)}_{\text{POD 通道（exp312 原样，冻结）}} + \underbrace{\text{UNet}_{\text{small}}(x)(R,Z)}_{\text{残差通道（学 } y - \psi_{\text{POD}}\text{）}}$$

- **POD 通道**：与 exp312/exp313 逐字节相同（SVD 基 p_psi=3/p_j=10 冻结
  buffer + 分支 MLP 16 标量 → 13 系数，275,984 参数）
- **残差通道**：UNet2d2608 瘦身（base 16 / depth 4，通道 16/32/64/128 +
  bottleneck 128，≈1.23M 参数），与 exp301 同构（同样 4 层下采样链、无
  归一化、bilinear skip 解码）——只是宽度 24→16。**纯 3×3 卷积 + 池化链，
  无谱混合、无注意力**
- 残差 proj 层 ×0.1 初始化（exp313 同款小起步）

**训练协议与 exp313 完全一致**（分阶段冻结，防通道竞争）：

| epochs | 训练 | 冻结 | 分工 |
|---|---|---|---|
| 1..N（60） | branch（POD 系数） | 残差 UNet（从 0 起步） | POD 通道先收敛 |
| N+1 起 | 残差 UNet | branch（锚定已学 POD） | 残差学 `y − ψ_pod` |

阶段1（纯监督）切换阶段2（加物理项）挂起到 pretrain 结束——残差开始
学习时物理权重 ramp 同步从 0 爬升。

两阶段损失与 exp102/exp313 逐项一致：

**阶段1 损失 = MSE(psi_plasma, z) + w_j·masked MSE(J, z)（core mask 内）**
**阶段2 损失 = 阶段1 + w_pde·‖Δ\*ψ_pred + μ0·R·J_pred‖²（core mask 内）+ w_ip·‖(ΣJ·dA − Ip)/ip_scale‖²**

## 2. 输入通道（18 通道，同 exp102/exp313）

| # | 通道 | 含义/单位 | 来源 |
|---|---|---|---|
| 1 | R | 网格 R 坐标（[-1,1]） | 固定 MAST 65×65（POD 通道不用，UNet 残差用） |
| 2 | Z | 网格 Z 坐标（[-1,1]） | 固定 MAST 65×65（POD 通道不用，UNet 残差用） |
| 3 | Ip | 等离子体电流 (A) | params[0]（branch + 残差输入） |
| 4 | paxis | 磁轴压强 (Pa) | params[1]（branch + 残差输入） |
| 5 | fvac | 真空通量函数 f (Wb/m) | params[2]（branch + 残差输入） |
| 6 | alpha_m | 剖面形状指数 m | params[3]（branch + 残差输入） |
| 7 | alpha_n | 剖面形状指数 n | params[4]（branch + 残差输入） |
| 8–17 | I_P2U … I_P6L | 上下偏滤器线圈电流 (A) | coil_currents[0:10]（branch + 残差输入） |
| 18 | I_P1 (Solenoid) | 中心螺线管电流 (A) | coil_currents[10]（branch + 残差输入） |

## 3. coil 分离（同 exp102）

网络**只预测 psi_plasma**（+Jφ）；评估时 `psi_total = psi_plasma_pred +
Σ_k I_k·G_k`（greens 恒等式验证 max diff 6e-8 Wb）。物理残差只定义在
等离子体场上——线圈场占 |psi_total| 幅值 183%、|Δ\*ψ_coils| 峰值 27.9 vs
等离子体 0.89，残差落在 psi_total 上会被线圈导体奇性主导。详见 exp101 README §3。

## 4. 训练设置

N=500（全量 2000 池嵌套子集，perm seed 12345），seed 1，AdamW lr 1e-3
wd 1e-4，ReduceLROnPlateau（patience 20，factor 0.5，min_lr 1e-5），batch 16，
800 epochs。模型 **PODResidual2d2608(pod_residual_unet)**：POD 通道
（p_psi=3 / p_j=10，能量 99.92%/99.93%，n=500，同 exp312）+ 残差 UNet
（base 16 / depth 4），**1,503,266 参数**（POD 275,984 + UNet 1,227,282；
exp313 的 1.73×、FNO 4.21M 的 0.36×）。PDE 残差 interior (1:-1,1:-1)
二阶中心差分，core mask 内 masked mean((res/pde_scale)²)，pde_scale=0.362
Wb/m²；Ip 损失用 ip_scale=5.51e5 A 归一化。

**分阶段冻结协议**：`--pod-pretrain-epochs 60`——epochs 1..60 只训 branch
（残差从 0 起步）；e61 冻结 branch、解冻残差学 `y − ψ_pod`；阶段1 切换
挂起到 e60 后。

**阶段切换**：switch_epoch 61（pretrain 60 结束立即切，与 exp313 精确一致）。
交接时 val ~1.9% → 阶段2 UNet 残差接手后缓慢下降：best **1.2810% @ e796**
（跑满 800 epochs 上限——UNet 残差收敛慢，75-epoch 早停未触发；exp313
FNO 残差 e475 即 best、e550 早停）。总训练 12.7 min（exp313 的 1.5×）。

## 5. 结果（test n=500，DN-only）

| 指标 | exp314（POD+UNet） | exp313（POD+FNO） | exp312（POD） | exp301（UNet 全任务） |
|---|---|---|---|---|
| rel_l2_total mean / median / p95 (%) | **1.099 / 0.867 / 2.57** | **0.883 / 0.645 / 2.05** | **1.51 / 1.27 / 3.26** | **2.202** |
| rel_l2_plasma mean / median / p95 (%) | 1.28 / 0.95 / 3.32 | 1.03 / 0.75 / 2.70 | 1.69 / 1.44 / 3.61 | 2.47 |
| GS 残差 core mask：pred / truth | 0.0307 / 0.0040 | 0.0219 / 0.0040 | 0.0812 / 0.0040 | 0.0643 |
| **Ip 相对误差** mean / median / p95 (%) | 0.403 / 0.283 / 1.15 | 0.338 / 0.246 / 0.935 | 0.25 / 0.17 / 0.72 | 0.587 |
| **J rel L2 mask 内** mean / median / p95 (%) | 2.33 / 1.97 / 4.79 | 1.76 / 1.41 / 3.86 | 2.52 / 2.13 / 4.98 | 3.86 |
| X 点定位误差 lo / up (cm) | 0.64 / 0.64 | 0.54 / 0.56 | 0.85 / 0.85 | 1.83 / 2.32 |
| O 点误差 (cm) | 0.36 | 0.34 | 0.65 | — |
| n_xpt_fail | 0 | 0 | 0 | 0 |

**解读**：
1. **科学问题答案：基外细节"部分需要"全局混合**——UNet 残差把 POD 从
   1.508% 拉到 1.099%（**−27%**）：X 点 0.85 → 0.64 cm（−25%）、O 点
   0.65 → 0.36 cm（−44%）、J 2.52 → 2.33、GS 残差 0.0812 → 0.0307（−62%）。
   纯卷积**不是完全失败**——POD 基承担全局形状后，基外细节的大部分可以
   靠局部-层级卷积表达
2. **但谱混合仍有边际增益（+0.22pp）**：1.099 vs 0.883——对比 exp301
   （完整任务下纯卷积 = FNO 的 2.75× 差距），残差任务的架构差距缩小到
   ~1.25×——**POD 基把"全局形状"这个最需要全局混合的部分拿走之后，
   残差通道的全局性需求大幅下降**。全局性在本问题的分层：完整任务
   （决定性）> 残差任务（边际但真实）
3. **X 点 0.64/0.64 与 FNO 全任务（0.60/0.63）相当、弱于 FNO 残差
   （0.54/0.56）**：X 点定位需要更精细的全局-局部协调，谱混合的价值在
   此最可见（−0.1 cm）
4. **Ip 0.403% 系列残差框架中最松**：与 exp313 同机制（残差放大 J 自由
   度 → 积分匹配略松 vs POD 0.253%），且 UNet 残差 J 分布更不准（2.33 vs
   1.76）→ Ip 也略松（0.403 vs 0.338）
5. **混合框架的稳健性（超出本实验问题的发现）**：即使残差通道用系列最
   弱架构（纯卷积，完整任务 2.202%），POD+残差框架仍把 1.508% 拉到
   1.099%——"低秩基 + 任意残差"都优于纯低秩基，POD 通道的贡献是框架性
   的（保底 +27% 收益），残差架构决定剩余差距
6. **收敛慢 + 跑满 800 epochs**：best @ e796（exp313 的 e475）、12.7 min
   训练——纯卷积局部优化特性与 exp301（阶段1 需 e300 兜底）一致

## 6. 局限

- **UNet 残差不是"最小局部"**：4 层下采样链有 65²→32²→16²→8²→4² 的感受
  野放大，加上 skip 连接，理论上能表达大尺度信息——若成功不能严格推出
  "残差是局部的"，只能推出"局部-层级卷积链够用"；更严的对照应是"去掉
  深下采样"的浅 UNet
- **收敛速度差异**：纯卷积局部优化慢（exp301 阶段1 需 e300 兜底）——
  若最终结果接近但低于 exp313，需警惕早停时机对结论的影响
- **容量不对称**：UNet 残差 1.23M vs FNO 残差 0.59M——若 UNet 残差更好，
  可能是容量差而非架构差；预算对齐的对照（base 10-12 的更瘦 UNet）未做
- **pretrain 长度仍是超参**：N=60 与 exp313 相同，未扫描（残差越弱越需要
  POD 通道多训，但 N 固定不能独立验证）
- **单 seed**：系列统一 seed 1，结论在单 seed 上（与 exp313 同条件可比）

## 7. 复现

```bash
bash dn_fno_2608/scripts/run_exp314_pino.sh train   # 训练（pretrain 60）
bash dn_fno_2608/scripts/run_exp314_pino.sh eval    # 评估
# 单独：
cd "$(git rev-parse --show-toplevel)"
C:/Users/HP/.conda/envs/torch5060/python.exe -u -m gs_pino_fno_phys.train_pino \
  --mode twostage --model pod_residual_unet --train-data dn_fno_2608/data_v5/dn/train.npz \
  --val-data dn_fno_2608/data_v5/dn/val.npz --n-train 500 --seed 1 \
  --epochs 800 --phys-weight 0.1 --ip-weight 1.0 --j-weight 1.0 \
  --stage1-threshold 0.03 --stage1-max-epochs 300 --stage2-ramp-epochs 30 \
  --pod-pretrain-epochs 60 \
  --out-dir dn_fno_2608/experiments/exp314_pod_residual_unet_pino_twostage_n500
```

POD 基自动计算（SVD 于 N=500 训练子集，能量 99.9% 封顶 256 模态）并存入
`best.pt` 的 `pod_basis` 键；评估从 checkpoint 恢复基，无需重算。产物同
exp102 约定：`best.pt`、`history.json`、`args.json`、`metrics.json`、
`train.log`/`eval.log`、`figures/`（fig1_best_worst_psi / fig2_field_stats /
fig3_geometry_stats + stats_per_sample.json）。
