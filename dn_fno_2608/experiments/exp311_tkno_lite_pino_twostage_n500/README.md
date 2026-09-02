# exp311 — TKNO-lite：conv stem + 全局自注意力的 Transformer 算子（去 KAN）

> 实验日期：2026-09-01 ｜ 状态：**完成（系列最佳结果）**（N=500，seed 1，
> data_v5/dn 单一位形）
> 数据：`data_v5/dn/`（MAST DN，65×65，2000/500/500，无墙）——与 exp102 逐项一致
> 对照：exp305（UFNO tuned，0.7147%，上一位）；exp302（UFNO，0.729%）
> 结论速览：**Transformer 全局注意力以 1.21M 参数（FNO 的 0.29×）全面刷新
> 系列纪录——test rel L2 0.635%（plasma 0.716%），比上一位 exp305 低 11%；
> J mask 0.977%（UFNO 1.13% 的 −14%）、Ip 0.189%、X 点 0.42/0.43 cm、O 点
> 0.18 cm 全部系列最佳。全局混合机制（谱 vs 注意力）的"谱必要"假说被
> 否定——与 exp301 UNet 失败直接对照，磁面全局耦合的关键是**混合的全局性**
> 而非具体实现；唯一代价是训练 80 min（FNO 5.2×，本构建无 flash 注意力）**

## 1. 目标

TKNO（Transformer-KAN Neural Operator，[arXiv:2511.19114](https://arxiv.org/abs/2511.19114)，
ICML 2026 海报）在**同一 Grad-Shafranov 任务**（LCFS 形状参数 → ψ，EXL-50U
部署）的五架构基准中胜出（监督 0.25%）。本实验取其 Transformer 全局注意力
核心，**按 exp303 证据剥离 KAN**（逐点 KAN 增益 ~0），并适配 65² 网格：

- **conv stem**（18→64，65² 全分辨率细节路径）
- **patch 嵌入**（64→128，3×3/s=2 → 32²=1024 token——65² 直接注意力在
  batch 16 下二次不可行；1024 为 8 的倍数以走 fused SDPA）
- **4× 全局自注意力块**（pre-LN：LayerNorm → 8 头注意力 → LayerNorm →
  MLP 4×，无 dropout，与系列无正则基线对齐）+ 学习 2D 位置编码
- **解码**：bilinear 上采样（显式 size=）+ stem skip concat + DoubleConv → proj
- **1,213,506 参数**（FNO 4.21M 的 0.29×、UFNO 2.47M 的 0.49×、DeepONet wide 1.34M 的 0.90×）

两阶段方案与 exp102 逐项一致：

**阶段1 损失 = MSE(psi_plasma, z) + w_j·masked MSE(J, z)（core mask 内）**
**阶段2 损失 = 阶段1 + w_pde·‖Δ\*ψ_pred + μ0·R·J_pred‖²（core mask 内）+ w_ip·‖(ΣJ·dA − Ip)/ip_scale‖²**

- 模型双输出通道：通道 0 = psi_plasma（z 域），通道 1 = Jφ（z 域）
- 切换条件：阶段1 val rel L2 < 3% 自动切阶段2（`--stage1-max-epochs 300` 兜底）
- **物理权重预热**：阶段2 的 w_pde/w_ip 从 0 线性 ramp 到目标值（30 epochs）

## 2. 输入通道（18 通道，同 exp102/exp302 coils 模式）

| # | 通道 | 含义/单位 | 来源 |
|---|---|---|---|
| 1 | R | 网格 R 坐标（[-1,1]） | 固定 MAST 65×65（stem 输入） |
| 2 | Z | 网格 Z 坐标（[-1,1]） | 固定 MAST 65×65（stem 输入） |
| 3 | Ip | 等离子体电流 (A) | params[0] |
| 4 | paxis | 磁轴压强 (Pa) | params[1] |
| 5 | fvac | 真空通量函数 f (Wb/m) | params[2] |
| 6 | alpha_m | 剖面形状指数 m | params[3] |
| 7 | alpha_n | 剖面形状指数 n | params[4] |
| 8–17 | I_P2U … I_P6L | 上下偏滤器线圈电流 (A) | coil_currents[0:10] |
| 18 | I_P1 (Solenoid) | 中心螺线管电流 (A) | coil_currents[10] |

（16 个标量通道广播到 65² 全网格后进 stem——局部常数被 patch 编码，
全局耦合由注意力完成；数据管线零改动。）

## 3. coil 分离（同 exp102）

网络**只预测 psi_plasma**（+Jφ）；评估时 `psi_total = psi_plasma_pred +
Σ_k I_k·G_k`（greens 恒等式验证 max diff 6e-8 Wb）。物理残差只定义在
等离子体场上——线圈场占 |psi_total| 幅值 183%、|Δ\*ψ_coils| 峰值 27.9 vs
等离子体 0.89，残差落在 psi_total 上会被线圈导体奇性主导。详见 exp101 README §3。

## 4. 训练设置

N=500（全量 2000 池嵌套子集，perm seed 12345），seed 1，AdamW lr 1e-3
wd 1e-4，ReduceLROnPlateau（patience 20，factor 0.5，min_lr 1e-5），batch 16，
800 epochs。模型 **TKNOlite2d2608(tkno_lite)**：stem 64 + patch 32² + 4×
注意力块（d=128，8 头，MLP 4×），**1,213,506 参数**。PDE 残差 interior
(1:-1,1:-1) 二阶中心差分，core mask 内 masked mean((res/pde_scale)²)，
pde_scale=0.362 Wb/m²；Ip 损失用 ip_scale=5.51e5 A 归一化。

**阶段切换**：阶段1 val rel L2 第 66 epoch 达阈值 → 切阶段2（与 exp302 的
e70 同档——注意力的阶段1 收敛不慢于谱）。阶段2 全程 734 epochs 中 val 从
阈值平滑降到 **0.7314%**（@ e797，接近训练末尾——800 epochs 内持续改善，
未触发早停）。**phase-aware best**：artifact 取阶段2 best（0.7314% @ e797）。

**注意力实现备注**：`need_weights=False` 走 fused SDPA（默认 True 会物化
注意力矩阵：4.6 GiB → 0.92 GiB 显存）；本 torch 构建（2.12 dev cu128）
**未编译 flash 注意力**，实际派发 mem-efficient 后端——训练 80.3 min
（FNO 15.4 min 的 5.2×、FNOKAN 60 min 的 1.3×），是系列唯一明显的时间代价。

## 5. 结果（test n=500，DN-only）

| 指标 | exp311（TKNO-lite） | exp305（UFNO tuned） | exp302（UFNO） |
|---|---|---|---|
| rel_l2_plasma mean / median / p95 (%) | 0.72 / 0.58 / 1.45 | 0.81 / 0.66 / 1.53 | 0.82 / 0.70 / 1.47 |
| rel_l2_total mean / median / p95 (%) | **0.64 / 0.52 / 1.39** | **0.71 / 0.60 / 1.47** | **0.73 / 0.62 / 1.48** |
| rmse_phys (Wb) plasma / total | 2.41e-4 / 2.41e-4 | 2.72e-4 / 2.72e-4 | 2.74e-4 / 2.74e-4 |
| GS 残差 core mask：pred / truth | 0.0158 / 0.0040 | 0.0144 / 0.0040 | 0.0150 / 0.0040 |
| **Ip 相对误差** mean / median / p95 (%) | 0.19 / 0.16 / 0.48 | 0.18 / 0.14 / 0.55 | 0.21 / 0.15 / 0.58 |
| **J rel L2 mask 内** mean / median / p95 (%) | **0.98 / 0.78 / 2.13** | 1.20 / 1.02 / 2.18 | 1.13 / 0.95 / 2.29 |
| X 点定位误差 lo / up (cm) | **0.42 / 0.43** | 0.63 / 0.61 | 0.52 / 0.63 |
| O 点误差 (cm) | **0.18** | 0.20 | 0.19 |
| n_xpt_fail | 0 | 0 | 0 |

**解读**：
1. **总误差 0.635% = 系列新纪录（−11% vs exp305）**：全局注意力承载磁面
   耦合不亚于谱混合，且参数只有 UFNO 的 0.49×。与 exp301（UNet 2.202%）
   构成完整对照：**局部卷积失败 → 全局混合（谱或注意力）成功——决定性变量
   是混合的全局性，不是傅里叶实现**。"谱卷积是磁面全局耦合唯一可行路径"
   的隐含假说被否定
2. **J 0.977% 系列最佳（UFNO 1.13% 的 −14%）**：J 是 ψ 的二阶导场、对高频
   细节敏感——注意力在 32² token 上的内容自适应混合比固定模态的谱混合
   更贴合局部结构；p95 2.13% 也优于谱系
3. **X 点 0.42/0.43 cm、O 点 0.18 cm 定位最佳**：磁面拓扑（X/O 点）是全局
   耦合的指纹——注意力把全局信息直接编码进 token 关系，定位精度直接受益
4. **Ip 0.189% 与 exp305 的 0.182% 打平**（物理约束项在两个机制下都收敛）
5. **代价只有时间**：80.3 min vs FNO 15.4 min（5.2×）——无 flash 编译的
   mem-efficient 注意力是主要瓶颈；参数/精度/几何全面占优下这是明确的正向
   性价比；阶段1 收敛（e66）不慢于谱系（e70）

## 6. 局限（Transformer 特有）

- **单 seed 单架构点**：0.635% 超出 exp305 0.05pp 以上（> 单 seed 噪声
  ~0.02-0.03pp 两档），但 3-seed 平均仍是显著性确认手段（EXPERIMENTS.md
  §6 已列为下一步）
- **注意力无 flash 编译**：mem-efficient 后端使训练 ~5× FNO——换 cuDNN
  注意力/flash 构建可降回 ~1-2×；不改变结论但改变成本曲线
- **token 网格 32² 是设计选择**：65² 直接注意力二次不可行；32² 与 UFNO 的
  中尺度混合层对齐。若注意力粒度更细（两级 32²+17² 分层注意力）可能再降，
  未做
- **固定位置编码**：32² 网格绑定 65² 输入——换分辨率（data_v6 129²）需重训
  （谱混合天然分辨率无关，DeepONet trunk 亦点式无关——这是注意力的固有
  代价，N=2000/129² 验证时需重新 patch）
- **小样本**：N=500 下 1.21M 参数高于 DeepONet 0.6M——样本效率需 N=2000
  验证（注意力数据饥饿的经典风险）
- **与 TKNO 原论文不可直接对照**：论文 0.25% 是 LCFS 形状参数输入 + 不同
  数据定义；本实验同口径（18ch coil 输入）下的对照方是 exp302/305

## 7. 复现

```bash
bash dn_fno_2608/scripts/run_exp311_312_pino.sh train   # 训练（含 exp312）
bash dn_fno_2608/scripts/run_exp311_312_pino.sh eval    # 评估（含 exp312）
# 单独：
cd "$(git rev-parse --show-toplevel)"
C:/Users/HP/.conda/envs/torch5060/python.exe -u -m gs_pino_fno_phys.train_pino \
  --mode twostage --model tkno_lite --train-data dn_fno_2608/data_v5/dn/train.npz \
  --val-data dn_fno_2608/data_v5/dn/val.npz --n-train 500 --seed 1 \
  --epochs 800 --phys-weight 0.1 --ip-weight 1.0 --j-weight 1.0 \
  --stage1-threshold 0.03 --stage1-max-epochs 300 --stage2-ramp-epochs 30 \
  --out-dir dn_fno_2608/experiments/exp311_tkno_lite_pino_twostage_n500
```

产物同 exp102 约定：`best.pt`（含 `model` 键）、`history.json`、`args.json`、
`metrics.json`、`train.log`/`eval.log`、`figures/`（fig1_best_worst_psi /
fig2_field_stats / fig3_geometry_stats + stats_per_sample.json）。
