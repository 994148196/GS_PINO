# exp303 — FNO-KAN 混合 + GS 物理残差（两阶段，自洽残差 + Ip 约束）

> 实验日期：2026-09-01 ｜ 状态：**完成**（N=500，seed 1，data_v5/dn 单一位形）
> 数据：`data_v5/dn/`（MAST DN，65×65，2000/500/500，无墙）——与 exp102 逐项一致
> 对照：exp102（FNO2d2608 两阶段，0.80%）；KAN 轨（点式 B-spline KAN 失败 20.87%）
> 结论速览：**FNO-KAN 混合（4.33M 参数）机制完全成立但无增益——test rel L2
> 0.857%（FNO 0.800%，+0.06pp 内为噪声级）、切换 e31（同 exp102 的 e29 档）、
> J 1.45%（+0.1）、Ip 0.238%（+0.03）——KANO 的"变系数 PDE 需要逐点可学习
> 激活"论点在本 65² 数据上未转化为收益（16×16 模态已充分覆盖 R 因子变化）；
> 代价是 4× 训练时长（~60 min）与 +3% 参数**。调研背景见
> [ARCHS_SURVEY.md](../../ARCHS_SURVEY.md)（KANO：变系数 PDE 上纯谱瓶颈失效；
> GS 的 Δ\* 恰为变系数算子）

## 1. 目标

验证 **FNO-KAN 混合骨架**：保留 FNO 的谱混合骨架，把块内 1×1 conv 换成
**per-pixel KAN 激活**。动机来自两点：(1) KANO 论点——纯谱瓶颈网络在
**变系数** PDE 上谱效率退化，GS 恰为变系数（Δ\* = ∂²/∂R² − (1/R)∂/∂R +
∂²/∂Z²，R 因子），需要逐点可学习非线性；(2) KAN 轨教训——点式 B-spline KAN
（无空间混合）失败于 20.87%，问题在缺乏空间混合而非 KAN 本身，本实验是
**相反组合**：空间混合保留（谱核）+ 激活升级（KAN）。两阶段方案与 exp102
逐项一致：

**阶段1 损失 = MSE(psi_plasma, z) + w_j·masked MSE(J, z)（core mask 内）**
**阶段2 损失 = 阶段1 + w_pde·‖Δ\*ψ_pred + μ0·R·J_pred‖²（core mask 内）+ w_ip·‖(ΣJ·dA − Ip)/ip_scale‖²**

- 模型双输出通道：通道 0 = psi_plasma（z 域），通道 1 = Jφ（z 域）
- 切换条件：阶段1 val rel L2 < 3% 自动切阶段2（`--stage1-max-epochs 300` 兜底）
- **物理权重预热**：阶段2 的 w_pde/w_ip 从 0 线性 ramp 到目标值（30 epochs）

**架构**：FNO2d2608 骨架（lift 18→64 + 4×FNOBlock + proj 64→2），块 =
`GELU(SpectralConv2dR(x) + KANConv1x1(x))`。KANConv1x1 每像素执行
`φ(x) = wb·silu(x) + ws·Σ_b c_b·B_b(x)`（同 `gs_pino_kan_2608.model_kan.KANLayer`
约定），grid 4 / degree 2（6 个 clamped B-spline 基函数），融合 matmul 实现
（不物化 (N,in,out) 中间张量）。

**与 FNOBlock 的有意偏差（全部记录在案）**：
1. **tanh 门控入样条域**：B-spline knot 固定 [-1,1] 而谱后激活无界；silu
   基分支保留无界路径；
2. **基函数 stop-grad**：基函数值在 `no_grad` 下计算——对 KAN 参数
   （coeffs/spline_w/base_w）的梯度零影响（dL/d(coeffs) = Bᵀ·dL/dh，B 是
   常数）；仅裁剪 x→样条分支的梯度路径（tanh 饱和区该路径本就趋零）。收益：
   fwd+bwd 从 55.7ms → ~22ms/块（否则 800 epochs 约 4.5h）；
3. **coeffs 零初始化**：φ(x) ≈ wb·silu(x) 起步，训练稳定（同 KANLayer 约定）。

**基函数计算**：批量 Cox-de Boor 递归（每 degree 2 次逐元素运算 + 移位）替代
`model_kan._bspline_bases` 的逐索引 Python 循环（GPU 上 96→39 ms/次，
与 scipy 验证实现交叉核对 max err 8e-6，见 `models_alt.py` 自测）。

## 2. 输入通道（18 通道，同 exp102/exp101 coils 模式）

| # | 通道 | 含义/单位 | 来源 |
|---|---|---|---|
| 1 | R | 网格 R 坐标（[-1,1]） | 固定 MAST 65×65 |
| 2 | Z | 网格 Z 坐标（[-1,1]） | 固定 MAST 65×65 |
| 3 | Ip | 等离子体电流 (A) | params[0] |
| 4 | paxis | 磁轴压强 (Pa) | params[1] |
| 5 | fvac | 真空通量函数 f (Wb/m) | params[2] |
| 6 | alpha_m | 剖面形状指数 m | params[3] |
| 7 | alpha_n | 剖面形状指数 n | params[4] |
| 8–17 | I_P2U … I_P6L | 上下偏滤器线圈电流 (A) | coil_currents[0:10] |
| 18 | I_P1 (Solenoid) | 中心螺线管电流 (A) | coil_currents[10] |

## 3. coil 分离（同 exp102）

网络**只预测 psi_plasma**（+Jφ）；评估时 `psi_total = psi_plasma_pred +
Σ_k I_k·G_k`（greens 恒等式验证 max diff 6e-8 Wb）。物理残差只定义在
等离子体场上——线圈场占 |psi_total| 幅值 183%、|Δ\*ψ_coils| 峰值 27.9 vs
等离子体 0.89，残差落在 psi_total 上会被线圈导体奇性主导。详见 exp101 README §3。

## 4. 训练设置

N=500（全量 2000 池嵌套子集，perm seed 12345），seed 1，AdamW lr 1e-3
wd 1e-4，ReduceLROnPlateau（patience 20，factor 0.5，min_lr 1e-5），batch 16，
800 epochs（约 60 min——KANConv 基函数计算仍为主要开销，见 §6）。模型 **FNOKAN2d2608**：lift 18→64 + 4 块
（width 64，modes 16×16，KANConv1x1 grid 4 / degree 2）+ proj 64→2，
**4,326,722 参数**（FNO 的 1.03×）。PDE 残差 interior (1:-1,1:-1) 二阶中心
差分，core mask 内 masked mean((res/pde_scale)²)，pde_scale=0.362 Wb/m²；
Ip 损失用 ip_scale=5.51e5 A 归一化。

**阶段切换**：阶段1 val rel L2 第 31 epoch 达 2.82% < 3% 阈值 → 切阶段2
（与 exp102 的 e29 同档，本系列最快之一）。阶段2 全程 769 epochs 中 val 从
2.82% 平滑降到 **1.026%**。**phase-aware best**：artifact 取阶段2 best
（best val rel L2 **1.0256%** @ epoch 791）。

## 5. 结果（test n=500，DN-only）

| 指标 | exp303（FNOKAN） | exp102（FNO twostage） | KAN 轨（点式） |
|---|---|---|---|
| rel_l2_plasma mean / median / p95 (%) | 0.99 / 0.73 / 2.38 | 0.90 / 0.74 / 1.70 | 20.87（整体，失败） |
| rel_l2_total mean / median / p95 (%) | **0.86 / 0.66 / 2.03** | **0.80 / 0.64 / 1.59** | — |
| rmse_phys (Wb) plasma / total | 3.26e-4 / 3.26e-4 | 2.97e-4 / 2.97e-4 | — |
| GS 残差 core mask：pred / truth | 0.0172 / 0.0040 | 0.0151 / 0.0040 | — |
| **Ip 相对误差** mean / median / p95 (%) | 0.24 / 0.18 / 0.65 | 0.21 / 0.16 / 0.54 | — |
| **J rel L2 mask 内** mean / median / p95 (%) | 1.45 / 1.10 / 3.30 | 1.36 / 1.11 / 2.66 | — |
| X 点定位误差 lo / up (cm) | 0.67 / 0.71 | 0.60 / 0.63 | — |
| O 点误差 / 分离面 mean (cm) | 0.24 / — | 0.21 / 0.34 | — |

**解读**：
1. **机制成立、增益为零（诚实中性结果）**：KAN 混合从 e31 正常切换、阶段2
   平滑收敛、无 NaN 无爆炸——"谱混合 + KAN 激活"组合完全可用；但所有指标
   与 FNO 差 +0.03~+0.1pp（p95 差最多 +0.44pp），同量级内无显著收益
2. **KANO 论点未转化为收益**：GS 的变系数（1/R 因子）在 65² × 16×16 模态
   下被谱卷积充分捕获（exp302 的 UFNO 甚至用更少参数更好）——"纯谱瓶颈
   在变系数 PDE 失效"在本问题规模/模态数下不成立，KAN 的逐点可学习激活
   成了冗余表达力
3. **与 KAN 轨的对照收束**：点式 KAN 失败（20.87%）是"无空间混合"所致，
   不是 KAN 本身——本实验证明 KAN 组件可以工作（不拖累精度），只是不带来
   收益；两条证据合起来：**本问题上 KAN 的价值在于证明"混合可以"而非
   "混合更好"**
4. **成本侧**：训练 ~60 min（FNO 15.4 min 的 4×，基函数计算是瓶颈）、
   参数 +3%——无收益时的净成本是选择时的决定性因素

## 6. 局限（FNOKAN 特有）

- **三处实现偏差的净效应**：tanh 门控压缩样条输入域、stop-grad 裁剪一条
  梯度路径、coeffs 零初始化——三者都偏向"KAN 混合 ≈ 微调版 FNO"起步；
  若与 exp102 持平，不能区分"变系数论点不成立"与"偏差抵消了增益"
- **样条基函数数是主要计算开销**：批量 Cox-de Boor 后 fwd+bwd ~22ms/块，
  仍是 FNO 同结构（1×1 conv）的 ~5×；800 epochs 训练 ⏳ min（对照 exp102
  的 FNO 15.4 min）
- **参数小幅增加**：+3% vs FNO（每像素 64×64×6 coeffs），N=500 小样本下
  过拟合风险略升；grid 4 不稳定可回退 grid 2（`FNOKAN2d2608(grid_size=2)`，
  未触发则本文档不生效）
- **KAN 轨对照的边界**：点式 KAN 失败含无谱混合 + 逐点激活两重因素，exp303
  只能归因"谱混合 + KAN 激活"组合，不能单独归因 KAN

## 7. 复现

```bash
bash dn_fno_2608/scripts/run_exp303_304_pino.sh train   # 训练（含 exp304）
bash dn_fno_2608/scripts/run_exp303_304_pino.sh eval    # 评估（含 exp304）
# 单独：
cd "$(git rev-parse --show-toplevel)"
C:/Users/HP/.conda/envs/torch5060/python.exe -u -m gs_pino_fno_phys.train_pino \
  --mode twostage --model fnokan --train-data dn_fno_2608/data_v5/dn/train.npz \
  --val-data dn_fno_2608/data_v5/dn/val.npz --n-train 500 --seed 1 \
  --epochs 800 --phys-weight 0.1 --ip-weight 1.0 --j-weight 1.0 \
  --stage1-threshold 0.03 --stage1-max-epochs 300 --stage2-ramp-epochs 30 \
  --out-dir dn_fno_2608/experiments/exp303_fnokan_pino_twostage_n500
```

产物同 exp102 约定：`best.pt`（含 `model` 键）、`history.json`、`args.json`、
`metrics.json`、`train.log`/`eval.log`、`figures/`（fig1_best_worst_psi /
fig2_field_stats / fig3_geometry_stats + stats_per_sample.json）。
