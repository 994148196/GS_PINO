# exp304 — PI-DeepONet + GS 物理残差（两阶段，自洽残差 + Ip 约束）

> 实验日期：2026-09-01 ｜ 状态：**完成**（N=500，seed 1，data_v5/dn 单一位形）
> 数据：`data_v5/dn/`（MAST DN，65×65，2000/500/500，无墙）——与 exp102 逐项一致
> 对照：exp102（FNO2d2608 两阶段，0.80%）
> 结论速览：**PI-DeepONet（分支-主干点积，0.60M 参数 = FNO 的 1/7）与 FNO
> 打平——test rel L2 0.802%（FNO 0.800%）、X 点 0.73/0.70 cm（FNO 0.60/0.63）、
> 切换 e42 正常；psi 精度不输但 J/Ip 通道弱 2×（J 2.61% vs 1.36%、Ip 0.38%
> vs 0.21%）——分支-主干分解高度贴合本问题的标量输入→场输出结构，0.6M
> 容量足以学 psi 但不足以同时承载二阶导数场**。调研背景见
> [ARCHS_SURVEY.md](../../ARCHS_SURVEY.md)（SUNIST-2 IAEA 2025：GS 平衡约束
> ~100 放电样本即可；Cai et al. CMAME 2022：DeepONet 对带噪输入稳健）

## 1. 目标

验证 **PI-DeepONet**：把本问题的输入分解结构直接映射到 DeepONet 的
分支-主干架构——**分支网络**吃 16 个全局标量（5 等离子体参数 + 11 线圈电流），
**主干网络**吃 (R,Z) 网格坐标，输出为两者点积的场函数。这个映射是本问题
最自然的：输入大部分是全局标量而非场（FNO/UNet 靠广播铺开，DeepONet 天生
是这种分解）。文献证据：SUNIST-2 用物理约束 DeepONet 做 GS 平衡重构
（~100 放电样本）；Cai et al. 证明 DeepONet 对带噪输入的稳健性（线圈电流
实测值天然带噪）。两阶段方案与 exp102 逐项一致：

**阶段1 损失 = MSE(psi_plasma, z) + w_j·masked MSE(J, z)（core mask 内）**
**阶段2 损失 = 阶段1 + w_pde·‖Δ\*ψ_pred + μ0·R·J_pred‖²（core mask 内）+ w_ip·‖(ΣJ·dA − Ip)/ip_scale‖²**

- 模型双输出通道：通道 0 = psi_plasma（z 域），通道 1 = Jφ（z 域）
- 切换条件：阶段1 val rel L2 < 3% 自动切阶段2（`--stage1-max-epochs 300` 兜底）
- **物理权重预热**：阶段2 的 w_pde/w_ip 从 0 线性 ramp 到目标值（30 epochs）

**架构**：branch ×2（各 16→256×3 层 GELU，输入 = `x[:,2:].mean((2,3))`
即 z-score 标量），共享 trunk MLP（(R,Z)→256×3 层，输入 = 归一化网格坐标，
`x[:,0:2]` reshape (B,4225,2)），输出 einsum 点积 → (B,2,65,65)；
trunk 末层权重 ×0.1 小初始化（线性起步，稳定）；**601,600 参数**
（其他骨架的 1/4~1/7——容量差异是既定解读口径，不是失败前提）。

## 2. 输入通道（18 通道，同 exp102/exp101 coils 模式）

| # | 通道 | 含义/单位 | 来源 |
|---|---|---|---|
| 1 | R | 网格 R 坐标（[-1,1]） | 固定 MAST 65×65（trunk 输入） |
| 2 | Z | 网格 Z 坐标（[-1,1]） | 固定 MAST 65×65（trunk 输入） |
| 3 | Ip | 等离子体电流 (A) | params[0]（branch 输入） |
| 4 | paxis | 磁轴压强 (Pa) | params[1]（branch 输入） |
| 5 | fvac | 真空通量函数 f (Wb/m) | params[2]（branch 输入） |
| 6 | alpha_m | 剖面形状指数 m | params[3]（branch 输入） |
| 7 | alpha_n | 剖面形状指数 n | params[4]（branch 输入） |
| 8–17 | I_P2U … I_P6L | 上下偏滤器线圈电流 (A) | coil_currents[0:10]（branch 输入） |
| 18 | I_P1 (Solenoid) | 中心螺线管电流 (A) | coil_currents[10]（branch 输入） |

（DeepONet 的分支输入 = 18ch 中除 R/Z 外的 16 个标量通道——模型内部
`x[:,2:].mean((2,3))`，广播后每像素相同故取 mean 即原值；数据管线与
exp102 完全同口径，零改动。）

## 3. coil 分离（同 exp102）

网络**只预测 psi_plasma**（+Jφ）；评估时 `psi_total = psi_plasma_pred +
Σ_k I_k·G_k`（greens 恒等式验证 max diff 6e-8 Wb）。物理残差只定义在
等离子体场上——线圈场占 |psi_total| 幅值 183%、|Δ\*ψ_coils| 峰值 27.9 vs
等离子体 0.89，残差落在 psi_total 上会被线圈导体奇性主导。详见 exp101 README §3。

## 4. 训练设置

N=500（全量 2000 池嵌套子集，perm seed 12345），seed 1，AdamW lr 1e-3
wd 1e-4，ReduceLROnPlateau（patience 20，factor 0.5，min_lr 1e-5），batch 16，
800 epochs（约 20 min，本系列最快）。模型 **PIDeepONet2d**：branch×2
（16→256×3 层 GELU）+ 共享 trunk（2→256×3 层）+ einsum 点积，trunk 末层
权重 ×0.1 初始化，**601,600 参数**。PDE 残差 interior (1:-1,1:-1) 二阶
中心差分，core mask 内 masked mean((res/pde_scale)²)，pde_scale=0.362
Wb/m²；Ip 损失用 ip_scale=5.51e5 A 归一化。

**阶段切换**：阶段1 val rel L2 第 42 epoch 达 2.99% < 3% 阈值 → 切阶段2
（正常档位，0.6M 容量未拖慢阶段1 达标）。阶段2 全程 758 epochs 中 val 从
2.99% 平滑降到 **0.953%**。**phase-aware best**：artifact 取阶段2 best
（best val rel L2 **0.9528%** @ epoch 795）。

## 5. 结果（test n=500，DN-only）

| 指标 | exp304（PI-DeepONet） | exp102（FNO twostage） |
|---|---|---|
| rel_l2_plasma mean / median / p95 (%) | 0.92 / 0.69 / 2.06 | 0.90 / 0.74 / 1.70 |
| rel_l2_total mean / median / p95 (%) | **0.80 / 0.63 / 1.87** | **0.80 / 0.64 / 1.59** |
| rmse_phys (Wb) plasma / total | 3.04e-4 / 3.04e-4 | 2.97e-4 / 2.97e-4 |
| GS 残差 core mask：pred / truth | 0.0298 / 0.0040 | 0.0151 / 0.0040 |
| **Ip 相对误差** mean / median / p95 (%) | 0.38 / 0.34 / 0.84 | 0.21 / 0.16 / 0.54 |
| **J rel L2 mask 内** mean / median / p95 (%) | 2.61 / 2.36 / 4.87 | 1.36 / 1.11 / 2.66 |
| X 点定位误差 lo / up (cm) | 0.73 / 0.70 | 0.60 / 0.63 |
| O 点误差 / 分离面 mean (cm) | 0.24 / — | 0.21 / 0.34 |

**解读**：
1. **容量奇迹：0.6M 参数平 FNO**（rel_l2_total 0.802% vs 0.800%，统计不可
   区分）——分支(16 标量)→主干(R,Z) 点积的结构与本问题输入分解**精确同构**
   （11 线圈电流 + 5 等离子体参数全是全局标量），SUNIST-2 的 DeepONet 路线
   在本数据上再次成立；训练最快（~20 min）
2. **psi 强、J/Ip 弱（容量分配）**：J 2.61%（FNO 的 1.9×）、Ip 0.38%（1.8×）、
   GS 残差 0.0298（2×）——0.6M 容量学 ψ 主任务绰绰有余（0.80%），但 J 是
   ψ 的二阶导，需要承载微结构的容量，被压缩后靠 PDE 项部分补偿（阶段2
   的 GS 残差从更高起点下降，val 0.95% 仍达标）——**若下游需要 J/Ip 输出，
   容量需加大**（branch/trunk 宽度翻倍是直接选项，未做超参搜索）
3. **X 点定位略逊但可用**：0.73/0.70 cm vs FNO 0.60/0.63 cm（+20%）——
   全局磁面拓扑由共享 trunk 表达，无显式全局核，但误差仍在亚厘米级
4. **两阶段方案对 DeepONet 成立**：e42 正常切换、ramp 无爆炸、阶段2 继续
   压降——物理约束与骨架正交的前提（plan 的关键前提）在点积输出上也验证
   （逐点评估使 PDE 残差天然适用）

## 6. 局限（PI-DeepONet 特有）

- **0.60M 参数容量**：仅为其他骨架的 1/4~1/7——若阶段1 卡在 3% 阈值外
  （e300 兜底切阶段2），"容量不足"与"分支-主干分解不适合"无法由本实验
  单独区分；解读时以 exp301-303 的容量-精度关系作参照
- **trunk 逐点评估的谱局限**：MLP 主干没有卷积/谱的平移结构，65² 网格上
  的平滑性靠共享 trunk 隐式提供；磁面的大尺度耦合（O 点/X 点全局位置）
  依赖 trunk 的表达力而非显式全局核
- **分支输入是均值池化**：`x[:,2:].mean((2,3))` 要求标量通道是常数场
  （数据管线保证）；若未来输入含逐像素场（如 F 分布），该取法需改
- **训练效率**：einsum 点积 (B,4225,64)·(B,4225,64)→(B,4225,64) 是
  (R,Z) 网格上的逐点共享运算，与 FNO 同量级（冒烟 0.6 min/30 epochs，
  最快骨架）

## 7. 复现

```bash
bash dn_fno_2608/scripts/run_exp303_304_pino.sh train   # 训练（含 exp303）
bash dn_fno_2608/scripts/run_exp303_304_pino.sh eval    # 评估（含 exp303）
# 单独：
cd "$(git rev-parse --show-toplevel)"
C:/Users/HP/.conda/envs/torch5060/python.exe -u -m gs_pino_fno_phys.train_pino \
  --mode twostage --model deeponet --train-data dn_fno_2608/data_v5/dn/train.npz \
  --val-data dn_fno_2608/data_v5/dn/val.npz --n-train 500 --seed 1 \
  --epochs 800 --phys-weight 0.1 --ip-weight 1.0 --j-weight 1.0 \
  --stage1-threshold 0.03 --stage1-max-epochs 300 --stage2-ramp-epochs 30 \
  --out-dir dn_fno_2608/experiments/exp304_pideeponet_pino_twostage_n500
```

产物同 exp102 约定：`best.pt`（含 `model` 键）、`history.json`、`args.json`、
`metrics.json`、`train.log`/`eval.log`、`figures/`（fig1_best_worst_psi /
fig2_field_stats / fig3_geometry_stats + stats_per_sample.json）。
