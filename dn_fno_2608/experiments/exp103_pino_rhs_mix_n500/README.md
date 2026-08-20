# exp103 — FNO + GS 物理残差（做法1 RHS，混合 DN+SN 训练）

> 实验日期：2026-08-20 ｜ 状态：**完成**（N=500，seed 1，data_v5 DN+SN 混合）
> 数据：`data_v5/dn + data_v5/sn` 逗号拼接（各 2000/500/500，混合 4000/1000/1000）
> 对照：exp011（混合纯 MSE，coil 18ch，0.894%）；exp101（DN-only 做法1，0.72%）
> 结论速览：**把 exp101 的物理残差（做法1）推广到混合 DN+SN——test 整体
> rel L2 0.70%（DN 桶 0.69% / SN 桶 0.72%），GS 残差 core 0.0156 = 真值 FD
> 下限的 3.5 倍；与 exp011 混合纯 MSE（0.894%）相比全面更优（-21%），**混合
> 代价为零**（甚至略优于 exp101 DN-only 0.72%）；SN 桶 0.72% 好于 exp011 的
> DN 桶 0.84%——物理残差在混合数据上增益最大；无 config 通道下位形自推断
> 依然成立**

## 1. 目标

exp101（做法1，DN-only）证明物理残差可加进 FNO 训练且精度不降反升；exp011
证明 coil 18ch 输入（无 X 点/锚点/config）下混合 DN+SN 训练可行、位形由
11 线圈电流自推断。本实验把两者结合：**混合 DN+SN + coil 18ch（无 config）+
做法1 物理残差**——验证物理约束在跨位形混合数据上是否仍然成立、混合代价
多大、SN 桶是否与 DN 桶同机制。

**损失 = MSE(psi_plasma) + w_pde · ‖Δ\*ψ_pred + μ0·R·J_data‖²（core mask 内）**

- RHS 来自数据集：`J_data = R·p′ + F·F′/(μ0·R)`（float64 重构，DN/SN 均验证
  Ip 重构误差 1e-4 量级）
- 与 exp011 同口径混合：train/val/test 逗号拼接 dn+sn；4000 池嵌套抽 500
  （perm seed 12345 → **255 DN + 245 SN**）；stats 全池（4000）计算
- **无 config 通道**（`use_config=False`，18ch）——dn config=0 / sn config=1
  字段存在但不加载，位形信息只能由 11 线圈电流自推断（对 exp011 结论的
  再验证，且物理残差管线全程无位形标签）

## 2. 输入通道（18 通道，同 exp101/exp011 coils 模式）

| # | 通道 | 含义/单位 | 来源 |
|---|---|---|---|
| 1 | R | 网格 R 坐标（[-1,1]） | 固定 MAST 65×65（dn/sn 同网格，已验证 array_equal） |
| 2 | Z | 网格 Z 坐标（[-1,1]） | 固定 MAST 65×65 |
| 3 | Ip | 等离子体电流 (A) | params[0] |
| 4 | paxis | 磁轴压强 (Pa) | params[1] |
| 5 | fvac | 真空通量函数 f (Wb/m) | params[2] |
| 6 | alpha_m | 剖面形状指数 m | params[3] |
| 7 | alpha_n | 剖面形状指数 n | params[4] |
| 8–17 | I_P2U … I_P6L | 上下偏滤器线圈电流 (A) | coil_currents[0:10] |
| 18 | I_P1 (Solenoid) | 中心螺线管电流 (A) | coil_currents[10] |

注：`config` 字段（dn=0 / sn=1）存在但不加载（use_config=False）。

## 3. coil 分离（同 exp101）

网络**只预测 psi_plasma**；评估时 `psi_total = psi_plasma_pred + Σ_k I_k·G_k`
（greens 解析加回，恒等式 max diff 6e-8 Wb）。物理残差只定义在等离子体
场上——线圈场占 |psi_total| 幅值 183%，残差落在 psi_total 上会被线圈导体
奇性主导。详见 exp101 README §3。

## 4. 训练设置

N=500（4000 池嵌套子集，perm seed 12345，**255 DN + 245 SN**），seed 1，
AdamW lr 1e-3 wd 1e-4，ReduceLROnPlateau（patience 20，factor 0.5，
min_lr 1e-5），batch 16，800 epochs（X min）。模型 FNO2d2608：lift 18→64 +
4×FNOBlock（width 64, modes 16×16）+ proj 64→1，4.21M 参数。

**混合池统计（全 4000 训练样本）**：

| 统计量 | 混合（本实验） | DN-only（exp101） | SN-only（参考） |
|---|---|---|---|
| psi_plasma mean / std (Wb) | 0.0400 / 0.0373 | 0.03795 / 0.03470 | 0.0421 / 0.0397 |
| J mean / std (A/m²) | 4.05e5 / 4.85e5 | 3.47e5 / 4.20e5 | 4.86e5 / 5.53e5 |
| pde_scale (Wb/m²) | **0.439** | 0.362 | 0.549 |
| ip_scale (A) | 5.50e5 | 5.51e5 | 5.48e5 |

w_pde=0.1 不变：归一化（除以 pde_scale）抵消量级变化，真值残差下限两桶
同量级（DN l_pde floor 4.5e-5 / SN 6.9e-5）。

best val rel L2 **0.7844%** @ epoch 799（19.8 min）。

## 5. 结果（三桶 test：all=1000 / dn=500 / sn=500）

| 指标 | 整体（n=1000） | DN 桶（n=500） | SN 桶（n=500） |
|---|---|---|---|
| rel_l2_total mean / median / p95 (%) | **0.70 / 0.54 / 1.76** | 0.69 / 0.53 / 1.77 | 0.72 / 0.54 / 1.67 |
| rel_l2_plasma mean / median / p95 (%) | 0.81 / 0.64 / 1.96 | 0.82 / 0.66 / 2.10 | 0.79 / 0.62 / 1.82 |
| rmse_phys (Wb) | 2.87e-4 | 2.77e-4 | 2.98e-4 |
| GS 残差 core：pred / truth | 0.0156 / 0.0044 | 0.0147 / 0.0040 | 0.0164 / 0.0048 |
| X 点定位 lo / up (cm) | 0.46 / 0.57（SN 仅 lo） | 0.61 / 0.57 | 0.32 / NaN |
| O 点 / 分离面 mean (cm) | 0.29 / 0.37 | 0.27 / 0.40 | 0.30 / 0.34 |

**解读**（对照）：
1. **混合代价为零**：整体 rel_l2_total 0.70% vs exp101 DN-only 0.72%（甚至
   略优）vs exp011 混合纯 MSE 0.894%（-21%）——**物理残差把"跨位形共享
   容量"的代价（exp010/011 经验 +26~35%）直接吃掉了**：PDE 项约束的是
   位形不变的 Δ\*ψ 物理关系，混合训练反而利用共享容量更充分
2. **SN 桶增益最大**：SN 桶 0.72% 好于 exp011 的 DN 桶 0.84%；DN 桶 0.69%
   也优于 exp011 两桶——物理正则对数据量少的一桶（SN 仅 245 训练样本）
   帮助最明显（正则化等效于数据增广）
3. **物理约束仍成立**：core mask GS 残差 pred 0.0156 vs truth 0.0044——
   3.5× 于 FD 下限，与 exp101（3.3×）同量级；SN 桶 0.0164 vs 0.0048 同机制
4. **位形自推断再验证**：无 config 通道、训练样本仅 255 DN + 245 SN，
   SN 桶依然健康——11 线圈电流承载位形信息（exp010/011 结论在物理残差
   管线下的再验证）
5. **与 exp104 的取舍**：0.70% vs 0.76%（+0.06），同 exp101/102 的 DN-only
   对比（0.72 vs 0.80）——做法1 更省、做法2 多 J 通道 + 自洽 + Ip

## 6. SN 桶几何口径说明

- SN 单分离面 X 点（下方）：xpts_actual 混合拼接后第 2 行 NaN-pad →
  `x_up_cm` 恒 NaN（设计行为，非失败）；fig3 散点仅局部可定位样本
  （DN + 可定位 SN）显示，`stats_per_sample.json` 中 x_up_cm 全 NaN
- n_xpt_pred < 2 计数对 SN 语义不同（单 X 点位形本就 1 个）——SN 桶该计数
  不直接可比，以 X 点误差与 GS 残差为准

## 7. 复现

```bash
bash dn_fno_2608/scripts/run_exp103_104_pino.sh train   # 训练 exp103+104
bash dn_fno_2608/scripts/run_exp103_104_pino.sh eval    # 三桶评估 + 可视化
# 单独：
cd "$(git rev-parse --show-toplevel)"
C:/Users/HP/.conda/envs/torch5060/python.exe -u -m gs_pino_fno_phys.train_pino \
  --mode rhs \
  --train-data dn_fno_2608/data_v5/dn/train.npz,dn_fno_2608/data_v5/sn/train.npz \
  --val-data dn_fno_2608/data_v5/dn/val.npz,dn_fno_2608/data_v5/sn/val.npz \
  --n-train 500 --seed 1 --epochs 800 --phys-weight 0.1 \
  --out-dir dn_fno_2608/experiments/exp103_pino_rhs_mix_n500
```

产物：`best.pt`、`history.json`、`args.json`、`metrics.json`、
`train.log`/`eval_*.log`、`eval_all|dn|sn/`（各 metrics.json）、
`figures_all|dn|sn/`（fig1/2/3 + stats_per_sample.json，exp011 风格）。
