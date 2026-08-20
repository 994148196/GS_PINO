# exp104 — FNO + GS 物理残差（做法2 两阶段，混合 DN+SN 训练）

> 实验日期：2026-08-20 ｜ 状态：**完成**（N=500，seed 1，data_v5 DN+SN 混合）
> 数据：`data_v5/dn + data_v5/sn` 逗号拼接（各 2000/500/500，混合 4000/1000/1000）
> 对照：exp102（DN-only 做法2，0.80%，Ip 0.21%）；exp103（混合做法1）；
> exp011（混合纯 MSE，0.894%）
> 结论速览：**两阶段自洽 + Ip 约束推广到混合 DN+SN——test 整体 rel L2
> 0.76%（DN 桶 0.73% / SN 桶 0.79%），Ip 误差 0.21%、mask 内 J 1.55%；
> 阶段1 第 45 epoch 达 3% 阈值切阶段2（物理权重 30-epoch 预热无爆炸）；
> psi↔J 自洽与 Ip 约束在 SN 桶同样成立（Ip 0.26% / J 1.73%）；vs exp103
> +0.06 精度代价换 J 通道 + 自洽 + Ip（与 exp101/102 的 0.72→0.80 同向）**

## 1. 目标

exp102（做法2，DN-only）实现 psi↔J 自洽 + Ip 约束；exp103（做法1，混合）
验证冻结 RHS 残差在混合数据上成立。本实验把**做法2 两阶段自洽**推广到
混合 DN+SN——重点：SN 桶（单 X 点位形、mask 更小）上 J 通道与自洽残差
是否与 DN 桶同机制。

**阶段1 损失 = MSE(psi_plasma, z) + w_j·masked MSE(J, z)（core mask 内）**
**阶段2 损失 = 阶段1 + w_pde·‖Δ\*ψ_pred + μ0·R·J_pred‖²（core mask 内）+ w_ip·‖(ΣJ·dA − Ip)/ip_scale‖²**

- 模型双输出通道：通道 0 = psi_plasma（z 域），通道 1 = Jφ（z 域）
- 切换条件：阶段1 val rel L2（psi_plasma）< 3% 自动切阶段2（预期晚于
  exp102 的 e29——混合 SN 桶拖慢收敛，300 epochs 兜底）；物理权重
  30-epoch 线性预热（exp102 阶段2 首 epoch 爆炸教训，记忆
  twostage-physics-weight-ramp）
- 混合口径与 exp103 相同：逗号拼接、4000 池嵌套抽 500（255 DN + 245 SN）、
  stats 全池（pde_scale 0.439）、无 config 通道

## 2. 输入通道（18 通道，同 exp103）

| # | 通道 | 含义/单位 | 来源 |
|---|---|---|---|
| 1 | R | 网格 R 坐标（[-1,1]） | 固定 MAST 65×65（dn/sn 同网格） |
| 2 | Z | 网格 Z 坐标（[-1,1]） | 固定 MAST 65×65 |
| 3 | Ip | 等离子体电流 (A) | params[0] |
| 4 | paxis | 磁轴压强 (Pa) | params[1] |
| 5 | fvac | 真空通量函数 f (Wb/m) | params[2] |
| 6 | alpha_m | 剖面形状指数 m | params[3] |
| 7 | alpha_n | 剖面形状指数 n | params[4] |
| 8–17 | I_P2U … I_P6L | 上下偏滤器线圈电流 (A) | coil_currents[0:10] |
| 18 | I_P1 (Solenoid) | 中心螺线管电流 (A) | coil_currents[10] |

## 3. coil 分离（同 exp101/exp103）

网络只预测 psi_plasma（+Jφ）；`psi_total = psi_plasma_pred + Σ_k I_k·G_k`
（greens 解析加回）。详见 exp101 README §3。

## 4. 训练设置

N=500（4000 池嵌套子集，perm seed 12345，**255 DN + 245 SN**），seed 1，
AdamW lr 1e-3 wd 1e-4，ReduceLROnPlateau（patience 20，factor 0.5，
min_lr 1e-5），batch 16，800 epochs（21.1 min）。模型 FNO2d2608：lift 18→64 +
4×FNOBlock（width 64, modes 16×16）+ proj 64→**2**，4.21M 参数。

**混合池统计**：psi_plasma 0.0400/0.0373、J 4.05e5/4.85e5、pde_scale 0.439
（vs DN-only 0.362，归一化抵消量级变化）、ip_scale 5.50e5（详见 exp103
README §4 表）。

**阶段切换**：阶段1 val rel L2 第 **45** epoch 达 2.81% < 3% → 切阶段2
（scheduler 不重置；早停仅从阶段2 计数；比 exp102 的 e29 晚——SN 桶拖慢
收敛，符合预期）。阶段2 物理权重 30-epoch 预热，pde 项稳定（无 exp102
首版阶段2 的 5.38 爆炸）。phase-aware best：artifact 取阶段2 best
（best val rel L2 **0.8591%** @ epoch 793）。

## 5. 结果（三桶 test：all=1000 / dn=500 / sn=500）

| 指标 | 整体（n=1000） | DN 桶（n=500） | SN 桶（n=500） |
|---|---|---|---|
| rel_l2_total mean / median / p95 (%) | **0.76 / 0.61 / 1.80** | 0.73 / 0.56 / 1.81 | 0.79 / 0.65 / 1.76 |
| rel_l2_plasma mean / median / p95 (%) | 0.88 / 0.70 / 2.10 | 0.88 / 0.69 / 2.07 | 0.89 / 0.70 / 2.14 |
| rmse_phys (Wb) | 3.17e-4 | 2.97e-4 | 3.38e-4 |
| GS 残差 core：pred / truth | 0.0167 / 0.0044 | 0.0158 / 0.0040 | 0.0176 / 0.0048 |
| **Ip 误差** mean / median / p95 (%) | 0.21 / 0.16 / 0.53 | 0.16 / 0.12 / 0.44 | 0.26 / 0.19 / 0.62 |
| **J rel L2 mask 内** mean / median / p95 (%) | 1.55 / 1.26 / 3.03 | 1.36 / 1.08 / 2.80 | 1.73 / 1.40 / 3.26 |
| X 点定位 lo / up (cm) | 0.53 / 0.69（SN 仅 lo） | 0.67 / 0.69 | 0.38 / NaN |
| O 点 / 分离面 mean (cm) | 0.29 / 0.37 | 0.26 / 0.38 | 0.31 / 0.35 |

**解读**（对照）：
1. **两阶段链路在混合数据上成立**：阶段2 加自洽残差 + Ip 后 val 继续降到
   0.86%，无爆炸——ramp 机制与 exp102 一致（e45 切换，早于 300 兜底）
2. **SN 桶自洽性同机制**：SN 桶 Ip 误差 0.26%、J mask 内 1.73%——单 X 点
   位形下 psi↔J 自洽与积分约束依然有效（略高于 DN 桶 0.16%/1.36%，同向）
3. **vs exp103（混合做法1）**：rel_l2_total 0.76% vs 0.70%（+0.06）——
   与 exp101/102 DN-only 对比（0.72 vs 0.80，+0.08）同向的小代价，换来
   J 通道 + 自洽 + Ip（exp103 给不了）
4. **vs exp102（DN-only 做法2）**：0.76% vs 0.80%——**混合不仅无代价反而
   略优**（同 exp103 的结论：PDE 正则吃掉跨位形共享容量代价）
5. **位形自推断再验证**：无 config 通道下 SN 桶健康（同 exp103）

## 6. SN 桶几何口径说明

同 exp103 §6：SN 单 X 点 → x_up_cm 恒 NaN（设计行为）；n_xpt_pred<2 计数
对 SN 语义不同，以 X 点误差与 GS 残差为准。

## 7. 复现

```bash
bash dn_fno_2608/scripts/run_exp103_104_pino.sh train   # 训练 exp103+104
bash dn_fno_2608/scripts/run_exp103_104_pino.sh eval    # 三桶评估 + 可视化
# 单独：
cd "$(git rev-parse --show-toplevel)"
C:/Users/HP/.conda/envs/torch5060/python.exe -u -m gs_pino_fno_phys.train_pino \
  --mode twostage \
  --train-data dn_fno_2608/data_v5/dn/train.npz,dn_fno_2608/data_v5/sn/train.npz \
  --val-data dn_fno_2608/data_v5/dn/val.npz,dn_fno_2608/data_v5/sn/val.npz \
  --n-train 500 --seed 1 --epochs 800 --phys-weight 0.1 --ip-weight 1.0 \
  --j-weight 1.0 --stage1-threshold 0.03 --stage1-max-epochs 300 \
  --stage2-ramp-epochs 30 \
  --out-dir dn_fno_2608/experiments/exp104_pino_twostage_mix_n500
```

产物：`best.pt`（含 switch_epoch/artifact_stage）、`history.json`（逐 epoch
stage/ramp/l_psi/l_j/l_pde/l_ip）、`args.json`、`metrics.json`、
`train.log`/`eval_*.log`、`eval_all|dn|sn/`、`figures_all|dn|sn/`
（fig1/2/3 + stats_per_sample.json，exp011 风格；twostage fig1 含 J 行、
fig2 含 Ip/J 直方图）。
