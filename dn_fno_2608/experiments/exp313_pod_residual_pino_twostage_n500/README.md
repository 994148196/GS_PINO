# exp313 — POD 固定基 + 并联小 FNO 残差：低秩基管大体，谱卷积补细节

> 实验日期：2026-09-02 ｜ 状态：**完成（正面结果）**——POD 基管大体形状、
> 小 FNO 残差补子空间外细节，exp312 诊断的直接落地修复（N=500，seed 1，
> data_v5/dn 单一位形）
> 数据：`data_v5/dn/`（MAST DN，65×65，2000/500/500，无墙）——与 exp102 逐项一致
> 对照：exp312（POD-DeepONet，1.508%——低秩基封顶的诚实负面）；exp102（FNO
> 基线，0.800%）；exp311（TKNO，0.635% 系列最优）
> 结论速览：**"截断掉的 0.08% 能量恰是 X 点尺度结构"被直接修复：并联小 FNO
> 残差（870K 参数 = FNO 的 0.21×）后 test rel L2 1.508% → 0.883%（−41%），
> X 点 0.85 → 0.54/0.56 cm（−36%/−34%）、O 点 0.65 → 0.34 cm（−48%）、J
> 2.52% → 1.76%（−30%）、GS 残差 −73%——残差通道确实学到了基外细节。分阶段
> 冻结协议（先 branch 60 epochs 后残差）消除通道竞争。vs 纯 FNO 0.800% 仍
> 差 0.08pp（median 持平 0.645 vs 0.642），X 点反而反超——低秩基 + 谱残差的
> 混合方案以 1/5 参数达到 FNO 的 ~90% 精度****

## 1. 目标

exp312 给出了一个带"诊断结论"的负面结果：POD 基 p_psi=3（能量 99.924%）
封顶精度 ~1.5%，因为按方差排序的 SVD 把 X 点尺度的结构（占总方差极小，
但对磁面拓扑决定性）截掉了。本实验是 exp312 诊断的**直接落地修复**：不丢
低秩基的便宜（快、稳、可解释），并联一个学习残差通道：

$$\psi_z(R,Z) = \underbrace{\psi_0 + \sum_{k=1}^{p} b_k(x)\,\phi_k(R,Z)}_{\text{POD 通道（exp312 原样）}} + \underbrace{\text{FNO}_{\text{small}}(x)(R,Z)}_{\text{残差通道（学习 } y - \psi_{\text{POD}})}$$

- **POD 通道**：SVD 基（p_psi=3 / p_j=10）冻结为 buffer，分支 MLP 16 标量
  → 13 系数——与 exp312 逐字节相同（275,984 参数）
- **残差通道**：小 FNO（width 32、modes 12×12、4 层，≈0.59M 参数），吃
  全部 18 通道（含 R/Z 空间信息），学 POD 通道补不上的部分
- 残差选 **FNO 而非纯卷积**（如 UNet）：exp301–311 的结论链表明 GS 磁面的
  全局耦合需要全局混合（全局注意力 ≥ 多尺度谱 > 纯局部卷积）——残差要补
  的是 X 点这种跨域耦合细节，谱卷积是已验证的最低成本全局混合
- 残差 proj 层初始化 ×0.1（exp304 trunk 约定）：网络从 ~0 起步，保证
  pretrain 阶段残差通道不干扰 POD 通道收敛

**分阶段冻结协议（核心设计，`--pod-pretrain-epochs N`）**：POD 与残差同时
从零训练会发生**通道竞争**——两者都能单独拟合大体积形状，梯度会互相抢
信号（谁学得快谁压死谁）。解法是把分工写进时间：

| epochs | 训练 | 冻结 | 分工 |
|---|---|---|---|
| 1..N（60） | branch（POD 系数） | 残差 FNO（从 0 起步） | POD 通道先收敛（线性基 16 epoch 即可 <3%，见 exp312） |
| N+1 起 | 残差 FNO | branch（锚定已学 POD） | 残差学 `y − ψ_pod`——目标明确，无竞争 |

阶段1（纯监督）切换阶段2（加物理项）的判据**挂起到 pretrain 结束**：POD
通道常在第 16 epoch 就达 3% 阈值，但若此时切阶段2，物理 ramp 爬升期间
残差恰好在学——两件事叠加容易不稳。挂起保证残差开始学习时物理权重同步
从 0 爬升，阶段1/2 边界不割裂残差训练。

两阶段损失与 exp102/exp312 逐项一致：

**阶段1 损失 = MSE(psi_plasma, z) + w_j·masked MSE(J, z)（core mask 内）**
**阶段2 损失 = 阶段1 + w_pde·‖Δ\*ψ_pred + μ0·R·J_pred‖²（core mask 内）+ w_ip·‖(ΣJ·dA − Ip)/ip_scale‖²**

## 2. 输入通道（18 通道，同 exp102/exp312）

| # | 通道 | 含义/单位 | 来源 |
|---|---|---|---|
| 1 | R | 网格 R 坐标（[-1,1]） | 固定 MAST 65×65（POD 通道不用，残差 FNO 用） |
| 2 | Z | 网格 Z 坐标（[-1,1]） | 固定 MAST 65×65（POD 通道不用，残差 FNO 用） |
| 3 | Ip | 等离子体电流 (A) | params[0]（branch + 残差输入） |
| 4 | paxis | 磁轴压强 (Pa) | params[1]（branch + 残差输入） |
| 5 | fvac | 真空通量函数 f (Wb/m) | params[2]（branch + 残差输入） |
| 6 | alpha_m | 剖面形状指数 m | params[3]（branch + 残差输入） |
| 7 | alpha_n | 剖面形状指数 n | params[4]（branch + 残差输入） |
| 8–17 | I_P2U … I_P6L | 上下偏滤器线圈电流 (A) | coil_currents[0:10]（branch + 残差输入） |
| 18 | I_P1 (Solenoid) | 中心螺线管电流 (A) | coil_currents[10]（branch + 残差输入） |

（branch 输入 = 16 标量 `x[:,2:].mean((2,3))` 同 exp312；残差 FNO 吃全部
18 通道——R/Z 空间坐标在残差通道重新参与，这是对 exp312 §6 局限
"branch 忽略 R/Z"的直接回应。）

## 3. coil 分离（同 exp102）

网络**只预测 psi_plasma**（+Jφ）；评估时 `psi_total = psi_plasma_pred +
Σ_k I_k·G_k`（greens 恒等式验证 max diff 6e-8 Wb）。物理残差只定义在
等离子体场上——线圈场占 |psi_total| 幅值 183%、|Δ\*ψ_coils| 峰值 27.9 vs
等离子体 0.89，残差落在 psi_total 上会被线圈导体奇性主导。详见 exp101 README §3。

## 4. 训练设置

N=500（全量 2000 池嵌套子集，perm seed 12345），seed 1，AdamW lr 1e-3
wd 1e-4，ReduceLROnPlateau（patience 20，factor 0.5，min_lr 1e-5），batch 16，
800 epochs。模型 **PODResidual2d2608(pod_residual)**：POD 通道（p_psi=3 /
p_j=10，能量 99.92%/99.93%，n=500，同 exp312）+ 残差 FNO（width 32、modes
12×12、4 层），**870,706 参数**（POD 通道 275,984 + 残差 594,722；exp102
FNO 4.21M 的 0.21×、exp311 TKNO 1.21M 的 0.72×）。PDE 残差 interior
(1:-1,1:-1) 二阶中心差分，core mask 内 masked mean((res/pde_scale)²)，
pde_scale=0.362 Wb/m²；Ip 损失用 ip_scale=5.51e5 A 归一化。

**分阶段冻结协议**：`--pod-pretrain-epochs 60`——epochs 1..60 只训 branch
（残差从 0 起步）；e61 冻结 branch、解冻残差学 `y − ψ_pod`；阶段1 切换
挂起到 e60 后（POD 通道提前达标不生效）。

**阶段切换**：switch_epoch 61（pretrain 60 结束立即切，POD 通道 e16 前已
<3% 但被挂起）。交接时 val 2.19%（branch 冻结瞬间略升）→ 阶段2 残差接手
后持续下降：**best 1.0478% @ e475**，e550 早停（75 epochs 无改进），总
训练 8.3 min（870K 参数 = FNO 4.21M 的 0.21×，比 exp102 快 ~1.8×）。

## 5. 结果（test n=500，DN-only）

| 指标 | exp313（POD+残差） | exp312（POD） | exp102（FNO） |
|---|---|---|---|
| rel_l2_plasma mean / median / p95 (%) | 1.03 / 0.75 / 2.70 | 1.69 / 1.44 / 3.61 | 0.90 / 0.74 / 1.70 |
| rel_l2_total mean / median / p95 (%) | **0.883 / 0.645 / 2.05** | **1.51 / 1.27 / 3.26** | **0.800 / 0.642 / 1.59** |
| rmse_phys (Wb) plasma / total | 3.39e-4 / 3.39e-4 | 5.70e-4 / 5.70e-4 | 2.97e-4 / 2.97e-4 |
| GS 残差 core mask：pred / truth | 0.0219 / 0.0040 | 0.0812 / 0.0040 | 0.0151 / 0.0040 |
| **Ip 相对误差** mean / median / p95 (%) | 0.338 / 0.246 / 0.935 | 0.25 / 0.17 / 0.72 | 0.206 / 0.162 / 0.536 |
| **J rel L2 mask 内** mean / median / p95 (%) | 1.76 / 1.41 / 3.86 | 2.52 / 2.13 / 4.98 | 1.36 / 1.11 / 2.66 |
| X 点定位误差 lo / up (cm) | 0.54 / 0.56 | 0.85 / 0.85 | 0.60 / 0.63 |
| O 点误差 (cm) | 0.34 | 0.65 | 0.21 |
| n_xpt_fail | 0 | 0 | 0 |

**解读**：
1. **修复验证成立（核心结论）**：POD 1.508% → 0.883%（**−41%**）——exp312
   诊断"截断掉的 0.08% 能量恰是 X 点尺度结构"被直接证实 + 修复：X 点
   0.85 → 0.54/0.56 cm（−36%/−34%）、O 点 0.65 → 0.34 cm（−48%）、J
   2.52% → 1.76%（−30%）、GS 残差 0.0812 → 0.0219（−73%）。残差通道学的
   正是 POD 基的"基外细节"，且谱卷积能表达它
2. **vs 纯 FNO（0.800%）：+0.08pp（+10%），median 持平（0.645 vs 0.642）**：
   870K 参数（FNO 的 0.21×）达到 FNO 的 ~90% 精度——低秩基管大体 + 谱
   残差补细节的混合方案接近但未超越纯 FNO；p95 2.05 vs 1.59 尾部仍差
   （残差通道容量不足覆盖最难样本的基外成分）
3. **X 点反超 FNO（0.54/0.56 vs 0.60/0.63 cm）**：POD 基的大体形状先验 +
   残差细节的组合在 X 点定位上优于纯 FNO 从头学——磁面拓扑关键点被
   双通道分工精确照顾
4. **Ip 0.338% 反而是退化**（POD 0.253%、FNO 0.206%）：POD 通道的 13 系数
   在积分约束上是"紧"的（低维空间 Ip 容易精确）；并联全自由度的残差 FNO
   后 J 的自由度放大，mask 内分布更准（1.76 vs 2.52）但积分匹配略松
   （+0.09pp vs POD）。J 的"分布精度"与"Ip 积分精度"在残差通道下被交换
5. **物理自洽性 −73% vs POD、仍 1.45× vs FNO**：GS 残差 0.0219 表明残差
   通道学到的是物理上合理的修正（非过拟合噪声）；双通道混合的物理残差
   介于纯低秩（0.0812）与纯全自由度（0.0151）之间，符合"容量越大物理
   越自洽"的系列规律（对照 exp304/307）
6. **8.3 min 训练**：870K 参数、e550 早停——混合方案保留 POD 的廉价
   （exp312 3.7 min 的 2.2×，FNO ~15 min 的 0.55×）

## 6. 局限

- **通道分工是"时间冻结"而非架构约束**：残差理论上可以反向篡改 POD 通道
  的贡献（它是全自由度网络）——分阶段冻结协议保证分工，但阶段2 物理
  微调中两者同时可学，若 POD 通道被物理项进一步拉动，分工可能再混合
- **pretrain 长度是超参**：N=60 是设计值，未扫描（更长 → POD 更收敛但残差
  起步晚；更短 → 竞争风险）。冒烟验证过 N=15 同样能交接
- **残差 = 小 FNO 一种选择**：UNet 残差/attention 残差未对比；FNO 是基于
  "全局耦合需要谱混合"结论链的理性选择，非消融
- **低秩基的 p 仍是 exp312 的自动判据**（99.9% 能量）——残差补充的是
  截断模态，若 p 变大残差任务变轻、p 变小残差任务变重，平衡点未扫描

## 7. 复现

```bash
bash dn_fno_2608/scripts/run_exp313_pino.sh train   # 训练（pretrain 60）
bash dn_fno_2608/scripts/run_exp313_pino.sh eval    # 评估
# 单独：
cd "$(git rev-parse --show-toplevel)"
C:/Users/HP/.conda/envs/torch5060/python.exe -u -m gs_pino_fno_phys.train_pino \
  --mode twostage --model pod_residual --train-data dn_fno_2608/data_v5/dn/train.npz \
  --val-data dn_fno_2608/data_v5/dn/val.npz --n-train 500 --seed 1 \
  --epochs 800 --phys-weight 0.1 --ip-weight 1.0 --j-weight 1.0 \
  --stage1-threshold 0.03 --stage1-max-epochs 300 --stage2-ramp-epochs 30 \
  --pod-pretrain-epochs 60 \
  --out-dir dn_fno_2608/experiments/exp313_pod_residual_pino_twostage_n500
```

POD 基自动计算（SVD 于 N=500 训练子集，能量 99.9% 封顶 256 模态）并存入
`best.pt` 的 `pod_basis` 键；评估从 checkpoint 恢复基，无需重算。产物同
exp102 约定：`best.pt`、`history.json`、`args.json`、`metrics.json`、
`train.log`/`eval.log`、`figures/`（fig1_best_worst_psi / fig2_field_stats /
fig3_geometry_stats + stats_per_sample.json）。
