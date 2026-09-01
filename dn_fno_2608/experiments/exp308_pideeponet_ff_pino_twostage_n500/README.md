# exp308 — PI-DeepONet 加宽 + trunk 傅里叶特征（exp307 + NeRF 式多频编码）

> 实验日期：2026-09-01 ｜ 状态：**失败（诚实负面结果）**——阶段2 物理训练
> 崩溃（N=500，seed 1，data_v5/dn 单一位形）
> 数据：`data_v5/dn/`（MAST DN，65×65，2000/500/500，无墙）——与 exp102 逐项一致
> 对照：exp307（DeepONet wide，0.7575%）；exp304（DeepONet 0.802%）
> 结论速览：**trunk 傅里叶特征（γ(R,Z)，ff_l=6）与逐点 FD 物理残差不兼容——
> 阶段1 正常（切换 e63、val 2.89%，冒烟亦正常）但阶段2 首段 pde 项爆炸
> （l_pde 6e-4 → 22.0，3.4 万倍），val rel L2 从 2.89% 爆到 92.6% 且 75
> epochs 未恢复，e138 正确早停（artifact 实为阶段1 模型，test 2.539%）。
> 机制：FF 高频特征（最高 32π）给 trunk 逐点输出注入空间高频分量，GS
> 残差的二阶中心差分放大高频 → 物理项梯度风暴；加宽（exp307）本身无此
> 问题（pde 6.6e-4 正常），问题确系 FF 特征引入**

## 1. 目标

exp304 README §6 记录了 trunk 的"逐点谱局限"：MLP 主干没有卷积/谱的平移
结构，65² 网格上的平滑性靠共享 trunk 隐式提供，J（ψ 的二阶导）对高频
分量最敏感。本实验在 exp307 的加宽基础上（**保持同一配置 p=hidden=384**，
归因口径：exp307→exp308 的增量 = 傅里叶特征的净贡献），给 trunk 输入加
**NeRF 式多频位置编码**：

$$\gamma(R,Z) = [\,x,\ \sin(2^k\pi x),\ \cos(2^k\pi x)\ \text{for } k=0..5\,],\quad x=(R,Z)$$

- 输入维度 2 → 2(1+2×6) = **26 维**，再进原 MLP（首层权重按新维度初始化）
- 最高频 2⁵π = 32π，恰为网格 Nyquist（dx = 2/64 → π/dx = 32π），无混叠
- 固定特征（sin/cos 无参数），MLP 学习组合——trunk 获得显式频率通道，
  高分辨率 J 微结构不需要全部压在隐式权重里

两阶段方案与 exp102 逐项一致：

**阶段1 损失 = MSE(psi_plasma, z) + w_j·masked MSE(J, z)（core mask 内）**
**阶段2 损失 = 阶段1 + w_pde·‖Δ\*ψ_pred + μ0·R·J_pred‖²（core mask 内）+ w_ip·‖(ΣJ·dA − Ip)/ip_scale‖²**

- 模型双输出通道：通道 0 = psi_plasma（z 域），通道 1 = Jφ（z 域）
- 切换条件：阶段1 val rel L2 < 3% 自动切阶段2（`--stage1-max-epochs 300` 兜底）
- **物理权重预热**：阶段2 的 w_pde/w_ip 从 0 线性 ramp 到目标值（30 epochs）

**架构**：branch ×2（各 16→384×3 层 GELU）+ 共享 trunk（γ(R,Z) 26 维 →
384×3 层）+ einsum 点积 → (B,2,65,65)；trunk 末层权重 ×0.1 小初始化；
**1,353,984 参数**（exp307 1,344,768 的 +9,216 = γ 首层权重增量，FF 本身
零参数）。

## 2. 输入通道（18 通道，同 exp102/exp304 coils 模式）

| # | 通道 | 含义/单位 | 来源 |
|---|---|---|---|
| 1 | R | 网格 R 坐标（[-1,1]） | 固定 MAST 65×65（trunk 输入，γ 编码） |
| 2 | Z | 网格 Z 坐标（[-1,1]） | 固定 MAST 65×65（trunk 输入，γ 编码） |
| 3 | Ip | 等离子体电流 (A) | params[0]（branch 输入） |
| 4 | paxis | 磁轴压强 (Pa) | params[1]（branch 输入） |
| 5 | fvac | 真空通量函数 f (Wb/m) | params[2]（branch 输入） |
| 6 | alpha_m | 剖面形状指数 m | params[3]（branch 输入） |
| 7 | alpha_n | 剖面形状指数 n | params[4]（branch 输入） |
| 8–17 | I_P2U … I_P6L | 上下偏滤器线圈电流 (A) | coil_currents[0:10]（branch 输入） |
| 18 | I_P1 (Solenoid) | 中心螺线管电流 (A) | coil_currents[10]（branch 输入） |

（分支输入与 exp304/307 相同；trunk 输入由裸 (R,Z) 换成 γ(R,Z)，在模型
内部完成，数据管线零改动。）

## 3. coil 分离（同 exp102）

网络**只预测 psi_plasma**（+Jφ）；评估时 `psi_total = psi_plasma_pred +
Σ_k I_k·G_k`（greens 恒等式验证 max diff 6e-8 Wb）。物理残差只定义在
等离子体场上——线圈场占 |psi_total| 幅值 183%、|Δ\*ψ_coils| 峰值 27.9 vs
等离子体 0.89，残差落在 psi_total 上会被线圈导体奇性主导。详见 exp101 README §3。

## 4. 训练设置

N=500（全量 2000 池嵌套子集，perm seed 12345），seed 1，AdamW lr 1e-3
wd 1e-4，ReduceLROnPlateau（patience 20，factor 0.5，min_lr 1e-5），batch 16，
800 epochs。模型 **PIDeepONet2d(deeponet_ff)**：`p=384, hidden=384, layers=3,
ff_l=6`，**1,353,984 参数**。PDE 残差 interior (1:-1,1:-1) 二阶中心差分，
core mask 内 masked mean((res/pde_scale)²)，pde_scale=0.362 Wb/m²；Ip 损失
用 ip_scale=5.51e5 A 归一化。

**阶段切换与崩溃**：阶段1 val rel L2 第 63 epoch 达 2.89% < 3% 阈值 → 切
阶段2。**阶段2 立即崩溃**：e70（ramp 0.23）train loss 1.77（pde 22.0）、
val rel L2 **92.57%**；随后 75 epochs 缓慢改善但从未低于切换值 2.89%
（e138 时 83.2%）→ 早停触发于 e138（stage-2 无改进 75 epochs，机制正确）。
**phase-aware best 如实保护**：artifact 取阶段2 best = 切换瞬间的
2.8921% @ e63——即**阶段1 末期的模型**，后续评估（§5）是阶段1 水平，
不代表阶段2 成果。

## 5. 结果（test n=500，DN-only）

> ⚠️ 以下数字的 checkpoint = **阶段1 末期模型**（val 2.89%@e63，阶段2 崩溃
> 后 artifact 未更新）——只代表"阶段1 训练的 deeponet_ff"，不代表两阶段
> 成果；exp307 的完整两阶段数字见 exp307 README §5。

| 指标 | exp308（FF，阶段1 模型） | exp307（DeepONet wide） | exp304（DeepONet） |
|---|---|---|---|
| rel_l2_plasma mean / median / p95 (%) | 2.89 / 2.48 / 5.75 | 0.87 / 0.64 / 1.99 | 0.92 / 0.69 / 2.06 |
| rel_l2_total mean / median / p95 (%) | **2.54 / 2.29 / 4.68** | **0.76 / 0.57 / 1.69** | **0.80 / 0.63 / 1.87** |
| rmse_phys (Wb) plasma / total | 9.6e-4 / 9.6e-4 | 2.88e-4 / 2.88e-4 | 3.04e-4 / 3.04e-4 |
| GS 残差 core mask：pred / truth | 1.1412 / 0.0040 | 0.0234 / 0.0040 | 0.0298 / 0.0040 |
| **Ip 相对误差** mean / median / p95 (%) | 1.60 / 1.44 / 4.01 | 0.21 / 0.15 / 0.58 | 0.38 / 0.34 / 0.84 |
| **J rel L2 mask 内** mean / median / p95 (%) | 5.42 / 4.74 / 10.2 | 2.13 / 1.92 / 3.67 | 2.61 / 2.36 / 4.87 |
| X 点定位误差 lo / up (cm) | 5.22 / 5.39 | 0.59 / 0.51 | 0.73 / 0.70 |
| O 点误差 / 分离面 mean (cm) | 1.42 / — | 0.20 / — | 0.24 / — |

**解读（失败归因，三证据链）**：
1. **崩溃的时序证据**：e63 切换前一切正常（阶段1 val 3.8%→2.89%平滑下降、
   冒烟 30 epochs 有限）；e70（ramp 0.23）pde 项 22.0（exp307 同刻 6.6e-4，
   3.4 万倍）、val 92.6%——崩溃与物理项 ramp 的介入严格同步
2. **机制**：FF 特征 γ(R,Z) 含 sin/cos(32πR) 等空间高频通道，trunk MLP
   随机初始化即把高频分量带进逐点输出；GS 残差用二阶中心差分（65² 网格
   上 d²/dR² 的离散核放大高频），物理项梯度对高频振荡极度敏感 → 梯度
   风暴把模型推出正常解空间（val 83%+ 且缓慢爬回）。这与 exp301 UNet
   的失败同构：**逐点/局部表达 + 高频敏感物理项的组合危险**
3. **排他性**：exp307（同配置无 FF）pde 6.6e-4 正常、val 平滑降到 0.88%；
   自测/冒烟均有限——FF 实现无误（不是 bug），问题在 FF 特征与逐点 FD
   物理项的相互作用，而非加宽或 DeepONet 本身

## 6. 局限与教训（傅里叶特征特有）

- **FF 与逐点 FD 物理项不兼容（本次核心教训）**：γ 的固定高频通道 + 二阶
  中心差分的频谱放大 = 阶段2 梯度风暴。若再尝试，方向是**抑制高频注入
  路径**：(a) FF 输入 ×0.01 缩放或首层权重小初始化（让高频通道初始近零）；
  (b) L 减小（如 ff_l=2，最高 8π）；(c) 物理项在平滑后的场上评估（违背
  PINO 精神，不推荐）；(d) 换可学习频率（SIREN 式），从低频起步
- **归因口径失效**：原设计"exp307→308 增量 = FF 贡献"未能执行——308 的
  artifact 是阶段1 模型，与 307 无可比性；FF 在纯监督（阶段1）下其实
  正常（2.89% 与 307 的 2.96% 同档），问题只在物理微调阶段
- **L=6 频段上限是设计选择**：32π 恰在 Nyquist——修复后（若做）仍需
  验证 L 的敏感性
- **γ 是固定特征，无自适应频率**：频率由设计定死；SIREN 式可学习频率
  是后续选项（也顺带避开固定高频注入）
- **诚实记录的价值**：早停 + phase-aware best 机制在崩溃中正确工作
  （75 epochs 无改进即停、artifact 保住阶段1 模型）——崩溃被安全捕获，
  没有产出误导性的"阶段2 成果"

## 7. 复现

```bash
bash dn_fno_2608/scripts/run_exp307_308_pino.sh train   # 训练（含 exp307）
bash dn_fno_2608/scripts/run_exp307_308_pino.sh eval    # 评估（含 exp307）
# 单独：
cd "$(git rev-parse --show-toplevel)"
C:/Users/HP/.conda/envs/torch5060/python.exe -u -m gs_pino_fno_phys.train_pino \
  --mode twostage --model deeponet_ff --train-data dn_fno_2608/data_v5/dn/train.npz \
  --val-data dn_fno_2608/data_v5/dn/val.npz --n-train 500 --seed 1 \
  --epochs 800 --phys-weight 0.1 --ip-weight 1.0 --j-weight 1.0 \
  --stage1-threshold 0.03 --stage1-max-epochs 300 --stage2-ramp-epochs 30 \
  --out-dir dn_fno_2608/experiments/exp308_pideeponet_ff_pino_twostage_n500
```

产物同 exp102 约定：`best.pt`（含 `model` 键）、`history.json`、`args.json`、
`metrics.json`、`train.log`/`eval.log`、`figures/`（fig1_best_worst_psi /
fig2_field_stats / fig3_geometry_stats + stats_per_sample.json）。
