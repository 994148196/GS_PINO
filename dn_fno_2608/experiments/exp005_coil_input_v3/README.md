# exp005 — coil 电流输入在锚点大变化下的可靠性（data_v3，coil 11ch）

> 实验日期：2026-08-14
> 状态：**完成**（N=500，seed 1）
> 代码：`src/gs_pino_dn_fno_2608/train_dn_fno_coils.py`（与 exp003 同路径，零改动）
> 数据：`dn_fno_2608/data_v3/`（X 点 ±0.20 m + 锚点采样；**数据集已删除**（2026-08-14，
> 被 data_v4 取代），复现脚本 `scripts/run_generate_v3.sh` 保留）
> 结论速览：coil 输入**确实隐含锚点信息**——test rel L2 12.5%（vs X点版 31.1%、
> X点+锚点版 21.7%），各锚点桶平坦、近退化区最优（8.6%）；但仍远高于 data_v2
> 的 0.41%，N=500 是主要瓶颈。

---

## 1. 动机与科学问题

exp004 的科学问题同样适用这里：X 点与 isoflux 锚点都大范围变化时，**仅 X 点
坐标作输入**的模型不含锚点信息（一对多映射）。但本实验换一个输入视角：

freegs 在全部约束（2 X 点 × Br/Bz + 2 isoflux `psi(X点)=psi(锚点)`，共 6 个）
下对 4 个控制线圈电流做 Tikhonov 最小二乘——**电流解是锚点的确定性函数**，
即线圈电流隐含锚点（分离面形状）的全部信息。

假设：coil 11 通道输入在 data_v3 下依然可靠（相对退化远小于 X 点版），
与 exp004 的 A（X点 11ch）和 A'（X点+锚点 13ch）放在同一 test 上三方对比。

## 2. 方法

```
模型 B（11 通道）:  R, Z | Paxis, Ip, fvac, alpha_m, alpha_n | I_P1L, I_P1U, I_P2L, I_P2U   (9 标量)
```

- 与 exp003 完全相同的模型与训练流程（coil 变体脚本零改动，通道数从 stats 推断）；
- 同数据 data_v3、N=500 seed 1、同超参；
- 评估：test 500 全量 + `analyze_anchor_buckets.py` 分桶（与 exp004 的 A/A' 同表输出）。

## 3. 结果（test 499，N=500，seed 1）

| 指标 | B coil 11ch (data_v3) | exp003 (data_v2) 参考 | exp002 (data_v2) 参考 |
|---|---|---|---|
| rel L2 mean % | 12.47 | 0.412 ± 0.328 | 0.303 ± 0.227 |
| rel L2 median % | 8.37 | 0.324 | 0.242 |
| rel L2 <0.12% | 0.0% | — | — |
| RMSE phys (Wb) | 3.85e-3 | 1.12e-4 | 8.23e-5 |
| GS 残差比值 | 1.245 | 0.995 | 0.996 |
| find_critical 失败 | 35/499 | 0/500 | 1/500 |

训练：best val rel L2 12.99% @e220，早停 e295，4.1 min。

三方对比与分桶（A X点 / A' X点+锚点 / B coil）见
[exp004/README.md §3-4](../exp004_xpoints_anchor_v3/README.md) 与
[exp004/notes.md §5](../exp004_xpoints_anchor_v3/notes.md)。

## 4. 结论

1. **coil 输入隐含锚点信息被证实**：B（12.5%）优于仅 X 点输入 2.5×、优于
   X点+锚点输入 1.7×——Tikhonov 电流解是锚点的确定性函数，电流直接决定
   线圈磁通部分，映射更"因果"。
2. **误差分布平坦**（各桶 7–13%，锚点贴 X 点时仅 8.6%）→ B 是"均匀欠拟合"
   而非"局部一对多"，增大 N 预期可恢复精度。
3. 但 N=500 对 data_v3 复杂度仍严重不足（12.5% vs data_v2 的 0.41%）——
   与 exp004 结论一致：**当前数据量下任何输入模式都不可靠，coil 是最佳选择**。

## 5. 复现命令

```bash
PY="C:/Users/HP/.conda/envs/torch5060/python.exe"

"$PY" -u -m gs_pino_dn_fno_2608.train_dn_fno_coils \
  --train-data dn_fno_2608/data_v3/train.npz --val-data dn_fno_2608/data_v3/val.npz \
  --n-train 500 --seed 1 --out-dir dn_fno_2608/experiments/exp005_coil_input_v3
"$PY" -u -m gs_pino_dn_fno_2608.evaluate_dn_fno_coils \
  --test-data dn_fno_2608/data_v3/test.npz \
  --checkpoint dn_fno_2608/experiments/exp005_coil_input_v3/best.pt \
  --out-dir dn_fno_2608/experiments/exp005_coil_input_v3
```

分桶对比见 exp004 README §5（analyze_anchor_buckets，三模型同表）。

## 6. 产物

```
exp005_coil_input_v3/
├── README.md
├── notes.md
├── best.pt / history.json / args.json / metrics.json
└── figures/worst_best/        # test 上 best/worst 各 5 样本：truth vs pred
    ├── rank00_worst_idx274_...png   # 124%：isoflux 残差 1.33 core，最差样本
    ├── rank05_best_idx463_...png    # 1.8%
    └── samples_summary.json         # 10 样本的锚点/X点/参数/rel L2
```

本实验无 fig1/2/3 统计图（可视化需要 data_v3/test.npz 真值，数据集已删除，
按用户决定跳过），仅保留上述 worst_best 场对比图。

图注：白色 X = X 点约束位置，品红星 = isoflux 锚点，白色实线 = 分离面。
**注意**：worst 样本（如 idx274/idx303）的真值分离面并不经过锚点——这些是
isoflux 约束不可达（残差 >1 core）的数据点，见 exp004 notes §5b（data_v3 已删除）。
生成脚本：[plot_best_worst_coil.py](../../../src/gs_pino_dn_fno_2608/plot_best_worst_coil.py)。

训练日志：`dn_fno_2608/logs/exp005_*.log`
