# exp005 — coil 电流输入在锚点大变化下的可靠性（data_v3，coil 11ch）

> 实验日期：2026-08-14 ｜ 状态：**完成**（N=500，seed 1）
> 数据：`data_v3/`（**数据集已删除**，见 exp004 说明）
> 对照：exp004 的 A（X点 11ch）/ A'（X点+锚点 13ch），同数据同 test
> 结论速览：coil 输入**确实隐含锚点信息**——test rel L2 12.5%（vs X点版 31.1%、
> X点+锚点版 21.7%），各锚点桶平坦、近退化区最优（8.6%）；但 12.5% 仍远高于
> data_v2 的 0.41%——**数据量/数据质量是主要瓶颈（后者被 exp007 证实）**

## 1. 目标

exp004 科学问题的 coil 输入视角：freegs 在全部约束（2 X 点 × Br/Bz + 2 isoflux）
下对 4 控制线圈电流做 Tikhonov 最小二乘——**电流解是锚点的确定性函数**，
即线圈电流隐含锚点（分离面形状）的全部信息。验证 coil 11ch 在 data_v3 下相对
退化远小于仅 X 点版。

## 2. 输入通道（11 通道）

R, Z + Ip, paxis, fvac, alpha_m, alpha_n + **I_P1L, I_P1U, I_P2L, I_P2U**（coil 电流）。

与 exp003 完全相同的模型与训练流程（coil 变体脚本零改动）；同数据 data_v3、
N=500 seed 1、同超参。

## 3. 结果（test 499）

| 指标 | B coil 11ch (data_v3) | exp003 (data_v2) 参考 |
|---|---|---|
| rel L2 mean % | **12.47** | 0.412 ± 0.328 |
| rel L2 median % | 8.37 | 0.324 |
| RMSE phys (Wb) | 3.85e-3 | 1.12e-4 |
| GS 残差比 | 1.245 | 0.995 |
| find_critical 失败 | 35/499 | 0/500 |

训练：best val 12.99% @e220，早停 e295，4.1 min。
三方对比与分桶见 [exp004/README.md §3-4](../exp004_xpoints_anchor_v3/README.md)。

## 4. 结论

1. **coil 隐含锚点信息被证实**：B（12.5%）优于仅 X 点 2.5×、优于 X 点+锚点
   1.7×（Tikhonov 电流解是锚点的确定性函数）；
2. **误差分布平坦**（各桶 7–13%，锚点贴 X 点仅 8.6%）→ 均匀欠拟合而非局部
   一对多，增大 N 预期可恢复；
3. N=500 对 data_v3 复杂度严重不足（12.5% vs data_v2 0.41%）——与 exp004
   结论一致（**后被 exp006/007 修订：主因是 isoflux 约束不可达的数据污染**）。

## 5. 复现命令（数据已删除，仅参考）

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

## 6. 产物

`best.pt` / `history.json` / `args.json` / `metrics.json` + `figures/worst_best/`
（test 上 best/worst 各 5 样本 truth vs pred 场对比 + samples_summary.json）。
无 fig1/2/3 统计图（可视化需要 data_v3 真值，数据集已删除）。生成脚本：
`src/gs_pino_dn_fno_2608/plot_best_worst_coil.py`。训练日志：`logs/exp005_*.log`。

## 7. 偏差记录

- worst 样本（如 idx274/idx303）真值分离面不经过锚点——isoflux 约束不可达
  （残差 >1 core）的数据点，见 exp004 notes §5b（data_v3 已删除）；
- 本实验的 coil 镜像脚本为 v1 几何（同 exp001/003/007 路径）。
