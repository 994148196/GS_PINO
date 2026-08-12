# 历史实验复现手册

所有历史实验的**精确运行命令**（从每个 checkpoint 的 `args` 自动重建，与当时实际运行的参数完全一致）。
生成器：[gen_reproduce.py](gen_reproduce.py) —— 以后新增实验后重跑 `python scripts/gen_reproduce.py` 即可更新。

## 使用方法

```bash
# Windows：在 Git Bash（或任何 bash）中执行

bash scripts/reproduce_freegs.sh                 # 重跑全部自由边界实验（很慢，几百 epoch）
bash scripts/reproduce_freegs.sh plasma_coil     # 只重跑单个实验（推荐）
bash scripts/reproduce_fixed.sh v41_fixed        # 固定边界同理

bash scripts/reproduce_freegs.sh                  # 列出可用实验名
```

每个实验 = 训练 + 评估（自动执行 `train_freegs` → `evaluate_freegs`）。
**注意：重跑会覆盖该实验目录下的 best.pt / test_metrics.json**，历史结果见下方表格。

> ⚠️ 每个实验是几百个 epoch 的训练（CPU 上约 40-80 分钟），一键全部重跑请确认磁盘和时间。

## 自由边界（freegs 分支主线）

| 实验名 | 日期 | 架构 | 关键配置 | 等离子体 rel L2 | ψ_total rel L2 | 说明 |
|--------|------|------|----------|----------------|----------------|------|
| freegs_rhs_planet | 07-07 | PlaNetCore | 无 coil，seed 42，旧数据集 | — | — | 早期版本，未评估 |
| freegs_test_rmse | 07-07 | PlaNetCore | 2 epochs 冒烟 | — | — | 冒烟测试 |
| freegs_planet_v9 | 07-08 | PlaNetCore | 曲率 0.5，axis/ip 全开 | 9.57% | 12.25% | 预测 ψ_total 的基线 |
| freegs_ufno_v2 | 07-08 | U-FNO v2 | 全物理约束 | 18.03% | 30.06% | 强约束反而变差 |
| freegs_ufno_mse | 07-08 | U-FNO v2 | 纯 MSE | 10.02% | 13.68% | 无物理约束 |
| freegs_planet_plasma | 07-08 | PlaNetCore | predict_plasma，curv 0.1 | 8.64% | 10.57% | 首个预测 ψ_plasma |
| freegs_planet_plasma_mse | 07-08 | PlaNetCore | predict_plasma，纯 MSE | 8.29% | 10.19% | 约束全关 |
| freegs_planet_plasma_opt | 07-09 | PlaNetCore | +dropout/weight_decay/noise | 7.55% | 9.28% | 正则化调优 |
| **freegs_planet_plasma_coil** | 07-09 | PlaNetCore | +coil input +fourier 32 | **6.89%** | **8.33%** | **全局最优** |
| freegs_planet_plasma_soft | 08-12 | PlaNetCore | 同 coil（复现） | 7.27% | 8.81% | 复现确认，波动 ~0.4pp |

## 固定边界（暂不维护）

| 实验名 | 日期 | 数据集 | 说明 |
|--------|------|--------|------|
| large / large_v2 / large_v3 / large_v4 | 06-25~06-30 | gs_large / gs_large2k | V1→V4 演进 |
| large_v4.1 | 07-17 | gs_fixed_boundary_v41.npz | V4.1：masked_mse 2.36e-3，relL2 6.17% |
| v41_fixed | 07-17 | gs_fixed_boundary_v41_fixed.npz | 修复后：masked_mse 7.1e-4，relL2 3.14% |

固定边界指标来自 `test_metrics.json`（键 `rel_l2`）。

## 生成器说明

`gen_reproduce.py` 自动扫描 `outputs/*/best.pt`：

- 含 `scale_mse`/`hidden_dim` → 自由边界（`train_freegs.py` 的 CLI 映射）
- 含 `pde_weight`/`bc_weight` → 固定边界（`train.py` 的 CLI 映射）
- checkpoint 里不存在的键不输出（沿用 CLI 默认值，与历史一致）
- 每个实验附带：创建日期、模型变体、`test_metrics.json` 中的最优指标注释
