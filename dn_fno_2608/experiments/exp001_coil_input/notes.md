# exp001: coil 电流输入 vs X 点坐标输入（论文基线）

**实验内容**：把基线输入的 4 个 X 点坐标通道替换为 4 个自由边界控制线圈电流
（P1L/P1U/P2L/P2U），其余完全不变（9 通道结构、模型、MSE 训练超参、N=500
嵌套子集、seed 1）。数据集无需重新生成（coil_currents 字段预留）。

**代码**：src/gs_pino/data_dn_fno_coils.py / train_dn_fno_coils.py /
evaluate_dn_fno_coils.py（独立文件，基线代码未改动）。

## 结果对比（test 500，N=500 seed 1）

| 指标 | coil 输入 | 基线 (X 点) | 论文 N=500 |
|---|---|---|---|
| rel L2 mean % | 0.326 | 0.222 ± 0.007 | 0.286 ± 0.034 |
| rel L2 median % | 0.257 | 0.167 | — |
| RMSE phys (Wb) | 8.31e-5 | 5.48e-5 | 8.53e-5 |
| <0.12% 样本占比 | 2.6% | — | — |
| sep_mean_cm | 0.267 | (见 report) | 0.072 (N=5000) |
| x_up_cm | 1.007 | — | — |
| find_critical 失败 | 0/500 | 0/500 | 0 |
| GS 残差比值 | 0.995 | 0.999 | 0.998 |

## 结论

- **coil 输入在 N=500 时精度约差 1.5×**（0.326% vs 0.222%），但仍处同一量级，
  绝对精度可用：几何指标合理、find_critical 全通过、GS 残差比值 0.995。
- 变差原因推测：X 点坐标是平衡边界的**直接拓扑锚点**（上下 X 点位置与
  分界面形态强相关）；coil 电流则是经过自由边界控制环求解的非线性映射
  （电流组合 → X 点位形），与 ψ 场的关系更隐式，且 4 个电流之间存在
  线性约束冗余（控制环解空间），信息密度低于 X 点坐标。
- 用户策略：小数据集结果"不错"则不跑更大档。0.326% 属于可接受但
  明显逊于基线；若要追平基线（~0.22%），按缩放律 ε ∝ N^-0.61 估计
  需 N ≈ 1500–2000。是否跑更大档由用户决定。

## 复现命令

```bash
PY="C:/Users/HP/.conda/envs/torch5060/python.exe"
"$PY" -u -m gs_pino_dn_fno_2608.train_dn_fno_coils \
  --train-data dn_fno_2608/data/train.npz --val-data dn_fno_2608/data/val.npz \
  --n-train 500 --seed 1 --out-dir dn_fno_2608/experiments/exp001_coil_input
"$PY" -u -m gs_pino_dn_fno_2608.evaluate_dn_fno_coils \
  --test-data dn_fno_2608/data/test.npz \
  --checkpoint dn_fno_2608/experiments/exp001_coil_input/best.pt \
  --out-dir dn_fno_2608/experiments/exp001_coil_input
```

产物：best.pt / history.json / args.json / metrics.json
训练日志：dn_fno_2608/logs/exp001_coil_n500_s1.log（best val 0.327% @ 783，11.0 min）
