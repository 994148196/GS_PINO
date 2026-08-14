# exp003: data_v2 上用线圈电流替代 X 点坐标输入（N=500, seed 1）

**实验内容**：把 exp001 的输入替换（4 个控制线圈电流 P1L/P1U/P2L/P2U 替代
4 个 X 点坐标）重放到 data_v2（alpha 采样，5 参数）上：输入 11 通道
`R, Z | Paxis, Ip, fvac, alpha_m, alpha_n | I_P1L, I_P1U, I_P2L, I_P2U`，
与 exp002 的 X 点版本通道数相同、仅替换最后 4 个标量。训练超参与
exp002/基线完全一致。

**代码**：coils 变体（data/train/evaluate_dn_fno_coils）应用与 exp002 相同的
通道数推断小改（`build_model(in_channels=2+len(scalar_mean))` +
`(7,)`→`(len(scalars),)` 广播修复）；旧数据仍 9 通道、exp001 checkpoint
回归通过；基线脚本零改动。

## 结果对比（test 500，N=500 seed 1）

| 指标 | exp003 coil (data_v2) | exp002 X点 (data_v2) | exp001 coil (data) | 基线 N=500 (data) |
|---|---|---|---|---|
| rel L2 mean % | **0.412 ± 0.322** | 0.303 ± 0.227 | 0.326 | 0.222 ± 0.007 |
| rel L2 median % | 0.306 | 0.242 | 0.257 | 0.167 |
| RMSE phys (Wb) | 1.11e-4 | 8.23e-5 | 8.31e-5 | 5.48e-5 |
| sep_mean_cm | 0.182 | 0.333 | 0.267 | 0.150 |
| sep_hausdorff_cm | 1.765 | 1.709 | 1.872 | 1.305 |
| x_lo / x_up_cm | 0.445 / 0.598 | 0.711 / 0.468 | 0.386 / 1.007 | 0.494 / 0.477 |
| o_point_cm | 0.398 | 0.261 | 0.254 | 0.142 |
| find_critical 失败 | **0/500** | 1/500 | 0/500 | 0/500 |
| GS 残差比值 | 0.996 | 0.996 | 0.995 | 0.999 |

训练：best val rel L2 0.3803% @ epoch 795（exp002 0.2809% @ 788），11.7 min。

## 结论

- **线圈 vs X 点的相对损失与数据复杂度无关**：data_v2 上 0.412/0.303 = 1.36×，
  旧数据上 0.326/0.222 = 1.47×——两条数据上比例几乎一致（~1.4×），说明这是
  "输入编码信息量"的固有差异，而非数据难度放大。
- 物理一致性不输 X 点版本：GS 残差 0.996；几何上 find_critical 0/500、
  sep_mean 0.182 cm 甚至优于 exp002——线圈输入在边界/几何任务上没有明显短板。
- **建议**：能拿到 X 点坐标时用 X 点输入（精度最优）；只能拿到线圈电流时
  该编码可用（N=5000 外推 ~0.11–0.14%）。可选下一步：X 点 + 线圈电流混合输入
  （13 通道）验证信息互补性。

## 复现命令

```bash
PY="C:/Users/HP/.conda/envs/torch5060/python.exe"
"$PY" -u -m gs_pino_dn_fno_2608.train_dn_fno_coils \
  --train-data dn_fno_2608/data_v2/train.npz --val-data dn_fno_2608/data_v2/val.npz \
  --n-train 500 --seed 1 --out-dir dn_fno_2608/experiments/exp003_coil_input_v2
"$PY" -u -m gs_pino_dn_fno_2608.evaluate_dn_fno_coils \
  --test-data dn_fno_2608/data_v2/test.npz \
  --checkpoint dn_fno_2608/experiments/exp003_coil_input_v2/best.pt \
  --out-dir dn_fno_2608/experiments/exp003_coil_input_v2
```

产物：best.pt（in_channels=11, params 4,211,777）/ history.json / args.json / metrics.json
训练日志：dn_fno_2608/logs/exp003_coil_input_v2_n500_s1.log（best val 0.3803% @ 795，11.7 min）
评估日志：dn_fno_2608/logs/exp003_eval.log
