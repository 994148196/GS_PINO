# exp003 — data_v2 上 coil 电流替代 X 点坐标输入

> 实验日期：2026-08-14 ｜ 状态：**完成**（N=500，seed 1）
> 数据：`data_v2/`（alpha 采样） ｜ 对照：exp002（X 点输入 0.303%）、exp001（coil on data/ 0.326%）
> 结论速览：coil 输入在 data_v2 上仍可行——rel L2 0.412% vs exp002 0.303%
> （差 1.36×），与 exp001 的 1.47× 几乎一致 → **"coil 替代 X 点"的相对精度损失
> 与数据复杂度无关，恒定 ~1.4×**；GS 0.996、find_critical 0/500

## 1. 目标

exp001 在 data/ 上验证 coil 替代 X 点可行；exp002 把数据升级为 data_v2（5 参数，
11 通道）。本实验把输入替换重放到 data_v2：验证"coil 与 X 点等价性"是否随数据
复杂度保持。

## 2. 输入通道（11 通道）

| # | 通道 | 内容 |
|---|---|---|
| 1–2 | R, Z | 网格坐标（[-1,1]） |
| 3–7 | Ip, paxis, fvac, alpha_m, alpha_n | 运行参数（z-score） |
| 8–11 | **I_P1L, I_P1U, I_P2L, I_P2U** | 4 控制线圈电流（z-score；exp002 为 X 点坐标） |

数据/参数/求解器/超参与 exp002 完全一致（data_v2，N=500，seed 1）；模型
in_channels=11 与 exp002 相同。

## 3. 训练设置

同 exp002（MSE / AdamW / ReduceLROnPlateau / 早停 75 / 800 epoch）。best val
0.3803% @ epoch 795（exp002 0.2809% @ 788），11.7 min。

## 4. 结果（test 500）

| 指标 | exp003 coil (data_v2) | exp002 X点 (data_v2) | exp001 coil (data) | 基线 N=500 |
|---|---|---|---|---|
| rel L2 mean % | **0.412 ± 0.322** | 0.303 ± 0.227 | 0.326 | 0.222 ± 0.007 |
| rel L2 median % | 0.306 | 0.242 | 0.257 | 0.167 |
| RMSE phys (Wb) | 1.11e-4 | 8.23e-5 | 8.31e-5 | 5.48e-5 |
| sep_mean (cm) | 0.182 | 0.333 | 0.267 | 0.150 |
| x_lo / x_up (cm) | 0.445 / 0.598 | 0.711 / 0.468 | 0.386 / 1.007 | 0.494 / 0.477 |
| o_point (cm) | 0.398 | 0.261 | 0.254 | 0.142 |
| find_critical 失败 | **0/500** | 1/500 | 0/500 | 0/500 |
| GS 残差比 | 0.996 | 0.996 | 0.995 | 0.999 |

## 5. 结论

- **coil 输入在 data_v2 上仍可行**，相对 X 点输入的损失恒定 ~1.4×（与数据复杂度
  无关）；物理一致性保持（GS 0.996），几何反而略好（0/500、sep_mean 0.182 cm）；
- 机制：X 点是自由边界解的"直接结果"，coil 电流需模型隐式学习"电流→场形"
  （真实控制反演方向），信息传递链更长，N=500 下差 ~1.4× 属合理代价；
- 应用意义：若实际只有线圈电流可测（无法给出 X 点坐标），0.412%（N=500）仍可用
  ——按缩放律外推 N=5000 预计 ~0.11–0.14%。

## 6. 复现命令

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

## 7. 产物

`best.pt`（in_channels=11）/ `history.json` / `args.json` / `metrics.json` /
`figures/`。训练日志：`logs/exp003_coil_input_v2_n500_s1.log`。

## 8. 偏差记录

代码改动向后兼容：data_dn_fno_coils.py 标量广播 `(7,)`→`(len(scalars),)`；
train/evaluate_dn_fno_coils `build_model(in_channels=2+len(scalar_mean))`；基线
与 model_dn_fno 零改动。验证：data_v2→11ch（4,211,777 参数）、旧 data→9ch
（exp001 checkpoint 加载 + 前向通过）。
