# exp003 — data_v2 上用线圈电流替代 X 点坐标输入（探针）

> 实验日期：2026-08-14
> 状态：**完成**（N=500，seed 1）
> 代码：`src/gs_pino_dn_fno_2608/`（data/train/evaluate_dn_fno_coils 通道数按 stats 推断，旧数据行为不变）
> 数据：`dn_fno_2608/data_v2/`（6000/6000，见 [data_v2/README.md](../../data_v2/README.md)）
> 结论速览：coil 输入在 data_v2 上仍可行——rel L2 0.412% vs exp002 0.303%（差 1.36×），
> 与 exp001 在旧数据上的比例（1.47×）几乎一致；GS 残差 0.996、find_critical 0/500。

---

## 1. 动机

exp001 已在旧数据（data/，3 参数）上验证"用 4 个控制线圈电流（P1L/P1U/P2L/P2U）
替代 4 个 X 点坐标作为输入"可行（rel L2 0.326% vs 基线 0.222%）。exp002 把数据
升级为 data_v2（剖面形状 alpha_m/alpha_n 采样，5 参数，11 通道）。本实验把 exp001
的输入替换重放到 data_v2 上：**在更复杂数据下，线圈电流（物理控制量）能否像
X 点坐标一样作为输入**，验证两种"隐性边界条件编码"的等价性是否随数据复杂度
保持。

## 2. 方法

### 2.1 输入结构对比

```
基线 data_v2（11 通道）:  R, Z | Paxis, Ip, fvac, alpha_m, alpha_n | R_lo, Z_lo, R_up, Z_up   (9 标量)
exp003（11 通道）:        R, Z | Paxis, Ip, fvac, alpha_m, alpha_n | I_P1L, I_P1U, I_P2L, I_P2U (9 标量)
```

- 4 个线圈电流来自 freegs 控制线圈 P1L/P1U（内线圈，正）+ P2L/P2U（外线圈，负），
  保存在数据集的 `coil_currents` 字段（(N,4)，生成时即为解出该平衡所用电流）；
- 数据、参数范围、求解器、训练超参与 exp002 完全一致（data_v2，N=500，seed 1）；
- 模型唯一差异来源：输入标量相同数量（9），in_channels=11 与 exp002 相同。

### 2.2 代码改动（向后兼容）

与 exp002 相同的通道数推断模式应用到 coils 变体：

| 文件 | 改动 |
|---|---|
| `data_dn_fno_coils.py` | 标量广播形状硬编码 `(7,)` → `(len(scalars),)`（旧数据仍 7 标量 9 通道） |
| `{train,evaluate}_dn_fno_coils.py` | `build_model(in_channels=2+len(scalar_mean))`（旧 checkpoint 自动仍为 9 通道） |
| 基线 `{data,train,evaluate}_dn_fno.py` | **零改动**（沿用 exp002 的推断） |
| `model_dn_fno.py` | **零改动** |

验证：data_v2 → 11 通道（4,211,777 参数，与 exp002 相同）；旧 data → 9 通道，
exp001 checkpoint 加载 + 前向通过（回归 OK）。

## 3. 结果（test 500，N=500，seed 1）

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

训练：best val rel L2 **0.3803%** @ epoch 795（exp002 0.2809% @ 788），11.7 min。

## 4. 结论

- **coil 输入在 data_v2 上仍可行**：rel L2 0.412% vs exp002（X 点输入）0.303%，差
  1.36×；对比 exp001 在旧数据上的比例（0.326/0.222 = 1.47×）——**"线圈电流替代
  X 点坐标"的相对精度损失几乎与数据复杂度无关**，恒定 ~1.4×。
- 物理一致性保持：GS 残差比值 0.996 与 exp002 相同；几何上反而略好——
  find_critical 0/500（exp002 有 1/500 失败）、sep_mean 0.182 cm（exp002 0.333）。
- 可能的机制：X 点坐标是自由边界解的"直接结果"，线圈电流需经模型隐式学习
  "电流→场形"映射（真实控制反演问题的方向），信息传递链更长，N=500 下差
  ~1.4× 属合理代价。
- 若实际应用场景只有线圈电流可测（无法直接给出 X 点坐标），0.412%（N=500）
  仍可用：按 exp002 外推的缩放律 ε ∝ N^-0.61，N=5000 时预计 ~0.11–0.14%。

## 5. 复现命令

```bash
PY="C:/Users/HP/.conda/envs/torch5060/python.exe"

# 训练
"$PY" -u -m gs_pino_dn_fno_2608.train_dn_fno_coils \
  --train-data dn_fno_2608/data_v2/train.npz --val-data dn_fno_2608/data_v2/val.npz \
  --n-train 500 --seed 1 --out-dir dn_fno_2608/experiments/exp003_coil_input_v2

# 评估
"$PY" -u -m gs_pino_dn_fno_2608.evaluate_dn_fno_coils \
  --test-data dn_fno_2608/data_v2/test.npz \
  --checkpoint dn_fno_2608/experiments/exp003_coil_input_v2/best.pt \
  --out-dir dn_fno_2608/experiments/exp003_coil_input_v2
```

## 6. 产物

```
exp003_coil_input_v2/
├── README.md       # 本文档
├── notes.md        # 简短结论记录
├── best.pt         # 最优权重（in_channels=11）
├── history.json    # 训练曲线
├── args.json       # 训练参数
└── metrics.json    # test 评估指标（全 500 样本）
```

训练日志：`dn_fno_2608/logs/exp003_coil_input_v2_n500_s1.log`

## 7. 后续可做（供参考）

- 与 exp001 结论并读：旧数据上 coil 输入（0.326%）比 X 点输入（0.222%）差 1.47×，
  看 data_v2 上是否保持同一比例，判断"线圈电流作为输入"的信息量差距是否随
  数据复杂度稳定；
- 若差值稳定，可考虑混合输入（X 点 + 线圈电流，13 通道）或把线圈电流作为
  额外通道而非替换。
