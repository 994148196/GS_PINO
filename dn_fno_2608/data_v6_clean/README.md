# data_v6_clean — data_v6 清洗版（剔除病态样本 + 候选池补足）

> 生成日期：2026-08-21 ｜ 用途：exp013（清洗数据重训，与 exp012 同设置对照）
> 代码：`dn_fno_2608/scripts/filter_v6.py`（判据）+
> `dn_fno_2608/scripts/run_generate_v6_topup.sh`（补足候选池生成）
> 前置数据：`../data_v6/README.md`（字段/通道/调用全文档，本目录只记录差异）

## 1. 速览

| 项 | 值 |
|---|---|
| 来源 | data_v6 五配置（MASTU_simple，129²，21ch coil 输入） |
| 清洗 | 剔除病态样本（判据见 §3），**原 data_v6 不动**（只增不改） |
| 补足 | 被剔除部分由**新 seed 独立生成**的健康候选池补齐（分布同参数范围） |
| 规模 | 每配置 train 500 / val 100 / test 200（与 data_v6 相同） |
| 差异 | **仅样本构成**：健康子集与 data_v6 逐样本一致（相同 chunk 序），
  新增补足样本来自新 seed（20260821 系），与原样本参数不重叠 |

## 2. 目录与文件

```
dn_fno_2608/data_v6_clean/
├── {dn,sn,snow_single,snow_double,limiter}/
│   └── train.npz / val.npz / test.npz   # 与 data_v6 同结构同 keys
├── manifest.json   # 逐 (cfg,split): 剔除 idx、判据值明细、topup 来源与行号、最终规模
└── scores.json     # 逐 (cfg,split) 剔除数量与 idx（data_v6 原始打分）
```

## 3. 剔除判据（2026-08-21 研究结论，exp012 stats 关联验证）

| 配置 | 判据 | test 剔除 | 依据 |
|---|---|---|---|
| **sn** | `gs_true > 15` ∪ `midplane_ratio < 0` | 27/200 (13.5%) | 磁轴上瓣薄/GS 未收敛病态；被剔样本 rel_l2 mean 18.9%，桶 8.99→7.44%（−17%）；worst10 图抓 8 个 |
| **dn** | `gs_true > 15` | 4/200 (2%) | 仅清未收敛尾部（idx 36/49/124/172） |
| **snow_double** | X 点记录偏差 > 0.15 m | 4/200 (2%) | idx 30/158/189/197：雪点约束在目标处 B≈0 但 find_critical 配对到内柱区（触壁+结构破坏），含桶内最差样本 |
| **snow_single** | 无 | 0 | 2026-08-21 用户拍板：数据健康（gs 4.2 < 其余 7.4、rel_l2 1.70%），仅标注问题（雪点不在分离面→X 点标记画到上偏滤器）不筛 |
| **limiter** | 无 | 0 | gs_true 0.32 最健康；wall_contact 100% 是有限位形设计行为 |

指标口径（与 exp012 evaluate 一致）：
- `gs_true`：GS 残差比 = ‖Δ\*ψ−RHS‖/‖RHS‖（mask = ψ≥ψ_bndry，`evaluate_dn_fno.gs_residual_ratio`）
- `midplane_ratio`：(ψ(R_axis,Z≈0)−ψ_bndry)/(ψ_axis−ψ_bndry)，<0 = 上瓣塌陷
- `xpt_dev`：xpts_actual 与 x_coords 按配对顺序的最大偏差

## 4. 补足（topup）

- 生成：`run_generate_v6_topup.sh`，新 seed（train 20260821/val 20260822/test 20260823
  起，dn/snow_double 用 20260831+/20260911+ 系列），**不用同 seed 扩展**
  （chunk resume 不推进 rng，新 chunk 参数序列会与前段重复）
- 候选池 → filter_v6 判据 → 健康候选按序取前 `need = target − 原健康数` 个
- 候选池目录 `dn_fno_2608/data_v6_topup/` 为一次性中间产物（已删除，无残留）
- 补足量与来源行号见 `manifest.json` 的 `topup_n`/`topup_rows`

## 5. 如何调用

```bash
PY="C:/Users/HP/.conda/envs/torch5060/python.exe"
CLEAN=dn_fno_2608/data_v6_clean
# exp013 训练（= exp012 命令仅数据源换 clean）
bash dn_fno_2608/scripts/run_exp013_train.sh
# 6 桶评估 + 可视化
bash dn_fno_2608/scripts/run_exp013_eval_vis.sh
```

字段/输入通道/网格与 data_v6 完全相同（21ch coil：R,Z + Ip/paxis/fvac/alpha_m/alpha_n
+ 14 线圈电流），见 `../data_v6/README.md` §3-4。

## 6. 相关脚本

| 脚本 | 用途 |
|---|---|
| `filter_v6.py` | 判据打分（score，写 scores.json）+ 构建 clean（compose，写 manifest.json） |
| `run_generate_v6_topup.sh` | 补足候选池生成（新 seed 独立池 + merge） |

## 7. 历史与偏差记录

1. **候选池新 seed 而非同 seed 扩展**：run_generate_v6.sh 注释的"同 seed 更大 n"方式
   在 chunk resume 下（跳过不推进 rng）新 chunk 参数序列重复旧段前段 → 弃用；
2. **snow_double x_coords 顺序与 dn 相反**（[上雪点, 下雪点] vs [下, 上]）——filter 按
   xpts_actual 生成器配对顺序对齐，不依赖 lo/up 语义；
3. **snow_single 标注问题不筛**：xpts_actual 对 31/200 样本记录到上偏滤器 X 点
   （雪点不在最终 psi_bndry 分离面），数据本身健康——待生成侧修复（雪点分离面
   判据），供 future data_v7。
