# exp008 — MAST 混合位形训练（DN+SN，config 输入）

> 实验日期：2026-08-17
> 状态：**完成**（N=500，seed 1）
> 代码：`src/gs_pino_dn_fno_2608/`（`train_dn_fno --config-input` + data_dn_fno
> 多文件拼接）
> 数据：`dn_fno_2608/data_v5/`（MAST DN+SN 分开落盘，各 3000；见
> [data_v5/README.md](../../data_v5/README.md)）
> 对照：exp009（专职单一位形分开训练）
> 结论速览：**混合训练在两配置上都工作（整体 test rel L2 0.935%），代价是
> 相对专职模型 +26~35%（DN 桶 0.667% vs 0.492%，SN 桶 1.203% vs 0.953%）；
> 而专职模型跨配置外推崩溃（DN→SN 253%，SN→DN 4.0e8%）——config 通道被
> 模型有效利用，混合训练换来的是**跨位形泛化**，而非零代价。

---

## 1. 动机与科学问题

data_v4（exp006/007）验证了"约束可达 + 可行区采样"下 X 点输入家族恢复
（A' 13ch 0.442%）。data_v5 是**位形泛化**的第一步：单一位形（DN）训练的
模型能否用**混合位形数据**学到跨位形（DN↔SN）的一致映射？

两条设计决策（用户拍板）：
- **混合位形先做 DN+SN**（divertor 族内差异：X 点数目 2 vs 1）；
- **真实装置用 MAST**（11 控制线圈、无墙、实际装置几何）；
- **数据集按配置分开生成**（`data_v5/dn`、`data_v5/sn` 各自文件，后续可只用
  一种或混合训练）。

本实验验证混合训练本身（科学问题）：
1. 混合模型在**整体 test**（DN+SN 拼接）上的水平 vs 专职模型（exp009）各自水平；
2. 混合模型在**分桶 test**（DN-only / SN-only）上是否对两个配置都接近专职水平
   （跨配置泛化代价）；
3. config 通道是否被模型利用（14ch xa+config 设计，SN 的 up=(0,0) 占位通道
   由 config 通道区分）。

## 2. 方法

- 训练：N=500（拼接后 4000 样本上嵌套取 500）seed 1，MSE/AdamW/
  ReduceLROnPlateau，同 exp006 惯例；
- 输入：A' 家族 `--input-mode xa --config-input` → **14 通道**
  （R, Z + 5 params + 4 X点 + 2 锚点 + 1 config）；
- 数据：train/val = `dn/train.npz,sn/train.npz`（逗号分隔多文件拼接），
  test 分桶 = dn/test.npz / sn/test.npz 单独 + 拼接整体；
- 归一化：stats 来自**拼接后的全量 train pool**（DN/SN 共享均值/方差），
  config 通道 z-score（0/1 分布 → 均值 0.5、std 0.5）。

## 3. 结果

### 3a. 混合 vs 专职（核心对比，[exp009](../exp009_split_configs/)）

| 指标 | 混合（exp008）DN桶 | 专职 DN（exp009） | 混合 SN桶 | 专职 SN（exp009） |
|---|---|---|---|---|
| rel L2 mean % | **0.667** | **0.492** | **1.203** | **0.953** |
| rel L2 median % | 0.560 | 0.411 | 1.033 | 0.818 |
| rel L2 p95 % | 1.289 | 1.010 | 2.518 | 1.908 |
| RMSE phys (Wb) | 2.74e-04 | 1.85e-04 | 5.24e-04 | 4.31e-04 |
| GS 残差比值 | 0.9888 | 0.9876 | 0.9836 | 0.9906 |
| find_critical 失败 | 0/500 | 0/500 | 0/500 | 0/500 |
| X 点误差 lo/up (cm) | 3.19 / 2.37 | 2.14 / 1.91 | 1.06 / — | 0.97 / — |
| sep_mean 误差 (cm) | 0.56 | 0.40 | 0.43 | 0.34 |
| sep 面积相对误差 (%) | 0.68 | 0.47 | 0.40 | 0.33 |
| O 点误差 (cm) | 0.48 | 0.34 | 0.61 | 0.42 |

（SN 无上 X 点 → x_up 为 NaN；几何指标 v2：真值基准配对 + 射线法分离面，见 §5.3）

### 3b. 训练（val）

| 模型 | best val rel L2 | epoch | 总时长 |
|---|---|---|---|
| 混合 xa+config 14ch | 0.8789% | 797 | 14.3 min |

### 3c. 分桶明细（analysis/）

[config_buckets.json](model_a14ch_xa_mix/analysis/config_buckets.json)：整体 test
（n=1000）rel L2 mean 0.935%、RMSE 3.99e-04 Wb、GS 残差比 0.986；
分桶 n=500/500，DN 桶 rel L2 mean 0.667% / SN 桶 1.203%（同上表）。
figures：[DN test](model_a14ch_xa_mix/figures_dn/)、
[SN test](model_a14ch_xa_mix/figures_sn/)（fig1 best/worst + fig2/3 统计）。

## 4. 结论

1. **混合训练可行**：单一模型在 DN+SN 拼接 test（n=1000）上 rel L2 0.935%、
   GS 残差比 0.986——两个配置都物理合理（find_critical 0 失败）。
2. **混合 vs 专职代价约 +26~35%**：DN 桶 0.667% vs 专职 0.492%（+36%）、
   SN 桶 1.203% vs 专职 0.953%（+26%）。SN 桶绝对误差整体高于 DN 桶
   （两种训练一致）——SN 的分离面约束自由度少、X 点位置对场更敏感。
3. **config 通道被有效利用**：专职模型**跨配置外推完全失败**（exp009 §3b：
   DN→SN 253%、SN→DN 4.0e8%），而混合模型两个桶都健康——up=(0,0) 占位
   恒值通道 + config one-hot 的组合让模型能区分位形并切换映射。
4. **几何误差与 rel L2 同量级、两桶同趋势**（v2 修复后）：X 点 1–3 cm、
   sep_mean 0.34–0.56 cm、sep 面积 <0.7%、O 点 <0.7 cm。SN 桶 X 点略优于
   DN 桶（单 X 点无配对歧义），与 rel L2（SN > DN）呈弱反相关——X 点在
   平缓 psi 区，位置误差对场误差高度敏感，两指标度量不同的物理量。

## 5. 偏差记录

1. **visualize_dn_fno 不支持 config 输入（(11,) vs (12,) 广播错）**：exp008
   的 checkpoint stats 含 config 通道（scalar_mean 12 维），但 visualize 重建
   dataset 时只按 input_mode 传 use_anchor，未传 use_config → __getitem__ 拼出
   11 维 scalars 与 12 维 stats 广播失败。修复：与 evaluate_dn_fno 同样的
   stats 过滤（排除 input_mode/config_input）+ `use_config=bool(ckpt["config_input"])`。
   回归：exp009（无 config）路径不受影响（改动为条件分支）。
2. **visualize 不支持逗号分隔多文件**：exp008 figures 按配置分开生成
   （figures_dn/、figures_sn/），未做多文件拼接可视化。
3. **几何指标 v2（2026-08-17，用户报告 fig3 大误差后修复）**：旧版
   geometry_metrics 从 find_critical 全候选按 Z 符号取第一个点配对（MAST
   无墙 → 混入 5–6 个真空假鞍点，配对失败时报数十 cm 假误差）、SN 的
   psi_bndry 错取"前两个 xpt 均值"（= 分离面 X 点 + 假鞍点均值）、分离面
   追踪依赖 matplotlib 闭合环（pred 场等高线穿出网格不闭合 → fallback 抓
   全域散点 → sep_mean 虚报 25–30 cm / 面积 100%+）。v2 修复：X 点按
   数据集真值（xpts_actual/o_point/axes[2]）配对（psi≥bnd−2e-3 + 距离
   ≤0.2 m，漏检走 ±0.35 m 局部鞍点找回，仍无则 NaN 诚实报告）；psi_bndry
   = 配对 X 点 psi 均值（DN）/唯一 X 点 psi（SN）；分离面 = 磁轴射线法
   （首穿 level，天然闭合）。修复后各实验几何数字见本文 §3a（旧版备份：
   eval*/metrics_v1.json、figures*/stats_per_sample_v1.json）。X 点仍不可
   定位率：DN ~10%、SN ~16%（FNO 平滑场中鞍点在平缓 psi 区消失，非配对
   错误）。fig1 新增装置结构（墙/线圈）+ 仅分离面 X 点/磁轴/分离面，
   R/Z 等比例（--machine 参数）；fig3 散点上限取 P95。
