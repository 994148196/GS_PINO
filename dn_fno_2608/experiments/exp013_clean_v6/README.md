# exp013 — data_v6_clean 清洗数据重训：剔除病态样本的收益验证

> 实验日期：2026-08-21 ｜ 状态：**完成**（N=500，seed 1）
> 数据：`data_v6_clean/`（data_v6 剔除病态 + 候选池补足，构成见
> [data_v6_clean/README.md](../../data_v6_clean/README.md)）
> 对照：exp012（同设置、原 data_v6 数据）
> 结论速览：**清洗后指标一致改善**——sn 桶 8.988%→7.738%（−13.9%）、
> 整体 3.435%→3.045%（−11.4%）、snow_double 1.307%→1.074%（−17.8%）。
> **收益分解（同 test 集严格对照）**：dn −10.7% / snow_double −9.6% 为
> **训练数据清洗的真实模型收益**；sn 桶改善主要来自 **test 集剔除病态**
> （模型在原 test 上 +3.1%，未变强）——sn 剔除 18.4% 训练样本后对边缘
> 样本泛化略降，提示病态剔除应保守

## 1. 目标

exp012 结论：五配置混合成立，但 **sn 桶 8.99% 异常 = 真值数据质量问题**
（磁轴偏下/上瓣薄/GS 未收敛病态，等高线曲折）。本实验验证：

**剔除病态样本（+ 补足）后重训，模型能否一致改善，尤其 sn 桶？**

## 2. 数据清洗（data_v6_clean，2026-08-21）

判据研究结论（filter_v6.py，全部由 npz 自身计算，exp012 stats 关联验证）：

| 配置 | 判据 | 剔除 (train/val/test) | 补足 |
|---|---|---|---|
| **sn** | `gs_true > 15` ∪ `midplane_ratio < 0` | 92 / 18 / 27 | 新 seed 候选池生成 137 |
| **dn** | `gs_true > 15` | 9 / 0 / 4 | 13 |
| **snow_double** | X 点记录偏差 > 0.15 m | 7 / 0 / 4 | 11 |
| snow_single / limiter | 无 | 0 | 0 |

- 原 data_v6 **不动**；健康子集与 data_v6 逐样本一致，补足样本来自新 seed
  （20260821 系，同参数范围）；规模保持 train 500/val 100/test 200
- 候选池生成：sn 单解 8.4 s/solve（129² 雪点约束）

## 3. 训练设置

与 exp012 **完全相同**（仅数据源）：21ch coil 输入（R,Z + 5 params + 14 线圈
电流，无 config 通道），N=500（nested seed 12345）、seed 1、800 epochs、
AdamW/MSE/ReduceLROnPlateau、batch 16、lr 1e-3→1e-5、patience 75。
best val rel L2 **3.3678%** @ epoch 789（exp012 3.5645% @ 799，−5.5%），
69.5 min。

## 4. 结果（rel L2 mean %，test）

### 4.1 主对照：清洗 test（exp013）vs 原 test（exp012）

| test | **exp013**（clean test 200/配置） | exp012（原 test 200/配置） | 变化 |
|---|---|---|---|
| 整体（n=1000） | **3.045** | 3.435 | −11.4% |
| dn | **3.193** | 3.618 | −11.7% |
| **sn** | **7.738** | 8.988 | **−13.9%** |
| snow_single | **1.433** | 1.486 | −3.6% |
| snow_double | **1.074** | 1.307 | −17.8% |
| limiter | 1.788 | 1.776 | +0.7%（持平） |

### 4.2 严格对照：两模型在**同一原 data_v6 test** 上（隔离模型收益）

| test（原数据） | exp013 ckpt | exp012 ckpt | 变化 |
|---|---|---|---|
| 整体（n=1000） | 3.379 | 3.435 | **−1.6%** |
| dn | 3.230 | 3.618 | **−10.7%** |
| sn | 9.265 | 8.988 | +3.1% |
| snow_single | 1.433 | 1.486 | −3.6% |
| snow_double | 1.181 | 1.307 | **−9.6%** |
| limiter | 1.788 | 1.776 | +0.7% |

**收益分解**：dn/snow_double 在同 test 集上模型确实变强（训练数据清洗收益）；
sn 模型未变强（+3.1% 噪声级）——sn 桶的 7.74% 主要来自 test 集剔除病态。
参考：exp012 ckpt 在剔除后的 173 健康样本上为 7.44%（≈ exp013 7.74%，含
27 个未见过的 topup 样本）。

### 4.3 补充观测

- exp012 ckpt 复测 clean val（物理域 rel L2 3.30%）vs 原 val（3.57%）→
  clean test 构成本身对 exp012 模型无害（略好），收益主要来自重训
- sn 桶 p95：exp013 19.67（clean test）vs exp012 25.20（原 test）
- RMSE (Wb)：整体 3.04e-3 vs 3.37e-3（−10%）；sn 8.65e-3 vs 原 9.9e-3 量级
- GS 残差：pred 3.75 vs freegs truth 6.79（ratio 0.55）与 exp012 同水平

## 5. 结论

1. **清洗策略分两段收益**：
   - **模型变强（训练数据收益）**：dn −10.7%、snow_double −9.6%（同 test 集
     严格对照）——train 中剔除的 dn 9 个、snow_double 7 个病态样本（未收敛/
     触壁结构破坏）确会拖累训练；
   - **指标改善（test 集构成效应）**：sn 桶 8.99→7.74%（−13.9%）主要来自
     27 个病态 test 样本被剔除；模型在**保留的健康样本**上能力基本不变
     （exp012 173 健康子集 7.44% ≈ exp013 7.74%，后者含 27 个未见 topup）；
2. **sn 剔除过量的警示**：train 剔除 92/500（18.4%）后模型在原 test 上
   +3.1%（略降）——病态样本仍是合法 GS 解，剔除过多损失边缘分布。sn 的
   残余 7.7% 主要是固有难度（SN 位形 + 129²），非可清洗病态；
3. **snow_double 是清洗最大赢家**：4 个触壁/雪点结构破坏样本（含桶内最差
   10.3%）剔除 + 补足，尾部收敛、桶误差 −17.8%；
4. limiter 持平（+0.7% 噪声级）：该桶本就无病态（gs_true 0.32），验证实验
   可复现性；
5. **实操建议**：病态剔除阈值宜保守（sn 用 `gs>15 ∪ midplane<0` 已是最低
   有效组合；若追求模型能力提升优先清 dn/snow_double 类中度病态，sn 类的
   大规模剔除仅改善指标不动能力）；生成侧修复（SN 磁轴检查收紧）从源头
   改善数据分布，优于事后剔除，供 data_v7。

## 6. 复现命令

```bash
PY="C:/Users/HP/.conda/envs/torch5060/python.exe"

# 清洗（score 打分 + compose 构建，判据与 manifest 见 data_v6_clean/）
"$PY" dn_fno_2608/scripts/filter_v6.py score
"$PY" dn_fno_2608/scripts/filter_v6.py compose

# 补足候选池生成（新 seed）
bash dn_fno_2608/scripts/run_generate_v6_topup.sh

# 训练 + 六桶评估 + 可视化
bash dn_fno_2608/scripts/run_exp013_train.sh
bash dn_fno_2608/scripts/run_exp013_eval_vis.sh
```

## 7. 产物

- `best.pt / history.json / args.json`（best val 3.3678% @ epoch 789）
- `eval_{all,dn,sn,snow_single,snow_double,limiter}/`（clean test 6 桶）
- `eval_on_v6_*`：exp013 ckpt 在**原 data_v6 test** 上的复测（对照 §4.2）
- `figures_{cfg}/`（5 配置可视化，limiter 桶 fig3 几何指标 NaN 已防护）
- 日志：`logs/exp013_*.log`；数据产物：`data_v6_clean/{scores,manifest}.json`

## 8. 偏差记录

1. **训练日志 val 读数陷阱**：history.json 的 `val_rel_l2` 为 raw 值（×100 才
   是日志显示的百分数）；曾误读为 exp013 比 exp012 差 100 倍，实为同水平
   （3.37% vs 3.56%）；
2. **visualize fig3 崩**：limiter 桶几何指标全 NaN → hist 崩溃（exp012 时期
   同样失败未暴露）；已加 `np.isfinite` 过滤 + "no finite data" 占位
   （visualize_dn_fno.py，向后兼容）；
3. **filter_v6 踩坑**：np.load 惰性解压（逐样本重复解压整文件 → 300s 超时）、
   savez_compressed 压缩 15 大文件过慢 → 一次性解压 + `np.savez`（与 data_v6
   一致）；x_coords[i] 不能 zip 展开（(4,) vs 样本迭代器）；
4. **topup 用新 seed 而非同 seed 扩展**：chunk resume 不推进 rng，同 seed
   新 chunk 参数序列重复旧段前段（见 data_v6_clean/README.md §4）；
5. **limiter 桶与 exp012 同水平**：该桶未清洗（本就健康），重训后持平验证
   了实验的可复现性（1.788 vs 1.776，+0.7% 噪声级）。
