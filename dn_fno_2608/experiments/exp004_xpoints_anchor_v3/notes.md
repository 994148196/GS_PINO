# exp004 notes — X点/锚点大变化下的可靠性（data_v3，N=500 探针）

> 2026-08-14 生成与训练。配套 README.md（实验设计）。数据集 data_v3 已删除
> （2026-08-14，被 data_v4 取代；生成脚本 scripts/run_generate_v3.sh 保留）。
> 三模型同数据同规模：A 11ch X点（退化对照）、A' 13ch X点+锚点（上界）、B 11ch coil（exp005）。

## 1. 数据（data_v3）

- X 点抖动 ±0.20 m（基线 ±0.02 m）；锚点 R~U[1.2,1.8]×Z~U[-0.3,0.3]（基线固定 (1.5,0.0)）
- 生成结果：train 1999/2000、val 500/500、test 499/500 接受（~1.4 s/solve，重试主导）
- sanity：无 NaN；锚点/参数范围全部命中；coil 电流 ∈ [-1.6e6, 3.4e6] A

## 2. 训练（N=500 seed 1，全部早停）

| 模型 | 输入 | best val rel L2 | 早停 epoch | 时间 |
|---|---|---|---|---|
| A | X点 11ch | 29.57% (e18) | 93 | 1.3 min |
| A' | X点+锚点 13ch | 20.68% (e356) | 431 | 6.0 min |
| B | coil 11ch | 12.99% (e220) | 295 | 4.1 min |

通道数已核对（A 11 / A' 13 / B 11，input_mode 正确写入 checkpoint），无静默 bug。

## 3. test 500 全量评估

| 指标 | A (X点) | A' (X点+锚点) | B (coil) | data_v2 参考 (exp002/exp003) |
|---|---|---|---|---|
| rel L2 mean % | 31.09 | 21.72 | 12.47 | 0.303 / 0.412 |
| rel L2 median % | 25.88 | 12.70 | 8.37 | 0.242 / 0.324 |
| RMSE phys (Wb) | 1.04e-2 | 7.88e-3 | 3.85e-3 | 8.2e-5 / 1.1e-4 |
| GS 残差比值 | 1.034 | 1.219 | 1.245 | ~0.996 |
| find_critical 失败 | 21/499 | 48/499 | 35/499 | 0–1/500 |

**没有 0.12% 以下的样本（paper 声称 >95%），全部退化 30–100×。**

## 4. 锚点分桶（test 499，analyze_anchor_buckets.py）

rel L2 mean %，按锚点属性分桶：

### 4.1 锚点距默认位 (1.5,0.0) 的距离
| dist | n | A | A' | B |
|---|---|---|---|---|
| <0.2 m | 207 | 25.2 | 16.5 | 11.2 |
| 0.2–0.4 m | 291 | 35.3 | 25.4 | 13.4 |

### 4.2 锚点距最近 X 点
| d_min | n | A | A' | B |
|---|---|---|---|---|
| <0.3 m | 13 | 52.5 | 54.2 | 8.6 |
| 0.3–0.6 m | 179 | 39.5 | 31.5 | 12.9 |
| ≥0.6 m | 307 | 25.3 | 14.6 | 12.4 |

### 4.3 象限
四象限间 A/A' 差别小（±4 pct），B 也平坦——方位角不是主要变量。

## 5. 解读

1. **A（X点输入）不可靠——一对多映射被证实**：误差随锚点偏移增大（25→35%），
   锚点贴近 X 点时最高（52.5%）。同一输入对应多种分离面形状，模型只能学条件均值。
2. **A'（含锚点，信息完整）仍 21.7%**：说明在 data_v3 的复杂度下 N=500 是主要瓶颈，
   不是信息缺失。锚点贴 X 点的样本（13/499）映射病态敏感（54%），罕见样本学不到。
3. **B（coil）整体最好且各桶平坦**：线圈电流隐含锚点信息（Tikhonov 解是锚点的
   确定性函数），且电流直接决定线圈场——映射更"因果"、更好学。锚点贴 X 点时
   B 反而最优（8.6%），因为该病态区域的信息已经"压缩"在电流里。
4. **核心回答**：若 X 点与锚点都大范围变化，现 N=500 训练策略下程序不可靠——
   即便输入含完整信息（A'）或隐式信息（B）。要恢复可靠性需把数据量放大到
   data_v2 的等价信息量（N≈5000+ 或按锚点密度自适应采样）。

## 5b. 数据质量：isoflux 约束残差（2026-08-14 追加发现）

生成脚本的接受准则**不检查约束残差**；freegs Picard 环的收敛判据是 psi
迭代变化 < rtol，而控制步（`constrain(eq)`）在收敛检查**之后**应用。4 个控制
线圈对 6 个约束（2 X点×Br/Bz + 2 isoflux）是**过定系统**，Tikhonov 定点解
只满足 `Aᵀb=0`（残差⊥可达子空间），不可达方向的残差保留：

| 数据集 | isoflux 残差 |psi(Xpt)−psi(anchor)|/core |
|---|---|---|
| data/（基线，锚点固定） | 单样本 16% |
| data_v2（锚点固定） | mean 0.13 / med 0.13 / p95 0.27 |
| **data_v3（锚点采样）** | **mean 0.50 / med 0.43 / p95 1.11 / max 2.10**，>10% 占 93% |

方法学验证：RegularGridInterpolator 与 freegs 同款 RectBivariateSpline 两种
插值在保存的 psi_total 上结果一致（~1e-4），残差真实，非测量误差。

物理含义：
1. **锚点对 4 线圈大部分"不可达"**——data_v3 中多数样本的分离面并不通过给定
   锚点；锚点是让系统过定的真实自由度（X 点条件 4 个 + psi(lo)=psi(up) 自动 +
   psi(锚)=psi(X点) 1 个，4 线圈无法同时满足）。
2. A'（输入含锚点）的 21.7% 部分来自此数据内在不一致：输入"请求"了不可达的
   目标，输出是折中解——**提高 N 只能解决学习瓶颈，约束不可达性不会消失**。
3. B（coil）输入恰好就是折中解本身（控制决策）→ 输入输出严格因果 → 各桶平坦。
4. B 误差与残差相关系数仅 0.131：N=500 欠拟合主导，但最差样本全部落在高残差
   区（idx274 残差 1.33、idx303 残差 2.10 → rel L2 124%/79%）。
5. 修复方向（未来数据）：接受准则加 isoflux 残差阈值；或加控制线圈（8 线圈）
   使锚点可达；或收窄锚点采样范围。

## 6. 遗留问题

- [ ] N=5000（或更大）复训 A'/B 能否恢复到 1% 量级——区分"数据量"与"架构"瓶颈
- [ ] 锚点贴 X 点的近退化区域（d_min<0.3）在训练池中占比极低（13/1999），
      需要按 d_min 分层采样平衡
- [ ] exp005 的 B 相对 A' 的优势是否在更大 N 下保持（当前 1.7×）
- [ ] coil 输入的可解释性：对比 coil 输入 vs (X点+锚点) 在 test 上的逐样本误差分布

## 7. 复现

```bash
# 训练（已按此执行）
bash dn_fno_2608/scripts/run_exp004_005_train.sh
# 评估
"$PY" -u -m gs_pino_dn_fno_2608.evaluate_dn_fno --test-data dn_fno_2608/data_v3/test.npz \
  --checkpoint dn_fno_2608/experiments/exp004_xpoints_anchor_v3/model_a11ch_xpt/best.pt \
  --out-dir dn_fno_2608/experiments/exp004_xpoints_anchor_v3/model_a11ch_xpt
# （A' 同，--checkpoint model_a13ch_xa/best.pt；B 用 evaluate_dn_fno_coils）
# 分桶
"$PY" -u -m gs_pino_dn_fno_2608.analyze_anchor_buckets \
  --test-data dn_fno_2608/data_v3/test.npz \
  --checkpoint a11ch=.../model_a11ch_xpt/best.pt \
  --checkpoint a13ch=.../model_a13ch_xa/best.pt \
  --checkpoint b11ch=dn_fno_2608/experiments/exp005_coil_input_v3/best.pt \
  --out-dir dn_fno_2608/experiments/exp004_xpoints_anchor_v3/analysis
```

## 8. 偏差记录

- analyze_anchor_buckets.py 初版有 `list == int` 静默空桶 bug（`bkt[key] == b`），
  已修为 `np.array(bkt[key]) == b`；初版全 0 桶输出未误用。
- 评估脚本对 data_v3 无改动（通道数自动推断），直接复用。
