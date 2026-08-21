# exp106 — FNO + GS 物理残差（做法2 两阶段，data_v6_clean 五配置混合）

> 实验日期：2026-08-21 ｜ 状态：**完成**（N=500，seed 1，data_v6_clean 五配置）
> 数据：`data_v6_clean/` 五配置逗号拼接（同 exp105/exp013 口径）
> 对照：exp105（同数据做法1，2.362%）；exp013（同数据纯 MSE，3.045%）
> 结论速览：**做法2 在五配置混合上部分成立——test 整体 rel L2 2.625%
> （仍优于 exp013 纯 MSE −14%），但自洽链路未建立：GS 残差 pred 1.60 vs
> truth 0.003（550×），Ip 1.04%、mask 内 J 7.44%**。直接原因：阶段1 第
> **300** epoch 触发兜底切换（五配置混合下 psi_plasma 最低 3.416%，差
> 0.4 点未达 3% 阈值——limiter 触壁 + SN 病态拖慢），阶段2 只剩 75 epochs
> 即早停（patience 从切换起计），物理项刚开始收敛（l_pde 2.93→0.013 仍
> 在降）被掐断。**教训：五配置混合下阶段1 阈值 3% 不可达；阶段2 起始
> 质量差 + 窗口短 → 自洽无法建立**

## 1. 目标

exp104（v5 混合）证明做法2 两阶段自洽在双配置混合上成立（switch e45，
Ip 0.21% / J 1.55%）。本实验把做法2 推广到 data_v6_clean **五配置混合**
（含触壁 limiter、双雪点、SN 病态）——验证自洽链路在最大复杂度混合池
上是否依然成立。

**阶段1 损失 = MSE(psi_plasma) + w_j·masked MSE(J, z)（core mask 内）**
**阶段2 损失 = 阶段1 + w_pde·‖Δ\*ψ_pred + μ0·R·J_pred‖² + w_ip·‖(ΣJ·dA − Ip)/ip_scale‖²**

## 2. 输入通道 / coil 分离

同 exp105（21ch，五配置拼接，无 config 通道；网络只预测 psi_plasma+J，
greens 加回 psi_total）。详见 exp105 README §2–3。

## 3. 训练设置

N=500（2500 池嵌套子集，perm seed 12345，五配置各 ~100），seed 1，
AdamW lr 1e-3 wd 1e-4，ReduceLROnPlateau，batch 16，800 epochs 上限
（**35.4 min**，early stop e375）。模型 FNO2d2608 双输出通道（psi_plasma
+ J），4.21M 参数。混合池统计同 exp105（pde_scale 0.718）。

**阶段切换（关键失败点）**：
- 阶段1 val rel L2 最低 **3.416% @ e294**——差 0.4 个点未达 3% 阈值 →
  **第 300 epoch 触发 `--stage1-max-epochs` 兜底切换**
- 阶段2 从 e301（切换扰动后 val 4.70%）开始，l_pde 2.93 → 0.013（e361）
  持续下降但未超过阶段2 起始 best（3.43% @ e300）→ **e375 早停**
  （patience 75 从阶段2 起计），物理项在收敛途中被掐断
- vs exp104（v5 混合）：switch e45（val 2.81% 达标）→ 阶段2 有 748 epochs
  精调窗口；本实验阶段2 只有 74 epochs

## 4. 结果（六桶 test：all=1000 / 各配置 200）

### 4.1 rel L2（psi_total 域，与 exp013 同口径）

| 桶 | exp106（本实验） | exp105（做法1） | exp013（纯 MSE） |
|---|---|---|---|
| **整体（n=1000）** | **2.625%** | **2.362%** | 3.045% |
| dn | 2.983 | 2.426 | 3.193 |
| sn | 5.841 | 5.882 | 7.738 |
| snow_single | 1.476 | 1.387 | 1.433 |
| snow_double | 1.043 | 0.787 | 1.074 |
| limiter | 1.783 | 1.325 | 1.788 |

plasma 域：all 3.446% / dn 3.340% / sn 7.838% / snow_single 1.909% /
snow_double 1.141% / limiter 3.004%。

### 4.2 物理项（自洽失败的直接证据）

| 指标（all / dn / sn / snow_single / snow_double / limiter） |
|---|
| GS 残差 core：pred | 1.60 / 1.67 / 1.44 / 1.08 / 1.28 / 2.53 |
| GS 残差 core：truth | 0.0029 / 0.0019 / 0.0021 / 0.0027 / 0.0013 / 0.0067 |
| Ip 误差 (%) | 1.04 / 0.75 / 1.88 / 0.77 / 0.60 / 1.20 |
| J rel L2 mask 内 (%) | 7.44 / 6.39 / **19.8** / 4.19 / 2.28 / 4.52 |

J 通道在五配置上没学好（sn 桶 19.8%——病态样本 mask 形状复杂）→
阶段2 自洽残差从 550× 起步，75 epochs 内只降到 ~500× 量级（评估值），
远未建立 psi↔J 一致。

### 4.3 解读

1. **阶段1 3% 阈值在五配置混合上不可达**（最低 3.416%）——与用户此前
   "阶段1 等到 1%" 的问题同向：数据越复杂（limiter 触壁、SN 病态），
   psi_plasma 拟合越慢，严阈值只会导致兜底切换（本实验 e300 兜底）
2. **阶段2 窗口被早停压缩到 74 epochs**：patience 75 从切换起计，而切换
   扰动（4.70%）未在 75 epochs 内恢复到阶段1 最佳（3.43%）→ 早停；
   物理项 l_pde 仍在单调下降——若 patience 更长或切换后重置计数，自洽
   有望继续建立
3. **精度仍优于纯 MSE**（2.625 vs 3.045，−14%）：阶段1 的 J 监督 + 阶段2
   短程物理项未损害精度（同 exp104 结论的弱形式）——做法2 在五配置上
   的底线是"不低于纯 MSE，但自洽收益需要更长的阶段2"
4. **vs exp105（做法1）**：+0.26 整体代价（2.625 vs 2.362），物理项全面
   劣化（GS 残差 550× vs 20×）——本次切换失败是主因，非做法2 固有缺陷
   （exp104 在 v5 上证明做法2 自洽可达）
5. **改进方向**：阶段1 阈值放宽（4–5% 或固定 e250–300）、阶段2 独立
   patience（切换后重计 150+）、或 w_pde 分桶加权——留待后续实验

## 5. 几何指标口径说明

同 exp105 §6：limiter 桶全 NaN（无分离面）；snowflake 桶 O 点/分离面
受检测算法局限污染（雪点高阶零点混淆临界点分类，对应样本 rel L2 仅
0.5–1.5%）；sn 桶 x_up 恒 NaN。

## 6. 复现

```bash
bash dn_fno_2608/scripts/run_exp105_106_pino.sh train   # 训练 exp105+106
bash dn_fno_2608/scripts/run_exp105_106_pino.sh eval    # 六桶评估 + 可视化
```

产物同 exp105（best.pt 含 switch_epoch=300/artifact_stage、history.json
逐 epoch stage/ramp/l_psi/l_j/l_pde/l_ip、六桶 metrics/figures/stats）。
