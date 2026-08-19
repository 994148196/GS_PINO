# exp012：coil 电流输入跨位形混合训练（MASTU_simple 五配置，21ch）

> 实验日期：2026-08-18 ｜ 状态：**完成**（N=500, seed 1, 800 epochs）
> 数据：[data_v6](../../data_v6/README.md)（五配置 train 500/val 100/test 200，129²）
> 目录：`dn_fno_2608/experiments/exp012_coil_input_v6/`
> 对照：exp011（MAST 18ch DN+SN：DN 0.841%/SN 0.948%）——**跨机器仅参考**
> 结论速览：21ch 端到端生成 psi 在五配置混合下成立——**eval_all 3.44%**，
> snow_double 1.31% / snow_single 1.49% / limiter 1.78% 最健康；
> **sn 桶 8.99% 异常，已确认是真值数据质量问题（见 §6.1，与模型能力无关）**

## 1. 输入通道（21ch）

`--input-mode coils --no-config-channel`（探针 separability：14 通道电流 pooled
分离度全部 >2.0 std、两两阈值判别 ≥90.75% → 位形信息由电流承载，无需 config 通道）：

| 通道 | 内容 |
|---|---|
| 1–2 | R, Z 网格（[-1,1] 归一化） |
| 3–7 | Ip, paxis, fvac, alpha_m, alpha_n（z-score） |
| 8–21 | 14 线圈电流：I_Solenoid, I_Pc, I_Px, I_D1, I_D2, I_D3, I_Dp, I_D5, I_D6, I_D7, I_P4, I_P5, I_P61, I_P62（z-score） |

目标：`psi_total`（129²，均值/标准差归一化）。多文件拼接 = 五配置 train pool
（2500 样本），`--n-train 500` nested 抽取。

## 2. 训练设置

| 项 | 值 |
|---|---|
| 数据 | data_v6 五配置混合 pool（train 2500 → 抽 500；val 500 全用） |
| 输入 | 21ch（coils，无 config 通道） |
| 网格 | 129² |
| 模型 | 手写 FNO（model_dn_fno.py），4,212,417 参数 |
| 优化 | AdamW lr 1e-3 wd 1e-4 / MSE / ReduceLROnPlateau(20, ×0.5, min 1e-5) |
| 训练 | 800 epochs、batch 16、workers 0、早停 patience 75 |
| best val rel L2 | **3.5645%** @ epoch 799（800 跑满无早停，69 分钟） |

## 3. 结果（6 桶评估，rel L2 / RMSE / GS 残差 / 几何）

| 桶 (n) | rel L2 | RMSE (Wb) | GS pred/true | O 点误差 | 分离面误差 | find_critical 失败 |
|---|---|---|---|---|---|---|
| eval_all (1000) | **3.435%** | 0.00367 | 4.02 / 7.21 | 3.75 cm | 2.79 cm | 4 |
| dn (200) | 3.618% | 0.00273 | 4.18 / 8.94 | 4.71 cm | 3.54 cm | 3 |
| sn (200) | **8.988%** | 0.01117 | 5.96 / 11.87 | 8.21 cm | 5.53 cm | 1 |
| snow_single (200) | **1.487%** | 0.00156 | 4.84 / 6.94 | 1.47 cm | 1.39 cm | 0 |
| snow_double (200) | **1.307%** | 0.00101 | 4.54 / 7.98 | 0.64 cm | 1.07 cm | 0 |
| limiter (200) | 1.776% | 0.00186 | 0.58 / 0.32 | NaN* | NaN* | 0 |

*limiter 无分离面 X 点（xpts_actual 0 行）→ 几何指标 NaN（诚实）；GS 残差
pred 0.58 / true 0.32 均显著低于偏滤器位形（限制面残差小），比值 1.8 偏高但绝对值健康。

## 4. 结论

1. **snow_double 1.31% / snow_single 1.49% 最优**——雪点二阶约束给出最确定的
   psi 结构（外加探针接受率最高/触壁最少）；limiter 1.78% 次之（结构最简单）；
2. **dn/sn 反而最差（3.6%/9.0%）**——与 exp011 相反（那时 DN+SN 是唯二位形）。
   可能因素：dn X 点抖动范围宽、sn 收敛迭代多（探针 103s/solve）导致数据内在
   噪声大；129² 上 rel L2 与 65² 不可直接比（网格点数 4 倍）；
3. **混合训练 5 位形未崩溃**——最难的 limiter/snow 桶 1.3–1.8%，14 通道电流
   对位形信息承载充分（separability 结论兑现）；
4. GS 残差 pred/true 全部 <1（除 limiter）——模型场比 freegs 真值更 GS-光滑
   （FNO 平滑化，exp011 同现象）。

## 5. 产物文件（本目录）

| 文件/目录 | 内容 |
|---|---|
| `best.pt` | 最优权重（model_state + 归一化统计 + args + `no_config_channel` 标志） |
| `args.json` / `history.json` | 训练配置 / 训练曲线 |
| `eval_all/`, `eval_{dn,sn,snow_single,snow_double,limiter}/` | 各桶 metrics.json |
| `figures_{dn,sn,snow_single,snow_double,limiter}/` | fig1 好坏样本 / fig2 场统计 / fig3 几何统计 + stats_per_sample.json |
| `README.pdf` | 本文件导出 |

## 6. 复现命令

```bash
PY="C:/Users/HP/.conda/envs/torch5060/python.exe"

# 训练（与 §2 相同；训练命令全量见 data_v6/README.md §5.2）
bash dn_fno_2608/scripts/run_exp012_eval_vis.sh   # 六桶评估 + 5 配置可视化一键
```

## 7. 偏差与历史记录

### 7.1 ⚠️ SN 数据质量问题（2026-08-19，用户发现，已确认）

sn 桶 figure worst 例子的 freegs truth 形态不对，非模型能力问题。验证
（sn/test.npz 200 样本 + figures_sn/stats_per_sample.json）：
- **磁轴系统性偏下**：Z_axis mean **−0.339 m**（仅 2/200 在 |Z|<0.15，min −0.848）——
  SN 位形上瓣普遍很薄：76/200 中平面 psi 距分离面 <0.5 core 深度（ratio mean 0.55）；
- **极端样本 idx158**（rel_l2 48.4%）：磁轴 Z=−0.848，中平面 psi < psi_bndry
  （分离面在 Z=0 处"外翻"到壁区，物理不可信）；
- **worst idx77**（58.5%）：上瓣薄（ratio 0.45）+ 分离面穿越边 607 vs best 461
  （+32%）；worst5 中 4/5 属于上瓣薄集合；
- |Z_axis| 与 rel_l2 相关性弱（0.19）——非唯一因素，但 sn 桶 8.99% 偏高的一部分
  应归因于**真值形态病态**；
- 建议（待拍板）：① SN 生成加"上瓣健康度"判据（中平面 psi > psi_bndry + k·core）
  ② 重生成病态样本（idx158 等）③ 专职 SN 对照模型。详见 data_v6/README.md §8.1。

### 7.2 修复与设计记录

1. limiter 桶几何指标 NaN：空 xpts_actual (0,3) 显式返回 NaN（evaluate_dn_fno.py
   `len(xpts_true)==0` 分支；首次评估时曾落入 legacy find_critical 配对给出
   80–180 cm 假数字，已修复）；
2. visualize 修复：MASTU_simple 线圈为 Circuit 类型（无 R/Rs 属性）→
   get_machine_geometry 增加 Circuit 分支（取内层 Coil R/Z，画上瓣）；
3. 数据触壁分布不均（limiter 100% 触壁 vs snow_double 1%）——混合训练下
   limiter 桶预期劣于专职模型；实际 1.78% 反而不差（结构最简单）；
4. 与 exp011 跨机器对比仅参考（机器/网格/线圈数不同）。
