# exp012：coil 电流输入跨位形混合训练（MASTU_simple 五配置，21ch）

> 训练日期：2026-08-18
> 数据：[data_v6](../data_v6/README.md)（5 配置 train 500/val 100/test 200，129²）
> 对比：exp011（MAST 18ch DN+SN：0.841%/0.948%）——跨机器仅参考
> 目录：`dn_fno_2608/experiments/exp012_coil_input_v6/`

## 1. 动机与输入

MASTU_simple 五种位形（dn/sn/snow_single/snow_double/limiter）端到端生成 psi。
探针 separability（probe_v6.py，5×80 样本）：14 通道线圈电流 pooled 分离度全部
>2.0 std、两两配对阈值判别 ≥90.75% → 电流本身携带位形信息，**无需 config 通道**
（decision_no_config_channel=true）。

**输入 21ch**（`--input-mode coils --no-config-channel`）：

| 通道 | 内容 |
|---|---|
| 1–2 | R, Z 网格（[-1,1] 归一化） |
| 3–7 | Ip, paxis, fvac, alpha_m, alpha_n（z-score） |
| 8–21 | 14 线圈电流：I_Solenoid, I_Pc, I_Px, I_D1, I_D2, I_D3, I_Dp, I_D5, I_D6, I_D7, I_P4, I_P5, I_P61, I_P62（z-score） |

目标：`psi_total`（129²，均值/标准差归一化）。

## 2. 训练设置

| 项 | exp011 | **exp012** |
|---|---|---|
| 数据 | data_v5（MAST，2 配置） | **data_v6（MASTU_simple，5 配置）** |
| 输入 | 18ch（无 config 字段） | **21ch（--no-config-channel）** |
| 网格 | 65² | **129²** |
| n_train | 500 | **500**（5 文件 pool 共 2500，nested 抽 500） |
| val | 2 文件 pool | 5 文件 pool（500） |
| 模型参数 | 4,212,225 | 4,212,417 |
| 其余 | seed 1 / 800 epochs / batch 16 / AdamW lr 1e-3 wd 1e-4 / ReduceLROnPlateau(20, ×0.5, min 1e-5) / early-stop 75 | 同左 |

## 3. 结果

best val rel L2 **3.5645%** @ epoch 799（800 跑满无早停，69 分钟）。
6 桶评估（rel L2 / RMSE 物理 Wb / GS 残差 pred|true / O 点误差 / 分离面误差）：

| 桶 (n) | rel L2 | RMSE (Wb) | GS pred/true | O 点误差 | 分离面误差 | find_critical 失败 |
|---|---|---|---|---|---|---|
| eval_all (1000) | **3.435%** | 0.00367 | 4.02 / 7.21 | 3.75 cm | 2.79 cm | 4 |
| dn (200) | 3.618% | 0.00273 | 4.18 / 8.94 | 4.71 cm | 3.54 cm | 3 |
| sn (200) | **8.988%** | 0.01117 | 5.96 / 11.87 | 8.21 cm | 5.53 cm | 1 |
| snow_single (200) | **1.487%** | 0.00156 | 4.84 / 6.94 | 1.47 cm | 1.39 cm | 0 |
| snow_double (200) | **1.307%** | 0.00101 | 4.54 / 7.98 | 0.64 cm | 1.07 cm | 0 |
| limiter (200) | 1.776% | 0.00186 | 0.58 / 0.32 | NaN* | NaN* | 0 |

*limiter 无分离面 X 点（xpts_actual 0 行）→ 几何指标 NaN（诚实）；GS 残差
pred 0.58 与真值 0.32 均显著低于偏滤器位形（限制面残差小），比值 1.8 偏高
但绝对值健康。

**初步解读**（跨机器仅参考：exp011 65² MAST DN 0.841%/SN 0.948%）：
1. **snow_double 1.31% / snow_single 1.49% 最优**——雪点二阶约束给出最确定的
   psi 结构（外加探针接受率最高/触壁最少），limiter 1.78% 次之（触壁由定义
   决定，结构最简单）；
2. **dn/sn 反而最差（3.6%/9.0%）**——与 exp011 相反（那时 DN+SN 是唯二位形）。
   可能因素：dn X 点抖动范围宽、sn 收敛迭代多（探针 103s/solve）导致数据内在
   噪声大；129² 上 rel L2 与 65² 不可直接比（网格点数 4 倍）。
3. **混合训练 5 位形未崩溃**——最难的 limiter/snow 桶 1.3-1.8%，说明 14 通道
   电流对位形信息承载充分（separability 结论兑现）；
4. GS 残差 pred/true 全部 <1（除 limiter），模型场比 freegs 真值更 GS-光滑
   （FNO 平滑化，exp011 同现象）。

## 4. 偏差记录

1. limiter 桶几何指标 NaN（无分离面 X 点）；空 xpts_actual (0,3) 显式返回 NaN
   （不再落入 legacy find_critical 配对——修复记录：首次评估时 80-180cm 假数字，
   evaluate_dn_fno.py `len(xpts_true)==0` 分支）。
2. 与 exp011 跨机器对比仅参考（机器/网格/线圈数不同）。
3. 数据触壁分布不均（limiter 100% 触壁 vs snow_double 1%）——混合训练下
   limiter 桶预期劣于专职模型；实际 limiter 1.78% 反而不差（结构最简单）。
4. visualize 修复：MASTU_simple 线圈为 Circuit 类型（无 R/Rs）→
   get_machine_geometry 增加 Circuit 分支（取内层 Coil R/Z）。
5. **SN 数据质量问题（用户发现，2026-08-19）：sn 桶 figure worst 例子的 freegs
   truth 形态不对。** 验证（sn/test.npz 200 样本 + figures_sn/stats_per_sample.json）：
   - **磁轴系统性偏下**：Z_axis mean **-0.339 m**（仅 2/200 在 |Z|<0.15，min -0.848，
     max +0.006）——SN 位形上瓣普遍很薄：76/200 样本中平面 psi 距分离面
     <0.5 core 深度（ratio = (psi(Z0,R_axis)-psi_bndry)/core，mean 0.55）；
   - **极端病态样本 idx158**（rel_l2 48.4%）：磁轴 Z=-0.848，**中平面 psi <
     psi_bndry**（ratio -0.85）——等离子体完全不跨中平面，分离面在 Z=0 处
     "外翻"到 R 1.4–1.87（壁区），真值形态物理上不可信；
   - **worst idx77**（58.5%）：上瓣薄（ratio 0.45）+ 分离面穿越边 607 vs best
     461（+32%，结构复杂/波动大）；worst5 中 4/5 属于上瓣薄集合；
   - |Z_axis| 与 rel_l2 相关性弱（0.19）——不是唯一因素，但 SN 桶 8.99% 偏高的
     一部分应归因于**真值形态病态**（对模型难学且目标本身离群）。
   - **建议**（待用户拍板）：① SN 生成加"上瓣健康度"判据（中平面 psi >
     psi_bndry + k·core，k 标定）或收紧磁轴 Z 允许范围；② 或对 SN 单独出
     mask/重生成受影响的样本（idx158 等）；③ 专职 SN 对照模型评估数据质量
     影响。
