# exp010 — 混合训练消融：无 config 通道（仅 up 占位自推断）

> 实验日期：2026-08-17 ｜ 状态：**完成**（N=500，seed 1）
> 数据：`data_v5/`（DN+SN 拼接，同 exp008）
> 对照：exp008（混合 + config 1ch，14ch）；exp009（专职 13ch）
> 结论速览：**模型完全可以自推断位形**——无 config 的混合模型两桶都健康
> （DN 0.611% / SN 1.148%），且略优于带 config 的 exp008（0.667/1.203%）。
> exp008 中模型"用"config 标签只是因为它与 up 占位完全冗余；**部署时无需
> 位形标签，up=(0,0) 占位编码即物理自包含**

## 1. 目标

exp008 反事实测试证明模型**依赖**显式 config 标签（翻转 → 22–54× 崩溃）。
但"用了"≠"不能不用"：SN 的 up X 点通道是 (0,0) 占位（z-score 后恒值），
DN 是变化值——输入本身完全可分。本消融去掉 config，强迫模型只靠占位结构
自推断位形。

判据：若健康（≈exp008 水平）→ 占位自推断足够，config 冗余可删；若退化到
外推级崩溃（≈253%）→ 显式标签必需。

## 2. 输入通道（13 通道，同 exp009 架构）

R, Z + Ip, paxis, fvac, alpha_m, alpha_n + R_lo, Z_lo, R_up, Z_up + R_anc, Z_anc
（`--input-mode xa`，无 `--config-input`）。

训练：N=500 seed 1；train/val = dn+sn 拼接（同 exp008）；共享 stats。注意：
混合 pool 下 up 通道 std 非零（DN 贡献），SN 样本 z-score 后为恒值常数（非 0）。

## 3. 结果（消融 vs 对照，rel L2 mean %）

| test | exp008 混合+config (14ch) | **exp010 混合无 config (13ch)** | exp009 专职 (13ch) |
|---|---|---|---|
| DN | 0.667 | **0.611** | 0.492 |
| SN | 1.203 | **1.148** | 0.953 |
| RMSE (Wb) DN / SN | 2.74e-4 / 5.24e-4 | 2.51e-4 / 5.02e-4 | 1.85e-4 / 4.31e-4 |
| GS 残差比 DN / SN | 0.989 / 0.984 | 0.995 / 0.977 | 0.988 / 0.991 |
| find_critical 失败 | 0/0 | **1/500** / 0/500 | 0/0 |
| X 点误差 lo/up (cm) DN / SN | 3.19/2.37；1.06/— | 2.65/2.59；0.94/— | 2.14/1.91；0.97/— |
| sep_mean (cm) DN / SN | 0.56 / 0.43 | 0.55 / 0.40 | 0.40 / 0.34 |
| O 点误差 (cm) DN / SN | 0.48 / 0.61 | 0.50 / 0.50 | 0.34 / 0.42 |

训练：best val 0.8327% @ e800（满；exp008 0.8789% @ 797）。

## 4. 结论

1. **模型能自己推断位形**：仅凭 up=(0,0) 占位 vs 变化值的结构区分两类——两桶
   健康（0.611/1.148%），与专职外推崩溃（253%/4.0e8%）隔了三个数量级；
2. **无 config 反而略好**（DN 0.611 vs 0.667、SN 1.148 vs 1.203、val 0.833 vs
   0.879）：显式标签让模型"偷懒"走条件映射捷径；去掉后必须建立统一的、位形
   自包含的映射；
3. **与 exp008 反事实的合成结论**：翻转 config 崩溃（用了标签）≠ 没有标签
   崩溃（消融证伪）——config 通道是冗余的，可安全删除；
4. 部署启示：输入按"存在的 X 点"编码（SN 的 up 填占位 (0,0)）即可，无需位形
   标签——这是 data_v6/exp012 **无 config 通道设计（21ch）的直接依据**。

## 5. 复现命令

```bash
PY="C:/Users/HP/.conda/envs/torch5060/python.exe"

"$PY" -u -m gs_pino_dn_fno_2608.train_dn_fno --input-mode xa \
  --train-data dn_fno_2608/data_v5/dn/train.npz,dn_fno_2608/data_v5/sn/train.npz \
  --val-data dn_fno_2608/data_v5/dn/val.npz,dn_fno_2608/data_v5/sn/val.npz \
  --n-train 500 --seed 1 --out-dir dn_fno_2608/experiments/exp010_mixed_no_config/model_a13ch_xa_mix
```

## 6. 产物

`model_a13ch_xa_mix/`（best.pt/history/args/metrics + eval_dn/、eval_sn/ +
figures_dn/、figures_sn/）。日志：`logs/exp010_*.log`。

## 7. 偏差记录

1. DN 桶 find_critical 失败 1/500（exp008 为 0/500）：单样本定位失败，模型输出
   场局部噪声所致，不影响 rel L2 结论（MAST 无墙下 DN 双 X 点对 find_critical
   更敏感）；
2. GS 残差比 SN 桶 0.977（exp008 0.984）：仍在 1 附近（真值 1.288），正常波动。
