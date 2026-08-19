# KAN 实验注册表

> 规则同 `dn_fno_2608/EXPERIMENTS.md`：每次实验一行，先写占位再回填结果；
> README 必须写明输入通道明细（通道数 + 每通道构成）。

## exp001_kan_v5（进行中）

- 日期：2026-08-19 ｜ 数据：data_v5（MAST DN+SN 混合，N=500，seed 1）
- 输入：19ch = R,Z + 5 params + config + 11 线圈电流（含 config 通道）
- 模型：手写 B 样条 KAN，1×10 隐含层，1260 params，网格 2、阶 3、基函数 silu
- 对照：FNO exp011（18ch 无 config，rel L2 0.894%）
- 结果：待 M1 回填（rel L2 / GS 残差比 / X 点 / sep / Jψ 自检，DN/SN/整体六桶）
- 产物：`model_b19ch_kan_mix/` + `eval_*/` + `figures_*/` + README 对比表

## M2/M3（待启）

- exp001 续：`sr/`（符号回归 KAN-SR）、`ext/`（OOD 外推 + DN/SN 分类）
