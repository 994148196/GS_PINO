# 一次性调试脚本（历史诊断，已归档）

M1 ckpt（exp001_kan_v5）上的诊断脚本，用途与结论均已在
`../experiments/exp001_kan_v5/README.md` §7 与 `kan/README.md` §6 记录。

| 脚本 | 诊断内容 | 结论 |
|---|---|---|
| `_dbg_sr.py` | 分层（L1/L2）符号回归差异定位 | 首版按 [-1,1] 拟合隐藏激活（实际达 [-12,1]）→ 多项式外推爆炸 |
| `_dbg_sr2.py` | 最差边缘样本定位（KAN-SR 爆炸源） | 边缘/角落样本 vs 内部：SR 误差集中在边界环 |
| `_dbg_sr3.py` | 角点 vs 内部 h0/h1 误差细分 | 边界环 6% 点误差主导 → 指向 torch 样条 eps 舍入 bug |
| `_sum_m1.py` | 六桶指标汇总 | M1 六桶 rel L2 一致性（20.9±0.1%） |

修复见 `src/gs_pino_kan_2608/model_kan.py`（eps=1e-6·span）与
`src/gs_pino_kan_2608/symbolic_kan.py`（观测范围拟合 + 逐层 clip + silu 不 clip）。

as-trained 语义评估器 `_eval_m1_astrained.py`（保留在 `kan/` 根）不是调试脚本，
是 exp001 记录数字的复现工具（README §7 引用）。
