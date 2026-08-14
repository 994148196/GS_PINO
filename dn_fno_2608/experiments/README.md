# dn_fno_2608 目录组织约定（2026-08-14）

论文复现（第一阶段）已冻结；后续模型改进实验按以下约定组织，**数据集全实验通用**。

## 目录结构

```
dn_fno_2608/
├── data/                      # 共享数据集（冻结，只读，勿改动）
│   ├── train.npz / val.npz / test.npz    # 6000 样本 + PINO 全字段
│   └── (train/ val/ test/ 分块目录可删，合并后即冗余)
├── outputs/                   # 第一阶段论文复现结果（冻结基线）
│   ├── fno_n{500,1000,2000,5000}_s*/    # 8 个 checkpoint + history/args
│   └── report/                # REPORT.md、latency.json、figures/、各指标 JSON
├── experiments/               # ★ 后续改进实验都放这里（本目录）
│   └── exp001_<slug>/         # 每个实验一个自包含目录
│       ├── args.json          # 训练/模型配置（从 args + 手动补充记录）
│       ├── best.pt            # 最优权重
│       ├── history.json       # 训练曲线
│       ├── metrics.json       # 评估指标（test）
│       ├── figures/           # 该实验的可视化
│       └── notes.md           # 结论：改了什么、比基线好/坏多少、为什么
├── logs/                      # 所有实验日志 logs/<exp_name>.log
├── scripts/                   # 共享编排脚本（每实验一个 run_*.sh 或参数化）
└── PLAN.md
```

## 约定

1. **数据只读共享**：所有实验直接用 `dn_fno_2608/data/*.npz`。若要换数据
   （新参数范围、新分辨率），新建 `data_v2/` 并记录差异，不覆盖原数据。
2. **基线**：改进实验一律与 `outputs/fno_n5000_s1`（N=5000 seed=1，test rel L2
   0.0561%，RMSE 1.50e-5 Wb）对比，写进 notes.md。
3. **实验自包含**：一个实验目录内必须能回答"配置是什么、结果是多少、
   对比基线如何"。训练脚本（`gs_pino.train_dn_fno`）已支持 `--out-dir` 直接
   指向 `experiments/expXXX_.../`。
4. **命名**：`exp001_<slug>` 递增（如 exp001_gelu→relu、exp002_physics_loss）。
5. **代码**：公共代码在 `src/gs_pino/`（data_dn_fno / model_dn_fno /
   train_dn_fno / evaluate_dn_fno / visualize_dn_fno / latency_dn_fno），
   新模型实现优先加进 `model_dn_fno.py` 或新建 `model_*.py` 并注册到训练 CLI。
   一次性实验脚本放 `dn_fno_2608/scripts/`。

## 示例：跑一个新实验

```bash
PY="C:/Users/HP/.conda/envs/torch5060/python.exe"
"$PY" -u -m gs_pino.train_dn_fno \
  --train-data dn_fno_2608/data/train.npz --val-data dn_fno_2608/data/val.npz \
  --n-train 5000 --seed 1 \
  --out-dir dn_fno_2608/experiments/exp001_xxx \
  2>&1 | tee dn_fno_2608/logs/exp001_xxx.log
"$PY" -u -m gs_pino.evaluate_dn_fno \
  --test-data dn_fno_2608/data/test.npz \
  --checkpoint dn_fno_2608/experiments/exp001_xxx/best.pt \
  --out-dir dn_fno_2608/experiments/exp001_xxx
"$PY" -u -m gs_pino.visualize_dn_fno \
  --checkpoint dn_fno_2608/experiments/exp001_xxx/best.pt \
  --out-dir dn_fno_2608/experiments/exp001_xxx/figures
```

## 实验台账

| 实验 | 内容 | test rel L2 | 结论 |
|---|---|---|---|
| (基线) outputs/fno_n5000_s1 | 论文复现 FNO | 0.0561% | 论文 0.061%（3-seed 均值） |
