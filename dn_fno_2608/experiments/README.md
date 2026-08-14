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
   **已建立 `dn_fno_2608/data_v2/`**（2026-08-14）：唯一差异 = 剖面形状指数
   alpha_m/alpha_n 从固定 (1.0, 2.0) 改为采样 αm~U[1,2]、αn~U[1.5,2.5]，其余
   与基线完全一致；输入通道 9→11。说明见 [data_v2/README.md](../data_v2/README.md)。
   **已建立 `dn_fno_2608/data_v3/`**（2026-08-14）：X 点抖动 ±0.02→±0.20 m +
   isoflux 锚点固定 (1.5,0.0)→采样 R~U[1.2,1.8]×Z~U[-0.3,0.3]，规模 3000
   （train 2000/val 500/test 500）；新增 anchor 字段；X点版输入 11 通道、
   X点+锚点版 13 通道。
   ⚠️ **已被 data_v4 取代并删除**（2026-08-14；isoflux 约束不可达缺陷，生成脚本
   `scripts/run_generate_v3.sh` 保留）。
   **已建立 `dn_fno_2608/data_v4/`**（2026-08-14）：可行区采样（X点中心
   (1.2,±0.6)、锚点中平面 Z=0 R∈[1.35,1.65]）+ **7 项物理合理性接受约束**
   （三角形内、四边形余量、墙内、isoflux 残差 ≤0.35、X 点偏差 ≤0.10、
   锚点-X点距离、core 深度）+ 8 个约束诊断字段，2972/3000，isoflux 残差
   mean 0.172/max 0.350（v3: 0.50/2.10）。说明见 [data_v4/README.md](../data_v4/README.md)。
2. **基线**：改进实验一律与 `outputs/fno_n5000_s1`（N=5000 seed=1，test rel L2
   0.0561%，RMSE 1.50e-5 Wb）对比，写进 notes.md。
3. **实验自包含**：一个实验目录内必须能回答"配置是什么、结果是多少、
   对比基线如何"。训练脚本（`gs_pino_dn_fno_2608.train_dn_fno`）已支持 `--out-dir` 直接
   指向 `experiments/expXXX_.../`。
4. **命名**：`exp001_<slug>` 递增（如 exp001_gelu→relu、exp002_physics_loss）。
5. **代码**：公共代码在 `src/gs_pino/`（data_dn_fno / model_dn_fno /
   train_dn_fno / evaluate_dn_fno / visualize_dn_fno / latency_dn_fno），
   新模型实现优先加进 `model_dn_fno.py` 或新建 `model_*.py` 并注册到训练 CLI。
   一次性实验脚本放 `dn_fno_2608/scripts/`。

## 示例：跑一个新实验

```bash
PY="C:/Users/HP/.conda/envs/torch5060/python.exe"
"$PY" -u -m gs_pino_dn_fno_2608.train_dn_fno \
  --train-data dn_fno_2608/data/train.npz --val-data dn_fno_2608/data/val.npz \
  --n-train 5000 --seed 1 \
  --out-dir dn_fno_2608/experiments/exp001_xxx \
  2>&1 | tee dn_fno_2608/logs/exp001_xxx.log
"$PY" -u -m gs_pino_dn_fno_2608.evaluate_dn_fno \
  --test-data dn_fno_2608/data/test.npz \
  --checkpoint dn_fno_2608/experiments/exp001_xxx/best.pt \
  --out-dir dn_fno_2608/experiments/exp001_xxx
"$PY" -u -m gs_pino_dn_fno_2608.visualize_dn_fno \
  --checkpoint dn_fno_2608/experiments/exp001_xxx/best.pt \
  --out-dir dn_fno_2608/experiments/exp001_xxx/figures
```

## 实验台账

| 实验 | 内容 | test rel L2 | 结论 |
|---|---|---|---|
| (基线) outputs/fno_n5000_s1 | 论文复现 FNO | 0.0561% | 论文 0.061%（3-seed 均值） |
| exp001_coil_input | coil 电流替代 X 点输入（N=500） | 0.326% | 可行但比基线差 1.5× |
| exp002_profile_alphas | data_v2（alpha 采样）+ 11 通道（N=500） | 0.303% | 比基线 0.222% 差 1.36×，同量级可接受 |
| exp003_coil_input_v2 | data_v2 上 coil 电流替代 X 点输入（N=500） | 0.412% | 比 exp002 差 1.36×，与 exp001 比例（1.47×）一致；GS 0.996、几何不输 |
| (data_v2) | 新数据集：αm/αn 采样，见 data_v2/README.md | — | — |
| (data_v3) | 新数据集：X点 ±0.20 m + 锚点采样，2998/3000（**已删除**，被 data_v4 取代；生成脚本 run_generate_v3.sh 保留） | — | — |
| exp004_xpoints_anchor_v3 | data_v3 上 X点(11ch) vs X点+锚点(13ch)，N=500 | 31.09% / 21.72% | 仅 X 点输入不可靠（一对多 100×退化）；含锚点仍 21.7% → 当时归因数据量瓶颈（**已被 exp006 修订为约束不可达**） |
| exp005_coil_input_v3 | data_v3 上 coil 输入（11ch），N=500 | 12.47% | coil 隐含锚点信息（优于 X点+锚点版 1.7×）；**data_v4 复测该优势不成立**（见 exp007） |
| (data_v4) | 新数据集：可行区采样 + 7 项接受约束 + 8 个诊断字段，2972/3000，见 data_v4/README.md | — | — |
| exp006_xpoints_anchor_v4 | data_v4 上 X点(11ch) vs X点+锚点(13ch)，N=500 | 3.38% / **0.442%** | **约束不可达是 exp004 退化主因**：A' 恢复 49× 至 data_v2 水平；A 一对多真实代价 7.7×；A' ≥ B |
| exp007_coil_input_v4 | data_v4 上 coil 输入（11ch），N=500 | **0.499%** | coil 恢复到 data_v2 水平（25×）；与 A'（0.442%）同档、略逊；exp005 的 1.7× 优势为污染伪差 |
