# dn_fno_2608 实验目录组织约定与台账

> 论文复现（第一阶段）已冻结；后续改进实验按以下约定组织。
> 实验按编号递增，**每个实验目录自包含**（README 单独能看懂：目标/输入通道/
> 设置/结果/结论/复现/偏差）。

## 目录结构

```
dn_fno_2608/
├── data/ ... data_v6/          # 数据集（版本演进，见 §2；npz 不入 git）
├── outputs/                    # 第一阶段论文复现结果（冻结基线）
├── experiments/                # ★ 改进实验（本目录）
│   └── expXXX_<slug>/          # 自包含实验目录
│       ├── README.md           # 实验卡：目标/输入通道/设置/结果/结论/复现/偏差
│       ├── args.json / best.pt / history.json / metrics.json
│       ├── figures/ 或 figures_<桶>/   # 可视化
│       └── notes.md            # 过程记录（可选）
├── logs/                       # 所有实验日志 logs/<exp>_*.log
├── scripts/                    # 共享编排脚本（run_*.sh、probe_*.py 等）
└── EXPERIMENTS.md              # 改进实验总览（摘要级）
```

## 约定

1. **数据版本化只增不改**：新参数范围/新分辨率/新机器 → 新建 `data_vX/` 并写
   README（速览/字段/通道/调用/脚本/生成/历史），不覆盖原数据。数据演进线见 §2。
2. **基线**：改进实验与 `outputs/fno_n5000_s1`（N=5000 seed 1，test rel L2
   0.0561%）或同期同数据对标实验对比，写进 README。
3. **实验自包含**：一个实验目录必须能回答"配置是什么、结果是多少、对比如何"。
4. **命名**：`expXXX_<slug>` 递增。
5. **代码**：公共代码在 `src/gs_pino_dn_fno_2608/`；新模型加进 `model_dn_fno.py`
   或新建 `model_*.py` 并注册到训练 CLI；一次性实验脚本放 `scripts/`。
6. **脚本**：每次实验的复现命令写入该实验 README §复现；一键链（训练+评估+
   可视化）放 `scripts/run_expXXX_*.sh`。

## 数据集版本线（各 README 是完整使用文档）

| 数据 | 机器/位形 | 网格 | 规模 | 输入通道 | 用途 |
|---|---|---|---|---|---|
| [data/](../data/README.md) | TestTokamak DN | 65² | 6000 | 9ch（X点） | 论文复现基线（fno_n*_s*） |
| [data_v2/](../data_v2/README.md) | TestTokamak DN（alpha 采样） | 65² | 6000 | 11ch（X点） | exp002/003 |
| ~~data_v3/~~ | TestTokamak DN（X点±0.20+锚点采样） | 65² | 3000 | 11/13ch | **已删除**：isoflux 约束不可达，被 data_v4 取代（脚本 run_generate_v3.sh 保留） |
| [data_v4/](../data_v4/README.md) | TestTokamak DN（可行区+7 项接受约束） | 65² | 2972 | 11/13ch | exp006/007 |
| [data_v5/](../data_v5/README.md) | **MAST** DN+SN | 65² | 各 3000 | 18ch（coil）/13ch（xa） | exp008-011 |
| [data_v6/](../data_v6/README.md) | **MASTU_simple** 五配置 | 129² | 各 500/100/200 | 21ch（coil） | exp012 |
| [data_gspack2_v1/](../data_gspack2_v1/README.md) | **MAST（gspack2_TRAE 复刻）** DN+SN | 65² | 首批各 1500（可 top-up） | 18ch（coil） | exp201/202 |

## 实验台账

| 实验 | 数据/输入 | test rel L2 | 结论 |
|---|---|---|---|
| (基线) outputs/fno_n5000_s1 | data/ 9ch X点 | 0.0561% | 论文 0.061%（3-seed 均值） |
| [exp001_coil_input](exp001_coil_input/README.md) | data/ 9ch coil | 0.326% | coil 替代 X 点可行但差 1.5× |
| [exp002_profile_alphas](exp002_profile_alphas/README.md) | data_v2 11ch X点 | 0.303% | alpha 采样可学（差 1.36×） |
| [exp003_coil_input_v2](exp003_coil_input_v2/README.md) | data_v2 11ch coil | 0.412% | coil 相对损失恒定 ~1.4× |
| [exp004_xpoints_anchor_v3](exp004_xpoints_anchor_v3/README.md) | data_v3 11/13ch | 31.1% / 21.7% | X 点输入锚点大变化不可靠；**已被 exp006 修订为约束不可达** |
| [exp005_coil_input_v3](exp005_coil_input_v3/README.md) | data_v3 11ch coil | 12.47% | coil 隐含锚点信息、最可靠；**data_v4 复测优势不成立** |
| [exp006_xpoints_anchor_v4](exp006_xpoints_anchor_v4/README.md) | data_v4 11/13ch | 3.38% / **0.442%** | **约束不可达是 exp004 退化主因**：A' 49× 恢复；A 一对多真实代价 7.7× |
| [exp007_coil_input_v4](exp007_coil_input_v4/README.md) | data_v4 11ch coil | **0.499%** | coil 25× 恢复到 data_v2 水平；与 A' 同档略逊 |
| [exp008_mixed_configs](exp008_mixed_configs/README.md) | data_v5 14ch xa+config 混合 | DN 0.667% / SN 1.203% / 整体 0.935% | **混合训练可行**：+26~35% vs 专职；config 有效利用 |
| [exp009_split_configs](exp009_split_configs/README.md) | data_v5 13ch xa 专职 | DN 0.492% / SN 0.953% | 专职自身最优；**跨配置外推崩溃**（253% / 4.0e8%）——混合训练是跨位形必要条件 |
| [exp010_mixed_no_config](exp010_mixed_no_config/README.md) | data_v5 13ch xa 无 config | DN 0.611% / SN 1.148% | **模型能自推断位形**：无 config 略优于带 config；config 冗余可删（exp012 21ch 的直接依据） |
| [exp011_coil_input_v5](exp011_coil_input_v5/README.md) | data_v5 18ch coil 混合 | DN 0.841% / SN 0.948% / 整体 0.894% | **端到端 psi 生成**：只给可测量量直接出物理正确的场；位形识别为隐含能力 |
| [exp012_coil_input_v6](exp012_coil_input_v6/README.md) | data_v6 21ch coil 五配置混合 | eval_all 3.435%；sn 8.988%；snow_double 1.307% | 五配置混合成立；雪点/limiter 桶最优；**sn 桶异常 = 真值数据质量问题**（data_v6 README §8.1） |
| [exp013_clean_v6](exp013_clean_v6/README.md) | data_v6_clean 21ch coil（剔除病态+补足） | eval_all 3.045%；sn 7.738%；snow_double 1.074% | **清洗收益分解**：dn −10.7%/snow_double −9.6% 为模型真实收益（同 test 集）；sn −13.9% 主要来自 test 剔除病态（模型原 test +3.1% 未变强，剔除过量警示） |
| [exp201_pino_rhs_mix_gspack2_n500](exp201_pino_rhs_mix_gspack2_n500/README.md) | **data_gspack2_v1**（gspack2_TRAE 求解，MAST 11 线圈复刻）dn+sn 混合 18ch coil（无 config，同 exp103 口径） | rel_l2_total **1.197%**（DN 1.06% / SN 1.33%） | exp103（同口径 freegs data_v5，0.703%） | **数据源替换实验**：管线零改动仅换求解包——g2 与 v5 **分布等价**（输入/场统计/频谱/收敛全一致，交叉评估模型互通），但**可达误差下限更高**（~1.1% vs ~0.7%，+0.50pp 超 ±0.2pp 判据）；交叉评估证明差异是数据固有（exp103 模型在 g2 train/test 均 ~1.11%），非训练失败或子集运气；物理残差机制本身在 g2 上成立（GS 残差 pred/truth 同量级、X 点亚厘米、n_xpt_fail=0） |
| [exp202_pino_twostage_mix_gspack2_n500](exp202_pino_twostage_mix_gspack2_n500/README.md) | 同 exp201 数据（g2 dn+sn 混合 18ch coil，同 exp104 口径） | rel_l2_total **1.316%**（DN 1.17% / SN 1.46%）；Ip 0.935%；J mask 2.82% | exp104（同口径 v5，0.76% / Ip 0.21% / J 1.55%） | **两阶段在 g2 上机制成立但全面退化**：+0.56pp 与 exp201 的 +0.50pp 同量级（g2 固有下限两做法一致 ~1.1%，exp104 模型在 g2 上 1.13% 佐证）；J/Ip 退化是 psi 的 2 倍+（J 是二阶导数场，微结构差异被放大；Ip 下限被 g2 数据内重构精度 0.58% 抬高，0.935% 不代表约束失效）；阶段切换 e61 正常、ramp 无爆炸 |

> 几何指标 v2（2026-08-17）：exp008/009/010 已按"真值基准 X 点配对 + 射线法
> 分离面"重评估（旧版 sep_mean 数十 cm 为 find_critical 假鞍点伪差），几何数字
> 见各实验 README；修复记录见 exp008 README §7.3，旧版备份为 eval*/metrics_v1.json。
