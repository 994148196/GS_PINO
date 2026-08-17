# data_v5 混合位形（DN+SN）：freegs 可行性探针报告 + 实施计划

> 2026-08-17 创建。方向（用户拍板，2026-08-14 讨论）：
> **混合位形先做 DN+SN**，真实装置用 **MAST**，数据集**按配置分开生成**
> （后续可只用一种或混合）。本文件记录 freegs 位形能力探针结果
> （决定哪些位形可生成、怎么判据）与 data_v5 实施计划。
>
> **实施状态（2026-08-17 完成）**：
> - ✅ 生成脚本重构完成（--config/--machine 注册表 + SN 分离面判据 + MAST 适配 +
>   config 字段 + 分开落盘）；默认路径字节级回归 PASS；临时调试代码（_rej/
>   DN_FNO_DEBUG_REJECT）已移除
> - ✅ 探针完成：MAST DN 80/80、MAST SN 80/80（原始接受率 100%，详见
>   [data_v5/README.md](data_v5/README.md) §2/§5 偏差记录）
> - ✅ 全量 data_v5 生成完成（dn/sn 各 train 2000 / val 500 / test 500，
>   500/500 全接受）；README + .gitignore 就绪
> - ✅ 模型侧完成：data_dn_fno use_config + 多文件拼接、train/evaluate
>   --config-input / 逗号分隔多文件；N=64 冒烟 + 多文件评估通过；
>   visualize_dn_fno 补 use_config（exp008 figures 修复）
> - ✅ **exp008 混合（xa+config 14ch）**：best val 0.8789%；test 分桶
>   DN 桶 0.667% / SN 桶 1.203% / 整体 0.935%（GS 0.986，find_critical 0/1000）
> - ✅ **exp009 分开（xa 13ch）**：专职 DN 0.492% / 专职 SN 0.953%；
>   交叉外推崩溃（DN→SN 253%、SN→DN 4.0e8%）——**混合训练是跨位形泛化
>   的必要条件，代价 +26~35%**
> - ✅ 文档全套：exp008/exp009 README（结果/结论/偏差）+ notes + figures +
>   analysis/config_buckets.json + EXPERIMENTS.md/experiments 台账更新
> - v6 预留：snowflake（freegs_snow fork 二阶约束，MASTU_simple）与 limiter
>   （fork 23-diiid-limiter.py 两步法，DIIID 域 R∈[0.7,2.5] 需全链域适配），
>   架构注册表已预留

---

## 1. freegs 位形能力探针（2026-08-17，torch5060 env，egg 9fbec20）

| 位形 | 探针配置 | 结果 | 结论 |
|---|---|---|---|
| DN（双 X 点） | TestTokamak，xpoints=2 对称 + isoflux→锚点（现 data_v4 配置） | 已量产（data_v4，2972/3000） | ✅ 基线 |
| **SN（单 X 点）** | TestTokamak，`xpoints=[(1.2,-0.6)]` 单点 + isoflux→(1.5, 0) | 收敛 8/10，**分离面单 X 点 8/8**，~2s/solve | ✅ 可行 |
| **SN（MAST）** | MAST，`xpoints=[(0.7,-1.1)]` + isoflux→(1.45, 0) | 收敛 5/5，分离面单 X 点 5/5，**4.9s/solve** | ✅ 可行（用户指定机器） |
| snowflake | 上游 egg：4 个一阶 X 点近似 | 收敛但 4 约束点合并成 2 个分离面 X 点（近邻对不分离） | ❌ 上游无二阶约束能力 |
| snowflake | **freegs_snow fork** 二阶约束（MASTU_simple，129²，`maxits=200, rtol=5e-3`，snowflake_weight=0.05, eps=1e-3） | 收敛 2.0s，Br/Bz/psiZZ/psiRZ 均 ~1e-4（二阶零点） | ✅ fork 可行，**推迟** |
| limiter | TestTokamakLimited + `check_limited=True, limit_it=0`（12-limited.py 官方配置复刻 + PaxisIp 随机化） | **0/20 收敛**（RuntimeError: Picard 不收敛） | ❌ 已装 egg 不可用，**推迟**（需专门调试或固定边界近似） |
| 波形/时间演化 | — | freegs 无时间演化模块 | ❌ 需自建约束插值脚本，**推迟** |

## 2. 关键发现（决定生成判据）

1. **SN 判据必须是"分离面 X 点"**：find_critical 在全网格找鞍点，会找到
   **墙外真空区假鞍点**（TestTokamak 每样本 2 个鞍点、MAST 5–6 个），但
   全部收敛样本中**恰好 1 个鞍点满足 psi ≥ psi_bndry**（在分离面上）。
   data_v4 的 `len(find_critical(...)) ≥ MIN_XPTS` 计数逻辑对 SN 无效
   （会把假鞍点计入），必须改为：**psi ≥ psi_bndry−eps 的 X 点数 == 1**
   且距目标 ≤ 阈值（v4 的 `--max-xpt-deviation` 继续用，SN 下 X 点
   滑走仍存在：1/10 样本偏离目标 >0.3 m）。
2. **对称锚点 (R_anc, 0) 是 SN 可行区**：锚点 Z=0 中平面 → TestTokamak
   收敛 8/10；**不对称锚点（Z≠0，含下侧）→ 0/10 全部失败**（isoflux 约束
   过紧，约束不可达）。SN 与 DN 共享同一锚点采样设计（v4 的
   `--anchor-midplane` R∈[1.35,1.65]）。
3. **MAST 无墙**（machine.MAST 无 wall）→ v4 的 `--require-wall` 检查
   在 MAST 分支必须禁用/跳过。
4. **MAST 速度不是瓶颈**：DN 0.68s/solve、SN 4.9s/solve（65² 网格）。
   全量 3000/配置（含 ~1.5× 重试、24 核分块）：DN ≈5 分钟、SN ≈15 分钟。
5. **limiter 位形当前不可行**（egg 官方示例配置也不收敛）；若后续要做，
   备选是固定边界近似（05-fixed-boundary）或专门调试 check_limited 流程。
6. **snowflake 可行但依赖 fork**：freegs_snow（3560aea）二阶约束 +
   MASTU_simple（≠ MAST 线圈布局）在 `maxits=200/rtol=5e-3` 收敛。
   作为第三后端引入（需固定 fork 版本），不在 v5 范围。

## 3. data_v5 实施计划

### 3.1 生成脚本重构（[generate_dn_dataset.py](../src/gs_pino_dn_fno_2608/generate_dn_dataset.py)，向后兼容）

- 新增 `--config {dn,sn}`（默认 dn = 现行为不变）与 `--machine {test,mast}`
  （默认 test）。**MAST 分支**：机器换 `freegs.machine.MAST()`、域/墙检查适配
  （MAST 无墙 → `--require-wall` 忽略）、X 点可行区探针校准（DN: (0.7,∓1.1)
  附近；SN: (0.7,−1.1) 附近）、锚点 R∈[1.2,1.6] 中平面（探针校准）。
- **SN 分支**：`xpoints=[(R_lo, Z_lo)]` 单点约束 + isoflux→锚点；接受检查
  改为分离面 X 点判据（§2.1）。
- 数据集**按配置分开落盘**：`data_v5/dn/…npz` 与 `data_v5/sn/…npz`，
  样本带 `config` 字段（0=DN, 1=SN）。（用户要求：后续可只用一种或混合。）

### 3.2 数据规模（延续 v4 惯例：每配置 3000 = train 2000/val 500/test 500）

- 先探针（~80 样本/配置）出可行区与接受率 → 全量（脚本 `run_generate_v5.sh`）。
- 复刻 data_v4 的 7 项接受检查（三角形/四边形/残差/X 点偏差/锚点距离/
  core 深度），SN 分支适配第 1 条。

### 3.3 模型输入编码（混合训练时）

- 统一输入向量：X 点 4ch（SN 样本 up 通道填 z-score 均值 0）+ 锚点 2ch +
  **config one-hot 1ch**。分开训练（exp009）可不含 config 通道。
- 训练脚本 `--train-data` 支持多文件拼接（`dn.npz,sn.npz`）+ `--config-input` 开关。

### 3.4 实验（N=500 seed 1，延续惯例）

- **exp008**：MAST DN+SN 混合训练（config 输入，11→12ch 通道族）
- **exp009**：MAST DN 单独 vs MAST SN 单独（分开训练基线对照）
- （可选，待用户拍板）TestTokamak SN 对照：隔离"配置效应 vs 机器效应"

### 3.5 涉及文件

| 文件 | 动作 |
|---|---|
| src/gs_pino_dn_fno_2608/generate_dn_dataset.py | --config/--machine + SN 判据 + MAST 适配 |
| src/gs_pino_dn_fno_2608/data_dn_fno.py | config 通道 + 多文件拼接 |
| src/gs_pino_dn_fno_2608/train_dn_fno.py | --config-input / 多数据文件 |
| dn_fno_2608/scripts/run_generate_v5.sh | 新建：探针 + 全量一键 |
| dn_fno_2608/data_v5/ | 新建：dn/、sn/ 子目录 + README（npz 不入 git） |
| dn_fno_2608/experiments/exp008_mixed_configs/ | 新建 |
| dn_fno_2608/experiments/exp009_split_configs/ | 新建 |
| dn_fno_2608/EXPERIMENTS.md + experiments/README.md | 台账更新 |

## 4. 风险与回退

- MAST 可行区窄（SN 锚点必须 Z=0 中平面）：探针定接受率，若 <40% 收窄
  范围或放宽阈值（记录偏差）。
- 混合训练的 config 通道学习：SN 样本的 up 通道恒 0 → 模型须学会忽略；
  若混合后 SN 分桶退化，fallback 是分开训练（exp009 即对照）。
- fork 依赖（snowflake 方向）：fork 固定 3560aea，本地路径，不入 git 依赖。
