# dn_fno_2608 改进实验总览（exp001–exp106）

> 2026-08-21 更新（exp105/106：物理残差推广到 data_v6_clean 五配置混合）。
> 2026-08-20 更新（exp103/104：物理残差推广到混合 DN+SN）。
> 2026-08-17 更新（data_v5 + exp008–011；几何指标 v2 修复，figures
> 重生成）。
> 本文档是**改进实验的总地图**：数据集怎么演进、每个实验改了什么、结果如何、
> 串成一条什么结论线。论文复现（冻结基线）见 [README.md](README.md)；
> 每个实验/数据集的详细文档在各自目录，本文档末尾有索引。

**一句话总结**：论文基线（N=5000，test rel L2 0.056%）冻结后，改进实验沿着
"输入信息架构 → 参数空间扩展 → 数据物理合理性 → **位形泛化 → 端到端**"四条线推进；
中间踩到 data_v3 的**约束不可达**数据缺陷（曾误判为数据量瓶颈），data_v4
修复后三模型全部恢复 9–49×，得到干净数据下的可靠对比：**缺锚点信息的 X 点
输入退化 7.7×、X点+锚点与 coil 输入精度相当（~0.44–0.50%）**；data_v5 把
场景扩展到 MAST 真实装置的 DN+SN 混合位形，**混合训练以 +26~35% 代价换来
跨位形泛化，专职模型跨配置外推崩溃**（DN→SN 253%、SN→DN 4.0e8%）；且
**模型能自推断位形**——去掉 config 标签后仅靠 up=(0,0) 占位结构两桶仍健康
（exp010，甚至略优于带标签版），config 通道冗余、部署无需位形标签；**最终
端到端路线成立**（exp011）：只给可测量量（R,Z + 5 参数 + 11 线圈电流），
模型直接生成物理正确的 psi 场，位形识别成为隐含能力（信息完全由线圈电流
承载），SN 桶甚至反超 xa 输入——真实装置上无需位形判定/磁面重构中间环节；
**exp012 把端到端推到五配置**（data_v6，MASTU_simple 真实壁 + DN/SN/snow_single/
snow_double/limiter，129²，21ch = R,Z + 5 参数 + 14 线圈电流，separability 证实
电流已识别位形无需 config 通道）：snowflake/limiter 桶 1.3-1.8% 健康（雪点二阶
约束位形最优），DN/SN 桶 3.6%/9.0% 明显偏高（129² 与 65² 不可直接比；**SN 还有真值形态
质量问题：磁轴系统性偏下、上瓣薄，极端样本中平面在分离面外**，2026-08-19
用户发现并记录于 exp012 README），limiter 桶几何指标按设计为 NaN（无分离面
X 点）；**exp013 验证清洗收益**（data_v6_clean = data_v6 剔除病态样本
[sn: gs_true>15∪midplane<0 27/200；dn: gs>15 4/200；snow_double: X 点偏差>0.15 4/200]
+ 新 seed 候选池补足，原数据不动）：指标一致改善（sn 8.99→7.74%、整体 3.44→3.05%、
snow_double 1.31→1.07%），但**收益分解**（同 test 集严格对照）显示 dn −10.7%/
snow_double −9.6% 是训练数据清洗的模型真实收益，**sn 桶改善主要来自 test 剔除
病态**（模型在原 test 上 +3.1% 未变强；train 剔除 18.4% 后边缘泛化略降——病态
剔除宜保守，生成侧修复优于事后筛选，供 data_v7）；**最后把 GS 方程物理残差
加进训练**（exp101/102，PINO 阶段，data_v5/dn）：
做法1 单阶段把预测场 Δ\* 拉到数据 RHS 上（test rel L2 0.72%，GS 残差降到真值
FD 下限的 3.3×，精度不降反升）；做法2 两阶段学习 psi↔J 自洽（阶段1 监督
psi_plasma+J，e29 达标切阶段2 加自洽残差 + Ip 约束，物理权重 30-epoch 预热修复
首版阶段2 爆炸）：test 0.80%、Ip 误差 0.21%、mask 内 J 1.36%——**纯数据管线拿
不到的自洽性与 Ip 约束，物理损失提供**；**exp103/104 把两条做法推广到混合
DN+SN（coil 18ch 无 config，与 exp011 同口径）**：做法1 混合 test 0.70%（DN
0.69% / SN 0.72%）——**混合代价为零**（vs exp101 DN-only 0.72% 略优、vs exp011
混合纯 MSE 0.894% 好 21%），PDE 项约束位形不变的物理关系、把"跨位形共享容量"
代价（exp010/011 的 +26~35%）直接吃掉，SN 桶（仅 245 训练样本）增益最大；
做法2 混合 test 0.76%（Ip 误差 0.21%、mask 内 J 1.55%），阶段1 e45 达标切阶段2、
30-epoch 预热无爆炸——两阶段自洽链路在混合数据上成立，且混合略优于 exp102
DN-only（0.76 vs 0.80）；**exp105/106 把两条做法推到 data_v6_clean 五配置
混合（21ch、129²，含触壁 limiter/双雪点/SN 病态，对照 exp013 纯 MSE）**：
做法1（exp105）整体 test 2.362%，**全面优于 exp013（每桶 −3~−27%）**，
X 点定位 3.09 vs 6.11 cm——"PDE 吃掉跨位形共享容量代价"在五配置上更强
成立；做法2（exp106）2.625%（仍优于 exp013 −14%）但**自洽未建立**（GS
残差 550×）：阶段1 最低 3.416% 差 0.4 点未达 3% 阈值 → e300 兜底切换，
阶段2 仅 74 epochs 即早停（patience 从切换起计），物理项在收敛途中被
掐断——**五配置混合下阶段1 严阈值不可达，阶段2 窗口被早停压缩**。

---

## 1. 实验脉络总览

```
论文复现（data/、outputs/）──────────────► 冻结基线（test 0.0561% @ N=5000）
    │
    ├─ 输入架构：X点坐标 ──► coil 电流（exp001 → exp003 → exp005/007）
    │
    ├─ 参数空间：固定剖面 ──► alpha 采样（data_v2，exp002/003）
    │
    ├─ 几何自由度：X点 ±0.02 ──► ±0.20 + 锚点采样（data_v3，exp004/005）
    │       │
    │       ▼ 发现 data_v3 约束不可达（isoflux 残差 mean 0.50 / max 2.10）
    │   data_v4：可行区采样 + 7 项接受检查 + 8 个诊断字段（exp006/007）──► 三模型复测
    │
    └─ 位形泛化：MAST 真实装置 + DN/SN 混合位形（data_v5，exp008/009）
            │  SN 判据 = 分离面 X 点（find_critical 假鞍点适配）
            ▼ 混合训练（config 1ch）vs 专职分开训练
    exp008（混合 14ch：DN桶 0.667% / SN桶 1.203%）──► 跨位形泛化可行，代价 +26~35%
    exp009（专职 13ch：DN 0.492% / SN 0.953%）───► 跨配置外推崩溃（253% / 4.0e8%）
            ▼ 端到端：coil 电流输入（无 X点/锚点/config）
    exp011（coil 18ch 混合：DN 0.841% / SN 0.948%）──► 只给可测量量直接出 psi，
            位形识别是隐含能力（信息完全由 11 线圈电流承载，无需任何显式信号）
            ▼ 五配置扩展：MASTU_simple 真实壁（data_v6，exp012）
    exp012（coil 21ch 混合 5 位形：DN 3.62% / SN 8.99% / snow 1.31-1.49% /
            limiter 1.78% / 整体 3.44%）──► snowflake+limiter 位形端到端可行，
            雪点二阶约束位形最优；跨机器不可直接比（129² vs 65²）
            ▼ 数据清洗：剔除病态 + 补足（data_v6_clean，exp013）
    exp013（同设置重训：SN 7.74% / snow_double 1.07% / 整体 3.05%）
            ──► 收益分解：dn/snow_double 模型真实变强（同 test 集 −10%），
            sn 改善主要来自 test 剔除病态（模型未变强——剔除宜保守）
```

| 阶段 | 内容 | 关键结论 |
|---|---|---|
| 复现 | arXiv:2608.05555 全套（数据/模型/训练/评估） | 各档全面优于论文（0.056% vs 0.061%） |
| 输入架构 | coil 电流替代 X 点坐标（data/、data_v2） | 可行但略差（~1.4×），映射更"因果" |
| 参数空间 | alpha_m/alpha_n 采样（data_v2，11 通道） | N=500 下与基线同量级（0.30%） |
| 几何扩展 | X点 ±0.20 + 锚点采样（data_v3） | 三模型全退（12–31%）→ **缺陷定位** |
| 数据质量 | data_v4 可行区采样 + 接受约束 + 诊断字段 | 三模型恢复 9–49×，约束可达验证 |
| 位形泛化 | data_v5（MAST DN+SN）+ 混合训练（exp008/009/010） | **混合训练以 +26~35% 代价换来跨位形泛化；专职跨配置外推崩溃；模型可自推断位形（config 冗余）** |
| 五配置端到端 | data_v6（MASTU_simple 5 位形 129²）+ exp012（coil 21ch 混合） | **snowflake/limiter 位形端到端可行（1.3–1.8%）；separability 证实 14 线圈电流识别位形、无需 config 通道；DN/SN 桶 3.6%/9.0%（跨机器仅参考）；limiter 几何 NaN（无分离面）** |

## 2. 数据集演进

| 版本 | 规模 | 改动 | 问题 → 解决 |
|---|---|---|---|
| data/（基线） | 6000（5000/500/500） | 论文 Eq.4 设定：X点 ±0.02 m、锚点固定 (1.5,0)、固定剖面 | 冻结只读 |
| data_v2 | 6000 | alpha_m~U[1,2] × alpha_n~U[1.5,2.5] 采样；参数空间 7→9 维，通道 9→11 | 剖面固定限制表达力 → 扩展 |
| data_v3 | 3000（2000/500/500） | X点 ±0.20 m；锚点 R~U[1.2,1.8]×Z~U[-0.3,0.3] 采样 | **isoflux 约束不可达**（4 线圈对 6 约束过定，残差 mean 0.50/max 2.10）；越界采样。**已删除**（生成脚本 run_generate_v3.sh 保留） |
| data_v4 | 3000（2000/500/500） | 可行区采样（X点中心 (1.2,±0.6)、锚点中平面 Z=0 R∈[1.35,1.65]）+ **7 项物理合理性接受检查**（三角形内/四边形余量/墙内/isoflux 残差 ≤0.35/X点偏差 ≤0.10/锚点距离/core 深度）+ **8 个约束诊断字段** | 残差降至 mean 0.172 / max 0.350，2972/3000 接受；诊断字段随样本落盘（超定系统的不可达信息无法从 4 线圈电流恢复） |
| data_v5 | 6000（各 3000 = 2000/500/500，**dn/、sn/ 分开落盘**） | **MAST 真实装置**（11 线圈、无墙）+ **混合位形**：DN（双 X 点 (0.7,±1.1)）与 SN（单 X 点，判据改为**分离面 X 点** psi≥psi_bndry 计数==1，磁轴按位置过滤）分开生成；参数范围 MAST 校准（paxis 1–5 kPa / Ip 0.3–0.8 MA / fvac 0.3–0.8）；新增 `config` 字段（0=DN, 1=SN） | **6000/6000 全接受**（500/500 各 split，SN 求解慢 7×：n_iter 40 vs 9）；SN 的 x_coords up=(0,0) 占位（恒值通道 + config 区分） |
| data_v6 | 4000（各配置 800 = 500/100/200，5 配置分开落盘） | **MASTU_simple 真实装置**（14 线圈、**真实真空室壁**）+ **5 位形**：dn（X 点中心 R=0.80 物理修正）/ sn / snow_single / snow_double（雪点二阶约束 + 初始电流种子选分支）/ limiter（两步法触壁）；**129²** 网格；参数范围例 18/19 强等离子体（paxis 40–80 kPa / Ip 0.7–1.5 MA / fvac 0.4–0.9）；新增**标注字段**（wall_contact/excess、inwall_sep_frac）+ limiter 专有字段（is_limited/Rlim/Zlim/psi_limit，无 anchor/xpts 0 行） | 接受率 55–100%（snow/limiter 全达标，dn/sn 物理失败为主）；**浅触壁保留 + 标注**（深触壁 >20% core 排除）；壁内结构判据 2% 阈值排除 Solenoid 柱区病态；separability：14 通道电流识别位形 → exp012 21ch 无 config 通道；同 seed 重跑可扩展（chunk resume + merge） |

详见 [data/README.md](data/README.md)、[data_v2/README.md](data_v2/README.md)、
[data_v4/README.md](data_v4/README.md)（data_v4 README §2 有阈值用 data_v2 校准的记录）、
[data_v5/README.md](data_v5/README.md)。

## 3. 实验一览（全部 N=500 seed 1，test 全量评估）

| 实验 | 数据 | 模型输入 | test rel L2 mean | 相对基线 N=500 (0.222%) | 一句话结论 |
|---|---|---|---|---|---|
| exp001 | data/ | coil 9ch | 0.326% | 1.47× | coil 替代 X 点可行，略差 |
| exp002 | data_v2 | X点 11ch | 0.303% | 1.36× | alpha 采样扩展参数空间，同量级可接受 |
| exp003 | data_v2 | coil 11ch | 0.412% | 1.86× | coil vs X点 比例与 exp001 一致（1.36×） |
| exp004 | data_v3 | A: X点 11ch / A': X点+锚点 13ch | 31.09% / 21.72% | 140× / 98× | 几何扩展后全退；当时归因"数据量瓶颈" |
| exp005 | data_v3 | B: coil 11ch | 12.47% | 56× | coil 隐含锚点信息、相对最好（**污染数据上的假优势**） |
| exp006 | data_v4 | A: X点 11ch / A': X点+锚点 13ch | 3.38% / **0.442%** | 15× / 2.0× | **约束不可达是主因**；A' 恢复 49× 至 v2 水平；A 真实代价 7.7× |
| exp007 | data_v4 | B: coil 11ch | 0.499% | 2.2× | coil 恢复到 v2 水平；A' ≥ B（exp005 优势为伪差） |
| exp008 | data_v5 | 混合 DN+SN：X点+锚点+config 14ch | DN桶 0.667% / SN桶 1.203% / 整体 0.935% | —（跨机器，不与 N=500 基线直接比） | **混合训练可行**：两桶都健康（GS 0.986、find_critical 0/1000）；vs 专职代价 +26~35%；**反事实**：翻转 config → 54×/23× 崩溃（模型用了标签） |
| exp009 | data_v5 | 专职 DN / 专职 SN：X点+锚点 13ch | DN 0.492% / SN 0.953% | — | 专职自身最优；**跨配置外推崩溃**（DN→SN 253%、SN→DN 4.0e8%）→ 单一位形训练无跨位形泛化 |
| exp010 | data_v5 | 混合 DN+SN **无 config**：X点+锚点 13ch | DN桶 0.611% / SN桶 1.148% | — | **模型能自推断位形**：仅靠 up=(0,0) 占位结构区分，两桶健康且略优于 exp008（+config）——config 通道冗余可删，输入物理自包含 |
| exp011 | data_v5 | 混合 DN+SN **coil 电流**：R,Z+5 params+11 线圈电流 18ch（无 X点/锚点/config） | DN桶 0.841% / SN桶 **0.948%** / 整体 0.894% | — | **端到端 psi 生成**：只给可测量量（电流+工程参数）直接出物理正确的场，无需任何显式位形/拓扑信息；位形识别是隐含能力（信息完全由 11 线圈电流承载）；SN 桶反超 xa（0.948 < 1.148），整体略优于 exp008 |
| exp012 | data_v6 | 混合 5 位形 **coil 电流**：R,Z+5 params+14 线圈电流 21ch（`--no-config-channel`，separability 证实电流识别位形） | 整体 3.435% / dn 3.618% / sn 8.988% / snow_single 1.487% / snow_double 1.307% / limiter 1.776% | —（跨机器跨网格，仅参考） | **端到端扩展到五配置**：snowflake（雪点二阶约束）与 limiter 位形 1.3–1.8% 健康；dn/sn 桶偏高（SN 收敛困难、129² 噪声大）；limiter 几何指标按设计 NaN（无分离面 X 点） |
| exp101 | data_v5/dn | **FNO + GS 物理残差（做法1 单阶段）**：R,Z+5 params+11 线圈电流 18ch，`L = MSE(psi_plasma) + w_pde·‖Δ\*ψ_pred + μ0RJ_data‖²`；coil 分离（网络只预测 psi_plasma，psi_total 由 greens 解析加回） | rel_l2_total **0.72%**（plasma 0.81%）｜ GS 残差 core pred **1.33%** vs truth 0.40% | exp011 DN 桶 0.84（同口径 psi_total） | **纯 MSE 加 PDE 项精度不降反升**（0.72 < 0.84），预测场 GS 残差降到真值有限差分下限的 3.3×；残差 RHS 是冻结数据（J 不随预测自洽），仅输出侧平滑正则 |
| exp102 | data_v5/dn | **FNO + GS 物理残差（做法2 两阶段）**：同 18ch；阶段1 监督 psi_plasma+J 两通道，阶段2 加**自洽**残差（Δ\*ψ_pred + μ0RJ_pred）+ Ip 约束；阶段1 val rel L2<3% 自动切换（e29），物理权重 30-epoch 线性 ramp（修复首版阶段2爆炸） | rel_l2_total **0.80%**（plasma 0.90%）｜ GS 残差 core 1.51% vs 0.40% ｜ **Ip 误差 0.21%** ｜ **J mask 内 1.36%** | exp101（做法1） | **psi↔J 自洽成立**：J ≈ −Δ\*ψ/μ0R（J 1.36% + GS 1.51% 同时成立），Ip 积分约束有效；vs exp101 多 0.08 精度代价换 J 通道 + 自洽 + Ip（做法1 给不了）；全网格 J 36.9% 是 mask 外谱振铃假指标 |
| exp103 | data_v5 dn+sn 混合 | **做法1 推广到混合**（exp011 同口径：逗号拼接、4000 池抽 500 = 255 DN + 245 SN、stats 全池 pde_scale 0.439、无 config 通道）：`L = MSE(psi_plasma) + w_pde·‖Δ\*ψ_pred + μ0RJ_data‖²` | rel_l2_total **0.70%**（DN 0.69% / SN 0.72%）｜ GS 残差 core 0.0156 vs truth 0.0044（3.5× FD 下限） | exp101（DN-only 0.72%）；exp011（混合纯 MSE 0.894%） | **混合代价为零**：PDE 项约束位形不变的 Δ\*ψ 物理关系，把 exp010/011 的跨位形共享容量代价（+26~35%）直接吃掉，整体甚至略优于 exp101 DN-only；SN 桶（245 训练样本）增益最大——物理正则等效于数据增广（SN 0.72% < exp011 的 DN 桶 0.84%）；位形自推断在物理残差管线下再验证（11 线圈电流承载位形信息） |
| exp104 | data_v5 dn+sn 混合 | **做法2 推广到混合**：同 exp103 数据口径；阶段1 监督 psi_plasma+J，e45（val 2.81%<3%）切阶段2 加自洽残差 + Ip 约束，物理权重 30-epoch ramp | rel_l2_total **0.76%**（DN 0.73% / SN 0.79%）｜ GS 残差 core 0.0167 vs 0.0044 ｜ **Ip 0.21%**（SN 0.26%）｜ **J mask 内 1.55%**（SN 1.73%） | exp103（做法1 0.70%）；exp102（DN-only 0.80%） | **两阶段自洽链路在混合数据上成立**（ramp 无爆炸，SN 桶自洽同机制：单 X 点位形下 Ip/J 约束有效）；vs exp103 +0.06 与 exp101/102 同向（+0.08）——做法2 多 J 通道 + 自洽 + Ip；混合略优于 exp102 DN-only（0.76 vs 0.80，同 exp103 的"PDE 吃掉共享容量代价"） |
| exp105 | data_v6_clean 五配置 | **做法1 推广到五配置混合**（exp013 同口径：21ch 无 config、2500 池抽 500、129²）：`L = MSE(psi_plasma) + w_pde·‖Δ\*ψ_pred + μ0RJ_data‖²` | rel_l2_total **2.362%**（dn 2.43 / sn 5.88 / snow_single 1.39 / snow_double 0.79 / limiter 1.33）｜ GS 残差 core 0.060 vs truth 0.0029（20.8× FD 下限）｜ X 点 3.09 cm | exp013（纯 MSE 3.045%） | **物理残差在五配置混合上全面增益**：每桶 −3~−27%（整体 −22%，limiter −26% 触壁位形不破坏管线）；X 点定位全面更好（3.09 vs 6.11 cm）；GS 残差 20.8×（v5 是 3.5×——129² 差分噪声 + limiter/SN 病态使拉紧程度下降）；snowflake 桶 O 点/分离面指标受检测算法局限污染（雪点高阶零点混淆临界点分类，对应样本 rel L2 仅 0.5–1.4%） |
| exp106 | data_v6_clean 五配置 | **做法2 推广到五配置混合**：阶段1 监督 psi_plasma+J；**阶段1 最低 3.416% 差 0.4 点未达 3% 阈值 → e300 兜底切换**；阶段2 仅 74 epochs 即早停（patience 75 从切换起计） | rel_l2_total **2.625%**（dn 2.98 / sn 5.84 / snow_single 1.48 / snow_double 1.04 / limiter 1.78）｜ GS 残差 core **1.60** vs 0.0029（550×——自洽未建立）｜ Ip 1.04% ｜ J mask 7.44% | exp105（做法1 2.362%）；exp013（纯 MSE 3.045%） | **自洽链路未建立**：阶段1 严阈值在五配置下不可达（limiter 触壁 + SN 病态拖慢 psi_plasma 拟合）→ 兜底切换 + 阶段2 窗口被早停压缩（l_pde 2.93→0.013 仍在降被掐断）；精度仍优于纯 MSE −14%（阶段1 J 监督不损害精度）；改进方向：阈值放宽/切换后重置 patience/更长阶段2 |

v3→v4 恢复倍数（同模型同 N）：A 9.2× ｜ A' **49×** ｜ B 25×。
exp008/009 是**位形泛化**实验：data_v5 换了机器（TestTokamak→MAST）与位形
（DN/SN），rel L2 与 TestTokamak 基线（0.22–0.44%）同量级但不可直接比
（机器/位形/参数范围都变了）；内部对比看 exp008 vs exp009（同数据同 N）。

## 4. 三模型关键对比（data_v4，test 494）

| 指标 | A（X点 11ch） | A'（X点+锚点 13ch） | B（coil 11ch） |
|---|---|---|---|
| rel L2 mean % | 3.38 | **0.442** | 0.499 |
| rel L2 p95 % | 8.41 | **0.844** | 1.13 |
| RMSE phys (Wb) | 1.28e-3 | **1.59e-4** | 1.76e-4 |
| find_critical 失败 | 7/494 | **0/494** | 0/494 |
| 分离面平均误差 (cm) | 1.71 | **0.23** | 0.30 |
| GS 残差比值 | 0.991 | 0.994 | 0.997 |

训练 val：A 3.164% @e231 ｜ A' 0.4548% @e793 ｜ B 0.5101% @e792。

## 5. 核心结论

1. **数据物理合理性 > 输入信息架构**：同架构同 N=500，仅把数据从 v3 换成
   v4，三模型恢复 9–49×。生成侧"接受准则"不是形式主义——超定控制系统的
   约束残差必须显式检查。
2. **exp004 的"数据量瓶颈"结论被修订**：A'（信息完整）在 v3 上 21.7% 的主
   成分是**真值污染**（多数样本的分离面并不通过输入的锚点），而非 N=500
   学不会；v4 上恢复到 0.442%（与 data_v2 的 0.303% 同量级）。
3. **A（缺锚点信息）的一对多真实代价 = 7.7×**：v3 上 31.1 vs 21.7 只有 1.4×
   是被污染掩盖的假对比；v4 干净数据下量化。若锚点由控制系统固定（data/、
   data_v2 设定），该问题不存在。
4. **A' ≥ B（0.442 vs 0.499）**：coil 电流确实隐含锚点信息（Tikhonov 解是
   锚点的确定性函数），但直接输入锚点坐标信息更无损；exp005 的 1.7× 优势
   是污染数据制造的伪差。
5. **N=500 对"信息完整 + 目标物理一致"的数据足够**：v4 的残余差距
   （0.44% vs v2 0.30%）来自更大的 X 点/锚点变化范围，属正常扩展代价而非
   数据量不足。
6. **跨位形泛化需要混合训练（data_v5/exp008/009/010）**：混合模型两桶都
   健康（exp008 14ch：0.667/1.203%，整体 0.935%）；代价 vs 专职 +26~35%
   （exp009：DN 0.492 / SN 0.953）。**专职模型跨配置外推完全失败**
   （DN→SN 253%、SN→DN 4.0e8%）：FNO 是分布内插值器，训练分布覆盖是
   必要条件（与结论 1"数据物理合理性 > 输入架构"同构：**训练分布覆盖
   > 架构**）。
7. **模型能自推断位形，config 通道冗余（exp008 反事实 + exp010 消融）**：
   翻转 config 标签 → 54×/23× 崩溃（模型确实在用标签），但去掉 config
   通道（exp010 13ch）后两桶 0.611%/1.148%、**略优于带标签版**——模型
   仅靠 SN 的 up=(0,0) 占位恒值结构即可区分位形。显式标签诱导"条件映射"
   捷径；自推断解更稳。**部署无需位形标签**：输入按"存在的 X 点"编码
   （up 占位）即物理自包含。
8. **SN 桶误差系统性高于 DN 桶**（专职 0.953 vs 0.492，混合同趋势）：SN
   约束自由度少（3 约束对 4 线圈）+ 求解 n_iter 40 vs 9，非 isoflux 残差
   问题（SN 残差为 0）。几何指标 v2 修复后（2026-08-17，真值基准配对 +
   射线法分离面，见 exp008 README §5.3）：X 点 1–3 cm、sep_mean 0.34–0.56 cm、
   sep 面积 <0.7%、O 点 <0.7 cm，两桶同健康；旧版"sep_mean DN 30 cm vs
   SN 2 cm 相反趋势"是 MAST 无墙下 find_critical 假鞍点对 DN 双 X 点贪心
   配对的诊断伪差，不影响 rel L2 结论。
9. **端到端可用（exp011，coil 电流输入）**：**只给可测量量（R,Z + 5 物理
   参数 + 11 线圈电流），不喂 X 点/锚点坐标、无 config 标签，模型直接
   生成物理正确的 psi 场**（DN 0.841% / SN 0.948%，GS 残差比 ≈1.01，
   几何 cm 级）。位形识别是**隐含能力**——5 个物理参数在 DN/SN 间统计
   不可区分（单通道阈值判别 96% 证实位形信息完全由 11 线圈电流承载），
   模型从电流模式中自动区分位形。SN 桶反超 xa 输入（0.948 vs 1.148：
   xa 的 up=(0,0) 占位是死通道，coil 输入 16 个标量全活跃）；整体略优于
   exp008。部署含义：真实装置可测输入直接可用，无需位形判定/磁面重构
   中间环节。

## 6. 遗留问题与后续方向

- **A 的一对多**：实际控制场景锚点通常固定，data/、data_v2 的设定仍贴近
  实际；如需 X 点输入覆盖锚点变化，需把锚点信息显式并入输入或改预测目标。
- **N 扩容收益**：v4 上 A 3.38% 仍有明显空间；A'/B 已接近 v2 水平，扩容
  收益小（exp006 notes §6）。
- **诊断字段下游**：data_v4 已落盘 8 个约束诊断字段（实际 X 点、O 点、
  约束残差、n_iter 等），可用于按真实残差分桶分析，或训练"残差修正模型"
  （把超定系统不可达信息作为输入/目标）。
- **位形泛化扩展**：data_v5 架构（machine×config 注册表 + 多文件拼接）
  为 snowflake/limiter 预留（v6：freegs_snow fork 二阶约束 /
  DIIID 两步法，需全链域适配）；第三位形加入混合池是下一自然步。
- **混合代价归因**：+26~35% 是共享容量代价（exp010 已排除 config 通道
  因素）；可用 N=2000 或分位形独立归一化继续消融。
- **PINO 阶段**：PLAN.md 为物理约束训练预留了全部字段（greens/dpdpsi/
  FdFdpsi），**已启用**（exp101–106，`src/gs_pino_fno_phys/`：exp101/102
  data_v5/dn 做法1 RHS + 做法2 两阶段自洽 + Ip 约束；exp103/104 推广到
  混合 DN+SN（混合代价为零）；exp105/106 推广到 data_v6_clean 五配置
  （做法1 全面优于纯 MSE exp013 −22%；做法2 因阶段1 严阈值不可达 +
  阶段2 窗口被早停压缩而自洽未建立）。后续方向：exp106 修复（阈值放宽
  4–5% / 切换后重置 patience / 阶段2 加长）、全量 N=2000、
  `--pde-mask-erode` 边界差分消融、w_pde 分桶加权（sn/limiter 桶拉紧）。
- **延迟/部署**：复现阶段已证 GPU 前向 1.6 ms（665× vs freegs）；改进模型
  结构未变，延迟结论直接沿用。

## 7. 文档索引

| 文档 | 内容 |
|---|---|
| [README.md](README.md) | 论文复现详解（数据/模型/训练/Table I–III/接口） |
| [PLAN.md](PLAN.md) | 复现计划（阶段 0–5） |
| 本文档 | 改进实验总览 |
| [data_v2/README.md](data_v2/README.md) | alpha 采样数据集说明 |
| [data_v4/README.md](data_v4/README.md) | data_v4：差异、阈值校准、7 项检查、8 字段、统计、复测结果 |
| [data_v5/README.md](data_v5/README.md) | data_v5：MAST DN/SN 差异、SN 判据与磁轴适配、探针/全量统计、偏差记录 |
| [PLAN_v5_mixed_configs.md](PLAN_v5_mixed_configs.md) | data_v5 可行性探针报告 + 实施计划（含 v6 snow/limiter 路线） |
| [experiments/README.md](experiments/README.md) | 实验目录组织约定 + 一页台账 |
| exp001–exp012/ | 每实验 README（设计/结果/结论）+ notes（解读/偏差记录）+ metrics.json |
| exp101/102_pino_*_n500/ | FNO + 物理残差（PINO 阶段，DN-only）：做法1 rhs（exp101）/ 做法2 两阶段（exp102）各自 README（18ch 通道表 + 结果表）+ metrics.json + figures/（exp011 风格 fig1/2/3 + stats_per_sample.json）+ train/eval.log |
| exp103/104_pino_*_mix_n500/ | 同上两做法在**混合 DN+SN**（coil 18ch 无 config）上的推广：README（结论速览 + 混合池统计 + 三桶结果表：all/dn/sn）+ eval_all\|dn\|sn/ + figures_all\|dn\|sn/（exp011 风格 fig1/2/3 + stats_per_sample.json，twostage fig1 含 J 行/fig2 含 Ip·J 直方图）+ train/eval_*.log（日志落各自实验目录） |
| exp105/106_pino_*_v6clean_n500/ | 同上两做法在 **data_v6_clean 五配置混合**（21ch、129²、MASTU_simple）上的推广：README（结论速览 + 混合池统计 + 六桶结果表：all + dn/sn/snow_single/snow_double/limiter）+ eval_all\|5 配置/ + figures_all\|5 配置/（exp011 风格，fig3 含 limiter 全 NaN 防护）+ train/eval_*.log（日志落各自实验目录） |

## 8. 可视化产物

每个实验目录下有 `figures/`（与 `outputs/report/figures/` 同格式，
`visualize_dn_fno.py --title` 标注实验名）。exp008/009/010 的 figures 于
2026-08-17 重生成（几何指标 v2）：fig1 现绘制装置结构（MAST 墙/线圈）+
**仅分离面 X 点**+磁轴+separatrix（R/Z 等比例，`--machine` 参数），fig3
散点上限取 P95 保险（旧版曾把真空假鞍点/等高线穿出网格误报为数十 cm
误差）。exp001–007 的 figures 未重跑（基线，见用户决策）：

| 实验 | figures 位置 | 内容 |
|---|---|---|
| exp001 | exp001_coil_input/figures/ | fig1_best_worst_psi（best/worst 场对比）、fig2_field_stats（rel L2/RMSE/GS 残差分布）、fig3_geometry_stats（X 点/O 点/分离面误差）+ stats_per_sample.json |
| exp002 | exp002_profile_alphas/figures/ | 同上 |
| exp003 | exp003_coil_input_v2/figures/ | 同上 |
| exp004 | — | data_v3 已删（可视化需真值场），按用户决定跳过 |
| exp005 | exp005_coil_input_v3/figures/worst_best/ | 旧版 best/worst 单样本图（10 张 + summary），无 fig1/2/3 |
| exp006 | model_a11ch_xpt/figures/ + model_a13ch_xa/figures/ | 同上（A / A' 各一套） |
| exp007 | exp007_coil_input_v4/figures/ | 同上 |
| exp008 | model_a14ch_xa_mix/figures_dn/ + figures_sn/ | 混合模型在 DN test / SN test 各一套（fig1/2/3 + stats_per_sample.json） |
| exp009 | model_dn_13ch/figures/ + model_sn_13ch/figures/ | 专职 DN / SN 各一套 |
| exp010 | model_a13ch_xa_mix/figures_dn/ + figures_sn/ | 消融（无 config）混合模型在 DN / SN test 各一套 |
| exp011 | model_b18ch_coils_mix/figures_dn/ + figures_sn/ | coil 18ch 端到端混合模型在 DN / SN test 各一套（fig1 含 MAST 装置/11 线圈） |
| exp101 | exp101_pino_rhs_n500/figures/ | exp011 风格 fig1/2/3 + stats_per_sample.json（fig1 = best/worst psi_total 真值/预测/|diff|，装置线圈 + 等高线 + 分离面 + X 点 + 磁轴；fig3 = 几何误差） |
| exp102 | exp102_pino_twostage_n500/figures/ | 同 exp101，fig1 追加 J 行、fig2 追加 Ip/J 直方图 |
| exp103 | exp103_pino_rhs_mix_n500/figures_all\|dn\|sn/ | 同 exp101 风格，三桶各一套（all=1000 拼接 / dn=500 / sn=500）；SN 桶 fig3 x_up 按设计 NaN（单 X 点） |
| exp104 | exp104_pino_twostage_mix_n500/figures_all\|dn\|sn/ | 同 exp102 风格（fig1 含 J 行、fig2 含 Ip/J），三桶各一套；预测场假临界点数（4–8，中位 5）少于 exp011（3–13，中位 6）——物理正则使预测场更干净 |
| exp105 | exp105_pino_rhs_v6clean_n500/figures_all\|dn\|sn\|snow_*\|limiter/ | exp101 风格六桶各一套（MASTU_simple 真实壁 + 14 线圈）；limiter 桶 fig3 全 NaN 防护（"no finite data"）；snowflake 桶 O 点/分离面受检测算法局限污染（见 README §6） |
| exp106 | exp106_pino_twostage_v6clean_n500/figures_all\|dn\|sn\|snow_*\|limiter/ | 同 exp102 风格（fig1 含 J 行、fig2 含 Ip/J），六桶各一套；阶段2 自洽未建立（GS 残差 550×）在 fig2 GS 直方图可见 |

示例（v4 三模型，test 494）：A' best #337 0.136% / worst #124 3.17%；
B best #39 0.147% / worst #411 3.44%；A best #22 0.444% / worst #126 13.45%。
最差样本集中在锚点贴 X 点/残差近阈值区（见 exp006/007 README）。

**产物现状**：data_v3 已删除（被 data_v4 取代）；exp004/exp005 保留为
v3 污染数据的诊断性对比基线；data_v4/data_v5 的 npz 不入 git（生成脚本
可复现，见 .gitignore）；exp008 的 analysis/config_buckets.json 为分桶明细。
