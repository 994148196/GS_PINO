# data_v5 数据集说明（MAST 混合位形 DN+SN，分开落盘）

> 生成日期：2026-08-17
> 代码：`generate_dn_dataset.py`（`--config {dn,sn}` + `--machine {test,mast}`，
> 机器×位形注册表为 snowflake/limiter 预留；默认组合字节级不变，回归 PASS）
> 方向（用户拍板）：**混合位形先做 DN+SN**，真实装置用 **MAST**，数据集**按配置分开
> 生成**（`dn/`、`sn/` 子目录，后续可只用一种或混合训练）
> 可行性探针报告：[PLAN_v5_mixed_configs.md](../PLAN_v5_mixed_configs.md)
> 用途：exp008（混合训练，config 输入）与 exp009（专职单一位形对照）的数据

## 1. 与 data_v4 的差异

| 项 | data_v4 (TestTokamak DN) | **data_v5 (MAST DN / SN)** |
|---|---|---|
| 机器 | TestTokamak（4 线圈，有墙） | **MAST**（11 控制线圈：10 Coil + 1 Solenoid，**无墙**） |
| X 点中心 | (1.2, ±0.6) | **(0.7, ±1.1)**（MAST 下 divertor 区域，探针校准） |
| X 点抖动 | R ±0.10 / Z ±0.15 m | **R ±0.06 / Z ±0.10 m**（探针校准：MAST 可行区更窄） |
| 锚点 | 中平面 (R, 0.0)，R~U[1.35,1.65] | 中平面 (R, 0.0)，**R~U[1.2,1.6]**（MAST 外中平面） |
| 参数范围 | paxis (200,3000) / Ip (5e4,4e5) / fvac (0.5,3.0) | **paxis (1e3,5e3) / Ip (3e5,8e5) / fvac (0.3,0.8)** |
| 位形 | 双 X 点（DN） | **DN（双 X 点）+ SN（单 X 点）** |
| 接受检查 | 7 项（§3 v4） | **同 7 项 + SN 判据适配**（§3） |
| 新字段 | — | **`config`**（0=DN, 1=SN，标量数组） |
| 落盘 | 单一 out-dir | **`dn/`、`sn/` 分开**（out-dir 区分，字段同构） |

## 2. 探针校准（probe_v5.py，seed 123，80 样本/配置，max_retries=1 = 原始接受率）

| 配置 | 接受率 | isoflux res 最大 | X 点偏差最大 | core 深度 (mean/min) | 收敛迭代 |
|---|---|---|---|---|---|
| MAST+DN | **80/80 (100%)** | 2.0e-4 | 0.016 m | 0.077 / 0.039 | 8.7± |
| MAST+SN | **80/80 (100%)** | 2.0e-5 | 0.0015 m | 0.085 / 0.044 | 40.6±（max 51 < MAXITS） |

- 全部参数命中采样范围（X 点 R∈[0.64,0.76]/Z∈[-1.16,-1.04]，锚点 R∈[1.21,1.60]，
  Ip∈[0.30,0.79] MA、paxis∈[1.0,4.9] kPa、fvac∈[0.30,0.79]）；
- 探针发现并修复 3 个 MAST/SN 适配 bug（§5 偏差记录）后达到 100%；
- SN 收敛迭代显著高于 DN（40.6 vs 8.7）→ 求解 ~0.63s/solve（24 核均摊；
  单核 ~15s），全量 3000 SN 样本（train+val+test）约 31 分钟（24 核）。

## 3. 接受检查（data_v4 7 项 + SN 适配）

data_v4 的 7 项检查全部保留（`--max-isoflux-residual 0.35 --max-xpt-deviation 0.10
--min-anchor-xpt-dist 0.15 --coil-margin 0.05 --min-core-depth 0.005`），SN 分支适配：

1. **SN 分离面 X 点判据**：`find_critical` 会报告真空区假鞍点（TestTokamak ~2 个/
   样本、MAST 5–6 个），SN 判定改为 **psi ≥ psi_bndry − 1e-6 的 X 点数 == 1**
   （分离面恰一个 X 点），替换 v4 的 `len(xpt) >= MIN_XPTS`；psi_bndry = 该 X 点 psi
   （DN 仍为两 X 点 psi 均值，paper 定义）；
2. **SN 磁轴**：SN 无"三角形"——磁轴检查改为 **O 点位于 X 点与中平面锚点之间**
   （R_lo < R_axis < R_anc 且 |Z_axis| < |Z_lo|）。注意 psi_total 含线圈贡献，
   `find_critical` 会报告线圈场真空假极值（TestTokamak 固定点 (1.00,−1.10) 附近、
   MAST 墙外 R>1.8），磁轴必须**按位置过滤后再取 psi 最大**（直接 max(psi) 会选到假轴）；
3. **无墙适配**：MAST `tokamak.wall` 为 None → `--require-wall` 忽略（打印 WARNING），
   墙检查跳过（TestTokamak 路径不变）；
4. **Solenoid 线圈**：`_control_coil_centers` 对 Solenoid（MAST 中心螺线管，属性
   Rs/Zsmin/Zsmax）取中点 (Rs, Z 中点)；凸包 sanity 按线圈数跳过硬编码 COIL_HULL
   （11 线圈机器打印 skipped）。

## 4. 字段说明

data_v4 全部字段（含 anchor + 8 约束诊断） + 新增 `config`（(N,1)，0=DN / 1=SN）。
**x_coords 固定 4 通道**：SN 样本上 X 点对存 **(0.0, 0.0) 占位**（恒值通道 →
z-score 后为 0，由 `config` 通道区分；模型侧 `np.maximum(std, 1e-8)` 防护）。
SN 的诊断字段形状缩小：xpt_constraint_res (2,)、isoflux_res (1,)、psi_at_constraints
[p_lo, p_anc]；o_point = 位置过滤后的磁轴。

## 5. 偏差记录（探针与全量过程中发现并修复的问题）

1. **SN 磁轴取错（0/8 全拒）**：初版 `max(opt, key=psi)` 选到线圈场真空假极值
   （TestTokamak 固定 (1.002,−1.102)、MAST 墙外 (1.87,−0.20)）→ 改为位置过滤后
   再取 psi 最大。修复后 TestTokamak+SN 冒烟 8/8、MAST+SN 探针 80/80。
2. **SN 轴检查负 Z 比较 bug**：`abs(za) < lo[1]`（lo[1] 为负）恒假 → `abs(za) < abs(lo[1])`。
3. **MAST Solenoid 无 `.R`**：`_control_coil_centers` 报 AttributeError → 按
   `hasattr(coil, "Rs")` 分支取 (Rs, Zsmin/Zsmax 中点)。
4. **MAST 无墙崩溃**：`tokamak.wall.R`（NoneType）→ wall_verts 仅在 `tokamak.wall
   is not None` 时构造。
5. **SN 配对统计 bug（不影响数据，探针与 merge 打印各一处）**：SN 的
   x_coords (0,0) 占位被当成配对目标（|lo−(0,0)|≈1.30–1.37 m 假信号）→
   probe_v5.py 与 merge 的 X-pt 偏差打印均改为按 xpts_actual 实际 X 点行数
   分支配对（数据字段 xpts_actual 本身正确，SN 实际偏差 ≤0.002 m）。
6. **SN 求解耗时显著高于 DN**（单核 ~15s vs ~4.4s/solve，n_iter 40 vs 9）：
   全量 3000 SN 样本（train+val+test）~31 分钟（24 核并行），不影响正确性。

## 6. 复现命令

```bash
# 全量（探针 + 6 个 split 一键；后台运行，实际耗时 DN ~6 min + SN ~31 min）
bash dn_fno_2608/scripts/run_generate_v5.sh

# 探针（原始接受率统计，输出 JSON）
python dn_fno_2608/scripts/probe_v5.py --machine mast --config sn --n 80
```

数据规模：dn、sn 各 train 2000 / val 500 / test 500（seed 123/456/789，同 v4 惯例）。
npz 不入 git（本 README 除外，见 .gitignore）。
