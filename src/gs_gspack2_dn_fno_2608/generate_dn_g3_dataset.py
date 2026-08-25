"""data_gspack2_v2 数据集生成器——gspack2_TRAE 版 MASTU_simple 五配置高质量数据。

exp105（data_v6_clean，freegs 求解，test 2.362%）的**数据源替换 + 数据质量改进**
版本：用 gspack2_TRAE（gspack v2.0.0）生成 MASTU_simple 五配置（dn/sn/
snow_single/snow_double/limiter）数据集，重点**从生成侧修复 data_v6 的 SN
数据质量差**（exp012 §7.1：磁轴偏下、上瓣薄、中平面外翻，sn 病态率 18.4%），
再零管线改动重跑 exp105（做法1 rhs）得 exp203。

与 data_v6 同构（21ch 输入 / 129² 网格 / 14 单元控制），差异：
  - 机器：gspack 内置 MASTU_simple() 的 26 物理线圈（= freegs_snow MASTU_simple
    1:1 复刻，Solenoid control=False 保持不变），经 UnitMachine 子类聚合为
    **14 控制单元**（Solenoid, Pc, Px, D1, D2, D3, Dp, D5, D6, D7, P4, P5,
    P61, P62——data_v6 通道序；P61/P62 下线圈 sign=-1 同 freegs Circuit
    multiplier）。control.py 的 _init_precomp/__call__ 自动走 override 的
    controlBr/Bz/Psi/PsiZZ/PsiRZ/controlAdjust（nc=14），零改动。
  - 求解：gspack.Equilibrium（129², order=2, method="lu"）+ ConstrainPaxisIp
    + constrain（DN/SN/雪点/limiter 分支同 v6）+ picard.solve
    （convergenceInfo=True 必打迭代表 → 包 redirect_stdout 屏蔽；
    gspack 不 raise 于不收敛，跑满 maxits 返回 → 收敛门在循环后判）
  - 质量门（本数据集核心，全部 CLI 可调，指标无条件写入 result）：
    * 收敛门：psi_relchange_final <= 10*rtol（limiter 取 step1）
    * SN midplane_ratio 门：中平面健康度 (psi(R_axis,Z~0)-psi_bndry)/core
      >= --sn-midplane-ratio-min（filter_v6 同款公式；v6 根因修复）
    * SN Z_axis 收紧门：|Z_axis| <= 0.5*|Z_lo|（叠加 v6 原 |Z_axis|<|Z_lo|）
    * gs_true 门：GS 残差比 <= --max-gs-true（evaluate_dn_fno 同款公式，
      mask=psi>=psi_bndry；limiter 触壁 J 不连续 → 仅报告不判定）
    * snow_double 雪点偏差门：xpts_actual vs 目标 <= --max-snow-xpt-dev
  - 确定性：每样本种子 = seed*100_000 + i（重试 + attempt*1_000_000），
    参数重采样 rng = default_rng(i_seed + 7_000_003)——与 v6/g2 逐位一致；
    chunk（500/块，存在则整体跳过=断点续跑）→ merge 按 idx 重读全部 chunk
    重建 {split}.npz（补数据=同种子更大 n 重跑，行 0..N0-1 比特级不变）

用法（仓库根目录）:
  python -m gs_gspack2_dn_fno_2608.generate_dn_g3_dataset --machine mastu_g3 \
      --config sn --split train --n-samples 500 --seed 123 \
      --out-dir dn_fno_2608/data_gspack2_v2 --chunk-size 500 --n-jobs 24 \
      --alpha-sampling --xpt-jitter 0.06 --xpt-jitter-z 0.10 --isoflux-sampling \
      --anchor-midplane --max-isoflux-residual 0.35 --max-xpt-deviation 0.10 \
      --min-anchor-xpt-dist 0.15 --coil-margin 0.05 --min-core-depth 0.005 \
      --save-constraint-diag --max-retries 20
  python -m gs_gspack2_dn_fno_2608.generate_dn_g3_dataset --split train \
      --out-dir dn_fno_2608/data_gspack2_v2 --config sn --merge
"""
from __future__ import annotations

import argparse
import io
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
from tqdm import trange

# gspack2_TRAE 包（不在 pip 环境时经 sys.path 注入；与 g2 生成器同款）
GSPACK2_ROOT = r"D:\D_F\Fusion\AI\PINN\gspack2_TRAE"
if GSPACK2_ROOT not in sys.path:
    sys.path.insert(0, GSPACK2_ROOT)
import gspack
from gspack import critical, picard
import gspack.backend as bk

bk.set_backend("cpu")

# 本仓库 src（复用 v6 生成器的 freegs 无关辅助函数/常量）
_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from gs_pino_dn_fno_2608 import generate_dn_dataset as g5  # noqa: E402

# ---- 网格与求解参数（= data_v6：129² 才能解析雪点；order=2 = freegs 同阶） ----
G3_RMIN, G3_RMAX = 0.1, 2.0
G3_ZMIN, G3_ZMAX = -2.0, 2.0
NX, NY = 129, 129
ORDER = 2                        # FDM 阶（freegs 同 2 阶；order=4 探针对照）
METHOD = "lu"                    # 稀疏求解器（129²>16384 → "auto" 走 AMG）
GAMMA = 1e-12                    # Tikhonov 正则（DN/SN/limiter，= v6）
RTOL, MAXITS = 1e-3, 80          # Picard 容差/最大迭代（DN/SN；v6 同）
IP_TOL = 0.10                    # |Ip_sol - Ip_tgt| / Ip_tgt <= 10%

# ---- 质量门默认阈值（探针校准后由 run 脚本显式传入） ----
CONV_GATE_FACTOR = 10.0          # 收敛门：rel_change 最后值 <= 10*rtol（g2 先例）
SN_MIDPLANE_RATIO_MIN = 0.05     # SN 中平面健康度下界（v6_clean 判据是 <0 病态）
SN_ZAXIS_RATIO_MAX = 0.5         # SN |Z_axis|/|Z_lo| 上界（v6 根因：轴偏下）
GS_TRUE_MAX = 15.0               # gs_residual_ratio 上界（v6_clean 同款阈值）
SNOW_XPT_DEV_MAX = 0.15          # snow_double 雪点/实际 X 点偏差（v6_clean 同款）

# ---- 14 控制单元规格（名字/顺序 = data_v6 通道序 = data_dn_fno_coils.py
# L48-50；sign = freegs_snow Circuit multiplier，P61/P62 下线圈 = -1；
# Solenoid/Pc 是单物理线圈（turns 编码在 greens 单位响应里）） ----
UNIT_SPECS = [
    ("Solenoid", [("Solenoid", +1.0)]),
    ("Pc",       [("Pc", +1.0)]),
    ("Px",  [("PxU", +1.0), ("PxL", +1.0)]),
    ("D1",  [("D1U", +1.0), ("D1L", +1.0)]),
    ("D2",  [("D2U", +1.0), ("D2L", +1.0)]),
    ("D3",  [("D3U", +1.0), ("D3L", +1.0)]),
    ("Dp",  [("DpU", +1.0), ("DpL", +1.0)]),
    ("D5",  [("D5U", +1.0), ("D5L", +1.0)]),
    ("D6",  [("D6U", +1.0), ("D6L", +1.0)]),
    ("D7",  [("D7U", +1.0), ("D7L", +1.0)]),
    ("P4",  [("P4U", +1.0), ("P4L", +1.0)]),
    ("P5",  [("P5U", +1.0), ("P5L", +1.0)]),
    ("P61", [("P61U", +1.0), ("P61L", -1.0)]),
    ("P62", [("P62U", +1.0), ("P62L", -1.0)]),
]
UNIT_NAMES = [spec[0] for spec in UNIT_SPECS]


class UnitMachine(gspack.Machine):
    """14 控制单元的 MASTU_simple（物理 26 线圈 = gspack 内置版 1:1）。

    单元电流语义 = freegs_snow Circuit 电流（子线圈电流 = 单元电流 × sign；
    turns 编码在单位响应里）。override 后 gspack control.py 的
    _init_precomp（nc=len(controlCurrents())=14）与 __call__ 的
    controlAdjust(dI) 自动走 14 单元，control.py 零改动。
    """

    def __init__(self):
        base = gspack.machine.MASTU_simple()   # 26 物理线圈 + 116 顶点壁
        super().__init__(base.coils, base.wall, base.R0)
        self._by_name = {nm: co for nm, co in self.coils}

    # -- 14 单元电流接口（子线圈 U/L 同步不变式：单元电流 = 首子线圈） --
    def controlCurrents(self):
        return [self._by_name[spec[1][0][0]].current for spec in UNIT_SPECS]

    def setControlCurrents(self, cs):
        for (_, subs), v in zip(UNIT_SPECS, cs):
            for phys, sign in subs:
                self._by_name[phys].current = sign * float(v)

    def controlAdjust(self, dI):
        for (_, subs), d in zip(UNIT_SPECS, dI):
            for phys, sign in subs:
                self._by_name[phys].current += sign * float(d)

    # -- 每单元单位响应 = Σ sign·子线圈响应（含 turns） --
    def _unit_response(self, meth, R, Z, *a):
        return [sum(sign * getattr(self._by_name[p], meth)(R, Z, *a)
                    for p, sign in subs)
                for _, subs in UNIT_SPECS]

    def controlPsi(self, R, Z):
        return self._unit_response("controlPsi", R, Z)

    def controlBr(self, R, Z):
        return self._unit_response("controlBr", R, Z)

    def controlBz(self, R, Z):
        return self._unit_response("controlBz", R, Z)

    def controlPsiZZ(self, R, Z, eps=1e-3):
        return self._unit_response("controlPsiZZ", R, Z, eps)

    def controlPsiRZ(self, R, Z, eps=1e-3):
        return self._unit_response("controlPsiRZ", R, Z, eps)


def make_mastu_unit() -> UnitMachine:
    return UnitMachine()


MACHINE_FACTORIES_G3 = {"mastu_g3": make_mastu_unit}

# 与 v6 CONFIG_SPECS 同形（五配置几何/参数全部照抄；machine 键 = mastu_g3）。
# 注：g5.CONFIG_SPECS 的 mastu_simple 行与本表逐键一致（含雪点权重链、
# 二阶阈值、limiter 轴范围）——仅 machine 名不同。
CONFIG_SPECS_G3 = {
    ("mastu_g3", "dn"): {"xpt_n": 2, "has_wall": True,
        "param_ranges": g5.MASTU_PARAM_RANGES, "anchor_r_range": g5.MASTU_ANCHOR_R_RANGE,
        "xpt_r0": 0.80, "xpt_z0": 1.20},
    ("mastu_g3", "sn"): {"xpt_n": 1, "has_wall": True,
        "param_ranges": g5.MASTU_PARAM_RANGES, "anchor_r_range": g5.MASTU_ANCHOR_R_RANGE,
        "xpt_r0": 0.65, "xpt_z0": 1.20},
    ("mastu_g3", "snow_single"): {"kind": "snowflake", "xpt_n": 1,
        "has_wall": True, "param_ranges": g5.MASTU_PARAM_RANGES,
        "anchor_r_range": g5.MASTU_ANCHOR_R_RANGE,
        "snow_r0": 0.509, "snow_z0": 1.291, "snow_jitter_r": 0.02,
        "snow_jitter_z": 0.04, "weights": g5.SNOW_SINGLE_WEIGHTS,
        "second_thresh": g5.SNOW_SINGLE_2ND_THRESH},
    ("mastu_g3", "snow_double"): {"kind": "snowflake", "xpt_n": 2,
        "has_wall": True, "param_ranges": g5.MASTU_PARAM_RANGES,
        "anchor_r_range": g5.MASTU_ANCHOR_R_RANGE,
        "snow_r0": 0.65, "snow_z0": 1.20, "snow_jitter_r": 0.02,
        "snow_jitter_z": 0.04, "weights": g5.SNOW_DOUBLE_WEIGHTS,
        "second_thresh": g5.SNOW_DOUBLE_2ND_THRESH},
    ("mastu_g3", "limiter"): {"kind": "limiter", "has_wall": True,
        "machine": "mastu_g3",   # 原生壁（gspack 内置 MASTU_simple 带壁）
        "param_ranges": g5.MASTU_PARAM_RANGES, "axis_r_range": g5.LIMITER_AXIS_R_RANGE},
}

# data_v6 npz 键集（34 键 = v6 28 stacked + 新指标 3 + R + Z；新指标无条件写）
STACKED_KEYS_G3 = list(g5.STACKED_KEYS) + [
    "gs_true", "midplane_ratio", "zaxis_ratio",
]

MU0 = 4.0 * np.pi * 1e-7


# ---- 质量指标（与 filter_v6 / evaluate_dn_fno 逐行同款公式） ----
def _lap_star(psi: np.ndarray, R: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """Δ*ψ = d²/dR² − (1/R)d/dR + d²/dZ²，2 阶中心差分（evaluate_dn_fno 同款）。"""
    dR = R[1, 0] - R[0, 0]
    dZ = Z[0, 1] - Z[0, 0]
    d2r = (psi[2:, 1:-1] - 2.0 * psi[1:-1, 1:-1] + psi[:-2, 1:-1]) / (dR**2)
    dr = (psi[2:, 1:-1] - psi[:-2, 1:-1]) / (2.0 * dR)
    d2z = (psi[1:-1, 2:] - 2.0 * psi[1:-1, 1:-1] + psi[1:-1, :-2]) / (dZ**2)
    return d2r - dr / (R[1:-1, 1:-1] + 1e-8) + d2z


def _gs_true_ratio(psi_total, R, Z, dpdpsi, fdFdpsi, psi_bndry) -> float:
    """GS 残差比（mask = psi >= psi_bndry，evaluate_dn_fno 同款）；core<10 → nan。"""
    mask = (psi_total >= psi_bndry).astype(np.float32)
    lap = _lap_star(psi_total, R, Z)
    R_c = R[1:-1, 1:-1]
    rhs = -MU0 * R_c**2 * dpdpsi[1:-1, 1:-1] - fdFdpsi[1:-1, 1:-1]
    core = mask[1:-1, 1:-1] > 0.5
    if core.sum() < 10:
        return float("nan")
    return float(np.linalg.norm((lap - rhs)[core]) / np.linalg.norm(rhs[core]))


def _midplane_ratio(psi_total, R, Z, axes) -> float:
    """中平面健康度 = (psi(R_axis, Z~0) − psi_bndry)/core（filter_v6 同款）。"""
    r_ax, _, psi_bndry, psi_ax = axes[0], axes[1], axes[2], axes[3]
    j0 = int(np.argmin(np.abs(Z[0, :])))
    psi_mid = float(np.interp(r_ax, R[:, j0], psi_total[:, j0]))
    core = max(psi_ax - psi_bndry, 1e-12)
    return (psi_mid - psi_bndry) / core


def _zaxis_ratio(axis_pt, lo) -> float:
    """|Z_axis| / |Z_lo|（SN 轴偏下程度的直接度量）。"""
    return abs(axis_pt[1]) / max(abs(lo[1]), 1e-12)


def _g3_control_coil_centers(tokamak) -> list[tuple[float, float]]:
    """导体位置（gspack 版）：Solenoid -> (R, Zsmin/Zsmax) 端点（gspack 用
    R/Zsmin/Zsmax 属性而非 freegs 的 Rs/Zs）；无 Circuit 分支（U/L 已独立）。"""
    centers = []
    for label, coil in tokamak.coils:
        if not coil.control:
            continue
        if hasattr(coil, "Zsmin"):  # gspack Solenoid：沿 Z 的分布式绕组
            centers.append((float(coil.R), float(coil.Zsmin)))
            centers.append((float(coil.R), float(coil.Zsmax)))
        else:                       # gspack Coil：单导体
            centers.append((float(coil.R), float(coil.Z)))
    # 绕质心极角排序 -> CCW 凸序（quad 精确）
    c = np.asarray(centers, dtype=np.float64)
    c0 = c.mean(axis=0)
    idx = np.argsort(np.arctan2(c[:, 1] - c0[1], c[:, 0] - c0[0]))
    return [(float(c[j, 0]), float(c[j, 1])) for j in idx]


def build_cfg_g3(machine: str = "mastu_g3", config: str = "dn",
                 xpt_jitter: float = 0.06, xpt_jitter_z: float | None = None,
                 isoflux_sampling: bool = True, anchor_midplane: bool = True,
                 coil_margin: float = 0.05,
                 max_isoflux_residual: float | None = None,
                 max_xpt_deviation: float | None = None,
                 min_anchor_xpt_dist: float | None = None,
                 require_wall: bool = False,
                 min_core_depth: float | None = None,
                 save_constraint_diag: bool = False,
                 xpt_r0: float | None = None, xpt_z0: float | None = None,
                 order: int = ORDER, method: str = METHOD,
                 sn_midplane_ratio_min: float | None = SN_MIDPLANE_RATIO_MIN,
                 sn_zaxis_ratio_max: float | None = SN_ZAXIS_RATIO_MAX,
                 max_gs_true: float | None = GS_TRUE_MAX,
                 max_snow_xpt_dev: float | None = SNOW_XPT_DEV_MAX,
                 snow_blend: float = 0.3) -> dict:
    """(machine, config) -> 求解/接受 cfg（v6 build_cfg + g2 扩展 + 新门阈值）。

    新门阈值传 None = 跳过判定（仅报告指标）——探针跑门关闭模式的机制。
    """
    if xpt_jitter_z is None:
        xpt_jitter_z = xpt_jitter
    spec = CONFIG_SPECS_G3.get((machine, config), {})
    xpt_n = spec.get("xpt_n", 2)
    if xpt_r0 is None:
        xpt_r0 = spec.get("xpt_r0", g5.XPT_R)
    if xpt_z0 is None:
        xpt_z0 = spec.get("xpt_z0", g5.XPT_Z)
    kind = spec.get("kind")  # None=DN/SN, "snowflake", "limiter"
    return {
        "xpt_n": xpt_n,
        "kind": kind,
        "config_name": config,
        "machine": spec.get("machine", machine),
        "machine_factory": MACHINE_FACTORIES_G3[spec.get("machine", machine)],
        "nx": NX, "ny": NY, "order": order, "method": method,
        "maxits": MAXITS,
        "has_wall": spec.get("has_wall", True),
        "param_ranges": spec.get("param_ranges", g5.MASTU_PARAM_RANGES),
        "anchor_r_range": spec.get("anchor_r_range", g5.MASTU_ANCHOR_R_RANGE),
        "xpt_r0": xpt_r0, "xpt_z0": xpt_z0,
        "xpt_jitter": xpt_jitter, "xpt_jitter_z": xpt_jitter_z,
        # data_v6: snowflake / limiter spec passthrough
        "snow_r0": spec.get("snow_r0", 0.7), "snow_z0": spec.get("snow_z0", 1.1),
        "snow_jitter_r": spec.get("snow_jitter_r", 0.02),
        "snow_jitter_z": spec.get("snow_jitter_z", 0.04),
        "weights": spec.get("weights", g5.SNOW_SINGLE_WEIGHTS),
        "second_thresh": spec.get("second_thresh", g5.SNOW_SINGLE_2ND_THRESH),
        # gspack 雪点松弛 Picard 混合系数（实测 0.3 收敛 + 下瓣分支；见 _solve_snow_relaxed）
        "snow_blend": snow_blend,
        "axis_r_range": spec.get("axis_r_range", g5.LIMITER_AXIS_R_RANGE),
        # data_gspack2_v2 质量门阈值（None = 报告不判定）
        "sn_midplane_ratio_min": sn_midplane_ratio_min,
        "sn_zaxis_ratio_max": sn_zaxis_ratio_max,
        "max_gs_true": max_gs_true,
        "max_snow_xpt_dev": max_snow_xpt_dev,
        "isoflux_sampling": isoflux_sampling, "anchor_midplane": anchor_midplane,
        "coil_margin": coil_margin,
        "max_isoflux_residual": max_isoflux_residual,
        "max_xpt_deviation": max_xpt_deviation,
        "min_anchor_xpt_dist": min_anchor_xpt_dist,
        "require_wall": require_wall and spec.get("has_wall", True),
        "min_core_depth": min_core_depth,
        "save_constraint_diag": save_constraint_diag,
    }


def _acceptance_checks_g3(cfg, eq, profiles, tokamak, lo, up, anchor,
                          opt, xpt, sep_xpt, Ip, dpdpsi, fdFdpsi):
    """共享接受检查（= v6 全量 + gspack API + data_gspack2_v2 新质量门）。

    返回 11 元组 (axis_pt, psi_bndry, wall_contact, wall_contact_excess,
    inwall_frac, p_lo, p_up, p_anc, midplane_ratio, gs_true, zaxis_ratio)
    或 None。新指标无条件计算；门判定按 cfg 阈值（None = 仅报告）。
    """
    p_lo = p_up = p_anc = 0.0
    inwall_frac = 0.0
    if eq.psi_axis is None or eq.psi_bndry is None:
        return None
    ip_err = abs(eq.plasmaCurrent() - Ip) / Ip
    if ip_err > IP_TOL:
        return None
    if cfg["kind"] == "snowflake":
        if len(sep_xpt) != cfg["xpt_n"]:
            return None
        psi_bndry = eq.psi_bndry
    elif cfg["xpt_n"] == 1:
        if len(sep_xpt) != 1:
            return None
    else:
        if len(sep_xpt) != cfg["xpt_n"]:
            return None
        if len(xpt) < g5.MIN_XPTS:
            return None
    if cfg["kind"] != "snowflake":
        psi_bndry = sep_xpt[0][2] if cfg["xpt_n"] == 1 else 0.5 * (xpt[0][2] + xpt[1][2])
    # 磁轴：SN 位置过滤取 max psi（gspack O 点按距网格中心排序，非 psi）
    if cfg["xpt_n"] == 1 and opt:
        cand = [o for o in opt if lo[0] < o[0] < anchor[0] and abs(o[1]) < abs(lo[1])]
        axis_pt = max(cand, key=lambda o: o[2]) if cand else opt[0]
    else:
        axis_pt = opt[0] if opt else None
    if not (np.isfinite(profiles.L) and 0.0 < profiles.Beta0 < 1.0):
        return None

    # ---- 新质量指标（无条件计算；psi_total 与 mask 都来自 eq.psi() 缓存） ----
    psi_total = eq.psi()
    midplane_r = _midplane_ratio(psi_total, eq.R, eq.Z,
                                 np.asarray([axis_pt[0], axis_pt[1],
                                             psi_bndry, eq.psi_axis]))
    gs_true = _gs_true_ratio(psi_total, eq.R, eq.Z, dpdpsi, fdFdpsi, psi_bndry)
    zaxis_r = _zaxis_ratio(axis_pt, lo)
    if cfg["xpt_n"] == 1 and cfg["sn_midplane_ratio_min"] is not None:
        # SN 中平面健康度硬门（v6 根因修复主门：上瓣薄/中平面外翻 → ratio<0）
        if not (midplane_r >= cfg["sn_midplane_ratio_min"]):
            return None
    if cfg["xpt_n"] == 1 and cfg["sn_zaxis_ratio_max"] is not None:
        # SN 轴偏下收紧门（叠加 v6 原 |Z_axis|<|Z_lo| 检查）
        if not (zaxis_r <= cfg["sn_zaxis_ratio_max"]):
            return None
    if cfg["kind"] != "limiter" and cfg["max_gs_true"] is not None:
        # GS 残差比门（nan = core<10 病态 → 拒绝）
        if not (gs_true <= cfg["max_gs_true"]):
            return None

    # ---- snowflake 一/二阶残差（v6 全量） ----
    if cfg["kind"] == "snowflake":
        for p in ([lo] if cfg["xpt_n"] == 1 else [lo, up]):
            if (abs(g5._at(eq, p, "Br")) >= g5.SNOW_1ST_THRESH
                    or abs(g5._at(eq, p, "Bz")) >= g5.SNOW_1ST_THRESH):
                return None
            if (abs(float(np.asarray(eq.hessianZZ(p[0], p[1], eps=g5.SNOWFLAKE_EPS)).reshape(-1)[0]))
                    >= cfg["second_thresh"]
                    or abs(float(np.asarray(eq.hessianRZ(p[0], p[1], eps=g5.SNOWFLAKE_EPS)).reshape(-1)[0]))
                    >= cfg["second_thresh"]):
                return None

    # ---- data_v4 physical-validity checks（v6 全量） ----
    diag = cfg["save_constraint_diag"]
    v4_active = (diag
                 or cfg["max_isoflux_residual"] is not None
                 or cfg["max_xpt_deviation"] is not None
                 or cfg["min_anchor_xpt_dist"] is not None
                 or cfg["require_wall"]
                 or cfg["min_core_depth"] is not None)
    if v4_active:
        core = eq.psi_axis - psi_bndry
        if core <= 0 or not np.isfinite(core):
            return None
        if cfg["min_core_depth"] is not None and core < cfg["min_core_depth"]:
            return None

        p_lo = g5._at(eq, lo)
        p_up = g5._at(eq, up)
        p_anc = g5._at(eq, anchor)

        if cfg["max_isoflux_residual"] is not None:
            if cfg["kind"] == "snowflake":
                res_terms = [abs(p_anc - g5._at(eq, (g5.SNOW_REF_R, 0.0)))]
            else:
                res_terms = [abs(p_lo - p_anc)]
                if cfg["xpt_n"] == 2:
                    res_terms.append(abs(p_up - p_anc))
            if max(res_terms) / core > cfg["max_isoflux_residual"]:
                return None

        if cfg["kind"] != "snowflake":
            pts = [lo, anchor] if cfg["xpt_n"] == 1 else [lo, up, anchor]
            hull_verts = g5._convex_hull(_g3_control_coil_centers(tokamak))
            wall_verts = None
            if tokamak.wall is not None:
                wall_verts = np.column_stack([
                    np.asarray(tokamak.wall.R, dtype=np.float64),
                    np.asarray(tokamak.wall.Z, dtype=np.float64),
                ])
            for p in pts:
                if not g5._inside_with_margin(p, hull_verts, cfg["coil_margin"]):
                    return None
                if cfg["require_wall"] and not g5._point_in_polygon(p, wall_verts):
                    return None
        if cfg["xpt_n"] == 1:
            ra, za = axis_pt[0], axis_pt[1]
            if not (lo[0] < ra < anchor[0] and abs(za) < abs(lo[1])):
                return None
        elif not g5._point_in_triangle(opt[0][:2], (lo, up, anchor)):
            return None
        if cfg["min_anchor_xpt_dist"] is not None:
            if np.hypot(anchor[0] - lo[0], anchor[1] - lo[1]) < cfg["min_anchor_xpt_dist"]:
                return None
            if cfg["xpt_n"] == 2 and np.hypot(anchor[0] - up[0], anchor[1] - up[1]) < cfg["min_anchor_xpt_dist"]:
                return None
        if cfg["max_xpt_deviation"] is not None and cfg["kind"] != "snowflake":
            cand = sep_xpt
            xpt_xy = np.asarray([(r, z) for r, z, _ in cand], dtype=np.float64)
            for tgt in ([lo] if cfg["xpt_n"] == 1 else [lo, up]):
                if np.hypot(xpt_xy[:, 0] - tgt[0], xpt_xy[:, 1] - tgt[1]).min() > cfg["max_xpt_deviation"]:
                    return None

        # ---- data_v6: wall contact on the flood-fill core mask ----
        wall_contact = 0.0
        wall_contact_excess = 0.0
        if tokamak.wall is not None:
            mask = critical.core_mask(eq.R, eq.Z, psi_total, opt, xpt)
            from scipy.interpolate import RectBivariateSpline
            Rw = np.asarray(tokamak.wall.R, dtype=np.float64)
            Zw = np.asarray(tokamak.wall.Z, dtype=np.float64)
            seg = []
            for i in range(len(Rw) - 1):
                t = np.linspace(0.0, 1.0, 60)
                seg.append(np.c_[Rw[i] + t * (Rw[i + 1] - Rw[i]),
                                 Zw[i] + t * (Zw[i + 1] - Zw[i])])
            wp = np.vstack(seg)
            maskf = RectBivariateSpline(eq.R[:, 0], eq.Z[0, :], mask)
            touch = maskf(wp[:, 0], wp[:, 1], grid=False) > 0.5
            if touch.any():
                wall_contact = 1.0
                # gspack psiRZ 仅标量（freegs 支持数组）→ 逐点
                exc = np.asarray([g5._at(eq, (r, z))
                                  for r, z in zip(wp[touch, 0], wp[touch, 1])]) - psi_bndry
                core_depth = eq.psi_axis - psi_bndry
                if core_depth > 0:
                    wall_contact_excess = float(exc.max() / core_depth)
                if wall_contact_excess > g5.WALL_CONTACT_TOL:
                    return None

            wRmin = float(np.asarray(tokamak.wall.R).min())
            ab = psi_total >= psi_bndry
            inw = eq.R < wRmin
            he = inw[:, :-1] & inw[:, 1:] & (ab[:, :-1] != ab[:, 1:])
            ve = inw[:-1, :] & inw[1:, :] & (ab[:-1, :] != ab[1:, :])
            n_cross = int(he.sum()) + int(ve.sum())
            n_edges = int((inw[:, :-1] & inw[:, 1:]).sum()) + int((inw[:-1, :] & inw[1:, :]).sum())
            inwall_frac = n_cross / n_edges if n_edges else 0.0
            if inwall_frac > g5.INWALL_SEP_FRAC_TOL:
                return None
    return (axis_pt, psi_bndry, wall_contact, wall_contact_excess, inwall_frac,
            p_lo, p_up, p_anc, midplane_r, gs_true, zaxis_r)


def _g3_greens(tokamak, eq) -> np.ndarray:
    """14 单元 greens = Σ sign·子线圈单位响应（eq._coil_psi_unit 缓存）。"""
    return np.stack([
        sum(sign * eq._coil_psi_unit[phys] for phys, sign in subs).astype(np.float32)
        for _, subs in UNIT_SPECS])


def _solve_snow_relaxed(eq, profiles, constrain, rtol, maxits, blend,
                        atol=1e-10):
    """带松弛的雪点 Picard 迭代（克隆 picard.solve 循环 + 关键修正）。

    实测（2026-08-24，对照 freegs_snow）：gspack 纯 Picard 在雪点问题上会
    进入周期-3 极限环发散（rel 0.85→1.00→1.41 循环）；freegs_snow 同参数
    收敛（n=63-71）说明迭代动力学差异（GS 求解器/等离子场数值细节）。

    修正点：混合必须放在 constrain(eq) 之后——constrain 会突变线圈电流，
    picard.solve 里 constrain 后 `psi = eq.psi()` 会把混合结果整体覆盖，
    松弛失去作用。本函数顺序：GS 解 → 收敛检查 → constrain → 混合
    psi = blend*psi_last + (1-blend)*psi_new2。

    blend=0.3 实测：snow_single/snow_double 8/8 收敛到正确下瓣形态
    （X 点距雪点目标 6-13mm，psi_axis ~0.15-0.17），rel ~2-7e-4；
    blend=0.5 会收敛但落上瓣假解（形态检查兜底拒绝）；0.7 收敛慢。
    """
    if constrain is not None:
        constrain(eq)
    psi = np.asarray(eq.psi(), dtype=float)
    bndry = 0.0
    bndry_change = float("inf")
    bndry_relchange = float("inf")
    rel_list = []
    for it in range(maxits):
        psi_last = psi.copy()
        bndry_last = bndry
        eq.solve(profiles, psi=psi)
        psi_new = np.asarray(eq.psi(), dtype=float)
        max_change = float(np.abs(psi_new - psi_last).max())
        span = float(psi_new.max() - psi_new.min())
        rel_change = max_change / (span + 1e-30)
        rel_list.append(rel_change)
        # psi_bndry 双判据（同 picard.solve：防线圈电流仍调整时提前收敛）
        if eq.psi_bndry is not None:
            bndry = eq.psi_bndry
            bndry_change = bndry_last - bndry
            bndry_relchange = abs(bndry_change / (bndry + 1e-30))
        else:
            bndry_relchange = 2.0 * rtol
        if ((max_change < atol) or (rel_change < rtol)) \
           and ((bndry_relchange < rtol) or (abs(bndry_change) < atol)):
            break
        if constrain is not None:
            constrain(eq)
            psi_new2 = np.asarray(eq.psi(), dtype=float)
        else:
            psi_new2 = psi_new
        # 混合在 constrain 之后（见 docstring）
        psi = blend * psi_last + (1.0 - blend) * psi_new2
    return np.array(rel_list)


def _solve_one(args: tuple) -> dict | None:
    """求解单个 DN/SN/雪点平衡（gspack 版；模块级供 joblib）。

    args = (params, i_seed, cfg)。接受 = v6 全量 + 收敛门 + data_gspack2_v2
    新质量门（SN midplane/Z_axis/gs_true/snow xpt_dev）。
    """
    params, i_seed, cfg = args
    t0 = time.perf_counter()

    # joblib spawn worker 里 cupy_ok 为 True 时 gspack 可能走 GPU 分支
    # （实测 module 模式 worker 的 backend 变 gpu，greens 烧 cupy CPU 100s+）；
    # 本机无 GPU 用途，强制 cpu 后端（幂等，主进程已 cpu）
    bk.set_backend("cpu")
    bk._state["cupy_ok"] = False

    paxis, Ip, fvac = params["paxis"], params["Ip"], params["fvac"]
    alpha_m = params.get("alpha_m", 1.0)
    alpha_n = params.get("alpha_n", 2.0)
    rng = np.random.default_rng(i_seed)
    if cfg["kind"] == "snowflake":
        sf = g5.sample_snowflake_points(rng, r0=cfg["snow_r0"], z0=cfg["snow_z0"],
                                        n=cfg["xpt_n"], jitter_r=cfg["snow_jitter_r"],
                                        jitter_z=cfg["snow_jitter_z"])
        lo, up = sf[0], sf[-1]
    else:
        lo, up = g5.sample_xpoints(rng, jitter=cfg["xpt_jitter"],
                                   jitter_z=cfg["xpt_jitter_z"],
                                   r0=cfg["xpt_r0"], z0=cfg["xpt_z0"])
    anchor = g5.sample_anchor(rng, isoflux=cfg["isoflux_sampling"],
                              midplane=cfg["anchor_midplane"],
                              r_range=cfg["anchor_r_range"])

    try:
        tokamak = cfg["machine_factory"]()
        if cfg["kind"] == "snowflake":
            # gspack 实证（2026-08-24，对照 freegs_snow）：MASTU_INIT_CURRENTS
            # 种子在 gspack 下使 Picard 进入上瓣假解并振荡（n=200 rel=0.15 极限
            # 环，轴 (1.013,1.173) 在上瓣）；零电流初始反而稳定收敛到与
            # freegs_snow seed 分支相同的下瓣正确解（X 点落在雪点目标 ~1cm、
            # 轴 (1.067,-0.435)、psi_axis/psi_bndry 与 freegs 解一致）。
            # freegs_snow 里"零电流振荡、种子改善"的行为在 gspack 正好相反
            # （Tikhonov 约束初始应用与迭代结构的实现差异）——保持零电流。
            pass

        eq = gspack.Equilibrium(tokamak,
                                Rmin=G3_RMIN, Rmax=G3_RMAX,
                                Zmin=G3_ZMIN, Zmax=G3_ZMAX,
                                nx=cfg["nx"], ny=cfg["ny"],
                                order=cfg["order"], method=cfg["method"])
        profiles = gspack.ConstrainPaxisIp(p_axis=paxis, Ip=Ip, fvac=fvac,
                                           alpha_m=alpha_m, alpha_n=alpha_n,
                                           Raxis=1.0)

        conv = None
        if cfg["kind"] == "snowflake":
            # 二阶雪点约束（gspack control.py 原生支持 snowflake 分支）+
            # 松弛 Picard（_solve_snow_relaxed，blend=0.3——纯 Picard 周期-3
            # 极限环发散，实测修正后 8/8 收敛到正确下瓣形态）；
            # weight 重试链：gspack 不 raise 于不收敛 → 每次新建 constrain
            # 对象（_precomp 按 n_coils 缓存、weight 变化不触发重建）
            sf_pts = [lo] if cfg["xpt_n"] == 1 else [lo, up]
            for w in cfg["weights"]:
                con = gspack.constrain(
                    snowflake=sf_pts,
                    isoflux=[(*anchor, g5.SNOW_REF_R, 0.0)] * len(sf_pts),
                    gamma=g5.SNOWFLAKE_GAMMA,
                    snowflake_weight=w,
                    snowflake_eps=g5.SNOWFLAKE_EPS)
                with redirect_stdout(io.StringIO()):
                    rel = _solve_snow_relaxed(eq, profiles, con,
                                              rtol=g5.SNOWFLAKE_RTOL,
                                              maxits=g5.SNOWFLAKE_MAXITS,
                                              blend=cfg["snow_blend"])
                conv = (np.array([0.0]), rel)  # (max_list, rel_list) 对齐
                if (rel is not None and np.isfinite(float(rel[-1]))
                        and float(rel[-1]) <= CONV_GATE_FACTOR * g5.SNOWFLAKE_RTOL):
                    break
        elif cfg["xpt_n"] == 1:
            con = gspack.constrain(xpoints=[lo], isoflux=[(*lo, *anchor)],
                                   gamma=GAMMA)
        else:
            con = gspack.constrain(xpoints=[lo, up],
                                   isoflux=[(*lo, *anchor), (*up, *anchor)],
                                   gamma=GAMMA)

        if conv is None:  # 非雪点分支：常规/SN 求解
            with redirect_stdout(io.StringIO()):
                conv = picard.solve(eq, profiles, constrain=con,
                                    rtol=RTOL, maxits=cfg["maxits"],
                                    convergenceInfo=True, verbose=False)

        # 收敛门：最后相对 psi 变化 <= 10*rtol（发散解在此拒绝）
        if conv is None or not np.isfinite(float(conv[1][-1])) \
                or float(conv[1][-1]) > CONV_GATE_FACTOR * RTOL:
            return None

        opt, xpt = critical.find_critical(eq.R, eq.Z, eq.psi())
        if not opt:
            return None
        # 磁轴：gspack O 点按距网格中心排序（非 psi）——按 v5 位置过滤取
        # 最高 psi 的物理芯区 O 点，opt 重排使 opt[0] = 轴
        cand = [o for o in opt
                if lo[0] < o[0] < anchor[0] and abs(o[1]) < abs(lo[1])]
        if not cand:
            return None
        axis_pt = max(cand, key=lambda o: o[2])
        opt = [axis_pt] + [o for o in opt if o is not axis_pt]

        sep_xpt = [p for p in xpt if float(p[2]) >= eq.psi_bndry - 1e-6]

        # GS RHS 成分（psi_norm 口径同 v6：用 eq.psi_bndry）
        psi_total = eq.psi()
        psi_norm = (psi_total - eq.psi_axis) / (eq.psi_bndry - eq.psi_axis + 1e-30)
        dpdpsi = profiles.pprime(psi_norm)
        fdFdpsi = profiles.ffprime(psi_norm)

        acc = _acceptance_checks_g3(cfg, eq, profiles, tokamak, lo, up, anchor,
                                    opt, xpt, sep_xpt, Ip, dpdpsi, fdFdpsi)
        if acc is None:
            return None
        (axis_pt, psi_bndry, wall_contact, wall_contact_excess, inwall_frac,
         p_lo, p_up, p_anc, midplane_r, gs_true, zaxis_r) = acc

        # SN midplane 几何门（g2 先例，叠加在主门之上）：分离面必须跨中平面
        if cfg["xpt_n"] == 1:
            Ri, Ro = eq.innerOuterSeparatrix(Z=0.0)
            degenerate = (Ri == G3_RMIN and Ro == G3_RMAX)
            if degenerate:
                return None

        # ---- xpts_actual（无条件计算：snow_double 雪点偏差门依赖） ----
        cand = sep_xpt if cfg["xpt_n"] == 1 else xpt
        xpt_xy = np.asarray([(r, z) for r, z, _ in cand], dtype=np.float64)
        remaining = list(range(len(xpt_xy)))
        xpts_actual = []
        for tgt in ([lo] if cfg["xpt_n"] == 1 else [lo, up]):  # greedy 配对
            j = min(remaining, key=lambda j: np.hypot(xpt_xy[j, 0] - tgt[0],
                                                      xpt_xy[j, 1] - tgt[1]))
            xpts_actual.append(cand[j][:3])
            remaining.remove(j)
        xpts_actual = np.asarray(xpts_actual, dtype=np.float64)
        if cfg["max_snow_xpt_dev"] is not None:
            dev = float(np.max(np.hypot(
                xpts_actual[:, 0] - np.asarray(
                    [lo, up] if cfg["xpt_n"] == 2 else [lo])[:, 0],
                xpts_actual[:, 1] - np.asarray(
                    [lo, up] if cfg["xpt_n"] == 2 else [lo])[:, 1])))
            if not (dev <= cfg["max_snow_xpt_dev"]):
                return None

        # ---- 字段提取（与 data_v6 逐键一致；greens/电流 = 14 单元） ----
        psi_plasma = eq.plasma_psi
        psi_coils = eq.tokamak.psi_coils(eq.R, eq.Z)
        psi_plasma_norm = (psi_plasma - eq.psi_bndry) / (eq.psi_axis - eq.psi_bndry + 1e-30)
        mask = critical.core_mask(eq.R, eq.Z, psi_total, opt, xpt)

        greens = _g3_greens(tokamak, eq)
        coil_currents = np.array(tokamak.controlCurrents(), dtype=np.float32)

        R_axis = axis_pt[0]
        Z_axis = axis_pt[1]
        result = {
            "psi_total": psi_total.astype(np.float32),
            "psi_plasma": psi_plasma.astype(np.float32),
            "psi_plasma_norm": psi_plasma_norm.astype(np.float32),
            "psi_coils": psi_coils.astype(np.float32),
            "mask": mask.astype(np.float32),
            "dpdpsi": dpdpsi.astype(np.float32),
            "FdFdpsi": fdFdpsi.astype(np.float32),
            "greens": greens,
            "coil_currents": coil_currents,
            "params": np.array([Ip, paxis, fvac, alpha_m, alpha_n],
                               dtype=np.float32),
            # SN/snow_single: x_coords 上对 (0,0) 占位（同 v6）
            "x_coords": np.array([*lo, *up] if cfg["xpt_n"] == 2
                                 else [*lo, 0.0, 0.0], dtype=np.float32),
            "config": np.array([g5.CONFIG_CODES[cfg["config_name"]]],
                               dtype=np.float32),
            "axes": np.array([R_axis, Z_axis, psi_bndry, eq.psi_axis],
                             dtype=np.float32),
            "L": np.array([profiles.L], dtype=np.float32),
            "Beta0": np.array([profiles.Beta0], dtype=np.float32),
            "solve_time": np.array([time.perf_counter() - t0], dtype=np.float32),
            "wall_contact": np.array([wall_contact], dtype=np.float32),
            "wall_contact_excess": np.array([wall_contact_excess], dtype=np.float32),
            "inwall_sep_frac": np.array([inwall_frac], dtype=np.float32),
            # data_gspack2_v2 新质量指标（无条件写入）
            "gs_true": np.array([gs_true], dtype=np.float32),
            "midplane_ratio": np.array([midplane_r], dtype=np.float32),
            "zaxis_ratio": np.array([zaxis_r], dtype=np.float32),
        }
        if cfg["isoflux_sampling"]:
            result["anchor"] = np.array(anchor, dtype=np.float32)
        if cfg["save_constraint_diag"]:
            result.update({
                "xpts_actual": xpts_actual.astype(np.float32),
                "o_point": np.array(axis_pt[:3], dtype=np.float32),
                "xpt_constraint_res": np.array(
                    [g5._at(eq, lo, "Br"), g5._at(eq, lo, "Bz")]
                    + ([g5._at(eq, up, "Br"), g5._at(eq, up, "Bz")]
                       if cfg["xpt_n"] == 2 else []),
                    dtype=np.float32),
                "isoflux_res": np.array(
                    ([p_anc - g5._at(eq, (g5.SNOW_REF_R, 0.0))]
                     if cfg["kind"] == "snowflake"
                     else [p_lo - p_anc]
                     + ([p_up - p_anc] if cfg["xpt_n"] == 2 else [])),
                    dtype=np.float32),
                "psi_at_constraints": np.array(
                    [p_lo, p_up, p_anc] if cfg["xpt_n"] == 2 else [p_lo, p_anc],
                    dtype=np.float32),
                "n_iter": np.array([len(conv[1])], dtype=np.float32),
                "psi_relchange_final": np.array([conv[1][-1]], dtype=np.float32),
            })
            if cfg["kind"] == "snowflake":
                res = []
                for p in ([lo] if cfg["xpt_n"] == 1 else [lo, up]):
                    res += [g5._at(eq, p, "Br"), g5._at(eq, p, "Bz"),
                            float(np.asarray(eq.hessianZZ(p[0], p[1], eps=g5.SNOWFLAKE_EPS)).reshape(-1)[0]),
                            float(np.asarray(eq.hessianRZ(p[0], p[1], eps=g5.SNOWFLAKE_EPS)).reshape(-1)[0])]
                result["snowflake_res"] = np.array(res, dtype=np.float32)
        return result
    except Exception as e:
        import traceback
        print(f"[_solve_one {cfg.get('config_name')}] {type(e).__name__}: {e}",
              file=sys.stderr)
        traceback.print_exc()
        return None


def _solve_one_limiter_g3(args: tuple) -> dict | None:
    """求解一个壁限制平衡（两步法，ex. 23；gspack 版）。

    Step 1: 钉磁轴（xpoints=[(R_axis,0)]）求解（check_limited=True，
    limiter_mode="flood"——连通性判据，修 freegs maxpsi 误报）；Step 2:
    冻结线圈电流（constrain=None）松弛到触壁。字段同 v6 limiter 分支 +
    gs_true（触壁解 J 不连续 → 默认仅报告不判定）。
    """
    params, i_seed, cfg = args
    t0 = time.perf_counter()

    # joblib spawn worker 后端修复（同 _solve_one）：强制 cpu
    bk.set_backend("cpu")
    bk._state["cupy_ok"] = False

    paxis, Ip, fvac = params["paxis"], params["Ip"], params["fvac"]
    alpha_m = params.get("alpha_m", 1.0)
    alpha_n = params.get("alpha_n", 2.0)
    rng = np.random.default_rng(i_seed)
    axis_tgt = (float(rng.uniform(*cfg["axis_r_range"])), 0.0)

    try:
        tokamak = cfg["machine_factory"]()
        eq = gspack.Equilibrium(tokamak,
                                Rmin=G3_RMIN, Rmax=G3_RMAX,
                                Zmin=G3_ZMIN, Zmax=G3_ZMAX,
                                nx=cfg["nx"], ny=cfg["ny"],
                                order=cfg["order"], method=cfg["method"],
                                check_limited=True, limiter_mode="flood")
        profiles = gspack.ConstrainPaxisIp(p_axis=paxis, Ip=Ip, fvac=fvac,
                                           alpha_m=alpha_m, alpha_n=alpha_n,
                                           Raxis=1.0)
        con = gspack.constrain(xpoints=[axis_tgt], gamma=g5.LIMITER_GAMMA)
        with redirect_stdout(io.StringIO()):
            conv1 = picard.solve(eq, profiles, constrain=con,
                                 rtol=g5.LIMITER_RTOL, maxits=g5.LIMITER_MAXITS,
                                 convergenceInfo=True, verbose=False)
        with redirect_stdout(io.StringIO()):
            picard.solve(eq, profiles, constrain=None,
                         rtol=g5.LIMITER_RTOL, maxits=g5.LIMITER_MAXITS,
                         convergenceInfo=False, verbose=False)
    except Exception as e:
        print(f"[_solve_one_limiter_g3 {cfg.get('config_name')}] "
              f"{type(e).__name__}: {e}", file=sys.stderr)
        return None

    # 收敛门（step1：limiter rtol 5e-3 → 10×rtol = 0.05）
    if conv1 is None or not np.isfinite(float(conv1[1][-1])) \
            or float(conv1[1][-1]) > CONV_GATE_FACTOR * g5.LIMITER_RTOL:
        return None

    # ---- 接受检查（= v6 limiter 全量） ----
    if not getattr(eq, "is_limited", False):
        return None
    if eq.psi_axis is None or eq.psi_bndry is None:
        return None
    ip_err = abs(eq.plasmaCurrent() - Ip) / Ip
    if ip_err > IP_TOL:
        return None
    opt, xpt = critical.find_critical(eq.R, eq.Z, eq.psi())
    cand = [o for o in opt if np.hypot(o[0] - axis_tgt[0], o[1] - axis_tgt[1]) < 0.5]
    axis_pt = max(cand, key=lambda o: o[2]) if cand else None
    if axis_pt is None:
        return None
    if any(float(p[2]) >= eq.psi_bndry - 1e-4 for p in xpt):
        return None
    core = eq.psi_axis - eq.psi_bndry
    if core <= 0 or not np.isfinite(core):
        return None
    if cfg["min_core_depth"] is not None and core < cfg["min_core_depth"]:
        return None
    if not (np.isfinite(profiles.L) and 0.0 < profiles.Beta0 < 1.0):
        return None
    if abs(g5._at(eq, (eq.Rlim, eq.Zlim)) - eq.psi_bndry) > 0.01:
        return None

    # ---- 质量指标（limiter：gs_true 报告；midplane/zaxis 不适用） ----
    psi_total = eq.psi()
    psi_norm = (psi_total - eq.psi_axis) / (eq.psi_bndry - eq.psi_axis + 1e-30)
    dpdpsi = profiles.pprime(psi_norm)
    fdFdpsi = profiles.ffprime(psi_norm)
    gs_true = _gs_true_ratio(psi_total, eq.R, eq.Z, dpdpsi, fdFdpsi, eq.psi_bndry)
    if cfg["max_gs_true"] is not None:
        if not (gs_true <= cfg["max_gs_true"]):
            return None

    # ---- 字段提取（同 v6 limiter + 新指标） ----
    psi_plasma = eq.plasma_psi
    psi_coils = eq.tokamak.psi_coils(eq.R, eq.Z)
    psi_plasma_norm = (psi_plasma - eq.psi_bndry) / (eq.psi_axis - eq.psi_bndry + 1e-30)
    mask = critical.core_mask(eq.R, eq.Z, psi_total, opt, xpt)
    greens = _g3_greens(tokamak, eq)
    coil_currents = np.array(tokamak.controlCurrents(), dtype=np.float32)

    R_axis, Z_axis = axis_pt[0], axis_pt[1]
    result = {
        "psi_total": psi_total.astype(np.float32),
        "psi_plasma": psi_plasma.astype(np.float32),
        "psi_plasma_norm": psi_plasma_norm.astype(np.float32),
        "psi_coils": psi_coils.astype(np.float32),
        "mask": mask.astype(np.float32),
        "dpdpsi": dpdpsi.astype(np.float32),
        "FdFdpsi": fdFdpsi.astype(np.float32),
        "greens": greens,
        "coil_currents": coil_currents,
        "params": np.array([Ip, paxis, fvac, alpha_m, alpha_n],
                           dtype=np.float32),
        "x_coords": np.array([R_axis, Z_axis, 0.0, 0.0], dtype=np.float32),
        "config": np.array([g5.CONFIG_CODES["limiter"]], dtype=np.float32),
        "axes": np.array([R_axis, Z_axis, eq.psi_bndry, eq.psi_axis],
                         dtype=np.float32),
        "L": np.array([profiles.L], dtype=np.float32),
        "Beta0": np.array([profiles.Beta0], dtype=np.float32),
        "solve_time": np.array([time.perf_counter() - t0], dtype=np.float32),
        "wall_contact": np.array([1.0], dtype=np.float32),
        "wall_contact_excess": np.array(
            [(eq.psi_limit - eq.psi_bndry) / core if core > 0 else 0.0],
            dtype=np.float32),
        "inwall_sep_frac": np.array([0.0], dtype=np.float32),
        # data_gspack2_v2 新质量指标（limiter：midplane/zaxis 不适用 → nan 占位）
        "gs_true": np.array([gs_true], dtype=np.float32),
        "midplane_ratio": np.array([np.nan], dtype=np.float32),
        "zaxis_ratio": np.array([np.nan], dtype=np.float32),
    }
    if cfg["save_constraint_diag"]:
        result.update({
            "xpts_actual": np.empty((0, 3), dtype=np.float32),
            "o_point": np.array(axis_pt[:3], dtype=np.float32),
            "xpt_constraint_res": np.array(
                [g5._at(eq, axis_tgt, "Br"), g5._at(eq, axis_tgt, "Bz")],
                dtype=np.float32),
            "isoflux_res": np.array([0.0], dtype=np.float32),
            "psi_at_constraints": np.array([eq.psi_axis], dtype=np.float32),
            "n_iter": np.array([len(conv1[1])], dtype=np.float32),
            "psi_relchange_final": np.array([conv1[1][-1]], dtype=np.float32),
            "is_limited": np.array([1.0], dtype=np.float32),
            "Rlim": np.array([eq.Rlim], dtype=np.float32),
            "Zlim": np.array([eq.Zlim], dtype=np.float32),
            "psi_limit": np.array([eq.psi_limit], dtype=np.float32),
        })
    return result


def _solve_with_retry(args: tuple, max_retries: int = 5,
                      rng: np.random.Generator | None = None) -> dict | None:
    """拒绝时整参重采样重试（= v6/g2；确定性种子逐位一致）。"""
    params, i_seed, cfg = args
    if rng is None:
        rng = np.random.default_rng(i_seed + 7_000_003)
    for attempt in range(max_retries):
        if cfg["kind"] == "limiter":
            result = _solve_one_limiter_g3((params, i_seed + attempt * 1_000_000, cfg))
        else:
            result = _solve_one((params, i_seed + attempt * 1_000_000, cfg))
        if result is not None:
            return result
        params = g5.sample_params(rng, alpha=("alpha_m" in params),
                                  ranges=cfg["param_ranges"])
    return None


def save_chunk(chunk_dir: Path, chunk_idx: int, results: list[dict],
               R_global: np.ndarray, Z_global: np.ndarray) -> Path:
    """存一块独立 .npz（断点续跑检查点）；键集 = data_v6 28 + 新指标 3。"""
    arrays = {key: np.stack([r[key] for r in results])
              for key in STACKED_KEYS_G3 if key in results[0]}
    arrays["R"] = R_global
    arrays["Z"] = Z_global
    chunk_dir.mkdir(parents=True, exist_ok=True)
    path = chunk_dir / f"chunk_{chunk_idx:03d}.npz"
    np.savez(path, **arrays)
    return path


def merge(out_dir: str, split: str) -> None:
    """按 idx 重读全部 chunk 重建 {split}.npz（幂等；绝不就地追加）。"""
    chunk_dir = Path(out_dir) / split
    chunks = sorted(chunk_dir.glob("chunk_*.npz"))
    if not chunks:
        raise FileNotFoundError(f"no chunks found in {chunk_dir}")

    merged, total = {}, 0
    for path in chunks:
        with np.load(path) as d:
            for key in STACKED_KEYS_G3:
                if key not in d:
                    continue
                merged.setdefault(key, []).append(d[key])
            merged.setdefault("R", d["R"])
            merged.setdefault("Z", d["Z"])
            total += d["psi_total"].shape[0]

    out_path = Path(out_dir) / f"{split}.npz"
    arrays = {key: np.concatenate(v) for key, v in merged.items()
              if isinstance(v, list)}
    arrays["R"] = merged["R"]
    arrays["Z"] = merged["Z"]
    np.savez(out_path, **arrays)

    print(f"\n  merged {len(chunks)} chunks -> {out_path} ({total} samples)")
    for key in sorted(arrays):
        print(f"    {key}: {arrays[key].shape} {arrays[key].dtype}")

    p = arrays["params"]
    print("\n  Parameter statistics:")
    print(f"    Ip:    [{p[:,0].min():.2e}, {p[:,0].max():.2e}] A   (MASTU [7e5, 1.5e6])")
    print(f"    paxis: [{p[:,1].min():.1f}, {p[:,1].max():.1f}] Pa (MASTU [4e4, 8e4])")
    print(f"    fvac:  [{p[:,2].min():.2f}, {p[:,2].max():.2f}]     (MASTU [0.4, 0.9])")
    print(f"    alpha_m: [{p[:,3].min():.3f}, {p[:,3].max():.3f}] / "
          f"alpha_n: [{p[:,4].min():.3f}, {p[:,4].max():.3f}]")
    x = arrays["x_coords"]
    print(f"    X-pt R: [{x[:,0].min():.3f}, {x[:,0].max():.3f}] m")
    if "anchor" in arrays:  # limiter 无锚点约束 → 无该键（同 v6 merge）
        a = arrays["anchor"]
        print(f"    anchor R: [{a[:,0].min():.3f}, {a[:,0].max():.3f}] m")
    c = arrays["config"]
    print("    config: " + " / ".join(f"{name} {int((c == code).sum())}"
                                      for name, code in g5.CONFIG_CODES.items()))

    # ---- data_gspack2_v2 质量指标汇总（新门验收依据） ----
    gt = arrays["gs_true"]
    print("\n  Quality gates (data_gspack2_v2):")
    print(f"    gs_true:          mean {np.nanmean(gt):.3f}, p95 "
          f"{np.nanpercentile(gt, 95):.3f}, max {np.nanmax(gt):.3f} "
          f"(gate <= {GS_TRUE_MAX})")
    mr = arrays["midplane_ratio"]
    print(f"    midplane_ratio:   mean {np.nanmean(mr):.3f}, min "
          f"{np.nanmin(mr):.3f} (SN gate >= {SN_MIDPLANE_RATIO_MIN}; "
          f"<0 count {int((mr < 0).sum())})")
    zr = arrays["zaxis_ratio"]
    print(f"    zaxis_ratio:      mean {np.nanmean(zr):.3f}, max "
          f"{np.nanmax(zr):.3f} (SN gate <= {SN_ZAXIS_RATIO_MAX})")

    if "isoflux_res" in arrays:
        res = np.abs(arrays["isoflux_res"]).max(axis=1)
        core = arrays["axes"][:, 3] - arrays["axes"][:, 2]
        norm = res / np.maximum(core, 1e-30)
        print("\n  Constraint diagnostics (--save-constraint-diag):")
        print(f"    isoflux res |psi(Xpt)-psi(anchor)|/core: mean {norm.mean():.3f}, "
              f"max {norm.max():.3f} (threshold 0.35)")
        n_actual = arrays["xpts_actual"].shape[1]
        if n_actual == 0:
            print("    X-pt |actual - target|: skipped (limiter: xpts_actual empty)")
        else:
            # SN/雪点: x_coords [loR,loZ,upR,upZ]（SN up=(0,0) 占位 → 只配对 lo）
            if n_actual == 1:
                dev = np.hypot(arrays["xpts_actual"][:, 0, 0] - x[:, 0],
                               arrays["xpts_actual"][:, 0, 1] - x[:, 1])
            else:
                # gspack 的 xpt 是 (r, z) 二元组（无 psi）→ xpts_actual (N,2,2)
                dev = np.hypot(arrays["xpts_actual"][:, :, 0] - x.reshape(-1, 2, 2)[:, :, 0],
                               arrays["xpts_actual"][:, :, 1] - x.reshape(-1, 2, 2)[:, :, 1]).max(axis=1)
            print(f"    X-pt |actual - target| max: mean {dev.mean():.4f} m, "
                  f"p95 {np.percentile(dev, 95):.4f} m, max {dev.max():.4f} m "
                  f"(threshold 0.10)")
        print(f"    n_iter: [{arrays['n_iter'].min():.0f}, "
              f"{arrays['n_iter'].max():.0f}], psi_relchange_final max "
              f"{arrays['psi_relchange_final'].max():.3e} (rtol {RTOL})")


def generate(out_dir: str, split: str, n_samples: int, seed: int,
             chunk_size: int, n_jobs: int, alpha_sampling: bool = False,
             max_retries: int = 5, xpt_jitter: float = 0.06,
             isoflux_sampling: bool = True, xpt_r0: float | None = None,
             xpt_z0: float | None = None, xpt_jitter_z: float | None = None,
             anchor_midplane: bool = True, max_isoflux_residual: float | None = None,
             max_xpt_deviation: float | None = None,
             min_anchor_xpt_dist: float | None = None, require_wall: bool = False,
             coil_margin: float = 0.05, min_core_depth: float | None = None,
             save_constraint_diag: bool = False,
             config: str = "dn", machine: str = "mastu_g3",
             order: int = ORDER, method: str = METHOD,
             sn_midplane_ratio_min: float | None = SN_MIDPLANE_RATIO_MIN,
             sn_zaxis_ratio_max: float | None = SN_ZAXIS_RATIO_MAX,
             max_gs_true: float | None = GS_TRUE_MAX,
             max_snow_xpt_dev: float | None = SNOW_XPT_DEV_MAX) -> None:
    """分块可续跑地生成一个 split。"""
    cfg = build_cfg_g3(machine, config, xpt_jitter, xpt_jitter_z,
                       isoflux_sampling, anchor_midplane, coil_margin,
                       max_isoflux_residual, max_xpt_deviation,
                       min_anchor_xpt_dist, require_wall, min_core_depth,
                       save_constraint_diag, xpt_r0, xpt_z0,
                       order, method,
                       sn_midplane_ratio_min, sn_zaxis_ratio_max,
                       max_gs_true, max_snow_xpt_dev)

    if not alpha_sampling:
        print("WARNING: --alpha-sampling 未开启 → params 3 通道（Ip/paxis/fvac），"
              "输入 17 通道，与 data_v6 的 21ch（5 params + 14 coils）不同构。"
              "data_gspack2_v2 请用 --alpha-sampling（与 data_v6 对齐）。")

    out_dir = Path(out_dir)
    chunk_dir = out_dir / split
    chunk_dir.mkdir(parents=True, exist_ok=True)

    eq_tmp = gspack.Equilibrium(cfg["machine_factory"](),
                                Rmin=G3_RMIN, Rmax=G3_RMAX,
                                Zmin=G3_ZMIN, Zmax=G3_ZMAX,
                                nx=cfg["nx"], ny=cfg["ny"],
                                order=cfg["order"], method=cfg["method"])
    R_GLOBAL = eq_tmp.R.astype(np.float32)
    Z_GLOBAL = eq_tmp.Z.astype(np.float32)

    rng = np.random.default_rng(seed)
    n_chunks = int(np.ceil(n_samples / chunk_size))

    print(f"\n{'='*70}")
    print(f"  {config.upper()} dataset generation | machine={machine} | "
          f"split={split} | n={n_samples} | seed={seed}")
    print(f"  grid {NX}x{NY}, R [{G3_RMIN},{G3_RMAX}], Z [{G3_ZMIN},{G3_ZMAX}], "
          f"order={order} method={method} (gspack v{gspack.__version__})")
    if cfg["kind"] == "snowflake":
        print(f"  snowflake pts ({cfg['snow_r0']},+-{cfg['snow_z0']}) "
              f"R+-{cfg['snow_jitter_r']} / Z+-{cfg['snow_jitter_z']} m, "
              f"weights {cfg['weights']}, 2nd thresh {cfg['second_thresh']}, "
              f"gamma={g5.SNOWFLAKE_GAMMA}, maxits={g5.SNOWFLAKE_MAXITS}")
    elif cfg["kind"] == "limiter":
        print(f"  limiter: axis R~U{cfg['axis_r_range']}, two-step method "
              f"(flood), gamma={g5.LIMITER_GAMMA}, maxits={g5.LIMITER_MAXITS}")
    else:
        print(f"  X-pts ({cfg['xpt_r0']},+-{cfg['xpt_z0']}) R+-{xpt_jitter} / "
              f"Z+-{xpt_jitter_z} m, isoflux->sampled, gamma={GAMMA}, "
              f"maxits={MAXITS}")
    gates = []
    if cfg["sn_midplane_ratio_min"] is not None:
        gates.append(f"SN midplane_ratio>={cfg['sn_midplane_ratio_min']}")
    if cfg["sn_zaxis_ratio_max"] is not None:
        gates.append(f"SN zaxis_ratio<={cfg['sn_zaxis_ratio_max']}")
    if cfg["max_gs_true"] is not None:
        gates.append(f"gs_true<={cfg['max_gs_true']}")
    if cfg["max_snow_xpt_dev"] is not None:
        gates.append(f"snow xpt_dev<={cfg['max_snow_xpt_dev']}")
    if gates:
        print(f"  quality gates (data_gspack2_v2): {', '.join(gates)}")
    if save_constraint_diag:
        print("  saving constraint diagnostics (--save-constraint-diag)")
    print(f"{'='*70}")

    from joblib import Parallel, delayed

    total_accepted, total_solves = 0, 0
    for ci in trange(n_chunks, desc=f"{split} chunks"):
        chunk_path = chunk_dir / f"chunk_{ci:03d}.npz"
        if chunk_path.exists():
            with np.load(chunk_path) as d:
                if int(np.unique(d["config"])[0]) != g5.CONFIG_CODES[cfg["config_name"]]:
                    raise RuntimeError(
                        f"chunk {ci:03d} belongs to config "
                        f"{int(np.unique(d['config'])[0])} != "
                        f"{cfg['config_name']} — split dir polluted; "
                        f"use per-config out-dir")
                total_accepted += d["psi_total"].shape[0]
            print(f"  chunk {ci:03d} exists, skipping (resume)")
            continue

        i0 = ci * chunk_size
        idx = range(i0, min(i0 + chunk_size, n_samples))
        chunk_params = [g5.sample_params(rng, alpha=alpha_sampling,
                                         ranges=cfg["param_ranges"])
                        for _ in idx]

        t0 = time.perf_counter()
        # batch_size=1：跑满 maxits 的慢样本不拖累同批其他 worker（g2 教训）
        results = Parallel(n_jobs=n_jobs, verbose=0, batch_size=1)(
            delayed(_solve_with_retry)(
                (p, seed * 100_000 + i, cfg), max_retries=max_retries)
            for i, p in zip(idx, chunk_params)
        )
        dt = time.perf_counter() - t0

        valid = [r for r in results if r is not None]
        total_accepted += len(valid)
        total_solves += len(results)
        if not valid:
            raise RuntimeError(
                f"chunk {ci:03d}: 0/{len(results)} accepted — adjust sampling "
                f"ranges / acceptance thresholds")
        save_chunk(chunk_dir, ci, valid, R_GLOBAL, Z_GLOBAL)
        print(f"  chunk {ci:03d}: {len(valid)}/{len(results)} accepted, "
              f"{dt:.1f}s ({dt/max(len(results),1):.2f}s/solve)")

    print(f"\n  {split}: {total_accepted}/{total_solves} accepted")
    print(f"  chunks saved to {chunk_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="data_gspack2_v2 dataset generation (gspack2_TRAE MASTU_simple "
                    "五配置高质量数据, exp105 对照版)")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--n-samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--out-dir", default="dn_fno_2608/data_gspack2_v2")
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--n-jobs", type=int, default=-1,
                        help="并行 worker 数（默认 -1 = 全部核；建议 ≤16 留系统余量）")
    parser.add_argument("--alpha-sampling", action="store_true",
                        help="sample alpha_m/alpha_n (data_v6 口径; 默认关)")
    parser.add_argument("--max-retries", type=int, default=5,
                        help="solve attempts per sample with full parameter "
                             "resampling on rejection (v6 用 20)")
    parser.add_argument("--xpt-jitter", type=float, default=0.06)
    parser.add_argument("--isoflux-sampling", action="store_true",
                        help="sample the isoflux anchor and save per-sample "
                             "anchor (data_v6 口径; 默认关=固定 (1.5,0.0))")
    parser.add_argument("--xpt-r0", type=float, default=None)
    parser.add_argument("--xpt-z0", type=float, default=None)
    parser.add_argument("--xpt-jitter-z", type=float, default=None)
    parser.add_argument("--anchor-midplane", action="store_true",
                        help="anchor = (R, 0.0), R~U[1.20,1.45] (data_v6 口径)")
    parser.add_argument("--max-isoflux-residual", type=float, default=None)
    parser.add_argument("--max-xpt-deviation", type=float, default=None)
    parser.add_argument("--min-anchor-xpt-dist", type=float, default=None)
    parser.add_argument("--require-wall", action="store_true")
    parser.add_argument("--coil-margin", type=float, default=0.05)
    parser.add_argument("--min-core-depth", type=float, default=None)
    parser.add_argument("--save-constraint-diag", action="store_true")
    parser.add_argument("--merge", action="store_true")
    # ---- data_gspack2_v2 求解与质量门 ----
    parser.add_argument("--order", type=int, default=ORDER,
                        help="FDM order (default 2 = freegs 同阶; 4 = 探针对照)")
    parser.add_argument("--method", type=str, default=METHOD, choices=["lu", "auto"],
                        help="sparse solver (default lu; auto = AMG on 129^2)")
    parser.add_argument("--sn-midplane-ratio-min", type=float, default=None,
                        help="SN midplane_ratio gate (default = data_v6_clean "
                             "口径扩展 0.05; None = 报告不判定)")
    parser.add_argument("--sn-zaxis-ratio-max", type=float, default=None,
                        help="SN |Z_axis|/|Z_lo| gate (default 0.5; None = 报告)")
    parser.add_argument("--max-gs-true", type=float, default=None,
                        help="gs_residual_ratio gate (default 15 = v6_clean 阈值; "
                             "None = 报告; limiter 默认报告)")
    parser.add_argument("--max-snow-xpt-dev", type=float, default=None,
                        help="snowflake xpt_dev gate (default 0.15 = v6_clean "
                             "snow_double 阈值; None = 报告)")
    parser.add_argument("--config", choices=["dn", "sn", "snow_single",
                                             "snow_double", "limiter"], default="dn")
    parser.add_argument("--machine", choices=["mastu_g3"], default="mastu_g3",
                        help="mastu_g3 = gspack 版 MASTU_simple 26 物理线圈 + "
                             "14 控制单元（唯一）")
    args = parser.parse_args()

    if args.merge:
        merge(args.out_dir, args.split)
    else:
        generate(args.out_dir, args.split, args.n_samples, args.seed,
                 args.chunk_size, args.n_jobs, args.alpha_sampling,
                 args.max_retries, args.xpt_jitter, args.isoflux_sampling,
                 args.xpt_r0, args.xpt_z0, args.xpt_jitter_z,
                 args.anchor_midplane, args.max_isoflux_residual,
                 args.max_xpt_deviation, args.min_anchor_xpt_dist,
                 args.require_wall, args.coil_margin, args.min_core_depth,
                 args.save_constraint_diag, args.config, args.machine,
                 args.order, args.method,
                 args.sn_midplane_ratio_min, args.sn_zaxis_ratio_max,
                 args.max_gs_true, args.max_snow_xpt_dev)


if __name__ == "__main__":
    main()
