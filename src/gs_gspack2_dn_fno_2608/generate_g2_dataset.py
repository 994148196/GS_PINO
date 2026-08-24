"""data_gspack2_v1 数据集生成器——gspack2_TRAE（gspack v2.0.0）版 MAST 11 线圈复刻。

与 data_v5 完全同构（26 个 npz 键逐键一致，dn/ sn 子目录，65x65 网格）：
  - 机器：freegs MAST 的 11 个控制线圈（P2U..P6L + P1 solenoid）用 gspack
    Coil / Solenoid 复刻（几何与 data_v5 线圈序一一对应，turns=1、无 wall）
  - 求解：gspack.Equilibrium（内部 von Hagenow 自由边界）+ ConstrainPaxisIp
    + constrain(xpoints/isoflux, gamma=1e-12) + picard.solve（rtol=1e-3,
    maxits=80；DN 通常 9 次迭代，SN 52-57 次）
  - 采样 / 接受约束 / 字段映射：克隆 v5（gs_pino_dn_fno_2608.generate_dn_dataset
    复用 sample_params / sample_xpoints / sample_anchor / _acceptance_checks 等
    freegs 无关部分）；差异见下：
    * gspack eq.psi_bndry = 轴下最大单 X 点（非均值）→ 仍按 v5 口径重算
      （DN = 两 X 点 psi 均值，实测差 ~1e-8）
    * gspack find_critical 的 O 点按距网格中心排序（非 psi）→ 按 v5 位置过滤
      （lo.R < o.R < anchor.R 且 |o.Z| < |lo.Z|）取 max psi 为磁轴，opt 重排后
      传 core_mask；过滤空则拒绝
    * greens 单位响应 = eq._coil_psi_unit[name]（gspack 构造时置 current=1.0
      缓存；Coil = greens(Rc,Zc)·turns，Solenoid = Σ 子绕组 greens）
    * 新增 SN midplane 健康门：innerOuterSeparatrix() 要求外中平面交点
      R_hi >= R_anchor - 0.05 且内交点 R_lo < R_axis（防 SN 上瓣退化）
  - 收敛门：picard.solve 最后相对 psi 变化 > 10*rtol 视为发散拒绝（v5 无显式
    门，其 Ip/X 点检查兜底；此处更严但留 10 倍余量）
  - 确定性：每样本种子 = seed*100_000 + i（重试 + attempt*1_000_000），
    参数重采样 rng = default_rng(i_seed + 7_000_003)——与 v5 逐位一致；
    chunk（500/块，存在则整体跳过=断点续跑）→ merge 按 idx 重读全部 chunk
    重建 {split}.npz 至恰好 N 行（补数据：同种子 --n-samples 更大重跑即可，
    行 0..N0-1 比特级不变，绝不就地追加）

用法（仓库根目录）:
  python -m gs_gspack2_dn_fno_2608.generate_g2_dataset --machine mast_g2 \
      --config dn --split train --n-samples 500 --seed 123 \
      --out-dir dn_fno_2608/data_gspack2_v1 --chunk-size 500 --n-jobs 24 \
      --alpha-sampling --xpt-jitter 0.06 --xpt-jitter-z 0.10 --isoflux-sampling \
      --anchor-midplane --max-isoflux-residual 0.35 --max-xpt-deviation 0.10 \
      --min-anchor-xpt-dist 0.15 --coil-margin 0.05 --min-core-depth 0.005 \
      --save-constraint-diag --max-retries 20
  python -m gs_gspack2_dn_fno_2608.generate_g2_dataset --split train \
      --out-dir dn_fno_2608/data_gspack2_v1 --config dn --merge
  # 补数据（同种子）：--n-samples 1500 重跑 train → 只生成 chunk 500-1499 → merge
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

# gspack2_TRAE 包（不在 pip 环境时经 sys.path 注入；examples 同款做法）
GSPACK2_ROOT = r"D:\D_F\Fusion\AI\PINN\gspack2_TRAE"
if GSPACK2_ROOT not in sys.path:
    sys.path.insert(0, GSPACK2_ROOT)
import gspack
from gspack import critical, picard
import gspack.backend as bk

bk.set_backend("cpu")

# 本仓库 src（复用 v5 生成器的 freegs 无关辅助函数）
_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from gs_pino_dn_fno_2608 import generate_dn_dataset as g5

# ---- 网格与求解参数（与 data_v5 一致；SN 需要 maxits 余量） ----
G2_RMIN, G2_RMAX = 0.1, 2.0
G2_ZMIN, G2_ZMAX = -2.0, 2.0
NX, NY = 65, 65
GAMMA = 1e-12                 # Tikhonov 正则（= v5）
RTOL, MAXITS = 1e-3, 80       # Picard 容差 / 最大迭代（v5: 50；SN 52-57 次）
IP_TOL = 0.10                 # |Ip_sol - Ip_tgt| / Ip_tgt <= 10%
CONV_GATE_FACTOR = 10.0       # 收敛门：rel_change 最后值 <= 10*rtol
SN_MIDPLANE_THRESH = 0.05     # SN 门：外中平面交点 >= R_anchor - 0.05

# 首批/补数据约定：每配置 train 500（seed 123）/ val 500（456）/ test 500（789）
# 补数据 = 同 seed 更大的 --n-samples（机制见模块 docstring 与 data README §9）

# ---- gspack 版 MAST 11 线圈复刻（几何 = freegs MAST, machine.py:1408-1428；
# 顺序 = data_v5 coil_currents/CHANNEL_NAMES_COILS 线圈序） ----
def make_mast_g2() -> gspack.Machine:
    coils = [
        ("P2U", gspack.Coil(0.49, 1.76)),
        ("P2L", gspack.Coil(0.49, -1.76)),
        ("P3U", gspack.Coil(1.1, 1.1)),
        ("P3L", gspack.Coil(1.1, -1.1)),
        ("P4U", gspack.Coil(1.51, 1.095)),
        ("P4L", gspack.Coil(1.51, -1.095)),
        ("P5U", gspack.Coil(1.66, 0.52)),
        ("P5L", gspack.Coil(1.66, -0.52)),
        ("P6U", gspack.Coil(1.5, 0.9)),
        ("P6L", gspack.Coil(1.5, -0.9)),
        ("P1", gspack.Solenoid(0.15, -1.4, 1.4, 100)),
    ]
    return gspack.Machine(coils, wall=None)

MACHINE_FACTORIES_G2 = {"mast_g2": make_mast_g2}

# 与 v5 CONFIG_SPECS 同形：X 点中心 (0.7, ±1.1)、锚点 R~U[1.2,1.6]、参数范围
# MAST_PARAM_RANGES（paxis U[1e3,5e3]、Ip U[3e5,8e5]、fvac U[0.3,0.8]）
CONFIG_SPECS_G2 = {
    ("mast_g2", "dn"): {"xpt_n": 2, "has_wall": False,
                        "param_ranges": g5.MAST_PARAM_RANGES,
                        "anchor_r_range": g5.MAST_ANCHOR_R_RANGE,
                        "xpt_r0": 0.7, "xpt_z0": 1.1},
    ("mast_g2", "sn"): {"xpt_n": 1, "has_wall": False,
                        "param_ranges": g5.MAST_PARAM_RANGES,
                        "anchor_r_range": g5.MAST_ANCHOR_R_RANGE,
                        "xpt_r0": 0.7, "xpt_z0": 1.1},
}

# data_v5 实际 npz 键集（26 键 = 24 stacked + R + Z；已实测 data_v5/dn/train.npz
# 无 v6 的 wall_contact 系列键——本机器无 wall，同样不存）
STACKED_KEYS_G2 = [
    "psi_total", "psi_plasma", "psi_plasma_norm", "psi_coils", "mask",
    "dpdpsi", "FdFdpsi", "greens", "coil_currents", "params", "x_coords",
    "axes", "L", "Beta0", "solve_time", "anchor", "config",
    "xpts_actual", "o_point", "xpt_constraint_res", "isoflux_res",
    "psi_at_constraints", "n_iter", "psi_relchange_final",
]


def build_cfg_g2(machine: str = "mast_g2", config: str = "dn",
                 xpt_jitter: float = g5.XPT_JITTER,
                 xpt_jitter_z: float | None = None,
                 isoflux_sampling: bool = False, anchor_midplane: bool = False,
                 coil_margin: float = 0.05,
                 max_isoflux_residual: float | None = None,
                 max_xpt_deviation: float | None = None,
                 min_anchor_xpt_dist: float | None = None,
                 require_wall: bool = False,
                 min_core_depth: float | None = None,
                 save_constraint_diag: bool = False,
                 xpt_r0: float | None = None, xpt_z0: float | None = None,
                 sn_midplane_thresh: float = SN_MIDPLANE_THRESH) -> dict:
    """(machine, config) -> 求解/接受 cfg（v5 build_cfg 的 gspack 版）。"""
    if xpt_jitter_z is None:
        xpt_jitter_z = xpt_jitter
    spec = CONFIG_SPECS_G2.get((machine, config), {})
    xpt_n = spec.get("xpt_n", 2)
    if xpt_r0 is None:
        xpt_r0 = spec.get("xpt_r0", g5.XPT_R)
    if xpt_z0 is None:
        xpt_z0 = spec.get("xpt_z0", g5.XPT_Z)
    return {
        "xpt_n": xpt_n,
        "kind": None,                       # DN/SN（无 snowflake/limiter 分支）
        "config_name": config,
        "machine": machine,
        "machine_factory": MACHINE_FACTORIES_G2[machine],
        "nx": NX, "ny": NY,
        "has_wall": False,
        "param_ranges": spec.get("param_ranges", g5.MAST_PARAM_RANGES),
        "anchor_r_range": spec.get("anchor_r_range", g5.MAST_ANCHOR_R_RANGE),
        "xpt_r0": xpt_r0, "xpt_z0": xpt_z0,
        "xpt_jitter": xpt_jitter, "xpt_jitter_z": xpt_jitter_z,
        "maxits": MAXITS,
        "sn_midplane_thresh": sn_midplane_thresh,
        "isoflux_sampling": isoflux_sampling, "anchor_midplane": anchor_midplane,
        "coil_margin": coil_margin,
        "max_isoflux_residual": max_isoflux_residual,
        "max_xpt_deviation": max_xpt_deviation,
        "min_anchor_xpt_dist": min_anchor_xpt_dist,
        "require_wall": False,              # MAST 无 wall
        "min_core_depth": min_core_depth,
        "save_constraint_diag": save_constraint_diag,
    }


def _solve_one(args: tuple) -> dict | None:
    """求解单个 DN/SN 平衡（gspack 版；模块级供 joblib）。

    args = (params, i_seed, cfg)。接受逻辑 = v5 全量 + 收敛门 + SN midplane 门。
    """
    params, i_seed, cfg = args
    t0 = time.perf_counter()

    paxis, Ip, fvac = params["paxis"], params["Ip"], params["fvac"]
    alpha_m = params.get("alpha_m", 1.0)
    alpha_n = params.get("alpha_n", 2.0)
    rng = np.random.default_rng(i_seed)
    lo, up = g5.sample_xpoints(rng, jitter=cfg["xpt_jitter"],
                               jitter_z=cfg["xpt_jitter_z"],
                               r0=cfg["xpt_r0"], z0=cfg["xpt_z0"])
    anchor = g5.sample_anchor(rng, isoflux=cfg["isoflux_sampling"],
                              midplane=cfg["anchor_midplane"],
                              r_range=cfg["anchor_r_range"])

    try:
        tokamak = cfg["machine_factory"]()
        eq = gspack.Equilibrium(tokamak,
                                Rmin=G2_RMIN, Rmax=G2_RMAX,
                                Zmin=G2_ZMIN, Zmax=G2_ZMAX,
                                nx=cfg["nx"], ny=cfg["ny"],
                                order=2, method="lu")
        # Raxis=1.0：与 freegs ConstrainPaxisIp 默认口径一致（Jtor 的 R/Raxis
        # 参考半径；自由边界下轴心 R 由求解决定，Raxis 只作形状参考）
        profiles = gspack.ConstrainPaxisIp(p_axis=paxis, Ip=Ip, fvac=fvac,
                                           alpha_m=alpha_m, alpha_n=alpha_n,
                                           Raxis=1.0)
        if cfg["xpt_n"] == 1:
            con = gspack.constrain(xpoints=[lo], isoflux=[(*lo, *anchor)],
                                   gamma=GAMMA)
        else:
            con = gspack.constrain(xpoints=[lo, up],
                                   isoflux=[(*lo, *anchor), (*up, *anchor)],
                                   gamma=GAMMA)
        # gspack 版 solve 在 convergenceInfo=True 时打印迭代表——重定向屏蔽
        with redirect_stdout(io.StringIO()):
            conv = picard.solve(eq, profiles, constrain=con,
                                rtol=RTOL, maxits=cfg["maxits"],
                                convergenceInfo=True, verbose=False)

        # 收敛门：最后相对 psi 变化 <= 10*rtol（v5 无显式门；发散解在此拒绝）
        if conv is None or not np.isfinite(float(conv[1][-1])) \
                or float(conv[1][-1]) > CONV_GATE_FACTOR * RTOL:
            return None

        opt, xpt = critical.find_critical(eq.R, eq.Z, eq.psi())
        if not opt:
            return None
        # 磁轴：gspack O 点按距网格中心排序（非 psi）——按 v5 位置过滤取
        # 最高 psi 的物理芯区 O 点（下 X 点 R 与锚点 R 之间、|Z| < |Z_lo|）
        cand = [o for o in opt
                if lo[0] < o[0] < anchor[0] and abs(o[1]) < abs(lo[1])]
        if not cand:
            return None
        axis_pt = max(cand, key=lambda o: o[2])
        opt = [axis_pt] + [o for o in opt if o is not axis_pt]  # opt[0] = 轴

        sep_xpt = [p for p in xpt if float(p[2]) >= eq.psi_bndry - 1e-6]
        acc = g5._acceptance_checks(cfg, eq, profiles, tokamak,
                                    lo, up, anchor, opt, xpt, sep_xpt, Ip)
        if acc is None:
            return None
        (axis_pt, psi_bndry, wall_contact, wall_contact_excess, inwall_frac,
         p_lo, p_up, p_anc) = acc

        # SN midplane 健康门（data_gspack2_v1 新增）：分离面必须在锚点外
        # R_hi >= R_anchor - thresh 且内交点 R_lo < R_axis——防上瓣退化
        # （v6 §8.1 病理：SN 解出上单瓣，分离面不跨中平面）。
        # innerOuterSeparatrix 在 <2 次穿越时回退 (Rmin, Rmax)（= 分离面不
        # 跨中平面，整体在锚点上方/下方）——此时直接拒绝（探测判定）。
        Ri = Ro = 0.0
        if cfg["xpt_n"] == 1:
            Ri, Ro = eq.innerOuterSeparatrix(Z=0.0)
            degenerate = (Ri == G2_RMIN and Ro == G2_RMAX)
            if degenerate or not (
                    Ro >= anchor[0] - cfg["sn_midplane_thresh"] and Ri < axis_pt[0]):
                return None

        diag = cfg["save_constraint_diag"]
        if diag:
            cand = sep_xpt if cfg["xpt_n"] == 1 else xpt
            xpt_xy = np.asarray([(r, z) for r, z, _ in cand], dtype=np.float64)
            remaining = list(range(len(xpt_xy)))
            xpts_actual = []
            for tgt in ([lo] if cfg["xpt_n"] == 1 else [lo, up]):  # greedy 配对
                j = min(remaining, key=lambda j: np.hypot(
                    xpt_xy[j, 0] - tgt[0], xpt_xy[j, 1] - tgt[1]))
                xpts_actual.append(cand[j][:3])
                remaining.remove(j)
            xpts_actual = np.asarray(xpts_actual, dtype=np.float64)
            o_point = np.asarray(axis_pt[:3], dtype=np.float64)

        # ---- 字段提取（与 v5 逐键一致；gspack 方法名差异见注释） ----
        psi_total = eq.psi()                      # = plasma + Σ I_k·G_k
        psi_plasma = eq.plasma_psi
        psi_coils = eq.tokamak.psi_coils(eq.R, eq.Z)   # v5: tokamak.psi(...)
        psi_plasma_norm = (psi_plasma - eq.psi_bndry) / (eq.psi_axis - eq.psi_bndry + 1e-30)
        mask = critical.core_mask(eq.R, eq.Z, psi_total, opt, xpt)
        psi_norm = (psi_total - eq.psi_axis) / (eq.psi_bndry - eq.psi_axis + 1e-30)
        dpdpsi = profiles.pprime(psi_norm)
        fdFdpsi = profiles.ffprime(psi_norm)

        # greens = 单位电流响应（gspack 构造时置 current=1.0 缓存于
        # eq._coil_psi_unit；机器定义顺序 = data_v5 线圈序）
        greens = np.stack([
            eq._coil_psi_unit[nm].astype(np.float32)
            for nm, co in tokamak.coils if co.control])
        coil_currents = np.array(
            [float(co.current) for nm, co in tokamak.coils if co.control],
            dtype=np.float32)

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
            # SN：x_coords 上对保持 (0,0) 占位（= v5 口径）
            "x_coords": np.array([*lo, *up] if cfg["xpt_n"] == 2
                                 else [*lo, 0.0, 0.0], dtype=np.float32),
            "config": np.array([g5.CONFIG_CODES[cfg["config_name"]]],
                               dtype=np.float32),
            "axes": np.array([R_axis, Z_axis, psi_bndry, eq.psi_axis],
                             dtype=np.float32),
            "L": np.array([profiles.L], dtype=np.float32),
            "Beta0": np.array([profiles.Beta0], dtype=np.float32),
            "solve_time": np.array([time.perf_counter() - t0], dtype=np.float32),
        }
        result["anchor"] = np.array(anchor, dtype=np.float32)
        if diag:
            result.update({
                "xpts_actual": xpts_actual.astype(np.float32),
                "o_point": o_point.astype(np.float32),
                "xpt_constraint_res": np.array(
                    [g5._at(eq, lo, "Br"), g5._at(eq, lo, "Bz")]
                    + ([g5._at(eq, up, "Br"), g5._at(eq, up, "Bz")]
                       if cfg["xpt_n"] == 2 else []),
                    dtype=np.float32),
                "isoflux_res": np.array(
                    [p_lo - p_anc]
                    + ([p_up - p_anc] if cfg["xpt_n"] == 2 else []),
                    dtype=np.float32),
                "psi_at_constraints": np.array(
                    [p_lo, p_up, p_anc] if cfg["xpt_n"] == 2
                    else [p_lo, p_anc], dtype=np.float32),
                "n_iter": np.array([len(conv[1])], dtype=np.float32),
                "psi_relchange_final": np.array([conv[1][-1]], dtype=np.float32),
            })
            if cfg["xpt_n"] == 1:
                # SN midplane 门诊断（探针用；不在 STACKED_KEYS_G2 中 →
                # chunk 落盘时被过滤，合并 npz 保持 26 键与 data_v5 一致）
                result["R_mid_in"] = np.array([Ri], dtype=np.float32)
                result["R_mid_out"] = np.array([Ro], dtype=np.float32)
        return result
    except Exception as e:
        print(f"[_solve_one {cfg.get('config_name')}] {type(e).__name__}: {e}",
              file=sys.stderr)
        return None


def _solve_with_retry(args: tuple, max_retries: int = 5,
                      rng: np.random.Generator | None = None) -> dict | None:
    """拒绝时整参重采样重试（= v5；确定性种子逐位一致）。"""
    params, i_seed, cfg = args
    if rng is None:
        rng = np.random.default_rng(i_seed + 7_000_003)
    for attempt in range(max_retries):
        result = _solve_one((params, i_seed + attempt * 1_000_000, cfg))
        if result is not None:
            return result
        params = g5.sample_params(rng, alpha=("alpha_m" in params),
                                  ranges=cfg["param_ranges"])
    return None


def save_chunk(chunk_dir: Path, chunk_idx: int, results: list[dict],
               R_global: np.ndarray, Z_global: np.ndarray) -> Path:
    """存一块独立 .npz（断点续跑检查点）；键集 = data_v5 26 键。"""
    arrays = {key: np.stack([r[key] for r in results])
              for key in STACKED_KEYS_G2 if key in results[0]}
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
            for key in STACKED_KEYS_G2:
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

    # 参数/约束诊断汇总（口径同 v5 merge）
    p = arrays["params"]
    print("\n  Parameter statistics:")
    print(f"    Ip:    [{p[:,0].min():.2e}, {p[:,0].max():.2e}] A   (MAST [3e5, 8e5])")
    print(f"    paxis: [{p[:,1].min():.1f}, {p[:,1].max():.1f}] Pa (MAST [1e3, 5e3])")
    print(f"    fvac:  [{p[:,2].min():.2f}, {p[:,2].max():.2f}]     (MAST [0.3, 0.8])")
    x = arrays["x_coords"]
    print(f"    X-pt R: [{x[:,0].min():.3f}, {x[:,0].max():.3f}] m (MAST 0.7 +- 0.06)")
    if p.shape[1] >= 5:
        print(f"    alpha_m: [{p[:,3].min():.3f}, {p[:,3].max():.3f}] (v2 [1.0, 2.0])")
        print(f"    alpha_n: [{p[:,4].min():.3f}, {p[:,4].max():.3f}] (v2 [1.5, 2.5])")
    a = arrays["anchor"]
    print(f"    anchor R: [{a[:,0].min():.3f}, {a[:,0].max():.3f}] m (MAST [1.2, 1.6])")
    c = arrays["config"]
    print("    config: " + " / ".join(f"{name} {int((c == code).sum())}"
                                      for name, code in g5.CONFIG_CODES.items()))

    if "isoflux_res" in arrays:
        res = np.abs(arrays["isoflux_res"]).max(axis=1)
        core = arrays["axes"][:, 3] - arrays["axes"][:, 2]
        norm = res / np.maximum(core, 1e-30)
        print("\n  Constraint diagnostics (--save-constraint-diag):")
        print(f"    isoflux res |psi(Xpt)-psi(anchor)|/core: mean {norm.mean():.3f}, "
              f"median {np.median(norm):.3f}, p95 {np.percentile(norm, 95):.3f}, "
              f"max {norm.max():.3f} (v5 threshold 0.35)")
        n_actual = arrays["xpts_actual"].shape[1]
        # DN: x_coords (N,4) = [lo_R,lo_Z,up_R,up_Z] → 成对; SN: 仅 lo 对
        # （[0,2] 列选择会拿 up_R 比 Z——曾误报 1.8 m，data 本身正确）
        tgt = arrays["x_coords"][:, :2].reshape(-1, 1, 2) if n_actual == 1 \
            else arrays["x_coords"].reshape(-1, 2, 2)
        dev = np.hypot(arrays["xpts_actual"][:, :, 0] - tgt[:, :, 0],
                       arrays["xpts_actual"][:, :, 1] - tgt[:, :, 1]).max(axis=1)
        print(f"    X-pt |actual - target| max: mean {dev.mean():.4f} m, "
              f"p95 {np.percentile(dev, 95):.4f} m, max {dev.max():.4f} m (v5 threshold 0.10)")
        print(f"    n_iter: [{arrays['n_iter'].min():.0f}, {arrays['n_iter'].max():.0f}], "
              f"psi_relchange_final max {arrays['psi_relchange_final'].max():.3e} "
              f"(rtol {RTOL})")


def generate(out_dir: str, split: str, n_samples: int, seed: int,
             chunk_size: int, n_jobs: int, alpha_sampling: bool = False,
             max_retries: int = 5, xpt_jitter: float = g5.XPT_JITTER,
             isoflux_sampling: bool = False, xpt_r0: float | None = None,
             xpt_z0: float | None = None, xpt_jitter_z: float | None = None,
             anchor_midplane: bool = False, max_isoflux_residual: float | None = None,
             max_xpt_deviation: float | None = None,
             min_anchor_xpt_dist: float | None = None, require_wall: bool = False,
             coil_margin: float = 0.05, min_core_depth: float | None = None,
             save_constraint_diag: bool = False,
             config: str = "dn", machine: str = "mast_g2") -> None:
    """分块可续跑地生成一个 split。"""
    cfg = build_cfg_g2(machine, config, xpt_jitter, xpt_jitter_z,
                       isoflux_sampling, anchor_midplane, coil_margin,
                       max_isoflux_residual, max_xpt_deviation,
                       min_anchor_xpt_dist, require_wall, min_core_depth,
                       save_constraint_diag, xpt_r0, xpt_z0)

    if not alpha_sampling:
        print("WARNING: --alpha-sampling 未开启 → params 3 通道（Ip/paxis/fvac），"
              "输入为 16 通道，与 exp103 的 18 通道（5 params + 11 coils）不同构。"
              "data_gspack2_v1 首批请用 --alpha-sampling（与 data_v5 的 5-param "
              "schema 对齐）。")

    out_dir = Path(out_dir)
    chunk_dir = out_dir / split
    chunk_dir.mkdir(parents=True, exist_ok=True)

    # 共享网格（与每样本一致；gspack 网格 = meshgrid('ij')）
    eq_tmp = gspack.Equilibrium(cfg["machine_factory"](),
                                Rmin=G2_RMIN, Rmax=G2_RMAX,
                                Zmin=G2_ZMIN, Zmax=G2_ZMAX,
                                nx=cfg["nx"], ny=cfg["ny"],
                                order=2, method="lu")
    R_GLOBAL = eq_tmp.R.astype(np.float32)
    Z_GLOBAL = eq_tmp.Z.astype(np.float32)

    rng = np.random.default_rng(seed)
    n_chunks = int(np.ceil(n_samples / chunk_size))

    print(f"\n{'='*70}")
    print(f"  {config.upper()} dataset generation | machine={machine} | "
          f"split={split} | n={n_samples} | seed={seed}")
    print(f"  grid {NX}x{NY}, R [{G2_RMIN},{G2_RMAX}], Z [{G2_ZMIN},{G2_ZMAX}], "
          f"chunk_size={chunk_size} (gspack v{gspack.__version__})")
    print(f"  X-pts ({cfg['xpt_r0']},+-{cfg['xpt_z0']}) R+-{xpt_jitter} / "
          f"Z+-{xpt_jitter_z} m, isoflux->sampled, gamma={GAMMA}, "
          f"maxits={MAXITS}")
    if alpha_sampling:
        print(f"  profile shapes: alpha_m ~ U{g5.ALPHA_RANGES['alpha_m']}, "
              f"alpha_n ~ U{g5.ALPHA_RANGES['alpha_n']} (v5 口径)")
    else:
        print("  profile shapes: alpha_m=1.0, alpha_n=2.0 fixed")
    checks = [f"isoflux_res<={max_isoflux_residual}" if max_isoflux_residual is not None else "",
              f"xpt_dev<={max_xpt_deviation}" if max_xpt_deviation is not None else "",
              f"anc_xpt_dist>={min_anchor_xpt_dist}" if min_anchor_xpt_dist is not None else "",
              f"core>={min_core_depth}" if min_core_depth is not None else ""]
    if any(checks):
        print(f"  acceptance+ : {', '.join([c for c in checks if c])}, "
              f"axis in band(lo,anchor), SN midplane gate (data_gspack2_v1)")
    if save_constraint_diag:
        print("  saving constraint diagnostics (8 fields, --save-constraint-diag)")
    print(f"{'='*70}")

    from joblib import Parallel, delayed

    total_accepted, total_solves = 0, 0
    for ci in trange(n_chunks, desc=f"{split} chunks"):
        chunk_path = chunk_dir / f"chunk_{ci:03d}.npz"
        if chunk_path.exists():
            # 防呆：跳过前校验 chunk 属于本 config（曾因 dn/sn 共用 out-dir 让
            # sn 的 chunk 与 dn 撞号被整块跳过 → merge 出 dn-only 数据）
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
        # batch_size=1：80 迭代收敛的边缘样本（DN 探针实测 3/500 到 maxits）不
        # 拖累同批其他 worker（joblib 默认按 n_jobs 分批、整批等最慢成员 → 慢
        # 样本把整批时间放大 ~6×，实测 500 样本 ~7 min → 改 1 后 ~1-2 min）
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
        description="data_gspack2_v1 dataset generation (gspack2_TRAE MAST replica, "
                    "arXiv:2608.05555 口径)")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--n-samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--out-dir", default="dn_fno_2608/data_gspack2_v1")
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--alpha-sampling", action="store_true",
                        help="sample alpha_m/alpha_n (v5 口径; 默认关=固定 1.0/2.0)")
    parser.add_argument("--max-retries", type=int, default=5,
                        help="solve attempts per sample with full parameter "
                             "resampling on rejection (v5 用 20)")
    parser.add_argument("--xpt-jitter", type=float, default=g5.XPT_JITTER)
    parser.add_argument("--isoflux-sampling", action="store_true",
                        help="sample the isoflux anchor and save per-sample "
                             "anchor (v5 口径; 默认关=固定 (1.5, 0.0))")
    # ---- v5 同款 physical-validity 接受约束 ----
    parser.add_argument("--xpt-r0", type=float, default=None,
                        help="X-point reference R (MAST spec 0.7)")
    parser.add_argument("--xpt-z0", type=float, default=None,
                        help="X-point reference |Z| (MAST spec 1.1)")
    parser.add_argument("--xpt-jitter-z", type=float, default=None,
                        help="X-point |dZ| jitter half-width (default = --xpt-jitter)")
    parser.add_argument("--anchor-midplane", action="store_true",
                        help="anchor = (R, 0.0), R~U[1.2,1.6] (v5 口径)")
    parser.add_argument("--max-isoflux-residual", type=float, default=None)
    parser.add_argument("--max-xpt-deviation", type=float, default=None)
    parser.add_argument("--min-anchor-xpt-dist", type=float, default=None)
    parser.add_argument("--require-wall", action="store_true",
                        help="ignored (MAST 复刻无 wall)")
    parser.add_argument("--coil-margin", type=float, default=0.05)
    parser.add_argument("--min-core-depth", type=float, default=None)
    parser.add_argument("--save-constraint-diag", action="store_true",
                        help="save 8 per-sample constraint-diagnostic fields "
                             "(v5 口径)")
    parser.add_argument("--merge", action="store_true",
                        help="merge existing chunks of --split into a single npz")
    parser.add_argument("--config", choices=["dn", "sn"], default="dn")
    parser.add_argument("--machine", choices=["mast_g2"], default="mast_g2",
                        help="mast_g2 = gspack 版 MAST 11 线圈复刻（唯一）")
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
                 args.save_constraint_diag, args.config, args.machine)


if __name__ == "__main__":
    main()
