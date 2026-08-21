#!/usr/bin/env python
"""data_v6 -> data_v6_clean: 按质量判据过滤 + 候选池补足到原规模。

研究结论（2026-08-21，exp012 stats 关联验证）:
  dn:          gs_true > 15              (GS 残差比阈值; test 4/200)
  sn:          gs_true > 15 或 midplane_ratio < 0
                                         (组合判据; test 27/200, 桶 rel_l2 8.99->7.44%)
  snow_double: X 点记录偏差 > 0.15 m     (test 4/200, 触壁+雪点结构破坏)
  snow_single: 无 (用户拍板: 数据健康, 标注问题不筛)
  limiter:     无

gs_true 口径与 exp012 evaluate 一致:
  gs_residual_ratio(psi_total, R, Z, dpdpsi, FdFdpsi, mask=psi>=psi_bndry)。

用法 (仓库根目录, 需 PYTHONPATH 含 src):
  python dn_fno_2608/scripts/filter_v6.py score
      只计算各 (cfg, split) 的判据统计, 写 data_v6_clean/scores.json, 不写数据
  python dn_fno_2608/scripts/filter_v6.py compose [--no-topup]
      构建 data_v6_clean/{cfg}/{split}.npz:
        原健康子集 (与 data_v6 逐样本一致) + topup 候选池健康样本 (取前 need 个)
      规模核对 train 500 / val 100 / test 200; 写 manifest.json
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from gs_pino_dn_fno_2608.evaluate_dn_fno import gs_residual_ratio  # noqa: E402

BASE = ROOT / "dn_fno_2608" / "data_v6"
CLEAN = ROOT / "dn_fno_2608" / "data_v6_clean"
TOPUP = ROOT / "dn_fno_2608" / "data_v6_topup"

CFGS = ["dn", "sn", "snow_single", "snow_double", "limiter"]
SPLIT_N = {"train": 500, "val": 100, "test": 200}
XPT_DEV_TOL = 0.15  # m, snow_double 雪点/记录 X 点偏差

# ---------------- predicates ----------------

def gs_true(psi, R, Z, dpdpsi, fdFdpsi, psi_bndry):
    """GS 残差比 (evaluate_dn_fno 同款; mask = psi>=psi_bndry)"""
    mask = (psi >= psi_bndry).astype(np.float32)
    return gs_residual_ratio(psi, R, Z, dpdpsi, fdFdpsi, mask)

def midplane_ratio(psi, R, Z, axes):
    """中平面健康度 = (psi(R_axis, Z~0) - psi_bndry) / (psi_axis - psi_bndry)"""
    r_ax, z_ax, psi_bndry, psi_ax = axes[0], axes[1], axes[2], axes[3]
    j0 = np.argmin(np.abs(Z[0, :]))
    psi_mid = np.interp(r_ax, R[:, j0], psi[:, j0])
    core = max(psi_ax - psi_bndry, 1e-12)
    return (psi_mid - psi_bndry) / core

def xpt_dev_max(xpts_actual, x_coords):
    """记录 X 点 vs 目标 (x_coords = [loR, loZ, upR, upZ] 或雪点+占位),
    按生成器贪心配对顺序逐目标取偏差的 max (NaN/空行跳过)"""
    devs = []
    xp = xpts_actual[0]  # (n_xpt, 3), 行顺序 = 配对顺序
    tgt = x_coords.reshape(2, 2)
    for k, x in enumerate(xp):
        if k >= len(tgt) or np.isnan(x).any():
            continue
        devs.append(np.hypot(x[0] - tgt[k, 0], x[1] - tgt[k, 1]))
    return max(devs) if devs else np.nan

def criteria_for(cfg):
    return {
        "dn":          [("gs_true",        lambda v: v > 15)],
        "sn":          [("gs_true",        lambda v: v > 15),
                        ("midplane_ratio", lambda v: v < 0)],
        "snow_single": [],
        "snow_double": [("xpt_dev",        lambda v: v > XPT_DEV_TOL)],
        "limiter":     [],
    }[cfg]

# ---------------- scoring ----------------

def score_sample(d, i, cfg):
    """返回 (is_bad, 判据值 dict)"""
    psi = d["psi_total"][i]
    R, Z = d["R"], d["Z"]
    vals = {}
    bad = False
    for name, cond in criteria_for(cfg):
        if name == "gs_true":
            v = gs_true(psi, R, Z, d["dpdpsi"][i], d["FdFdpsi"][i], d["axes"][i, 2])
        elif name == "midplane_ratio":
            v = midplane_ratio(psi, R, Z, d["axes"][i])
        elif name == "xpt_dev":
            v = xpt_dev_max(d["xpts_actual"][i:i + 1], d["x_coords"][i])
        else:
            v = np.nan
        vals[name] = float(v)
        if cond(v):
            bad = True
    return bad, vals

def score_file(cfg, split, src=BASE):
    """一次性解压全部数组后逐样本打分 (np.load 惰性解压 -> 避免重复 IO)"""
    with np.load(src / cfg / f"{split}.npz") as lz:
        d = {k: lz[k] for k in lz.files}
    N = d["params"].shape[0]
    bad_idx, vals = [], []
    for i in range(N):
        b, v = score_sample(d, i, cfg)
        if b:
            bad_idx.append(i)
        vals.append(v)
    return d, N, bad_idx, vals

# ---------------- compose ----------------

def row_filter(d, keep):
    """按行过滤 npz 全部 keys (R/Z 共享网格, 不选行)"""
    out = {}
    for k in d:
        a = d[k]
        if a.ndim >= 1 and a.shape[0] == d["params"].shape[0]:
            out[k] = a[keep]
        else:
            out[k] = a
    return out

def write_npz(path, d):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **d)  # 与 data_v6 一致 (merge 用 savez 不压缩)

def compose(args):
    topup = not args.no_topup
    manifest = {}
    for cfg in CFGS:
        for split in SPLIT_N:
            target = SPLIT_N[split]
            d, N, bad_idx, vals = score_file(cfg, split)
            keep = np.ones(N, bool)
            keep[bad_idx] = False
            n_healthy = int(keep.sum())
            removed = bad_idx

            src_rows = np.arange(N)[keep]
            rows = src_rows.copy()
            topup_rows, topup_source = [], []

            if topup and n_healthy < target:
                need = target - n_healthy
                td, tn, tbad, tvals = score_file(cfg, split, src=TOPUP)
                tkeep = np.ones(tn, bool)
                tkeep[tbad] = False
                t_idx = np.arange(tn)[tkeep][:need]
                if len(t_idx) < need:
                    raise SystemExit(
                        f"[filter_v6] {cfg}/{split}: 候选池健康样本不足 "
                        f"({len(t_idx)}/{need}), 需加大 topup 池")
                rows = np.concatenate([src_rows, t_idx])
                topup_rows = [int(x) for x in t_idx]
                topup_source = f"topup/{cfg}/{split}"

            merged = row_filter(d, rows)
            write_npz(CLEAN / cfg / f"{split}.npz", merged)
            manifest[f"{cfg}/{split}"] = {
                "target": target, "n_orig": N, "n_healthy": n_healthy,
                "removed_idx": removed,
                "topup_n": len(topup_rows), "topup_rows": topup_rows,
                "topup_source": topup_source, "n_final": len(rows),
                "criteria_values": vals,
            }
            if len(rows) != target:
                print(f"  WARN {cfg}/{split}: 最终 {len(rows)} != target {target}")
    (CLEAN / "manifest.json").write_text(
        json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
    n_ok = sum(1 for v in manifest.values() if v["n_final"] == v["target"])
    print(f"[filter_v6] compose done: {n_ok}/{len(manifest)} files at target size")

def score_all():
    scores = {}
    for cfg in CFGS:
        for split in SPLIT_N:
            _, N, bad_idx, _ = score_file(cfg, split)
            scores[f"{cfg}/{split}"] = {"n": N, "removed": bad_idx,
                                        "n_removed": len(bad_idx)}
            print(f"  {cfg}/{split}: {len(bad_idx)}/{N} removed"
                  + (f" idx={bad_idx[:12]}..." if bad_idx else ""))
    CLEAN.mkdir(parents=True, exist_ok=True)
    (CLEAN / "scores.json").write_text(
        json.dumps(scores, indent=1), encoding="utf-8")
    print("[filter_v6] score done -> data_v6_clean/scores.json")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["score", "compose"])
    ap.add_argument("--no-topup", action="store_true")
    a = ap.parse_args()
    if a.cmd == "score":
        score_all()
    else:
        compose(a)
