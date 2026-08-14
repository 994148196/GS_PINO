"""Count rejection reasons for the data_v4 acceptance checks (diagnostic only).

Runs the same per-sample attempt stream as generate_dn_dataset.py (same seeds,
same retry mechanism) but records WHY each attempt is rejected. Use to tune the
v4 thresholds before the full run.

Usage:
  python -m gs_pino_dn_fno_2608.diag_rejections --n-samples 80 --seed 123
"""
from __future__ import annotations

import argparse
import time
from collections import Counter

import numpy as np
from joblib import Parallel, delayed

import freegs
from freegs import boundary, control, critical, jtor

from gs_pino_dn_fno_2608 import generate_dn_dataset as G


def attempt(args):
    params, i_seed, cfg = args
    reasons = []
    paxis, Ip, fvac = params["paxis"], params["Ip"], params["fvac"]
    alpha_m = params.get("alpha_m", 1.0)
    alpha_n = params.get("alpha_n", 2.0)
    rng = np.random.default_rng(i_seed)
    lo, up = G.sample_xpoints(rng, jitter=cfg["xpt_jitter"], jitter_z=cfg["xpt_jitter_z"],
                              r0=cfg["xpt_r0"], z0=cfg["xpt_z0"])
    anchor = G.sample_anchor(rng, isoflux=cfg["isoflux_sampling"],
                             midplane=cfg["anchor_midplane"])
    try:
        tokamak = freegs.machine.TestTokamak()
        eq = freegs.Equilibrium(tokamak=tokamak, Rmin=G.RMIN, Rmax=G.RMAX,
                                Zmin=G.ZMIN, Zmax=G.ZMAX, nx=G.NX, ny=G.NY,
                                boundary=boundary.freeBoundaryHagenow)
        profiles = jtor.ConstrainPaxisIp(eq, paxis=paxis, Ip=Ip, fvac=fvac,
                                         alpha_m=alpha_m, alpha_n=alpha_n)
        constrain = control.constrain(xpoints=[lo, up],
                                      isoflux=[(*lo, *anchor), (*up, *anchor)],
                                      gamma=G.GAMMA)
        try:
            freegs.solve(eq, profiles, constrain, rtol=G.RTOL, maxits=G.MAXITS,
                         show=False)
        except RuntimeError as e:
            reasons.append("solve: " + str(e))
        if not reasons:
            if eq.psi_axis is None or eq.psi_bndry is None:
                reasons.append("psi axis/bndry None")
            ip_err = abs(eq.plasmaCurrent() - Ip) / Ip
            if ip_err > G.IP_TOL:
                reasons.append(f"ip_err {ip_err:.2f}")
            opt, xpt = critical.find_critical(eq.R, eq.Z, eq.psi())
            if len(xpt) < G.MIN_XPTS:
                reasons.append(f"n_xpt {len(xpt)}")
            if not (np.isfinite(profiles.L) and 0.0 < profiles.Beta0 < 1.0):
                reasons.append(f"L/Beta0 ({profiles.L:.0f}, {profiles.Beta0:.3f})")
        if not reasons:
            psi_bndry = 0.5 * (xpt[0][2] + xpt[1][2])
            core = eq.psi_axis - psi_bndry
            if core <= 0 or not np.isfinite(core):
                reasons.append(f"core<=0 ({core})")
            elif cfg["min_core_depth"] is not None and core < cfg["min_core_depth"]:
                reasons.append(f"core depth {core:.5f}")
        if not reasons:
            p_lo, p_up, p_anc = G._at(eq, lo), G._at(eq, up), G._at(eq, anchor)
            if cfg["max_isoflux_residual"] is not None:
                res = max(abs(p_lo - p_anc), abs(p_up - p_anc)) / core
                if res > cfg["max_isoflux_residual"]:
                    reasons.append(f"isoflux res {res:.3f}")
        if not reasons:
            hull_verts = np.asarray(G._control_coil_centers(tokamak), dtype=np.float64)
            wall_verts = np.column_stack([np.asarray(tokamak.wall.R, dtype=np.float64),
                                          np.asarray(tokamak.wall.Z, dtype=np.float64)])
            for name, p in (("lo", lo), ("up", up), ("anchor", anchor)):
                if not G._inside_with_margin(p, hull_verts, cfg["coil_margin"]):
                    reasons.append(f"hull margin {name} {p}")
                elif cfg["require_wall"] and not G._point_in_polygon(p, wall_verts):
                    reasons.append(f"wall {name} {p}")
        if not reasons:
            if not opt:
                reasons.append("no opt")
            elif not G._point_in_triangle(opt[0][:2], (lo, up, anchor)):
                reasons.append(f"axis {opt[0][:2]} not in tri")
        if not reasons and cfg["min_anchor_xpt_dist"] is not None:
            d = min(np.hypot(anchor[0] - lo[0], anchor[1] - lo[1]),
                    np.hypot(anchor[0] - up[0], anchor[1] - up[1]))
            if d < cfg["min_anchor_xpt_dist"]:
                reasons.append(f"anchor-xpt dist {d:.3f}")
        if not reasons and cfg["max_xpt_deviation"] is not None:
            xpt_xy = np.asarray([(r, z) for r, z, _ in xpt], dtype=np.float64)
            for name, tgt in (("lo", lo), ("up", up)):
                d = np.hypot(xpt_xy[:, 0] - tgt[0], xpt_xy[:, 1] - tgt[1]).min()
                if d > cfg["max_xpt_deviation"]:
                    reasons.append(f"xpt deviation {name} {d:.3f}")
        return reasons
    except Exception as e:
        return [f"EXC {type(e).__name__}: {e}"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-samples", type=int, default=80)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--n-jobs", type=int, default=24)
    args = ap.parse_args()

    cfg = {
        "xpt_r0": 1.2, "xpt_z0": 0.6, "xpt_jitter": 0.10, "xpt_jitter_z": 0.15,
        "isoflux_sampling": True, "anchor_midplane": True,
        "coil_margin": 0.05, "max_isoflux_residual": 0.35,
        "max_xpt_deviation": 0.10, "min_anchor_xpt_dist": 0.15,
        "require_wall": True, "min_core_depth": 0.005,
        "save_constraint_diag": True,
    }
    rng = np.random.default_rng(args.seed)
    chunk_params = [G.sample_params(rng, alpha=True) for _ in range(args.n_samples)]

    t0 = time.perf_counter()
    results = Parallel(n_jobs=args.n_jobs, verbose=0)(
        delayed(attempt)((p, args.seed * 100_000 + i, cfg))
        for i, p in enumerate(chunk_params))
    dt = time.perf_counter() - t0

    accepted = sum(1 for r in results if not r)
    reasons = Counter()
    for r in results:
        reasons.update(r)
    n_attempts = len(results) * 20
    print(f"\n{args.n_samples} samples, {n_attempts} attempts, "
          f"accepted {accepted}/{args.n_samples} "
          f"({100*accepted/args.n_samples:.1f}%), wall {dt:.0f}s")
    print(f"\nrejection reasons over {n_attempts} attempts "
          f"(only 1 reason counted per rejected attempt):")
    for reason, n in reasons.most_common():
        print(f"  {n:5d}  {100*n/n_attempts:6.2f}%  {reason[:90]}")


if __name__ == "__main__":
    main()
