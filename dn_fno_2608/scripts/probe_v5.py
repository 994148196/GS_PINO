#!/usr/bin/env python
"""data_v5 可行区探针：原始接受率（max_retries=1）+ 范围命中统计。

用法（在仓库根目录）:
    python dn_fno_2608/scripts/probe_v5.py --machine mast --config sn --n 80

输出：JSON（--out，接受率 + 诊断分布）。每次 solve 恰好一次
（max_retries=1）→ 接受率 = 原始采样接受率。
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from gs_pino_dn_fno_2608 import generate_dn_dataset as g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--machine", default="mast", choices=["test", "mast"])
    ap.add_argument("--config", default="dn", choices=["dn", "sn"])
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--n-jobs", type=int, default=24)
    ap.add_argument("--out", default="dn_fno_2608/data_v5/_probe_stats.json")
    args = ap.parse_args()

    cfg = g.build_cfg(
        machine=args.machine, config=args.config,
        xpt_jitter=0.06, xpt_jitter_z=0.10,
        isoflux_sampling=True, anchor_midplane=True,
        max_isoflux_residual=0.35, max_xpt_deviation=0.10,
        min_anchor_xpt_dist=0.15, coil_margin=0.05,
        min_core_depth=0.005, save_constraint_diag=True,
    )
    print(f"probe {args.machine}+{args.config}: n={args.n} seed={args.seed} "
          f"xpt=({cfg['xpt_r0']},+-{cfg['xpt_z0']}) jitter R/Z "
          f"{cfg['xpt_jitter']}/{cfg['xpt_jitter_z']} "
          f"anchor R~U{cfg['anchor_r_range']}")

    rng = np.random.default_rng(args.seed)
    params_list = [g.sample_params(rng, alpha=True, ranges=cfg["param_ranges"])
                   for _ in range(args.n)]
    jobs = [(p, args.seed * 100_000 + i, cfg) for i, p in enumerate(params_list)]
    results = Parallel(n_jobs=args.n_jobs)(
        delayed(g._solve_with_retry)(j, max_retries=1) for j in jobs)

    valid = [r for r in results if r is not None]
    n_ok = len(valid)
    acc = n_ok / args.n

    # collate diagnostics of the ACCEPTED samples
    def arr(key):
        return np.stack([r[key] for r in valid])

    stats = {
        "machine": args.machine, "config": args.config, "n": args.n,
        "n_accepted": n_ok, "acceptance": round(acc, 4),
        "core_depth": {"mean": float(np.mean(arr("axes")[:, 3] - arr("axes")[:, 2])),
                       "min": float(np.min(arr("axes")[:, 3] - arr("axes")[:, 2]))},
        "isoflux_res_max": float(np.max(np.abs(arr("isoflux_res")))),
        # xpts_actual (N, n_actual, 3) is greedy-paired to [lo, up]. x_coords is
        # (N, 4); SN's up pair is a (0,0) placeholder -> pair lo only there.
        # (naive reshape would treat (0,0) as a target: |lo_actual-(0,0)| ~ 1.3)
        "xpt_dev": float(np.max(np.linalg.norm(
            arr("xpts_actual")[:, :, :2]
            - (arr("x_coords")[:, :2].reshape(-1, 1, 2)
               if arr("xpts_actual").shape[1] == 1
               else arr("x_coords").reshape(-1, 2, 2)),
            axis=2))),
        "x_coords_r": {"lo_min": float(np.min(arr("x_coords")[:, 0])),
                       "lo_max": float(np.max(arr("x_coords")[:, 0])),
                       "lo_z_min": float(np.min(arr("x_coords")[:, 1])),
                       "lo_z_max": float(np.max(arr("x_coords")[:, 1]))},
        "anchor_r": {"min": float(np.min(arr("anchor")[:, 0])),
                     "max": float(np.max(arr("anchor")[:, 0]))},
        "params_hit": {k: [float(np.min(arr("params")[:, i])),
                           float(np.max(arr("params")[:, i]))]
                       for i, k in enumerate(
                           ["paxis", "Ip", "fvac", "alpha_m", "alpha_n"])},
        "n_control_coils": int(arr("coil_currents").shape[1]),
        "n_iter": {"mean": float(np.mean(arr("n_iter"))),
                   "max": float(np.max(arr("n_iter")))},
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
