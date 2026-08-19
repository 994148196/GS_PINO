#!/usr/bin/env python
"""data_v6 探针：MASTU_simple 上 5 配置（dn/sn/snow_single/snow_double/limiter）原始接受率。

用法（仓库根目录，需要 freegs_snow fork）:
    export PYTHONPATH="D:/D_F/Fusion/AI/PINN/freegs_snow"
    python dn_fno_2608/scripts/probe_v6.py --config snow_single --n 80 --n-jobs 24
    python dn_fno_2608/scripts/probe_v6.py --separability --n 80 --n-jobs 24

输出：JSON（--out；接受率 + 判据分布 + 计时）。每次 solve 恰好一次
（max_retries=1）→ 接受率 = 原始采样接受率。--separability 跑全 5 配置并对
coil_currents 做位形电流可分性判别（≥2 通道 pooled 分离 > 2.0 std 且全部
配对 ≥90% 阈值判别 → exp012 无需 config 通道）。
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from gs_pino_dn_fno_2608 import generate_dn_dataset as g

CONFIGS = ["dn", "sn", "snow_single", "snow_double", "limiter"]
PROBE_FLAGS = dict(
    xpt_jitter=0.06, xpt_jitter_z=0.10,
    isoflux_sampling=True, anchor_midplane=True,
    max_isoflux_residual=0.35, max_xpt_deviation=0.10,
    min_anchor_xpt_dist=0.15, coil_margin=0.05,
    min_core_depth=0.005, save_constraint_diag=True,
)


def probe_one(args):
    """Run the probe for a single (machine, config); returns (cfg, results)."""
    machine, config, n, seed, n_jobs = args
    cfg = g.build_cfg(machine=machine, config=config, **PROBE_FLAGS)
    print(f"probe {machine}+{config}: n={n} seed={seed} "
          f"kind={cfg['kind']} xpt_r0/z0={cfg['xpt_r0']}/{cfg['xpt_z0']} "
          f"anchor R~U{cfg['anchor_r_range']}", flush=True)
    if cfg["kind"] == "snowflake":
        print(f"  snow pts ({cfg['snow_r0']},+-{cfg['snow_z0']}) "
              f"R+-{cfg['snow_jitter_r']} / Z+-{cfg['snow_jitter_z']}, "
              f"weights {cfg['weights']}, 2nd thresh {cfg['second_thresh']}", flush=True)
    elif cfg["kind"] == "limiter":
        print(f"  axis R~U{cfg['axis_r_range']}, two-step, "
              f"gamma={g.LIMITER_GAMMA}, maxits={g.LIMITER_MAXITS}", flush=True)

    rng = np.random.default_rng(seed)
    params_list = [g.sample_params(rng, alpha=True, ranges=cfg["param_ranges"])
                   for _ in range(n)]
    jobs = [(p, seed * 100_000 + i, cfg) for i, p in enumerate(params_list)]
    t0 = time.perf_counter()
    results = Parallel(n_jobs=n_jobs)(
        delayed(g._solve_with_retry)(j, max_retries=1) for j in jobs)
    wall = time.perf_counter() - t0

    valid = [r for r in results if r is not None]
    n_ok = len(valid)
    print(f"  {machine}+{config}: {n_ok}/{n} accepted "
          f"({n_ok / max(n, 1):.1%}), wall {wall:.1f}s", flush=True)

    stats = collate(config, cfg, valid, n, wall)
    return config, stats, valid


def collate(config, cfg, valid, n, wall):
    """Per-config probe stats (accepted samples only, unless noted)."""
    stats = {
        "machine": "mastu_simple", "config": config, "n": n,
        "n_accepted": len(valid), "acceptance": round(len(valid) / max(n, 1), 4),
        "wall_s": round(wall, 1),
    }
    if not valid:
        return stats

    def arr(key):
        return np.stack([r[key] for r in valid])

    cc = arr("coil_currents")
    stats["n_control_coils"] = int(cc.shape[1])
    stats["solve_time_s"] = {
        "mean": round(float(np.mean(arr("solve_time"))), 3),
        "max": round(float(np.max(arr("solve_time"))), 3),
    }
    stats["n_iter"] = {"mean": round(float(np.mean(arr("n_iter"))), 1),
                       "max": int(np.max(arr("n_iter")))}
    stats["core_depth"] = {"mean": round(float(np.mean(arr("axes")[:, 3] - arr("axes")[:, 2])), 5),
                           "min": round(float(np.min(arr("axes")[:, 3] - arr("axes")[:, 2])), 5)}
    p = arr("params")  # saved order: [Ip, paxis, fvac, alpha_m, alpha_n]
    stats["params_hit"] = {k: [round(float(p[:, i].min()), 4),
                               round(float(p[:, i].max()), 4)]
                           for i, k in enumerate(["Ip", "paxis", "fvac", "alpha_m", "alpha_n"])}
    stats["axis_r"] = {"min": round(float(np.min(arr("axes")[:, 0])), 4),
                       "max": round(float(np.max(arr("axes")[:, 0])), 4)}
    # mean coil currents per channel (physics sanity: P61/P62 vertical field etc.)
    stats["coil_currents_mean"] = [round(float(v), 1) for v in cc.mean(axis=0)]
    # data_v6 labels: wall contact (fraction + max excess of core depth) and
    # wall-internal psi_bndry structure (max crossing-edge fraction; the
    # INWALL_SEP_FRAC_TOL gate already rejected everything above 2%)
    stats["wall_contact"] = {
        "n_touch": int(arr("wall_contact").sum()),
        "max_excess": round(float(np.max(arr("wall_contact_excess"))), 4),
    }
    stats["inwall_sep_frac"] = {"max": round(float(np.max(arr("inwall_sep_frac"))), 4)}

    if config in ("dn", "sn"):
        stats["isoflux_res_max"] = round(float(np.max(np.abs(arr("isoflux_res")))), 5)
        stats["xpt_dev"] = round(float(np.max(np.linalg.norm(
            arr("xpts_actual")[:, :, :2]
            - (arr("x_coords")[:, :2].reshape(-1, 1, 2)
               if arr("xpts_actual").shape[1] == 1
               else arr("x_coords").reshape(-1, 2, 2)),
            axis=2))), 4)
        stats["x_coords_r"] = {"lo_min": round(float(np.min(arr("x_coords")[:, 0])), 4),
                               "lo_max": round(float(np.max(arr("x_coords")[:, 0])), 4)}
        stats["anchor_r"] = {"min": round(float(np.min(arr("anchor")[:, 0])), 4),
                             "max": round(float(np.max(arr("anchor")[:, 0])), 4)}
    elif config in ("snow_single", "snow_double"):
        # snowflake_res: (N, 4*n_pts) = [Br, Bz, psi_ZZ, psi_RZ] per point
        sr = np.abs(arr("snowflake_res"))
        first = sr[:, 0::4].max(axis=1) if sr.shape[1] > 1 else sr[:, 0]
        second = sr[:, 2::4].max(axis=1) if sr.shape[1] > 1 else np.zeros(len(sr))
        stats["snow_1st"] = {"mean": round(float(first.mean()), 5),
                             "p95": round(float(np.percentile(first, 95)), 5),
                             "max": round(float(first.max()), 5),
                             "thresh": g.SNOW_1ST_THRESH}
        stats["snow_2nd"] = {"mean": round(float(second.mean()), 6),
                             "p95": round(float(np.percentile(second, 95)), 6),
                             "max": round(float(second.max()), 6),
                             "thresh": cfg["second_thresh"]}
        stats["isoflux_res_max"] = round(float(np.max(np.abs(arr("isoflux_res")))), 5)
        stats["xpt_dev"] = round(float(np.max(np.linalg.norm(
            arr("xpts_actual")[:, :, :2]
            - (arr("x_coords")[:, :2].reshape(-1, 1, 2)
               if arr("xpts_actual").shape[1] == 1
               else arr("x_coords").reshape(-1, 2, 2)),
            axis=2))), 4)
        stats["x_coords_r"] = {"lo_min": round(float(np.min(arr("x_coords")[:, 0])), 4),
                               "lo_max": round(float(np.max(arr("x_coords")[:, 0])), 4)}
    elif config == "limiter":
        rlim, zlim = arr("Rlim"), arr("Zlim")
        stats["Rlim"] = {"min": round(float(np.min(rlim)), 4),
                         "max": round(float(np.max(rlim)), 4)}
        stats["Zlim"] = {"min": round(float(np.min(zlim)), 4),
                         "max": round(float(np.max(zlim)), 4)}
        # contact point must lie inside the computational grid (no spline
        # extrapolation from wall points at |Z| > 2.0 / R > 2.0)
        in_grid = ((rlim >= g.RMIN) & (rlim <= g.RMAX)
                   & (zlim >= g.ZMIN) & (zlim <= g.ZMAX))
        stats["contact_in_grid"] = round(float(np.mean(in_grid)), 4)
        # psi_limit vs psi_bndry self-consistency (== 0 when is_limited).
        # Note: (N,1) - (N,) broadcasts to an (N,N) outer product — flatten
        # the first operand (found via debug: reported 0.067, true max 0.0)
        stats["psi_limit_vs_bndry_max"] = round(
            float(np.max(np.abs(arr("psi_limit")[:, 0] - arr("axes")[:, 2]))), 6)
    return stats


def separability(all_valid, names):
    """Coil-current separability across configs (accepted samples only).

    per-channel: max over config pairs of |mean_a - mean_b| / pooled_std.
    per-pair: best-channel midpoint threshold accuracy.
    Decision (exp011 logic): >= 2 channels with separation > 2.0 AND all
    pairs >= 90% accuracy -> config channel NOT needed (currents identify
    the configuration).
    """
    arrs = {name: np.stack([r["coil_currents"] for r in valid])
            for name, valid in zip(names, all_valid) if valid}
    n_ch = next(iter(arrs.values())).shape[1]
    out = {"n_control_coils": int(n_ch), "channels": {}}
    for c in range(n_ch):
        vals = {k: a[:, c] for k, a in arrs.items()}
        sep = {}
        for i, a in enumerate(vals):
            for b in list(vals)[i + 1:]:
                pooled = np.sqrt((vals[a].std() ** 2 + vals[b].std() ** 2) / 2)
                sep[f"{a}|{b}"] = round(float(abs(vals[a].mean() - vals[b].mean())
                                              / max(pooled, 1e-12)), 3)
        out["channels"][f"ch{c}"] = {"max_sep": max(sep.values()), "per_pair": sep}
    n_strong = sum(1 for ch in out["channels"].values() if ch["max_sep"] > 2.0)
    out["n_channels_sep_gt_2"] = n_strong

    pairs = {}
    for i, a in enumerate(list(arrs)):
        for b in list(arrs)[i + 1:]:
            best_ch, best_acc = None, 0.0
            for c in range(n_ch):
                va, vb = arrs[a][:, c], arrs[b][:, c]
                if va.std() < 1e-9 and vb.std() < 1e-9:
                    continue
                thr = 0.5 * (va.mean() + vb.mean())
                if va.mean() > vb.mean():
                    acc = np.mean([np.mean(va > thr), np.mean(vb < thr)])
                else:
                    acc = np.mean([np.mean(va < thr), np.mean(vb > thr)])
                if acc > best_acc:
                    best_ch, best_acc = c, acc
            pairs[f"{a}|{b}"] = {"best_ch": best_ch,
                                 "accuracy": round(float(best_acc), 4)}
    out["pairs"] = pairs
    out["all_pairs_ge_90"] = all(v["accuracy"] >= 0.90 for v in pairs.values())
    out["decision_no_config_channel"] = bool(
        n_strong >= 2 and out["all_pairs_ge_90"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--machine", default="mastu_simple",
                    choices=["test", "mast", "mastu_simple"])
    ap.add_argument("--config", default="dn", choices=CONFIGS)
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--n-jobs", type=int, default=24)
    ap.add_argument("--out-dir", default="dn_fno_2608/data_v6/_probe")
    ap.add_argument("--separability", action="store_true",
                    help="run all 5 configs and evaluate coil-current "
                         "separability (exp012 input-channel decision)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.separability:
        all_stats, all_valid = {}, []
        for config in CONFIGS:
            name, stats, valid = probe_one(
                (args.machine, config, args.n, args.seed, args.n_jobs))
            all_stats[name] = stats
            all_valid.append(valid)
            with open(out_dir / f"probe_v6_{name}.json", "w", encoding="utf-8") as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)
        sep = separability(all_valid, CONFIGS)
        report = {"machine": args.machine, "n": args.n, "seed": args.seed,
                  "per_config": all_stats, "separability": sep}
        out = out_dir / "probe_v6_separability.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"\n-> {out}")
    else:
        name, stats, _ = probe_one(
            (args.machine, args.config, args.n, args.seed, args.n_jobs))
        out = out_dir / f"probe_v6_{name}.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        print(json.dumps(stats, indent=2, ensure_ascii=False))
        print(f"\n-> {out}")


if __name__ == "__main__":
    main()
