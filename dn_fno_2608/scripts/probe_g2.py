#!/usr/bin/env python
"""data_gspack2_v1 可行区探针：原始接受率（max_retries=1）+ 范围命中统计
+ SN midplane 健康门诊断。

用法（在仓库根目录）:
    python dn_fno_2608/scripts/probe_g2.py --machine mast_g2 --config sn --n 80

输出：JSON（--out，接受率 + 诊断分布）。每次 solve 恰好一次
（max_retries=1）→ 接受率 = 原始采样接受率。

SN midplane 门（g2 新增，v5 无）：
  - 探针以阈值 +10（门恒过）跑原采样，再由探针用存下的 R_mid_in / R_mid_out
    自行判定门（Ro >= R_anchor - 0.05 且 Ri < R_axis，且非退化 (Rmin,Rmax)
    回退）→ 同时报告 raw / effective 接受率与门的拒绝率
  - 目标：effective >= 90%（v5 MAST 为 100%）
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, r"D:\D_F\Fusion\AI\PINN\gspack2_TRAE")
from gs_gspack2_dn_fno_2608 import generate_g2_dataset as g2

SN_MIDPLANE_THRESH = 0.05      # 生产阈值（= 生成器默认）
SN_MIDPLANE_PROBE = 10.0       # 探针用：门恒过


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--machine", default="mast_g2", choices=["mast_g2"])
    ap.add_argument("--config", default="dn", choices=["dn", "sn"])
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--n-jobs", type=int, default=24)
    ap.add_argument("--out", default="dn_fno_2608/data_gspack2_v1/_probe_stats.json")
    args = ap.parse_args()

    # SN：门以 +10 跑（恒过），探针自行判定；DN：门无作用，直接原始接受率
    sn_midplane_thresh = (SN_MIDPLANE_PROBE if args.config == "sn"
                          else g2.SN_MIDPLANE_THRESH)
    cfg = g2.build_cfg_g2(
        machine=args.machine, config=args.config,
        xpt_jitter=0.06, xpt_jitter_z=0.10,
        isoflux_sampling=True, anchor_midplane=True,
        max_isoflux_residual=0.35, max_xpt_deviation=0.10,
        min_anchor_xpt_dist=0.15, coil_margin=0.05,
        min_core_depth=0.005, save_constraint_diag=True,
        sn_midplane_thresh=sn_midplane_thresh,
    )
    print(f"probe {args.machine}+{args.config}: n={args.n} seed={args.seed} "
          f"xpt=({cfg['xpt_r0']},+-{cfg['xpt_z0']}) jitter R/Z "
          f"{cfg['xpt_jitter']}/{cfg['xpt_jitter_z']} "
          f"anchor R~U{cfg['anchor_r_range']}")
    if args.config == "sn":
        print(f"  SN midplane gate: probe-threshold +10 (disabled in solve), "
              f"evaluated by probe at thresh={SN_MIDPLANE_THRESH}")

    rng = np.random.default_rng(args.seed)
    params_list = [g2.g5.sample_params(rng, alpha=True, ranges=cfg["param_ranges"])
                   for _ in range(args.n)]
    jobs = [(p, args.seed * 100_000 + i, cfg) for i, p in enumerate(params_list)]
    results = Parallel(n_jobs=args.n_jobs, batch_size=1)(
        delayed(g2._solve_with_retry)(j, max_retries=1) for j in jobs)

    valid = [r for r in results if r is not None]
    n_ok = len(valid)
    acc = n_ok / args.n

    stats = {
        "machine": args.machine, "config": args.config, "n": args.n,
        "n_accepted_raw": n_ok, "acceptance_raw": round(acc, 4),
    }
    if args.config == "sn":
        # SN：门判定（用探针解存下的中平面交点）
        gate_ok = 0
        for r in valid:
            ri, ro = float(r["R_mid_in"][0]), float(r["R_mid_out"][0])
            anc_r = float(r["anchor"][0])
            axis_r = float(r["axes"][0])
            deg = (ri == g2.G2_RMIN and ro == g2.G2_RMAX)
            if not deg and ro >= anc_r - SN_MIDPLANE_THRESH and ri < axis_r:
                gate_ok += 1
        stats.update({
            "n_gate_fail": n_ok - gate_ok,
            "n_accepted": gate_ok,
            "acceptance": round(gate_ok / args.n, 4),
        })
        passed = [r for r in valid if not (
            (float(r["R_mid_in"][0]) == g2.G2_RMIN
             and float(r["R_mid_out"][0]) == g2.G2_RMAX)
            or float(r["R_mid_out"][0]) < float(r["anchor"][0]) - SN_MIDPLANE_THRESH
            or float(r["R_mid_in"][0]) >= float(r["axes"][0]))]
        if passed:
            out_margin = [float(r["R_mid_out"][0]) - float(r["anchor"][0])
                          for r in passed]
            in_margin = [float(r["axes"][0]) - float(r["R_mid_in"][0])
                         for r in passed]
            stats["midplane_margin_out_anchor"] = {
                "min": round(min(out_margin), 4),
                "mean": round(float(np.mean(out_margin)), 4)}
            stats["midplane_margin_axis_in"] = {
                "min": round(min(in_margin), 4),
                "mean": round(float(np.mean(in_margin)), 4)}
    else:
        stats["n_accepted"] = n_ok
        stats["acceptance"] = round(acc, 4)

    # collate diagnostics of the ACCEPTED samples
    def arr(key):
        return np.stack([r[key] for r in valid])

    stats.update({
        "core_depth": {"mean": float(np.mean(arr("axes")[:, 3] - arr("axes")[:, 2])),
                       "min": float(np.min(arr("axes")[:, 3] - arr("axes")[:, 2]))},
        "isoflux_res_max": float(np.max(np.abs(arr("isoflux_res")))),
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
                           ["Ip", "paxis", "fvac", "alpha_m", "alpha_n"])},
        "n_control_coils": int(arr("coil_currents").shape[1]),
        "n_iter": {"mean": float(np.mean(arr("n_iter"))),
                   "max": float(np.max(arr("n_iter")))},
        "psi_relchange_final_max": float(np.max(arr("psi_relchange_final"))),
    })
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
