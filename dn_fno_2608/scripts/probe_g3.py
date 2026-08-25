#!/usr/bin/env python
"""data_gspack2_v2 探针：原始接受率（max_retries=1）+ 新质量门分布诊断
+ order/method 对照。

用法（在仓库根目录）:
    python dn_fno_2608/scripts/probe_g3.py --config dn --n 80
    python dn_fno_2608/scripts/probe_g3.py --config sn --n 80

输出：JSON（--out，接受率 + 门分布统计）。每次 solve 恰好一次
（max_retries=1）→ 接受率 = 原始采样接受率。

门全关（None）跑原采样，由探针用 result dict 里的质量指标自行判定
（gs_true / midplane_ratio / zaxis_ratio / xpts_actual）——与生成器的
门阈值解耦，直接看分布定阈值。

order/method 对照（--order-check，默认 dn 10 样本 × 4 组）：
  order=2/lu、order=4/lu、order=2/auto、order=4/auto
  ——记录同 seed 下 psi 场 max diff 与耗时，供 S1 全量生成选型。
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, r"D:\D_F\Fusion\AI\PINN\gspack2_TRAE")
from gs_gspack2_dn_fno_2608 import generate_dn_g3_dataset as g3  # noqa: E402
from gs_pino_dn_fno_2608 import generate_dn_dataset as g5  # noqa: E402

CFGS = ["dn", "sn", "snow_single", "snow_double", "limiter"]


def cfg_gates_off(name):
    """门全关（探针模式）：求解层照常（基线门 + 收敛门），新质量门 None。"""
    return g3.build_cfg_g3(
        machine="mastu_g3", config=name,
        isoflux_sampling=True, anchor_midplane=True,
        max_isoflux_residual=0.35, max_xpt_deviation=0.10,
        min_anchor_xpt_dist=0.15, coil_margin=0.05, min_core_depth=0.005,
        save_constraint_diag=True,
        sn_midplane_ratio_min=None, sn_zaxis_ratio_max=None,
        max_gs_true=None, max_snow_xpt_dev=None)


def eval_gates(r, cfg):
    """对接受的样本判定新质量门（阈值 = 生产值），返回 (passes, dict)。"""
    gs_true = float(r["gs_true"][0])
    mid = float(r["midplane_ratio"][0])
    zaxis = float(r["zaxis_ratio"][0])
    checks = {
        "conv": float(r["psi_relchange_final"][0]) <= g3.CONV_GATE_FACTOR * 1e-3,
        "gs_true": gs_true <= g3.GS_TRUE_MAX,
    }
    if cfg["config_name"] == "sn":
        checks["midplane"] = mid >= g3.SN_MIDPLANE_RATIO_MIN
        checks["zaxis"] = zaxis <= g3.SN_ZAXIS_RATIO_MAX
    elif cfg["config_name"] == "snow_double":
        dev = float(np.max(np.hypot(
            r["xpts_actual"][:, 0] - r["x_coords"][:2:2],
            r["xpts_actual"][:, 1] - r["x_coords"][1::2])))
        checks["snow_xpt_dev"] = dev <= g3.SNOW_XPT_DEV_MAX
    return all(checks.values()), checks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="dn", choices=CFGS)
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--n-jobs", type=int, default=16)
    ap.add_argument("--out-dir", default="dn_fno_2608/data_gspack2_v2/_probe")
    ap.add_argument("--order-check", action="store_true",
                    help="order/method 对照（dn 10 样本 × 4 组，~1 min）")
    args = ap.parse_args()

    cfg = cfg_gates_off(args.config)
    rng = np.random.default_rng(args.seed)
    params_list = [g5.sample_params(rng, alpha=True, ranges=cfg["param_ranges"])
                   for _ in range(args.n)]
    jobs = [(p, args.seed * 100_000 + i, cfg) for i, p in enumerate(params_list)]
    t0 = __import__("time").perf_counter()
    results = Parallel(n_jobs=args.n_jobs, batch_size=1)(
        delayed(g3._solve_with_retry)(j, max_retries=1) for j in jobs)
    wall = __import__("time").perf_counter() - t0

    valid = [r for r in results if r is not None]
    n_ok = len(valid)

    stats = {
        "config": args.config, "n": args.n, "n_jobs": args.n_jobs,
        "n_accepted_raw": n_ok, "acceptance_raw": round(n_ok / args.n, 4),
        "wall_min": round(wall / 60, 2),
        "solve_time_s": {"mean": round(float(np.mean(
            np.stack([r["solve_time"] for r in valid]))), 2) if valid else None,
            "max": round(float(np.max(np.stack(
                [r["solve_time"] for r in valid]))), 2) if valid else None},
        "n_iter": {"mean": round(float(np.mean(
            np.stack([r["n_iter"] for r in valid]))), 1) if valid else None,
            "max": int(np.max(np.stack([r["n_iter"] for r in valid]))) if valid else None},
        "psi_relchange_final_max": float(np.max(np.stack(
            [r["psi_relchange_final"] for r in valid]))) if valid else None,
    }

    # ── 新质量门分布（对 raw 接受的样本） ──
    if valid:
        arr = lambda k: np.stack([r[k] for r in valid])  # noqa: E731
        stats["gs_true"] = {"min": float(np.min(arr("gs_true"))),
                            "mean": round(float(np.mean(arr("gs_true"))), 3),
                            "max": float(np.max(arr("gs_true"))),
                            "frac_gt_15": float(np.mean(arr("gs_true") > 15))}
        stats["midplane_ratio"] = {"min": float(np.min(arr("midplane_ratio"))),
                                   "mean": round(float(np.mean(arr("midplane_ratio"))), 3),
                                   "frac_lt_0": float(np.mean(arr("midplane_ratio") < 0)),
                                   "frac_lt_005": float(np.mean(arr("midplane_ratio") < 0.05))}
        stats["zaxis_ratio"] = {"mean": round(float(np.mean(arr("zaxis_ratio"))), 3),
                                "max": float(np.max(arr("zaxis_ratio"))),
                                "frac_gt_05": float(np.mean(arr("zaxis_ratio") > 0.5))}

        # 门后接受率（生产阈值）
        passed = [r for r in valid if eval_gates(r, cfg)[0]]
        stats["n_accepted_gated"] = len(passed)
        stats["acceptance_gated"] = round(len(passed) / args.n, 4)
        stats["gate_fail_breakdown"] = {}
        for r in valid:
            ok, chk = eval_gates(r, cfg)
            if not ok:
                for k, v in chk.items():
                    if not v:
                        stats["gate_fail_breakdown"][k] = stats["gate_fail_breakdown"].get(k, 0) + 1

        if cfg["config_name"] == "sn":
            # 额外：病态率（v6 口径 midplane<0 或 zaxis>|Z_lo| 全量）——门后应为 0
            stats["sn_disease_rate_raw"] = float(np.mean(
                (arr("midplane_ratio") < 0) | (arr("zaxis_ratio") > 1.0)))
            stats["sn_disease_rate_gated"] = float(np.mean(
                (np.stack([r["midplane_ratio"] for r in passed]) < 0)
                | (np.stack([r["zaxis_ratio"] for r in passed]) > 1.0))) if passed else None

    # ── order/method 对照（dn 快配置，10 样本 × 4 组） ──
    if args.order_check:
        om_cfg = cfg_gates_off("dn")
        rng2 = np.random.default_rng(args.seed + 1)
        om_params = [g5.sample_params(rng2, alpha=True, ranges=om_cfg["param_ranges"])
                     for _ in range(10)]
        rows = []
        for order, method in [(2, "lu"), (4, "lu"), (2, "auto"), (4, "auto")]:
            c = cfg_gates_off("dn")
            c["order"], c["method"] = order, method
            jobs = [(p, (args.seed + 1) * 100_000 + i, c)
                    for i, p in enumerate(om_params)]
            rs = Parallel(n_jobs=args.n_jobs, batch_size=1)(
                delayed(g3._solve_with_retry)(j, max_retries=1) for j in jobs)
            rs = [r for r in rs if r is not None]
            t = float(np.mean(np.stack([r["solve_time"] for r in rs]))) if rs else None
            rows.append({"order": order, "method": method, "n_ok": len(rs),
                         "solve_time_mean_s": round(t, 2) if t else None})
        stats["order_method_check"] = rows

    out = Path(args.out_dir) / f"probe_{args.config}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
