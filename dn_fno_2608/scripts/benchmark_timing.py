"""Phase 0.3: timing benchmark for DN data generation.

- Runs N sequential solves with random paper-range parameters, full acceptance checks.
- Reports per-solve mean/median/p95 and estimates total wall time for 6000 samples
  with joblib parallel across all cores.

Usage: python benchmark_timing.py [--n 10] [--parallel 24]
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from smoke_dn_solve import build_and_solve, GAMMA, MAXITS, RTOL


def sample_params(rng):
    return {
        "paxis": float(rng.uniform(200.0, 3000.0)),
        "Ip": float(rng.uniform(5e4, 4e5)),
        "fvac": float(rng.uniform(0.5, 3.0)),
    }


def solve_one(i_seed):
    rng = np.random.default_rng(1000 + i_seed)
    p = sample_params(rng)
    try:
        eq, profiles, tokamak, dt, n_xpt = build_and_solve(
            p["paxis"], p["Ip"], p["fvac"], with_isoflux=True, verbose=False)
        ip_err = abs(eq.plasmaCurrent() - p["Ip"]) / p["Ip"]
        ok = (n_xpt >= 2) and (ip_err <= 0.10)
        return dt, ok, eq, profiles
    except Exception as e:
        return None, False, None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--parallel", type=int, default=24)
    args = ap.parse_args()

    print(f"Sequential benchmark: {args.n} solves ...")
    times, n_ok = [], 0
    for i in range(args.n):
        t0 = time.perf_counter()
        dt, ok, _, _ = solve_one(i)
        dt = time.perf_counter() - t0
        if dt is not None:
            times.append(dt)
        n_ok += int(ok)
    times = np.array(times)
    print(f"accepted: {n_ok}/{args.n}")
    print(f"sequential per-solve: mean={times.mean():.3f}s, median={np.median(times):.3f}s, "
          f"p95={np.percentile(times, 95):.3f}s")
    est_seq = np.median(times) * 6000
    est_par = np.median(times) * 6000 / args.parallel
    print(f"6000 samples estimate: sequential ~{est_seq/60:.1f} min, "
          f"parallel({args.parallel} jobs) ~{est_par/60:.1f} min")


if __name__ == "__main__":
    main()
