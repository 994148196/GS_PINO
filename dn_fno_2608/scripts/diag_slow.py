#!/usr/bin/env python
"""诊断 dn 全量生成为何慢：5 样本 × 全量参数（含 xpt_jitter）+ max_retries=20，
单线程逐次打印每次 solve 的耗时/结果，验证接受率与重试分布。"""
import sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, r"D:\D_F\Fusion\AI\PINN\gspack2_TRAE")
from gs_gspack2_dn_fno_2608 import generate_dn_g3_dataset as g3
from gs_pino_dn_fno_2608 import generate_dn_dataset as g5

cfg = g3.build_cfg_g3(
    machine="mastu_g3", config="dn",
    isoflux_sampling=True, anchor_midplane=True,
    xpt_jitter=0.06, xpt_jitter_z=0.10,
    max_isoflux_residual=0.35, max_xpt_deviation=0.10,
    min_anchor_xpt_dist=0.15, coil_margin=0.05, min_core_depth=0.005,
    save_constraint_diag=True)

rng = np.random.default_rng(123)
params = [g5.sample_params(rng, alpha=True, ranges=cfg["param_ranges"])
          for _ in range(5)]

t_all0 = time.perf_counter()
for i, p in enumerate(params):
    t0 = time.perf_counter()
    for attempt in range(20):
        ta = time.perf_counter()
        r = g3._solve_one((p, 123 * 100_000 + i + attempt * 1_000_000, cfg))
        dt = time.perf_counter() - ta
        if r is not None:
            print(f"sample {i} attempt {attempt}: ACCEPTED {dt:.1f}s "
                  f"n_iter={int(r['n_iter'][0])} "
                  f"rel={r['psi_relchange_final'][0]:.1e} "
                  f"gs_true={r['gs_true'][0]:.1f}", flush=True)
            break
        print(f"sample {i} attempt {attempt}: rejected {dt:.1f}s", flush=True)
    else:
        print(f"sample {i}: ALL 20 REJECTED", flush=True)
    print(f"  sample {i} total {time.perf_counter()-t0:.1f}s", flush=True)
print(f"== 5 samples done in {time.perf_counter()-t_all0:.1f}s wall (single thread) ==")
