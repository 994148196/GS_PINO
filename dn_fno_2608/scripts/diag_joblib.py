#!/usr/bin/env python
"""joblib 卡死定位：verbose=10 + 10 样本 + n_jobs=2，stdout 直连（无管道）。
观察 task 分派/完成日志，判断卡在 solve、回传还是收集。"""
import sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, r"D:\D_F\Fusion\AI\PINN\gspack2_TRAE")
from joblib import Parallel, delayed
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
          for _ in range(10)]
jobs = [(p, 123 * 100_000 + i, cfg) for i, p in enumerate(params)]

print(f"== launching {len(jobs)} tasks, n_jobs=2, verbose=10 ==", flush=True)
t0 = time.perf_counter()
results = Parallel(n_jobs=2, verbose=10, batch_size=1)(
    delayed(g3._solve_with_retry)(j, max_retries=5) for j in jobs)
print(f"== DONE in {time.perf_counter()-t0:.1f}s, {len(results)} results "
      f"({sum(1 for r in results if r is not None)} valid) ==", flush=True)
