#!/usr/bin/env python
"""验证 worker 里 gspack backend 状态：2 样本 n_jobs=2，task 内打印 backend。"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, r"D:\D_F\Fusion\AI\PINN\gspack2_TRAE")
from joblib import Parallel, delayed
import gspack
import gspack.backend as bk
from gs_gspack2_dn_fno_2608 import generate_dn_g3_dataset as g3
from gs_pino_dn_fno_2608 import generate_dn_dataset as g5

print(f"MAIN backend={bk.get_backend()} cupy_ok={bk._state['cupy_ok']}", flush=True)

cfg = g3.build_cfg_g3(machine="mastu_g3", config="dn",
                      isoflux_sampling=True, anchor_midplane=True,
                      max_isoflux_residual=0.35, max_xpt_deviation=0.10,
                      min_anchor_xpt_dist=0.15, coil_margin=0.05,
                      min_core_depth=0.005, save_constraint_diag=True)

rng = np.random.default_rng(123)
params = [g5.sample_params(rng, alpha=True, ranges=cfg["param_ranges"])
          for _ in range(2)]


def solve_and_report(job):
    print(f"WORKER before solve: backend={bk.get_backend()} "
          f"cupy_ok={bk._state['cupy_ok']}", flush=True)
    r = g3._solve_one(job)
    print(f"WORKER after solve: backend={bk.get_backend()} "
          f"cupy_ok={bk._state['cupy_ok']} result={'OK' if r else 'None'}", flush=True)
    return r


jobs = [(p, 123 * 100_000 + i, cfg) for i, p in enumerate(params)]
results = Parallel(n_jobs=2, verbose=0, batch_size=1)(
    delayed(solve_and_report)(j) for j in jobs)
print(f"DONE: {len(results)} results", flush=True)
