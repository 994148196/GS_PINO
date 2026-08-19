"""Re-evaluate M1's ckpt with the TRAINING-TIME basis semantics.

M1 was trained with the pre-fix `_bspline_bases` (eps = 1e-10*span rounds to
0 in float32, so the spline contribution is exactly 0 on the R/Z boundary
ring |R|=|Z|=1). The weights were fit to THAT function; evaluating them with
the fixed basis (eps=1e-6) yields a function that was never trained (24.8%
rel L2 vs 20.9% as-trained). This wrapper restores the original basis so the
recorded metrics.json match what was actually trained.

Usage:
  PY _eval_m1_astrained.py --test-data T --out-dir O
"""
import sys

import numpy as np
import torch

import gs_pino_kan_2608.model_kan as M


def _old_bspline_bases(x, knots, degree):
    """Pre-fix basis: eps 1e-10*span -> float32 rounding leaves x=b unclamped
    -> basis (and hence the whole spline part) vanishes at |x| = 1 exactly."""
    k = knots
    eps = 1e-10 * (k[-1] - k[0])
    xc = torch.clamp(x, min=k[0], max=k[-1] - eps)
    b0 = torch.zeros(x.shape + (len(k) - 1,), dtype=x.dtype, device=x.device)
    for i in range(len(k) - 1):
        b0[..., i] = ((xc >= k[i]) & (xc < k[i + 1])).to(x.dtype)
    out = [b0]
    for p in range(1, degree + 1):
        bp = torch.zeros(x.shape + (len(k) - p - 1,), dtype=x.dtype,
                         device=x.device)
        bp_prev = out[-1]
        for i in range(len(k) - p - 1):
            d1 = (k[i + p] - k[i])
            d2 = (k[i + p + 1] - k[i + 1])
            t1 = (x - k[i]) / d1 if d1 > 0 else torch.zeros_like(x)
            t2 = (k[i + p + 1] - x) / d2 if d2 > 0 else torch.zeros_like(x)
            bp[..., i] = t1 * bp_prev[..., i] + t2 * bp_prev[..., i + 1]
        out.append(bp)
    return out


M._bspline_bases = _old_bspline_bases  # _basis_and_derivs looks this up at runtime

if __name__ == "__main__":
    from gs_pino_kan_2608 import evaluate_kan
    args = sys.argv[1:]
    sys.argv = ["evaluate_kan", "--checkpoint",
                "dn_fno_2608/kan/experiments/exp001_kan_v5/model_b19ch_kan_mix/best.pt"] + args
    evaluate_kan.main()
