"""Shared smooth-geometry helpers for the paper prediction figures.

The LCFS / separatrix is drawn from the flux field via the ray-traced
``separatrix_pts`` (identical to the experiment evaluation figures), NOT
from the binarised plasma mask — the mask contour is pixelated and jagged.
Truth geometry comes straight from the dataset (xpts_actual / o_point /
axes[2] = boundary flux); the predicted LCFS is ray-traced from the model's
total flux with its own matched X-points / magnetic axis.
"""
from __future__ import annotations

import numpy as np

from gs_pino_dn_fno_2608.evaluate_dn_fno import (
    match_xpoints_and_axis,
    separatrix_pts,
)


def close_separatrix(pts: np.ndarray, xpts: np.ndarray | None,
                     tol: float = 0.25) -> np.ndarray:
    """Fill NaN ray rows of a ray-traced separatrix (gaps at X-point cusps).

    The LCFS passes through the X-points; the missing arc of the ray-traced
    curve at each X-point cusp is therefore closed exactly by connecting the
    gap ends through the nearest X-point (two straight segments).  Without a
    matching X-point (``xpts`` empty / too far) the gap is bridged linearly.
    Handles a NaN run that wraps around theta = 0.
    """
    pts = np.array(pts, dtype=float)
    n = len(pts)
    nan = np.isnan(pts[:, 0])
    if not nan.any() or nan.all():
        return pts

    # normalize so the first row is valid (unwrap a wrap-around NaN run)
    if nan[0] and nan[-1]:
        s = int(np.argmax(~nan))
        pts = np.roll(pts, -s, axis=0)
        nan = np.roll(nan, -s)

    xpts = np.asarray(xpts, dtype=float).reshape(-1, 2) if xpts is not None \
        and len(xpts) else np.empty((0, 2))
    i = 1
    while i < n:
        if not nan[i]:
            i += 1
            continue
        j = i
        while j < n and nan[j]:
            j += 1
        p0, p1 = pts[i - 1], pts[j % n]          # gap ends (circular)
        xt = None
        if len(xpts):
            kx = int(np.argmin(np.minimum(np.linalg.norm(xpts - p0, axis=1),
                                          np.linalg.norm(xpts - p1, axis=1))))
            if min(np.linalg.norm(xpts[kx] - p0), np.linalg.norm(xpts[kx] - p1)) <= tol:
                xt = xpts[kx]
        k = j - i
        for t in range(k):
            if xt is not None:                   # arc through the X-point
                k0 = k // 2
                w = (t + 1) / (k0 + 1) if t < k0 else (t - k0) / max(k - k0, 1)
                a = p0 if t < k0 else xt
                b = xt if t < k0 else p1
            else:                                # plain linear bridge
                w, a, b = (t + 1) / (k + 1), p0, p1
            pts[(i + t) % n] = (1 - w) * a + w * b
        i = j
    return pts


def sample_geoms(ds, j, psi_tot_t, psi_tot_p):
    """Truth + predicted LCFS curves, X-points and magnetic axis for sample j.

    ``ds`` is a DNPinoDataset (provides xpts_actual / o_point / axes per
    sample). Returns a dict with:
      sep_t  (n_theta, 2)  ray-traced truth LCFS (NaN rows = rays that missed)
      sep_p  (n_theta, 2) | None   predicted LCFS (None if not localizable)
      xpt_t  (n, 2)        truth separatrix X-points
      xpt_p  (n, 2)        predicted X-points (row-aligned subset)
      o_t    (2,)          truth magnetic axis
      o_p    (2,)          predicted magnetic axis
    """
    bnd = float(ds.axes[j][2])                       # LCFS flux level (~0)
    o_t = np.asarray(ds.o_point[j][:2], dtype=float)
    xa = np.asarray(ds.xpts_actual[j])[:, :2]        # (n_xpt, 2), NaN-padded
    xpt_t = xa[np.isfinite(xa[:, 0])] if xa.size else np.empty((0, 2))

    sep_t = close_separatrix(
        separatrix_pts(psi_tot_t, ds.R_phys, ds.Z_phys, bnd, o_t), xpt_t)

    match = match_xpoints_and_axis(psi_tot_p, ds.R_phys, ds.Z_phys, xa, bnd)
    xpt_p = (np.array([p[:2] for p in match["xpt_pred"] if p is not None],
                      dtype=float).reshape(-1, 2)
             if match["xpt_pred"] else np.empty((0, 2)))
    sep_p = None
    if match["bnd_p"] is not None and np.isfinite(match["o_pred"]).all():
        sep_p = close_separatrix(
            separatrix_pts(psi_tot_p, ds.R_phys, ds.Z_phys,
                           match["bnd_p"], match["o_pred"]), xpt_p)
    return dict(sep_t=sep_t, sep_p=sep_p, xpt_t=xpt_t, xpt_p=xpt_p,
                o_t=o_t, o_p=np.asarray(match["o_pred"]))
