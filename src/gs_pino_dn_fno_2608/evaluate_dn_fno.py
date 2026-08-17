"""Evaluation for the arXiv:2608.05555 double-null FNO surrogate.

Metrics (paper Tables I/II + Sec. IV residual diagnostic):
  field-level : rel L2 in the normalized representation (Eq. 7) [mean/std/median/P95],
                physical RMSE in Wb after inverse transform
  geometry    : separatrix mean closest-point distance / Hausdorff / area relative
                error, lower & upper X-point errors, O-point error, |dpsi_bndry|
  GS residual : Delta* psi vs RHS = -mu0 R^2 dp/dpsi - F dF/dpsi (precomputed
                dataset fields, paper Eq. 10-11), normalized over the plasma mask;
                computed for prediction AND freegs truth (baseline comparison)

Geometry v2 (2026-08-17, after user-reported fig3 artifacts):
  - data_v4+ npz carry the true separatrix X-points (xpts_actual), O-point
    (o_point) and boundary flux (axes[2]); the prediction is paired against
    these per-X-point, excluding the vacuum saddle points that freegs
    find_critical reports on wall-less MAST (psi >= psi_bndry - 2e-3 + nearest
    distance <= 20 cm; else a +-35 cm local saddle recovery; else NaN = the
    structure is genuinely not localizable). psi_bndry = mean of the PAIRED
    X-point fluxes (DN) or the single X-point flux (SN).
  - separatrix = radial-ray sampling from the magnetic axis (first psi=level
    crossing per ray) — closed and robust on wall-less MAST where the
    matplotlib closed-contour fallback degrades to domain-wide scatter.
  - paper data/ & data_v2 (no truth fields) keep the legacy find_critical
    path with Z-sign pairing and closed-contour tracing.

Usage:
  python -m gs_pino_dn_fno_2608.evaluate_dn_fno --test-data dn_fno_2608/data/test.npz \
      --checkpoint dn_fno_2608/outputs/fno_n5000_s3/best.pt \
      --out-dir dn_fno_2608/outputs/report
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from gs_pino_dn_fno_2608.data_dn_fno import DNFnoDataset
from gs_pino_dn_fno_2608.model_dn_fno import build_model

MU0 = 4.0 * np.pi * 1e-7

try:
    from freegs import critical as freegs_critical
    HAS_FREEGS = True
except ImportError:
    HAS_FREEGS = False

# contouring with the Agg backend (no display)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------
# GS residual
# --------------------------------------------------------------------------

def lap_star(psi: np.ndarray, R: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """Delta* psi = d2/dR2 - (1/R) d/dR + d2/dZ2, 2nd-order central differences."""
    dR = R[1, 0] - R[0, 0]
    dZ = Z[0, 1] - Z[0, 0]
    d2r = (psi[2:, 1:-1] - 2.0 * psi[1:-1, 1:-1] + psi[:-2, 1:-1]) / (dR**2)
    dr = (psi[2:, 1:-1] - psi[:-2, 1:-1]) / (2.0 * dR)
    d2z = (psi[1:-1, 2:] - 2.0 * psi[1:-1, 1:-1] + psi[1:-1, :-2]) / (dZ**2)
    return d2r - dr / (R[1:-1, 1:-1] + 1e-8) + d2z


def gs_residual_ratio(psi_field: np.ndarray, R: np.ndarray, Z: np.ndarray,
                      dpdpsi: np.ndarray, fdFdpsi: np.ndarray, mask: np.ndarray) -> float:
    """Normalized GS residual over the plasma mask (paper Eq. 10-11)."""
    lap = lap_star(psi_field, R, Z)
    R_c = R[1:-1, 1:-1]
    rhs = -MU0 * R_c**2 * dpdpsi[1:-1, 1:-1] - fdFdpsi[1:-1, 1:-1]
    core = mask[1:-1, 1:-1] > 0.5
    if core.sum() < 10:
        return float("nan")
    return float(np.linalg.norm((lap - rhs)[core]) / np.linalg.norm(rhs[core]))


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------

def separatrix_points(psi: np.ndarray, R: np.ndarray, Z: np.ndarray, level: float) -> np.ndarray:
    """Closed-contour points at the given flux level (Q3+3*IQR filtered).

    All closed contour paths at `level` are collected (the balanced-DN
    separatrix may come out of matplotlib as the two touching lobes); paths with
    area below 5% of the largest closed path are dropped as artifacts; the
    remaining points are filtered once by a distance-from-centroid
    Q3+3*IQR criterion (paper: removes divertor-leg artifacts, applied equally
    to true and predicted fields).

    NOTE: intended for walled machines (TestTokamak) where the separatrix
    contour closes inside the domain. On wall-less MAST the level contour often
    runs off the grid (open path) and this function's fallback degrades to
    domain-wide scatter — data_v5 evaluation uses separatrix_pts() instead.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cs = ax.contour(R, Z, psi, levels=[level])
    # matplotlib >= 3.10 removed ContourSet.collections; allsegs[0] holds the
    # vertex arrays of the single requested level (keep the old path as fallback)
    if hasattr(cs, "allsegs"):
        seg_lists = cs.allsegs[0]
    else:
        seg_lists = [p.vertices for p in cs.collections[0].get_paths()]
    plt.close(fig)

    closed, areas = [], []
    for v in seg_lists:
        if len(v) < 4:
            continue
        if np.allclose(v[0], v[-1], atol=1e-6):  # closed path
            x, y = v[:, 0], v[:, 1]
            area = 0.5 * np.abs(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
            closed.append(v)
            areas.append(area)
    if not closed:
        # fallback: all vertices of the level contour
        pts = np.concatenate(seg_lists)
        return pts
    amax = max(areas)
    keep = [v for v, a in zip(closed, areas) if a >= 0.05 * amax]
    pts = np.concatenate(keep)

    # Q3 + 3*IQR distance filter (one pass, paper: removes leg artifacts)
    centroid = pts.mean(axis=0)
    dist = np.linalg.norm(pts - centroid, axis=1)
    q3, q1 = np.percentile(dist, 75), np.percentile(dist, 25)
    thr = q3 + 3.0 * (q3 - q1)
    return pts[dist <= thr]


def separatrix_pts(psi: np.ndarray, R: np.ndarray, Z: np.ndarray, level: float,
                   axis: np.ndarray, n_theta: int = 180, r_max: float = 3.5,
                   n_s: int = 400) -> np.ndarray:
    """Separatrix points via radial rays from the magnetic axis (N_theta, 2).

    For each of n_theta evenly-spaced directions from `axis` (R, Z), sample psi
    along the ray (bilinear, scipy map_coordinates) and take the FIRST crossing
    of `level` — psi is monotonically decreasing outward from the axis, so the
    first crossing is the core separatrix. Robust on wall-less MAST where the
    contour is not closed inside the grid (the failure mode of
    separatrix_points above); rows with no crossing are NaN.
    """
    from scipy.ndimage import map_coordinates

    axis = np.asarray(axis, dtype=float)
    R1, Z1 = R[:, 0], Z[0, :]
    dR, dZ = R1[1] - R1[0], Z1[1] - Z1[0]
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    dirs = np.stack([np.cos(theta), np.sin(theta)], axis=1)          # (n_theta, 2)
    s = np.linspace(0.02, r_max, n_s)
    pts = axis[None, None, :] + s[None, :, None] * dirs[:, None, :]  # (n_theta, n_s, 2)
    g = np.stack([(pts[..., 0] - R[0, 0]) / dR, (pts[..., 1] - Z[0, 0]) / dZ], axis=-1)
    vals = map_coordinates(psi, g.reshape(-1, 2).T, order=1,
                           mode="nearest").reshape(n_theta, n_s)
    cross = (vals[:, :-1] - level) * (vals[:, 1:] - level) <= 0.0
    out = np.full((n_theta, 2), np.nan)
    for j in range(n_theta):
        if not cross[j].any():
            continue
        i = int(np.argmax(cross[j]))
        a, b = vals[j, i], vals[j, i + 1]
        w = (level - a) / (b - a + 1e-30)
        r = s[i] * (1.0 - w) + s[i + 1] * w
        out[j] = axis + r * dirs[j]
    return out


def _fill_nan_theta(pts: np.ndarray) -> np.ndarray:
    """Interpolate NaN rows (missing ray crossings) by angle — the separatrix
    is closed, so only isolated gaps are expected."""
    bad = np.isnan(pts[:, 0])
    if not bad.any():
        return pts
    good = ~bad
    if good.sum() < 4:
        return pts
    th = np.linspace(0.0, 2.0 * np.pi, len(pts), endpoint=False)
    out = pts.copy()
    for c in (0, 1):
        out[bad, c] = np.interp(th[bad], th[good], pts[good, c])
    return out


def _local_saddles(psi: np.ndarray, R: np.ndarray, Z: np.ndarray,
                   r0: float, z0: float, half: float = 0.35) -> list:
    """Saddle points of `psi` in a (r0,z0) +- half neighborhood (cropped grid).

    FNO output is smooth and the separatrix X-point sits in a flat psi region,
    so the global grid scan of find_critical frequently misses it; the cropped
    sub-grid rescales the search and recovers the saddle. Sub-grids have no
    O-point — silence the freegs "No O points found" chatter.
    """
    from contextlib import redirect_stdout, redirect_stderr

    m = (R >= r0 - half) & (R <= r0 + half) & (Z >= z0 - half) & (Z <= z0 + half)
    rows = np.where(m.any(axis=1))[0]
    cols = np.where(m.any(axis=0))[0]
    if len(rows) < 4 or len(cols) < 4:
        return []
    with open(__import__("os").devnull, "w") as devnull, \
            redirect_stdout(devnull), redirect_stderr(devnull):
        try:
            _, xpts = freegs_critical.find_critical(
                R[np.ix_(rows, cols)], Z[np.ix_(rows, cols)], psi[np.ix_(rows, cols)])
        except Exception:
            return []
    return list(xpts)


def match_xpoints_and_axis(psi_pred: np.ndarray, R: np.ndarray, Z: np.ndarray,
                           xpts_true: np.ndarray, psi_bndry_true: float,
                           anchor: np.ndarray | None = None) -> dict:
    """Pair predicted separatrix X-points / magnetic axis against ground truth.

    data_v4+ datasets store the true separatrix X-points (xpts_actual); freegs
    find_critical additionally reports wall-less vacuum saddle points (MAST:
    5-6 per sample). Each true X-point is paired in three stages:
      1. global candidates with psi >= psi_bndry_true - EPS (vacuum saddles
         excluded) and distance <= MAX_DIST -> healthy pair;
      2. otherwise a local saddle search in a +-0.35 m neighborhood (recovers
         saddles the global grid scan missed on smooth FNO fields); the
         reported distance can exceed MAX_DIST — it is the model's true
         localization error there;
      3. otherwise the pair is None (structure genuinely not localizable ->
         NaN geometry, honest).
    The predicted axis is the highest-psi optimum inside the region bounded by
    the lower X-point and the anchor.
    Returns dict(xpt_pred: list[(r, z, psi)|None] row-aligned with xpts_true,
    o_pred: (r, z), bnd_p: float|None, n_xpt_pred: int).
    """
    EPS = 2e-3       # Wb: pred separatrix X-point psi is within ~1e-4..1e-3
    MAX_DIST = 0.2   # m: healthy localization error is ~1-5 cm
    LOCAL_PSI = 0.02 # Wb: local-recovery candidates must sit near the boundary
    out = {"xpt_pred": [], "o_pred": (np.nan, np.nan), "bnd_p": None, "n_xpt_pred": 0}
    opt_p, xpt_p = freegs_critical.find_critical(R, Z, psi_pred)
    out["n_xpt_pred"] = len(xpt_p)
    if not xpt_p or not opt_p:
        return out

    arr = np.array([[p[0], p[1], p[2]] for p in xpt_p])              # (n, 3)
    xpts_t = np.asarray(xpts_true)[:, :2]
    pairs = []                       # ("global", idx) | ("local", (r,z,psi)) | None
    avail = np.ones(len(arr), dtype=bool)  # each global candidate used once
    for t in xpts_t:
        d = np.linalg.norm(arr[:, :2] - t, axis=1)
        d[(arr[:, 2] < psi_bndry_true - EPS) | ~avail] = np.inf
        j = int(np.argmin(d))
        if d[j] <= MAX_DIST:
            pairs.append(("global", j))
            avail[j] = False
            continue
        # local recovery: saddle present but missed by the global scan
        used_pos = np.array([arr[p[1], :2] for p in pairs
                             if p is not None and p[0] == "global"])
        cand = [p for p in _local_saddles(psi_pred, R, Z, t[0], t[1])
                if p[2] >= psi_bndry_true - LOCAL_PSI
                and (len(used_pos) == 0
                     or np.linalg.norm(used_pos - np.array(p[:2]), axis=1).min() >= 0.05)]
        if cand:
            la = np.array([[p[0], p[1], p[2]] for p in cand])
            jl = int(np.argmin(np.linalg.norm(la[:, :2] - t, axis=1)))
            pairs.append(("local", (float(la[jl, 0]), float(la[jl, 1]), float(la[jl, 2]))))
        else:
            pairs.append(None)

    used_psi = [p[1][2] if p[0] == "local" else arr[p[1], 2]
                for p in pairs if p is not None]
    if used_psi:
        out["bnd_p"] = float(np.mean(used_psi))
    for j in pairs:
        if j is None:
            out["xpt_pred"].append(None)
        elif j[0] == "local":
            out["xpt_pred"].append(j[1])
        else:
            out["xpt_pred"].append((float(arr[j[1], 0]), float(arr[j[1], 1]), float(arr[j[1], 2])))

    # magnetic axis: highest-psi optimum inside (r_lo, r_anc) x |z| < |z_lo|
    i_lo = int(np.argmin(xpts_t[:, 1]))                              # lower X-point
    r_lo, z_lo = xpts_t[i_lo]
    r_anc = float(anchor[0]) if anchor is not None else float(xpts_t[:, 0].max())
    pad = 0.3
    inside = [o for o in opt_p if r_lo - pad < o[0] < r_anc + pad and abs(o[1]) < abs(z_lo) + pad]
    o_p = inside[0] if inside else opt_p[0]
    out["o_pred"] = (float(o_p[0]), float(o_p[1]))
    return out


def pointwise_sep_error(pred_pts: np.ndarray, true_pts: np.ndarray) -> tuple[float, float]:
    """Mean closest-point distance (pred->true) and Hausdorff distance (cm)."""
    d_pt = np.linalg.norm(pred_pts[:, None, :] - true_pts[None, :, :], axis=2)
    mean_err = float(d_pt.min(axis=1).mean())
    hausdorff = max(float(d_pt.min(axis=1).max()), float(d_pt.min(axis=0).max()))
    return 100.0 * mean_err, 100.0 * hausdorff  # cm


def poly_area(pts: np.ndarray) -> float:
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * np.abs(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def geometry_metrics(psi_pred: np.ndarray, psi_true: np.ndarray,
                     R: np.ndarray, Z: np.ndarray,
                     xpts_true: np.ndarray | None = None,
                     o_true: np.ndarray | None = None,
                     psi_bndry_true: float | None = None,
                     anchor: np.ndarray | None = None) -> dict:
    """Paper Table II metrics for one sample (distances in cm, area rel err in %).

    data_v4+ (xpts_true given): true separatrix X-points, O-point and boundary
    flux come from the dataset fields (xpts_actual / o_point / axes[2]); the
    prediction is paired via match_xpoints_and_axis (vacuum saddle points on
    wall-less MAST excluded), the boundary flux is the mean of the PAIRED X-point
    fluxes (DN) or the single X-point flux (SN), and the separatrix is traced
    by radial rays from the axis (wall-less MAST safe).
    Paper data/ & data_v2 (xpts_true=None): legacy find_critical path with
    Z-sign pairing and closed-contour tracing (unchanged).
    """
    out = {"n_xpt_pred": 0, "n_xpt_true": 0}
    if not HAS_FREEGS:
        return out

    opt_p, xpt_p = freegs_critical.find_critical(R, Z, psi_pred)
    opt_t, xpt_t = freegs_critical.find_critical(R, Z, psi_true)
    out["n_xpt_pred"], out["n_xpt_true"] = len(xpt_p), len(xpt_t)
    if not opt_t or not opt_p:
        return out

    have_truth = xpts_true is not None and len(xpts_true) > 0
    if not have_truth:
        # ---------- legacy path: paper data/ & data_v2 ----------
        # boundary flux = mean of the two X-point fluxes (paper)
        def psi_bndry(xpts, fallback):
            return 0.5 * (xpts[0][2] + xpts[1][2]) if len(xpts) >= 2 else fallback

        out["dpsi_bndry_Wb"] = float(abs(psi_bndry(xpt_p, opt_p[0][2]) - psi_bndry(xpt_t, opt_t[0][2])))

        # O-point error
        out["o_point_cm"] = 100.0 * float(np.hypot(opt_p[0][0] - opt_t[0][0], opt_p[0][1] - opt_t[0][1]))

        # X-point errors matched by Z sign (lower: Z<0, upper: Z>0)
        for tag, sign in (("x_lo_cm", -1), ("x_up_cm", 1)):
            t = [x for x in xpt_t if sign * x[1] >= 0]
            p = [x for x in xpt_p if sign * x[1] >= 0]
            if t and p:
                out[tag] = 100.0 * float(np.hypot(p[0][0] - t[0][0], p[0][1] - t[0][1]))
            else:
                out[tag] = float("nan")

        # separatrix at psi_bndry (closed contour, Q3+3IQR filtered)
        if len(xpt_t) >= 2 and len(xpt_p) >= 2:
            lvl_t, lvl_p = psi_bndry(xpt_t, opt_t[0][2]), psi_bndry(xpt_p, opt_p[0][2])
            sep_t = separatrix_points(psi_true, R, Z, lvl_t)
            sep_p = separatrix_points(psi_pred, R, Z, lvl_p)
            if len(sep_t) > 4 and len(sep_p) > 4:
                out["sep_mean_cm"], out["sep_hausdorff_cm"] = pointwise_sep_error(sep_p, sep_t)
                a_t, a_p = poly_area(sep_t), poly_area(sep_p)
                out["sep_area_rel_err_pct"] = 100.0 * abs(a_p - a_t) / (a_t + 1e-30)
        return out

    # ---------- ground-truth-anchored path (data_v4+) ----------
    bnd_t = float(psi_bndry_true if psi_bndry_true is not None else np.mean(xpts_true[:, 2]))
    o_t = np.asarray(o_true[:2], dtype=float) if o_true is not None else np.asarray(opt_t[0][:2])

    match = match_xpoints_and_axis(psi_pred, R, Z, xpts_true, bnd_t, anchor)
    xpts_t = np.asarray(xpts_true)[:, :2]
    out["dpsi_bndry_Wb"] = float(abs(match["bnd_p"] - bnd_t)) if match["bnd_p"] is not None else float("nan")
    out["o_point_cm"] = 100.0 * float(np.hypot(match["o_pred"][0] - o_t[0], match["o_pred"][1] - o_t[1]))

    # X-point errors per true X-point (Z sign decides lo/up; SN has only x_lo)
    out["x_lo_cm"], out["x_up_cm"] = float("nan"), float("nan")
    for i, t in enumerate(xpts_t):
        p = match["xpt_pred"][i] if i < len(match["xpt_pred"]) else None
        if p is None:
            continue
        tag = "x_lo_cm" if t[1] < 0 else "x_up_cm"
        if not np.isnan(out[tag]):
            continue
        out[tag] = 100.0 * float(np.hypot(p[0] - t[0], p[1] - t[1]))

    # separatrix: radial rays from the axis (wall-less MAST safe)
    if match["bnd_p"] is not None:
        sep_t = _fill_nan_theta(separatrix_pts(psi_true, R, Z, bnd_t, o_t))
        sep_p = _fill_nan_theta(separatrix_pts(psi_pred, R, Z, match["bnd_p"], match["o_pred"]))
        if len(sep_t) > 8 and len(sep_p) > 8:
            out["sep_mean_cm"], out["sep_hausdorff_cm"] = pointwise_sep_error(sep_p, sep_t)
            a_t, a_p = poly_area(sep_t), poly_area(sep_p)
            out["sep_area_rel_err_pct"] = 100.0 * abs(a_p - a_t) / (a_t + 1e-30)
    return out


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-data", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-samples", type=int, default=0, help="0 = all test samples")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    stats = ckpt["stats"]
    # input_mode is a string marker, config_input a bool — neither is a stat
    # (np.float32(str) throws; np.float32(bool) silently pollutes the scalars)
    stats = {k: (np.asarray(v, dtype=np.float32) if isinstance(v, list) else np.float32(v))
             for k, v in stats.items() if k not in ("input_mode", "config_input")}

    # input channels inferred from the checkpoint stats (9 baseline / 11 data_v2
    # / 13 data_v3-xa / 14 data_v5-mixed-xa); dataset mode follows the
    # checkpoint's input_mode / config_input
    model = build_model(in_channels=2 + len(stats["scalar_mean"])).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    use_anchor = ckpt.get("input_mode", "xpoints") == "xa"
    use_config = bool(ckpt.get("config_input", False))
    ds = DNFnoDataset(args.test_data, stats=stats, use_anchor=use_anchor,
                      use_config=use_config)
    n_eval = min(len(ds), args.max_samples) if args.max_samples else len(ds)

    print(f"\n{'='*70}")
    print(f"  DN-FNO evaluation | test samples: {n_eval} | device: {device}")
    print(f"  checkpoint: {args.checkpoint}")
    print(f"  best val rel L2: {ckpt['best_val_rel_l2']*100:.4f}% @ epoch {ckpt['best_epoch']}")
    print(f"  model params: {n_params} (paper: 4,770,241)")
    print(f"{'='*70}")

    # physical grid (meters) — needed by lap_star / RHS / find_critical;
    # ds.R/Z are the [-1,1]-normalized model input channels, not usable here.
    # --test-data may be a comma-separated list (mixed-config eval); all files
    # share the grid (data_v5 dn/sn are both MAST 65x65) — take the first.
    with np.load(args.test_data.split(",")[0].strip()) as d:
        R, Z = d["R"], d["Z"]

    rel_l2s, rmse_phys, residuals_pred, residuals_true = [], [], [], []
    geo_all = {k: [] for k in
               ["dpsi_bndry_Wb", "o_point_cm", "x_lo_cm", "x_up_cm",
                "sep_mean_cm", "sep_hausdorff_cm", "sep_area_rel_err_pct"]}
    n_fail_crit = 0

    for i in range(n_eval):
        x, y_norm = ds[i]  # y_norm: (1, 65, 65) normalized
        with torch.no_grad():
            pred_norm = model(x[None].to(device)).squeeze(0).cpu().numpy()  # (1,65,65)

        y_norm = y_norm.numpy()
        rel_l2s.append(float(np.linalg.norm(pred_norm - y_norm) / (np.linalg.norm(y_norm) + 1e-12)))

        # denormalize to physical flux (Wb)
        psi_pred = pred_norm[0] * float(stats["psi_std"]) + float(stats["psi_mean"])
        psi_true = ds.psi_total[ds.indices[i]]
        rmse_phys.append(float(np.sqrt(np.mean((psi_pred - psi_true) ** 2))))

        # GS residual on prediction and truth (paper Eq. 10-11)
        # paper plasma mask = geometric plasma region {psi >= psi_bndry}; freegs
        # core_mask additionally excludes the near-axis points whose FD
        # truncation error dominates the residual (with core_mask the truth
        # baseline is ~0.01, not the paper's 2.29; probe: {psi>=psi_bndry} gives
        # 2.2945 on the same field -- verified in scripts/probe_residual2.py)
        mask_i = (psi_true >= ds.axes[ds.indices[i]][2]).astype(np.float32)
        residuals_pred.append(gs_residual_ratio(
            psi_pred, R, Z, ds.dpdpsi[ds.indices[i]], ds.FdFdpsi[ds.indices[i]], mask_i))
        residuals_true.append(gs_residual_ratio(
            psi_true, R, Z, ds.dpdpsi[ds.indices[i]], ds.FdFdpsi[ds.indices[i]], mask_i))

        # geometry (Table II); data_v4+ npz carry the true separatrix X-points /
        # axis / boundary flux — pass them so pairing excludes vacuum saddles
        # (wall-less MAST) and SN gets its single-X-point psi_bndry
        j_i = ds.indices[i]
        gm = geometry_metrics(
            psi_pred, psi_true, R, Z,
            xpts_true=ds.xpts_actual[j_i] if ds.xpts_actual is not None else None,
            o_true=ds.o_point[j_i] if ds.o_point is not None else None,
            psi_bndry_true=float(ds.axes[j_i][2]) if ds.axes is not None else None,
            anchor=ds.anchor[j_i] if ds.anchor is not None else None)
        if gm["n_xpt_pred"] < 2:
            n_fail_crit += 1
        for k in geo_all:
            geo_all[k].append(gm.get(k, float("nan")))

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{n_eval} done")

    rel_l2s, rmse_phys = np.array(rel_l2s), np.array(rmse_phys)
    rp, rt = np.array(residuals_pred), np.array(residuals_true)

    def stats_dict(a):
        return {"mean": float(np.nanmean(a)), "std": float(np.nanstd(a)),
                "median": float(np.nanmedian(a)), "p95": float(np.nanpercentile(a, 95))}

    metrics = {
        "n_eval": n_eval,
        "n_params": n_params,
        "n_find_critical_fail": n_fail_crit,
        "rel_l2_pct": stats_dict(rel_l2s * 100),
        "rel_l2_lt_0.12_pct": float((rel_l2s < 0.0012).mean() * 100),
        "rmse_phys_Wb": stats_dict(rmse_phys),
        "gs_residual": {"pred": stats_dict(rp), "true_freegs": stats_dict(rt),
                        "ratio_pred_true": float(np.nanmean(rp) / (np.nanmean(rt) + 1e-30))},
        "geometry": {k: stats_dict(np.array(v)) for k, v in geo_all.items()},
    }

    # paper reference numbers for comparison (Table I N=5000 / Table II best model)
    metrics["paper_reference"] = {
        "rel_l2_mean_pct": 0.061, "rel_l2_best_model_pct": 0.052,
        "rmse_phys_Wb": 1.79e-5, "rmse_phys_best_Wb": 1.54e-5,
        "sep_mean_cm": 0.072, "sep_hausdorff_cm": 2.128, "sep_area_pct": 0.465,
        "x_lo_cm": 0.161, "x_up_cm": 0.112, "o_point_cm": 0.031,
        "dpsi_bndry_Wb": 9.82e-6, "gs_residual": 2.29,
    }

    print("\n  ---- field-level (normalized domain) ----")
    print(f"  rel L2:      mean {metrics['rel_l2_pct']['mean']:.4f}% "
          f"(paper N=5000: 0.061%, best seed: 0.052%)")
    print(f"               std {metrics['rel_l2_pct']['std']:.4f}% | "
          f"median {metrics['rel_l2_pct']['median']:.4f}% | P95 {metrics['rel_l2_pct']['p95']:.4f}%")
    print(f"               <0.12%: {metrics['rel_l2_lt_0.12_pct']:.1f}% of samples (paper >95%)")
    print(f"  RMSE phys:   mean {metrics['rmse_phys_Wb']['mean']:.3e} Wb (paper 1.79e-5)")
    print("\n  ---- geometry ----")
    for k in ["sep_mean_cm", "sep_hausdorff_cm", "sep_area_rel_err_pct",
              "x_lo_cm", "x_up_cm", "o_point_cm", "dpsi_bndry_Wb"]:
        v = metrics["geometry"][k]
        print(f"  {k:22s}: mean {v['mean']:.4f} | median {v['median']:.4f} | P95 {v['p95']:.4f}")
    print(f"  find_critical failures: {n_fail_crit}/{n_eval} (paper: 0)")
    print("\n  ---- GS residual (paper: 2.29, freegs baseline 2.29+-0.06) ----")
    print(f"  pred {metrics['gs_residual']['pred']['mean']:.3f} | "
          f"freegs truth {metrics['gs_residual']['true_freegs']['mean']:.3f} | "
          f"ratio {metrics['gs_residual']['ratio_pred_true']:.3f}")

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  saved -> {out_dir}/metrics.json")


if __name__ == "__main__":
    main()
