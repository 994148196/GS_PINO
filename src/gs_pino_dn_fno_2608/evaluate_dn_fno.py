"""Evaluation for the arXiv:2608.05555 double-null FNO surrogate.

Metrics (paper Tables I/II + Sec. IV residual diagnostic):
  field-level : rel L2 in the normalized representation (Eq. 7) [mean/std/median/P95],
                physical RMSE in Wb after inverse transform
  geometry    : separatrix mean closest-point distance / Hausdorff / area relative
                error, lower & upper X-point errors, O-point error, |dpsi_bndry|
                (X/O points via freegs find_critical; separatrix = closed contour
                at psi_bndry = 0.5*(psi_lo^X + psi_up^X), Q3+3*IQR distance filter)
  GS residual : Delta* psi vs RHS = -mu0 R^2 dp/dpsi - F dF/dpsi (precomputed
                dataset fields, paper Eq. 10-11), normalized over the plasma mask;
                computed for prediction AND freegs truth (baseline comparison)

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
                     R: np.ndarray, Z: np.ndarray) -> dict:
    """Paper Table II metrics for one sample (distances in cm, area rel err in %)."""
    out = {"n_xpt_pred": 0, "n_xpt_true": 0}
    if not HAS_FREEGS:
        return out

    opt_p, xpt_p = freegs_critical.find_critical(R, Z, psi_pred)
    opt_t, xpt_t = freegs_critical.find_critical(R, Z, psi_true)
    out["n_xpt_pred"], out["n_xpt_true"] = len(xpt_p), len(xpt_t)
    if not opt_t or not opt_p:
        return out

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
    stats = {k: (np.asarray(v, dtype=np.float32) if isinstance(v, list) else np.float32(v))
             for k, v in stats.items()}

    # input channels inferred from the checkpoint stats (9 baseline / 11 data_v2)
    model = build_model(in_channels=2 + len(stats["scalar_mean"])).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    ds = DNFnoDataset(args.test_data, stats=stats)
    n_eval = min(len(ds), args.max_samples) if args.max_samples else len(ds)

    print(f"\n{'='*70}")
    print(f"  DN-FNO evaluation | test samples: {n_eval} | device: {device}")
    print(f"  checkpoint: {args.checkpoint}")
    print(f"  best val rel L2: {ckpt['best_val_rel_l2']*100:.4f}% @ epoch {ckpt['best_epoch']}")
    print(f"  model params: {n_params} (paper: 4,770,241)")
    print(f"{'='*70}")

    # physical grid (meters) — needed by lap_star / RHS / find_critical;
    # ds.R/Z are the [-1,1]-normalized model input channels, not usable here
    with np.load(args.test_data) as d:
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

        # geometry (Table II)
        gm = geometry_metrics(psi_pred, psi_true, R, Z)
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
