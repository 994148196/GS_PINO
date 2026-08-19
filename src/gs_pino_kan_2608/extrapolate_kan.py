"""M3: linear extrapolation (paper Eq. 4) + DN/SN configuration classifier.

The trained KAN maps (R, Z, 17 scalars) -> psi inside the training parameter
envelope. For an OOD input x outside the 16D training convex hull (5 plasma
params + 11 coil currents; the discrete config channel is excluded):

    xbnd  = nearest boundary point of the hull to x            (SLSQP projection)
    grad  = central-difference d psi / d x_k at xbnd, k = 1..16
            (normalized-domain steps h_norm, mapped back to physical units)
    psi_ext(x) = psi(xbnd) + grad^T (x - xbnd)                 (paper Eq. 4)

The configuration (DN vs SN) is read off the separatrix of the extrapolated
flux: count saddle (X-) points with psi >= max_saddle - tol; >= 2 -> DN,
== 1 -> SN (wall-less MAST vacuum saddles sit below the separatrix value).
The tolerance is calibrated on the in-distribution test set (DN top-2 saddle
gap * 10, floored by the SN top-2 gap) — the plan's 1e-3 Wb starting point
with data-driven adaptation.

Products (--out-dir):
  classify_metrics.json   calibration + Accuracy/Precision/Recall/Specificity
                          (DN positive, paper Fig. 10) + distance bins (Fig. 11)
  samples.json            per-sample records (method, distance, labels, field error)
  fig_classify.png        2x2: confusion matrix / distance-misclassification
                          curve / DN + SN extrapolated-field examples

Usage:
  "$PY" -u -m gs_pino_kan_2608.extrapolate_kan \
    --checkpoint <exp>/model_b19ch_kan_mix/best.pt \
    --train-data dn_fno_2608/data_v5/dn/train.npz,dn_fno_2608/data_v5/sn/train.npz \
    --in-dist-data dn_fno_2608/data_v5/dn/test.npz,dn_fno_2608/data_v5/sn/test.npz \
    --ood-dir dn_fno_2608/kan/data_ood --out-dir <exp>/ext
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
from pathlib import Path

import numpy as np
import torch

from gs_pino_dn_fno_2608.data_dn_fno import _normalize_rz, nested_train_indices
from gs_pino_kan_2608.data_kan import _minmax
from gs_pino_kan_2608.model_kan import build_model

N_GRID = 65
N_DIM = 16                      # 5 params + 11 coil currents (config excluded)
CONFIG_NAMES = ["dn", "sn"]
HULL_FALLBACK_MODE = "convex"


# ---------------------------------------------------------------------------
# 16D training envelope (5 plasma params + 11 coil currents, raw physical)
# ---------------------------------------------------------------------------
def build_hull(points: np.ndarray, mode: str = "ellipsoid") -> dict:
    """Training envelope: 'ellipsoid' | 'box' | 'convex'.

    A (even PCA-reduced) ConvexHull of the data_v5 train subset does NOT
    terminate: the 11 coil currents are nearly determined by the 5 plasma
    params, the float32 point cloud sits on a near-degenerate 12D manifold
    and qhull churns on it at every k >= 8 (verified 2026-08-19: raw 16D and
    PCA k=12 both run > 2 min without returning). Default is therefore the
    Mahalanobis ellipsoid — a smooth convex envelope that by construction
    contains every training point (radial threshold = max training MD) and
    whose nearest-boundary projection is a closed-form radial scaling (no
    optimizer). 'convex' is kept for well-conditioned data (k <= 10) with an
    automatic ellipsoid fallback otherwise; 'box' is the plan's axis-aligned
    fallback.
    """
    pts = np.asarray(points, float)
    if mode == "box":
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        return {"mode": "box", "lo": lo.tolist(), "hi": hi.tolist()}
    mean = pts.mean(axis=0)
    _, sv, Vt = np.linalg.svd(pts - mean, full_matrices=False)
    k = min(12, int((sv / sv[0] > 1e-6).sum()))
    if mode == "convex" and k <= 10:
        from scipy.spatial import ConvexHull
        P = (pts - mean) @ Vt[:k].T                # PCA coords (k dims)
        h = ConvexHull(P, qhull_options="QJ")
        return {"mode": "pca", "k": int(k), "sv": sv.tolist(),
                "mean": mean.tolist(), "V": Vt[:k].tolist(),
                "H": h.equations[:, :-1].tolist(),
                "b": h.equations[:, -1].tolist(), "n_facets": h.equations.shape[0]}
    if mode == "convex":
        print("  WARNING: --hull-mode convex needs k>10 -> qhull unreliable; "
              f"falling back to ellipsoid (k={k})")
    # Mahalanobis ellipsoid: y = whitened coords, boundary = max train radius
    W = (Vt[:k] / sv[:k, None])                    # (k, 16): y = W (x - mean)
    y = W @ (pts - mean).T                         # (k, N)
    thr = float(np.sqrt((y ** 2).sum(axis=0).max()))
    return {"mode": "ellipsoid", "k": int(k), "sv": sv.tolist(),
            "mean": mean.tolist(), "W": W.tolist(), "Vt": Vt[:k].tolist(),
            "thr": thr}


def _whiten(x: np.ndarray, hull: dict) -> np.ndarray:
    """(N,16) -> (N,k) whitened coords ('ellipsoid'); PCA coords ('pca');
    identity otherwise."""
    if hull["mode"] == "ellipsoid":
        return (x - np.array(hull["mean"])) @ np.array(hull["W"]).T
    if hull["mode"] == "pca":
        return (x - np.array(hull["mean"])) @ np.array(hull["V"]).T
    return x


def _back_proj(y: np.ndarray, hull: dict) -> np.ndarray:
    """(N,k) -> (N,16) physical ('ellipsoid'/'pca' modes; identity otherwise)."""
    if hull["mode"] == "ellipsoid":
        sv = np.array(hull["sv"])[:hull["k"]]
        return np.array(hull["mean"]) + (y * sv) @ np.array(hull["Vt"])
    if hull["mode"] == "pca":
        return np.array(hull["mean"]) + y @ np.array(hull["V"])
    return y


def in_hull(x: np.ndarray, hull: dict, tol: float = 1e-9) -> np.ndarray:
    """(N,16) -> bool mask of points INSIDE the training envelope."""
    x = np.asarray(x, float)
    if hull["mode"] == "box":
        lo, hi = np.array(hull["lo"]), np.array(hull["hi"])
        return np.all((x >= lo - tol) & (x <= hi - tol), axis=1)
    if hull["mode"] == "pca":
        H, b = np.array(hull["H"]), np.array(hull["b"])
        return np.all(H @ _whiten(x, hull).T + b[:, None] <= tol, axis=0)
    # ellipsoid: ||W (x - mean)|| <= thr
    r = np.linalg.norm(_whiten(x, hull), axis=1)
    return r <= hull["thr"] + tol


def nearest_boundary(x: np.ndarray, hull: dict) -> np.ndarray:
    """Closest envelope point to x (16,).

    'ellipsoid': radial scaling in whitened coordinates (closed form);
    'box': per-dim clip; 'pca': SLSQP in k dims, lifted back to 16D.
    """
    x = np.asarray(x, float)
    if hull["mode"] == "box":
        return np.clip(x, hull["lo"], hull["hi"])
    y = _whiten(x[None], hull)                     # (1, k)
    r = float(np.linalg.norm(y[0]))
    if hull["mode"] == "ellipsoid":
        if r <= hull["thr"]:
            return x.copy()
        return _back_proj(y * (hull["thr"] / r), hull)[0]   # (1,16) -> (16,)
    if r == 0.0 or in_hull(x[None], hull)[0]:
        return x.copy()
    from scipy.optimize import minimize
    H = np.array(hull["H"], float)                 # (n_facets, k)
    b = np.array(hull["b"], float)
    cons = [{"type": "ineq", "fun": lambda u: -(H @ u + b), "jac": lambda u: -H}]
    res = minimize(lambda u: 0.5 * np.sum((u - y[0]) ** 2), x0=y[0],
                   jac=lambda u: u - y[0], constraints=cons, method="SLSQP",
                   options={"ftol": 1e-12, "maxiter": 300, "disp": False})
    xb = res.x
    # safety snap: push back across the worst-violated facet
    viol = H @ xb + b
    j = int(np.argmax(viol))
    if viol[j] > 1e-8:
        xb = xb - (viol[j] + 1e-9) * H[j] / np.dot(H[j], H[j])
    return _back_proj(xb[None], hull)[0]


def train_rows(train_data: str, n_train: int, seed: int):
    """(params (N,5), coils (N,11)) of the nested training subset (KAN parity)."""
    paths = [p.strip() for p in train_data.split(",")]
    ds0 = np.load(paths[0])
    n_full = int(ds0["params"].shape[0]) * len(paths)
    idx = nested_train_indices(n_full, n_train, seed)
    params = np.concatenate([np.load(p)["params"] for p in paths])[idx]
    coils = np.concatenate([np.load(p)["coil_currents"] for p in paths])[idx]
    return params, coils


# ---------------------------------------------------------------------------
# full-grid forward / Eq. (4) extrapolation
# ---------------------------------------------------------------------------
def _grid_input(Rn: np.ndarray, Zn: np.ndarray, scalars_n: np.ndarray) -> np.ndarray:
    """(4225, 19) float32 normalized input for one row (grid + broadcast scalars)."""
    rr, cc = np.indices((N_GRID, N_GRID))
    rr, cc = rr.ravel(), cc.ravel()
    return np.concatenate([Rn[rr, cc][:, None], Zn[rr, cc][:, None],
                           np.broadcast_to(scalars_n, (N_GRID * N_GRID, 17))],
                          axis=1).astype(np.float32)


def _forward_psi(model: torch.nn.Module, Rn: np.ndarray, Zn: np.ndarray,
                 scalars_n: np.ndarray, device: torch.device) -> np.ndarray:
    """psi z-scored on the full grid -> (65, 65)."""
    with torch.no_grad():
        out = model(torch.from_numpy(_grid_input(Rn, Zn, scalars_n)).to(device))
    return out[:, 0].cpu().numpy().reshape(N_GRID, N_GRID)


def extrapolate_batch(model: torch.nn.Module, x_phys: np.ndarray,
                      xbnd_phys: np.ndarray, config_codes: np.ndarray,
                      Rn: np.ndarray, Zn: np.ndarray, lo: np.ndarray, hi: np.ndarray,
                      psi_std: float, psi_mean: float, device: torch.device,
                      h_norm: float = 5e-3, chunk: int = 4) -> tuple:
    """Paper Eq. (4) for a batch of out-of-hull samples.

    x_phys / xbnd_phys: (B, 16) raw physical params+coils. Returns
      psi_ext (B,65,65)  physical extrapolated flux
      psi_bnd (B,65,65)  physical flux at the boundary point
      dist_norm (B,)     extrapolation distance, per-dim normalized by the
                         training range (dimensionless, comparable across dims)
    """
    B = x_phys.shape[0]
    rng = np.where(hi[:N_DIM] - lo[:N_DIM] < 1e-8, 1.0, hi[:N_DIM] - lo[:N_DIM])
    h_phys = h_norm * rng / 2.0
    dist = np.sqrt(np.sum(((x_phys - xbnd_phys) / rng) ** 2, axis=1))
    psi_ext = np.empty((B, N_GRID, N_GRID), np.float32)
    psi_bnd = np.empty((B, N_GRID, N_GRID), np.float32)
    rr, cc = np.indices((N_GRID, N_GRID))
    grid_rz = np.concatenate([Rn[rr, cc, None], Zn[rr, cc, None]], axis=1).ravel() \
        .reshape(-1, 2)                                  # (4225, 2)
    t_rz = torch.from_numpy(grid_rz.astype(np.float32))

    with torch.no_grad():
        for s in range(0, B, chunk):
            Bc = min(chunk, B - s)
            scal = np.empty((Bc, 1 + 2 * N_DIM, 17), np.float32)
            for b in range(Bc):
                base = xbnd_phys[s + b].copy()
                grid_list = [np.concatenate([base, [config_codes[s + b]]])]
                for k in range(N_DIM):
                    p = base.copy(); p[k] += h_phys[k]
                    grid_list.append(np.concatenate([p, [config_codes[s + b]]]))
                    m = base.copy(); m[k] -= h_phys[k]
                    grid_list.append(np.concatenate([m, [config_codes[s + b]]]))
                scal[b] = np.stack(grid_list)            # (33, 17) physical
            scal_n = _minmax(scal.reshape(-1, 17), lo, hi).reshape(Bc, 1 + 2 * N_DIM, 17)
            # (Bc*33, 4225, 19): grid part shared, scalars broadcast
            X = torch.zeros((Bc * (1 + 2 * N_DIM), N_GRID * N_GRID, 19),
                            dtype=torch.float32)
            X[:, :, :2] = t_rz
            X[:, :, 2:] = torch.from_numpy(
                np.broadcast_to(scal_n[:, :, None, :],
                                (Bc, 1 + 2 * N_DIM, N_GRID * N_GRID, 17)).copy())
            out = model(X.reshape(-1, 19).to(device)).cpu().numpy()[:, 0]
            out = out.reshape(Bc, 1 + 2 * N_DIM, N_GRID, N_GRID)   # psi z-scored
            for b in range(Bc):
                psi_z0 = out[b, 0]
                g = (out[b, 1:1 + N_DIM] - out[b, 1 + N_DIM:]) \
                    / (2.0 * h_phys)[:, None, None]               # (16,65,65)
                dpsi = np.einsum("kij,k->ij", g, x_phys[s + b] - xbnd_phys[s + b])
                psi_bnd[s + b] = psi_z0 * psi_std + psi_mean
                psi_ext[s + b] = (psi_z0 + dpsi) * psi_std + psi_mean
    return psi_ext, psi_bnd, dist


# ---------------------------------------------------------------------------
# DN/SN classifier: separatrix X-point count of the (extrapolated) flux
# ---------------------------------------------------------------------------
def find_saddles(psi: np.ndarray, R: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """All saddle points (n, 3) = [r, z, psi] of the field.

    freegs critical on wall-less MAST also reports vacuum-region saddles (5-6
    per sample) — they sit BELOW the separatrix value and are filtered by the
    caller. Smooth KAN fields can make the global grid scan miss a flat
    separatrix X-point (FNO experience): if fewer than 3 saddles are found,
    rescan a 4x cubic-upsampled grid and merge (dedupe within 0.05 m,
    upsampled estimates win — they sit on the finer grid).
    """
    from freegs import critical as fc

    def _scan(ps, Rg, Zg):
        with open(os.devnull, "w") as devnull, \
                contextlib.redirect_stdout(devnull), \
                contextlib.redirect_stderr(devnull):
            try:
                _, xpts = fc.find_critical(Rg, Zg, ps)
            except Exception:
                return []
        return [np.array([p[0], p[1], p[2]], float) for p in xpts]

    def _merge(a, b):
        pts = list(b) + list(a)                 # upsampled first (preferred)
        kept = []
        for p in pts:
            if all(np.linalg.norm(p[:2] - q[:2]) >= 0.05 for q in kept):
                kept.append(p)
        return np.array(kept).reshape(-1, 3) if kept else np.zeros((0, 3))

    out = _scan(psi, R, Z)
    if len(out) >= 3:
        return np.array([np.array(p) for p in out]).reshape(-1, 3)
    # cubic upsampling on exact coordinates (RectBivariateSpline)
    from scipy.interpolate import RectBivariateSpline
    r = np.linspace(R[0, 0], R[-1, 0], 257)
    z = np.linspace(Z[0, 0], Z[0, -1], 257)
    try:
        spl = RectBivariateSpline(R[:, 0], Z[0, :], psi, kx=3, ky=3)
        psi_up = spl(r, z)
    except Exception:
        return np.array([np.array(p) for p in out]).reshape(-1, 3)
    Ru, Zu = np.meshgrid(r, z, indexing="xy")
    return _merge(out, _scan(psi_up, Ru, Zu))


def classify_dn_sn(psi: np.ndarray, R: np.ndarray, Z: np.ndarray, tol: float):
    """("dn"|"sn"|"?", n_sep_xpts, saddle_psi array) from the separatrix count."""
    saddles = find_saddles(psi, R, Z)
    if len(saddles) == 0:
        return "?", 0, saddles
    ps = saddles[:, 2]
    n = int((ps >= ps.max() - tol).sum())
    return ("dn" if n >= 2 else "sn"), n, saddles


def calibrate_tol(dn_fields, sn_fields, R, Z, tol_fallback: float = 1e-3,
                  max_n: int = 120) -> tuple:
    """tol from in-distribution saddle-gap statistics (plan: 1e-3 + adapt).

    tol = max(1e-3, 10 * median(DN top-2 saddle gap)), capped at half the
    minimum SN top-2 gap (so a SN's second saddle is never swallowed).
    Returns (tol, gaps_dn, gaps_sn, per-config accuracy dict).
    """
    def top2_gaps(fields):
        out = []
        for f in fields:
            s = find_saddles(f, R, Z)
            if len(s) >= 2:
                ps = np.sort(s[:, 2])[::-1]
                out.append(float(ps[0] - ps[1]))
        return out

    g_dn = top2_gaps(dn_fields[:max_n])
    g_sn = top2_gaps(sn_fields[:max_n])
    tol = max(tol_fallback, 10.0 * float(np.median(g_dn))) if g_dn else tol_fallback
    if g_sn:
        sn_floor = 0.5 * min(g_sn)
        if sn_floor > 0:
            tol = min(tol, sn_floor)
    acc = {}
    for name, fields in (("dn", dn_fields[:max_n]), ("sn", sn_fields[:max_n])):
        preds = [classify_dn_sn(f, R, Z, tol)[0] for f in fields]
        acc[name] = {"n": len(preds), "correct": sum(p == name for p in preds),
                     "unclassified": sum(p == "?" for p in preds)}
    return tol, g_dn, g_sn, acc


# ---------------------------------------------------------------------------
# metrics / figures
# ---------------------------------------------------------------------------
def _confusion(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """DN = positive (1), SN = negative (0). Returns counts + 4 metrics."""
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    acc = (tp + tn) / max(tp + fp + fn + tn, 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    return {"confusion": [[tn, fp], [fn, tp]], "n": tp + fp + fn + tn,
            "accuracy": float(acc), "precision": float(prec),
            "recall": float(rec), "specificity": float(spec)}


def _distance_bins(dist: np.ndarray, misclass: np.ndarray) -> list:
    """Per-bin misclassification rate vs extrapolation distance (paper Fig. 11)."""
    bins = [0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 1.0, 2.0]
    out = []
    for i in range(len(bins) - 1):
        m = (dist >= bins[i]) & (dist < bins[i + 1])
        n = int(m.sum())
        out.append({"bin": i, "lo": bins[i], "hi": bins[i + 1], "n": n,
                    "n_misclass": int(misclass[m].sum()) if n else 0,
                    "misclass_rate": float(misclass[m].mean()) if n else None})
    tail = dist >= bins[-1]
    n = int(tail.sum())
    out.append({"bin": len(bins) - 1, "lo": bins[-1], "hi": None, "n": n,
                "n_misclass": int(misclass[tail].sum()) if n else 0,
                "misclass_rate": float(misclass[tail].mean()) if n else None})
    return out


def _plot(results, tol, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    # (0,0) confusion matrix (paper Fig. 10)
    ax = axes[0, 0]
    cm = np.array(results["ood"]["confusion"])           # [[tn, fp], [fn, tp]]
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], ["SN", "DN"])
    ax.set_yticks([0, 1], ["SN", "DN"])
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"OOD classification (DN positive)\n"
                 f"acc {results['ood']['accuracy']*100:.1f}% | prec "
                 f"{results['ood']['precision']*100:.1f}% | recall "
                 f"{results['ood']['recall']*100:.1f}% | spec "
                 f"{results['ood']['specificity']*100:.1f}%")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i][j]}", ha="center", va="center",
                    color="white" if cm[i][j] > cm.max() / 2 else "black", fontsize=16)
    fig.colorbar(im, ax=ax, fraction=0.046)

    # (0,1) distance-misclassification (paper Fig. 11)
    ax = axes[0, 1]
    dbs = results["distance_bins"]
    xs = [0.0] + [(d["lo"] + (d["hi"] or 0.7 * d["lo"])) / 2 for d in dbs[1:]]
    rates = [results["distance_bins"][0]["misclass_rate"]] + \
            [d["misclass_rate"] for d in dbs[1:]]
    ns = [results["distance_bins"][0]["n"]] + [d["n"] for d in dbs[1:]]
    xb = [0.0] + [d["lo"] for d in dbs[1:]]
    ax.bar(range(len(dbs)), ns, alpha=0.3, color="gray", label="n per bin")
    ax.plot(range(len(dbs)), [r if r is not None else 0 for r in rates],
            "o-", color="crimson", label="misclass. rate")
    ax.set_xticks(range(len(dbs)),
                  ["in-hull"] + [f"{xb[i]:.2f}+" for i in range(1, len(dbs))],
                  rotation=45)
    ax.set_ylabel("misclassification rate / n")
    ax.set_title(f"misclass. vs extrapolation distance (tol={tol:.1e} Wb)")
    ax.legend()

    # (1,0)/(1,1) example extrapolated fields (DN / SN)
    for k, cfg in enumerate(("dn", "sn")):
        ax = axes[1, k]
        ex = results["examples"][cfg]
        psi_ext, psi_true, R, Z, saddles_top, xpts_true = \
            ex["psi_ext"], ex["psi_true"], ex["R"], ex["Z"], \
            ex["saddles_top"], ex["xpts_true"]
        vm = max(abs(psi_ext.min()), abs(psi_ext.max()))
        cf = ax.contourf(R, Z, psi_ext, levels=24, cmap="RdBu_r",
                         vmin=-vm, vmax=vm)
        ax.contour(R, Z, psi_true, levels=16, colors="k", linewidths=0.6,
                   linestyles="--", alpha=0.45)
        if len(saddles_top):
            ax.plot(saddles_top[:, 0], saddles_top[:, 1], "r*", ms=13,
                    label=f"{len(saddles_top)} separatrix X-pts (pred)")
        if xpts_true is not None and len(xpts_true):
            ax.plot(xpts_true[:, 0], xpts_true[:, 1], "k^", ms=9,
                    label="true X-points")
        ax.set_title(f"{cfg.upper()} example | d={ex['d_norm']:.2f} "
                     f"| field rel L2 {ex['rel_l2_pct']:.2f}% | "
                     f"pred={ex['pred']}, true={cfg}")
        ax.legend(loc="upper right", fontsize=8)
        fig.colorbar(cf, ax=ax, fraction=0.046)
    fig.suptitle(results["title"], fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--train-data", required=True,
                    help="dn+sn train npz (hull from the nested subset)")
    ap.add_argument("--in-dist-data", required=True,
                    help="dn+sn test npz (tol calibration)")
    ap.add_argument("--ood-dir", default="dn_fno_2608/kan/data_ood",
                    help="dir with {dn,sn}/test.npz + hull.json (gen_ood_kan)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-train", type=int, default=500)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--h-norm", type=float, default=5e-3,
                    help="central-difference step in the normalized domain")
    ap.add_argument("--chunk", type=int, default=4, help="samples per gradient batch")
    ap.add_argument("--calib-max", type=int, default=120,
                    help="in-dist rows per config used for calibration")
    ap.add_argument("--max-samples", type=int, default=0, help="0 = all OOD rows")
    ap.add_argument("--hull-mode", choices=["ellipsoid", "convex", "box"],
                    default="ellipsoid",
                    help="training envelope (convex: qhull needs k<=10 on this data)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--skip-fig", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else ("cpu" if args.device == "auto" else args.device))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    stats = {k: (np.asarray(v, dtype=np.float32) if isinstance(v, list) else np.float32(v))
             for k, v in ckpt["stats"].items()}
    lo, hi = stats["scalar_lo"], stats["scalar_hi"]
    psi_std, psi_mean = float(stats["psi_std"]), float(stats["psi_mean"])
    model = build_model(**ckpt.get("arch", {})).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"\n{'='*72}")
    print(f"  KAN-EXT extrapolation + DN/SN classification")
    print(f"  phase {ckpt.get('phase','?')} | val rel L2 "
          f"{ckpt['best_val_rel_l2']*100:.4f}% @ {ckpt.get('best_epoch')} | "
          f"prune {ckpt.get('prune_ratio',0)*100:.1f}% | device {device}")
    print(f"  h_norm={args.h_norm:.1e} (normalized-domain central difference)")
    print(f"{'='*72}")

    # ---- 16D hull (training envelope) ----
    params, coils = train_rows(args.train_data, args.n_train, args.seed)
    hull = build_hull(np.hstack([params, coils]), args.hull_mode)
    hull_path = Path(args.ood_dir) / "hull.json"
    if hull_path.exists():
        with open(hull_path) as f:
            saved = json.load(f)
        same = saved["mode"] == hull["mode"] and \
            saved.get("n_facets", 0) == hull.get("n_facets", 0) and \
            saved.get("k", 16) == hull.get("k", 16)
        if not same:
            print(f"  WARNING: hull.json differs from the rebuilt hull "
                  f"({saved.get('mode')} k={saved.get('k', 16)} "
                  f"f={saved.get('n_facets')} vs {hull['mode']} "
                  f"k={hull.get('k', 16)} f={hull.get('n_facets')}) "
                  f"— regenerate with matching args")
    print(f"  hull: {hull['mode']}, {hull.get('n_facets', 'box')} facets, "
          f"{len(params)} train points")

    # ---- in-distribution calibration ----
    in_paths = {"dn": [], "sn": []}
    for p in [s.strip() for s in args.in_dist_data.split(",") if s.strip()]:
        parts = p.replace("\\", "/").split("/")
        tag = parts[-2] if parts[-2] in ("dn", "sn") else \
            ("dn" if "dn" in p else "sn")       # config from the parent dir
        in_paths[tag].append(p)
    fields = {}
    grid = None
    for cfg in CONFIG_NAMES:
        d = np.load(in_paths[cfg][0])
        fields[cfg] = {k: d[k] for k in ("params", "coil_currents", "config",
                                         "psi_total")}
        if grid is None:
            grid = (d["R"], d["Z"])
    R, Z = grid
    Rn, Zn = _normalize_rz(R.astype(np.float32), Z.astype(np.float32))
    calib_fields = {cfg: [] for cfg in CONFIG_NAMES}
    for cfg in CONFIG_NAMES:
        for i in range(min(len(fields[cfg]["psi_total"]), args.calib_max)):
            s = np.concatenate([fields[cfg]["params"], fields[cfg]["coil_currents"],
                                fields[cfg]["config"]], axis=1)[i].astype(np.float32)
            calib_fields[cfg].append(
                _forward_psi(model, Rn, Zn, _minmax(s[None], lo, hi)[0], device))
    tol, g_dn, g_sn, calib_acc = calibrate_tol(
        calib_fields["dn"], calib_fields["sn"], R, Z)
    gap_dn = float(np.median(g_dn)) if g_dn else float("nan")
    gap_sn = float(min(g_sn)) if g_sn else float("nan")
    print(f"  calibration: tol = {tol:.2e} Wb (DN top-2 gap median "
          f"{gap_dn:.2e}, SN min {gap_sn:.2e})")
    for cfg in CONFIG_NAMES:
        a = calib_acc[cfg]
        print(f"    in-dist {cfg}: {a['correct']}/{a['n']} correct"
              + (f" ({a['unclassified']} unclassified)" if a["unclassified"] else ""))

    # ---- OOD: direct (in-hull) vs Eq. (4) extrapolation ----
    rec = []                       # per-sample records
    for cfg in CONFIG_NAMES:
        npz_path = Path(args.ood_dir) / cfg / "test.npz"
        if not npz_path.exists():
            print(f"  WARNING: {npz_path} missing — skipping {cfg}")
            continue
        with np.load(npz_path) as d:
            n = len(d["params"])
            n = min(n, args.max_samples) if args.max_samples else n
            x16 = np.hstack([d["params"], d["coil_currents"]])[:n].astype(np.float64)
            cfg_code = np.full(n, CONFIG_NAMES.index(cfg), np.float32)
            psi_true = d["psi_total"][:n].astype(np.float32)
            xs = np.concatenate([d["params"], d["coil_currents"]], axis=1)[:n]
            has_xpts = "xpts_actual" in d.files
            xpts_true_all = d["xpts_actual"][:n] if has_xpts else None
        inside = in_hull(x16, hull)
        print(f"  OOD {cfg}: {n} rows, {int((~inside).sum())} out-of-hull "
              f"({int(inside.sum())} in-hull, direct prediction)")
        for i in range(n):
            if inside[i]:
                psi_pred = _forward_psi(
                    model, Rn, Zn, _minmax(xs[i][None], lo, hi)[0], device)
                d_norm, method = 0.0, "direct"
            else:
                xbnd = nearest_boundary(x16[i], hull)
                (psi_pred, _, d_norm), method = extrapolate_batch(
                    model, x16[i:i + 1], xbnd[None], cfg_code[i:i + 1],
                    Rn, Zn, lo, hi, psi_std, psi_mean, device,
                    h_norm=args.h_norm, chunk=1)[:3], "extrap"
                psi_pred = psi_pred[0]
                d_norm = float(d_norm[0])
            pred, n_sep, saddles = classify_dn_sn(psi_pred, R, Z, tol)
            y_psi_z = (psi_true[i] - psi_mean) / psi_std
            rel_l2 = float(np.linalg.norm(psi_pred - psi_true[i]) /
                           np.linalg.norm(y_psi_z * psi_std + psi_mean)) \
                if np.linalg.norm(psi_true[i]) else float("nan")
            xpts_true = None
            if has_xpts and np.isfinite(xpts_true_all[i]).any():
                xpts_true = xpts_true_all[i][np.isfinite(xpts_true_all[i]).all(axis=1)]
            rec.append({"config": cfg, "y_true": int(cfg == "dn"),  # DN = positive
                        "y_pred": 1 if pred == "dn" else 0, "pred": pred,
                        "method": method, "d_norm": float(d_norm),
                        "n_sep_xpts": int(n_sep),
                        "rel_l2_pct": rel_l2 * 100,
                        "psi_pred": psi_pred, "psi_true": psi_true[i],
                        "xpts_true": xpts_true, "R": R, "Z": Z})
        print(f"    classified {sum(r['y_pred'] == 1 for r in rec if r['config'] == cfg)}"
              f"/{n} as DN")

    if not rec:
        raise SystemExit("no OOD data — check --ood-dir")

    # ---- metrics ----
    y_true = np.array([r["y_true"] for r in rec])
    y_pred = np.array([r["y_pred"] for r in rec])
    classif = [r["pred"] != "?" for r in rec]
    ood_m = _confusion(y_true[classif], y_pred[classif])
    per_cfg = {cfg: _confusion(y_true[[r["config"] == cfg for r in rec]],
                               y_pred[[r["config"] == cfg for r in rec]])
               for cfg in CONFIG_NAMES}
    n_in = sum(r["method"] == "direct" for r in rec)
    dists = np.array([r["d_norm"] for r in rec])
    mis = (y_pred != y_true).astype(int)
    dbs = _distance_bins(dists[classif], mis[classif])
    rel = np.array([r["rel_l2_pct"] for r in rec])
    def _mean(vals):
        v = [x for x in vals if np.isfinite(x)]
        return float(np.mean(v)) if v else float("nan")
    examples = {}
    for cfg in CONFIG_NAMES:
        cands = [r for r in rec if r["config"] == cfg and r["method"] == "extrap"
                 and r["pred"] == cfg]
        if not cands:
            cands = [r for r in rec if r["config"] == cfg and r["method"] == "extrap"]
        if not cands:
            cands = [r for r in rec if r["config"] == cfg]
        if cands:
            cands.sort(key=lambda r: abs(r["d_norm"] - np.median(dists)))
            r = cands[len(cands) // 2]
            sep = find_saddles(r["psi_pred"], r["R"], r["Z"])
            sep_top = sep[sep[:, 2] >= sep[:, 2].max() - tol] if len(sep) else sep
            examples[cfg] = {"psi_ext": r["psi_pred"], "psi_true": r["psi_true"],
                             "R": r["R"], "Z": r["Z"],
                             "saddles_top": sep_top, "xpts_true": r["xpts_true"],
                             "d_norm": r["d_norm"], "rel_l2_pct": r["rel_l2_pct"],
                             "pred": r["pred"]}
    results = {
        "examples": examples,
        "title": f"KAN-EXT | {ckpt.get('best_epoch','?')} ep | tol {tol:.1e} Wb | "
                 f"OOD n={len(rec)} (in-hull {n_in})",
        "calibration": {"tol_Wb": tol,
                        "dn_gap_median_Wb": float(np.median(g_dn)) if g_dn else None,
                        "sn_gap_min_Wb": float(min(g_sn)) if g_sn else None,
                        "in_dist": {cfg: calib_acc[cfg] for cfg in CONFIG_NAMES}},
        "ood": {**ood_m, "n_in_hull": n_in, "n_extrap": len(rec) - n_in,
                "n_unclassified": sum(r["pred"] == "?" for r in rec),
                "per_config": per_cfg},
        "field_rel_l2_pct": {
            "direct": _mean([r["rel_l2_pct"] for r in rec if r["method"] == "direct"]),
            "extrap": _mean([r["rel_l2_pct"] for r in rec if r["method"] == "extrap"])},
        "distance_bins": dbs,
        "paper_reference": {"accuracy": 89.1, "precision": 72.1,
                            "recall": 98.9, "specificity": 85.3},
    }
    with open(out_dir / "classify_metrics.json", "w") as f:
        json.dump({k: v for k, v in results.items() if k != "examples"},
                  f, indent=2, allow_nan=True)
    samples_out = [{k: v for k, v in r.items()
                    if k not in ("psi_pred", "psi_true", "xpts_true", "R", "Z")}
                   for r in rec]
    with open(out_dir / "samples.json", "w") as f:
        json.dump(samples_out, f, indent=1)
    print(f"\n  ---- OOD classification (paper: 89.1/72.1/98.9/85.3%) ----")
    print(f"  accuracy  {ood_m['accuracy']*100:5.1f}%   (n={ood_m['n']}, "
          f"{results['ood']['n_unclassified']} unclassified)")
    print(f"  precision {ood_m['precision']*100:5.1f}%  recall "
          f"{ood_m['recall']*100:5.1f}%  specificity {ood_m['specificity']*100:5.1f}%")
    for cfg in CONFIG_NAMES:
        m = per_cfg[cfg]
        print(f"    {cfg}: acc {m['accuracy']*100:5.1f}% (n={m['n']})")
    print(f"  field rel L2: direct {results['field_rel_l2_pct']['direct']:.3f}% | "
          f"extrap {results['field_rel_l2_pct']['extrap']:.3f}%")
    print(f"  distance bins (misclass rate): " +
          ", ".join(f"{d['lo']:.2f}+: {d['n']} ({d['misclass_rate']*100:.0f}%)"
                    if d["misclass_rate"] is not None else f"{d['lo']:.2f}+: {d['n']}"
                    for d in dbs))
    if not args.skip_fig:
        _plot(results, tol, out_dir / "fig_classify.png")
        print(f"  saved -> {out_dir}/fig_classify.png")
    print(f"  saved -> {out_dir}/classify_metrics.json, samples.json")


if __name__ == "__main__":
    main()
