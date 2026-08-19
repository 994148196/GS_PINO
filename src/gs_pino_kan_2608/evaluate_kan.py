"""Full-grid KAN evaluation CLI (mirrors evaluate_dn_fno.py).

Per test sample: forward the whole 65x65 grid -> psi (normalized) -> physical
flux; metrics (metrics.json, exp011 schema + KAN additions):
  rel_l2_pct          per-sample relative L2 in the normalized representation
  rmse_phys_Wb        physical psi RMSE
  r_squared           per-sample R^2 = 1 - ||p-t||^2/||t-mean(t)||^2 (paper 0.9953)
  gs_residual         pred/true_freegs/ratio (reuses gs_residual_ratio, paper
                      Eq. 10-11, geometric plasma mask {psi >= psi_bndry})
  geometry            v2 geometry metrics (X-points/O-point/separatrix; FNO
                      geometry_metrics reuse, incl. xpts_actual pairing)
  j_self_check_pct    per-sample |sum J dR dZ - Ip|/Ip on the predicted J
                      (full grid, exact) — the L_Ip constraint's test-time check
  paper_reference     aps.75.20260331 numbers (KAN-1..3: 0.631/0.912/1.022%)

Usage:
  "$PY" -u -m gs_pino_kan_2608.evaluate_kan \
    --test-data dn_fno_2608/data_v5/sn/test.npz \
    --checkpoint <out>/best.pt --out-dir <out>/eval_sn
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from gs_pino_dn_fno_2608.evaluate_dn_fno import (geometry_metrics,
                                                 gs_residual_ratio)
from gs_pino_kan_2608.data_kan import PointKANDataset
from gs_pino_kan_2608.model_kan import build_model


def stats_dict(a: np.ndarray) -> dict:
    return {"mean": float(np.nanmean(a)), "std": float(np.nanstd(a)),
            "median": float(np.nanmedian(a)), "p95": float(np.nanpercentile(a, 95))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-data", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-samples", type=int, default=0, help="0 = all test samples")
    args = ap.parse_args()

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else ("cpu" if args.device == "auto" else args.device))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    stats = {k: (np.asarray(v, dtype=np.float32) if isinstance(v, list) else np.float32(v))
             for k, v in ckpt["stats"].items()}
    arch = dict(ckpt.get("arch", {}))
    device_config = arch.pop("device_config", "mast")
    target = arch.pop("target", "total")
    model = build_model(**arch).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())

    ds = PointKANDataset(args.test_data, stats=stats, device=device_config,
                         target=target)
    ds.load_truth_fields()  # GS-residual target + geometry truth (lazy)
    n_eval = min(len(ds), args.max_samples) if args.max_samples else len(ds)
    with np.load(args.test_data.split(",")[0].strip()) as d:
        R, Z = d["R"], d["Z"]
    dA = float(R[1, 0] - R[0, 0]) * float(Z[0, 1] - Z[0, 0])
    psi_std, psi_mean = float(stats["psi_std"]), float(stats["psi_mean"])
    j_std, j_mean = float(stats["j_std"]), float(stats["j_mean"])

    print(f"\n{'='*70}")
    print(f"  KAN evaluation | test samples: {n_eval} | device: {device} | "
          f"target: {target}")
    print(f"  checkpoint: {args.checkpoint}")
    print(f"  phase {ckpt.get('phase', '?')} | best val rel L2 "
          f"{ckpt['best_val_rel_l2']*100:.4f}% @ epoch {ckpt.get('best_epoch')} | "
          f"prune ratio {ckpt.get('prune_ratio', 0)*100:.1f}%")
    print(f"  model params: {n_params} (paper KAN-2: 960 @ 14ch)")
    print(f"{'='*70}")

    rel_l2s, rel_l2_total, rmse_phys, r2s, j_self, residuals_pred, \
        residuals_true = [], [], [], [], [], [], []
    geo_all = {k: [] for k in
               ["dpsi_bndry_Wb", "o_point_cm", "x_lo_cm", "x_up_cm",
                "sep_mean_cm", "sep_hausdorff_cm", "sep_area_rel_err_pct"]}
    n_fail_crit = 0

    with torch.no_grad():
        ng = ds.n_grid
        for i in range(n_eval):
            b = ds.full_grid(i)
            out = model(b["x"].to(device))
            psi_z = out[:, 0].cpu().numpy().reshape(ng, ng)
            j_z = out[:, 1].cpu().numpy().reshape(ng, ng)
            psi_pred = psi_z * psi_std + psi_mean
            y_psi = b["y_psi"].numpy().reshape(ng, ng)
            psi_true = ds.psi[i]
            if target == "plasma":
                # add the known coil field back: psi_total = psi_plasma + psi_coils
                # (psi_coils stored; greens@I agrees to 1e-7 — exact reconstruction)
                psi_pred = psi_pred + ds.psi_coils[i]
                psi_true = ds.psi_total[i]

            rel_l2s.append(float(np.linalg.norm(psi_z - y_psi) /
                                 (np.linalg.norm(y_psi) + 1e-12)))
            if target == "plasma":
                rel_l2_total.append(float(np.linalg.norm(psi_pred - psi_true) /
                                          (np.linalg.norm(psi_true) + 1e-12)))
            rmse_phys.append(float(np.sqrt(np.mean((psi_pred - psi_true) ** 2))))
            denom = np.sum((psi_true - psi_true.mean()) ** 2) + 1e-30
            r2s.append(1.0 - float(np.sum((psi_pred - psi_true) ** 2)) / denom)
            j_phys = j_z * j_std + j_mean
            est = float(np.sum(j_phys) * dA)
            j_self.append(abs(est - ds.ip[i]) / (abs(ds.ip[i]) + 1e-30))

            # GS residual on prediction and truth (geometric plasma mask
            # {psi_true >= psi_bndry}, FNO parity)
            mask_i = (psi_true >= ds.axes[i][2]).astype(np.float32)
            residuals_pred.append(gs_residual_ratio(
                psi_pred, R, Z, ds.dpdpsi[i], ds.FdFdpsi[i], mask_i))
            residuals_true.append(gs_residual_ratio(
                psi_true, R, Z, ds.dpdpsi[i], ds.FdFdpsi[i], mask_i))

            gm = geometry_metrics(
                psi_pred, psi_true, R, Z,
                xpts_true=ds.xpts_actual[i] if ds.xpts_actual is not None else None,
                o_true=ds.o_point[i],
                psi_bndry_true=float(ds.axes[i][2]),
                anchor=None)
            if gm["n_xpt_pred"] < 2:
                n_fail_crit += 1
            for k in geo_all:
                geo_all[k].append(gm.get(k, float("nan")))

            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{n_eval} done")

    rel_l2s, rmse_phys, r2s, j_self = map(np.array, [rel_l2s, rmse_phys, r2s, j_self])
    rp, rt = np.array(residuals_pred), np.array(residuals_true)

    metrics = {
        "n_eval": n_eval,
        "n_params": n_params,
        "n_find_critical_fail": n_fail_crit,
        "target": target,
        "rel_l2_pct": stats_dict(rel_l2s * 100),
        "rmse_phys_Wb": stats_dict(rmse_phys),
        "r_squared": stats_dict(r2s),
        "j_self_check_pct": stats_dict(j_self * 100),
        "gs_residual": {"pred": stats_dict(rp), "true_freegs": stats_dict(rt),
                        "ratio_pred_true": float(np.nanmean(rp) / (np.nanmean(rt) + 1e-30))},
        "geometry": {k: stats_dict(np.array(v)) for k, v in geo_all.items()},
    }
    metrics["paper_reference"] = {
        "rel_l2_kan1_pct": 0.631, "rel_l2_kan2_pct": 0.912, "rel_l2_kan3_pct": 1.022,
        "r_squared": 0.9953, "rmse_phys_Wb": 1.06e-4,
    }
    if target == "plasma":
        metrics["rel_l2_total_pct"] = stats_dict(np.array(rel_l2_total) * 100)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n  ---- field-level ----")
    print(f"  rel L2:      mean {metrics['rel_l2_pct']['mean']:.4f}% "
          f"(paper KAN-1..3: 0.631/0.912/1.022%)")
    if target == "plasma":
        print(f"  rel L2 (total, coil added back): mean "
              f"{metrics['rel_l2_total_pct']['mean']:.4f}% | "
              f"std {metrics['rel_l2_total_pct']['std']:.4f}%")
    print(f"               std {metrics['rel_l2_pct']['std']:.4f}% | "
          f"median {metrics['rel_l2_pct']['median']:.4f}% | P95 {metrics['rel_l2_pct']['p95']:.4f}%")
    print(f"  RMSE phys:   mean {metrics['rmse_phys_Wb']['mean']:.3e} Wb")
    print(f"  R^2:         mean {metrics['r_squared']['mean']:.4f} (paper 0.9953)")
    print(f"  J self-check: mean {metrics['j_self_check_pct']['mean']:.3f}% "
          f"(|sum J dS - Ip|/Ip)")
    g = metrics["geometry"]
    print("  ---- geometry ----")
    for k in ("x_lo_cm", "x_up_cm", "o_point_cm", "sep_mean_cm",
              "sep_hausdorff_cm", "sep_area_rel_err_pct", "dpsi_bndry_Wb"):
        print(f"  {k:22s} mean {g[k]['mean']:.4g}")
    print(f"  ---- GS residual ----")
    print(f"  pred {metrics['gs_residual']['pred']['mean']:.3f} | "
          f"freegs truth {metrics['gs_residual']['true_freegs']['mean']:.3f} | "
          f"ratio {metrics['gs_residual']['ratio_pred_true']:.3f}")
    print(f"  find_critical fail: {n_fail_crit}/{n_eval}")
    print(f"  saved -> {out_dir}/metrics.json")


if __name__ == "__main__":
    main()
