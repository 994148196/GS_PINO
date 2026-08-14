"""Evaluation CLI for the coil-current-input DN FNO experiment (exp001).

Mirrors evaluate_dn_fno.py (field-level rel L2 / physical RMSE, geometry
Table II metrics, GS residual diagnostic) but loads the coil-input dataset
and checkpoint. All physics diagnostics are imported from the baseline
evaluate_dn_fno.py (unchanged), so metrics are directly comparable.

Usage:
  python -m gs_pino_dn_fno_2608.evaluate_dn_fno_coils --test-data dn_fno_2608/data/test.npz \
      --checkpoint dn_fno_2608/experiments/exp001_coil_input/best.pt \
      --out-dir dn_fno_2608/experiments/exp001_coil_input
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from gs_pino_dn_fno_2608.data_dn_fno_coils import DNFnoDatasetCoils
from gs_pino_dn_fno_2608.evaluate_dn_fno import gs_residual_ratio, geometry_metrics
from gs_pino_dn_fno_2608.model_dn_fno import build_model


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
             for k, v in stats.items() if k != "input_mode"}

    model = build_model(in_channels=2 + len(stats["scalar_mean"])).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    ds = DNFnoDatasetCoils(args.test_data, stats=stats)
    n_eval = min(len(ds), args.max_samples) if args.max_samples else len(ds)

    print(f"\n{'='*70}")
    print(f"  DN-FNO coil-input evaluation | test: {n_eval} | device: {device}")
    print(f"  checkpoint: {args.checkpoint}")
    print(f"  best val rel L2: {ckpt['best_val_rel_l2']*100:.4f}% @ epoch {ckpt['best_epoch']}")
    print(f"  model params: {n_params}")
    print(f"{'='*70}")

    with np.load(args.test_data) as d:
        R, Z = d["R"], d["Z"]

    rel_l2s, rmse_phys, residuals_pred, residuals_true = [], [], [], []
    geo_all = {k: [] for k in
               ["dpsi_bndry_Wb", "o_point_cm", "x_lo_cm", "x_up_cm",
                "sep_mean_cm", "sep_hausdorff_cm", "sep_area_rel_err_pct"]}
    n_fail_crit = 0

    for i in range(n_eval):
        x, y_norm = ds[i]
        with torch.no_grad():
            pred_norm = model(x[None].to(device)).squeeze(0).cpu().numpy()

        y_norm = y_norm.numpy()
        rel_l2s.append(float(np.linalg.norm(pred_norm - y_norm) / (np.linalg.norm(y_norm) + 1e-12)))

        psi_pred = pred_norm[0] * float(stats["psi_std"]) + float(stats["psi_mean"])
        psi_true = ds.psi_total[ds.indices[i]]
        rmse_phys.append(float(np.sqrt(np.mean((psi_pred - psi_true) ** 2))))

        mask_i = (psi_true >= ds.axes[ds.indices[i]][2]).astype(np.float32)
        residuals_pred.append(gs_residual_ratio(
            psi_pred, R, Z, ds.dpdpsi[ds.indices[i]], ds.FdFdpsi[ds.indices[i]], mask_i))
        residuals_true.append(gs_residual_ratio(
            psi_true, R, Z, ds.dpdpsi[ds.indices[i]], ds.FdFdpsi[ds.indices[i]], mask_i))

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
        "input_mode": "coils",
        "rel_l2_pct": stats_dict(rel_l2s * 100),
        "rel_l2_lt_0.12_pct": float((rel_l2s < 0.0012).mean() * 100),
        "rmse_phys_Wb": stats_dict(rmse_phys),
        "gs_residual": {"pred": stats_dict(rp), "true_freegs": stats_dict(rt),
                        "ratio_pred_true": float(np.nanmean(rp) / (np.nanmean(rt) + 1e-30))},
        "geometry": {k: stats_dict(np.array(v)) for k, v in geo_all.items()},
    }
    # reference: N=500 seed 1, same train-subset size, X-point input
    # (baseline = old data/, exp002 = data_v2 — pick per test data)
    metrics["baseline_reference_n500_s1"] = {
        "rel_l2_pct_mean": 0.222, "rel_l2_pct_std": 0.007, "rmse_phys_Wb": 5.726e-5,
    }
    metrics["exp002_reference_dv2_n500_s1"] = {
        "rel_l2_pct_mean": 0.303, "rel_l2_pct_std": 0.227, "rmse_phys_Wb": 8.23e-5,
    }

    print("\n  ---- field-level ----")
    print(f"  rel L2:      mean {metrics['rel_l2_pct']['mean']:.4f}% "
          f"(baseline N=500: 0.222% ± 0.007%)")
    print(f"               std {metrics['rel_l2_pct']['std']:.4f}% | "
          f"median {metrics['rel_l2_pct']['median']:.4f}% | P95 {metrics['rel_l2_pct']['p95']:.4f}%")
    print(f"               <0.12%: {metrics['rel_l2_lt_0.12_pct']:.1f}% of samples")
    print(f"  RMSE phys:   mean {metrics['rmse_phys_Wb']['mean']:.3e} Wb "
          f"(baseline N=500: 5.73e-5)")
    print("\n  ---- geometry ----")
    for k in ["sep_mean_cm", "sep_hausdorff_cm", "sep_area_rel_err_pct",
              "x_lo_cm", "x_up_cm", "o_point_cm", "dpsi_bndry_Wb"]:
        v = metrics["geometry"][k]
        print(f"  {k:22s}: mean {v['mean']:.4f} | median {v['median']:.4f} | P95 {v['p95']:.4f}")
    print(f"  find_critical failures: {n_fail_crit}/{n_eval}")
    print("\n  ---- GS residual ----")
    print(f"  pred {metrics['gs_residual']['pred']['mean']:.3f} | "
          f"freegs truth {metrics['gs_residual']['true_freegs']['mean']:.3f} | "
          f"ratio {metrics['gs_residual']['ratio_pred_true']:.3f}")

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  saved -> {out_dir}/metrics.json")


if __name__ == "__main__":
    main()
