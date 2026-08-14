"""Best/worst sample plots for the coil-input model (exp005, data_v3).

Selects the 5 worst and 5 best test samples by per-sample rel L2 for the coil
model checkpoint, and plots truth vs prediction for each: psi_total contourf
(shared levels), white separatrix, white X = X-point constraints, magenta
star = isoflux anchor. Also dumps a summary JSON of the 10 samples.

Usage:
  python -m gs_pino_dn_fno_2608.plot_best_worst_coil \
      --test-data dn_fno_2608/data_v3/test.npz \
      --checkpoint dn_fno_2608/experiments/exp005_coil_input_v3/best.pt \
      --out-dir dn_fno_2608/experiments/exp005_coil_input_v3/figures/worst_best
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gs_pino_dn_fno_2608.data_dn_fno import rel_l2_normalized
from gs_pino_dn_fno_2608.data_dn_fno_coils import DNFnoDatasetCoils
from gs_pino_dn_fno_2608.model_dn_fno import build_model

REF_ANCHOR = (1.5, 0.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-data", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    stats = {k: (np.asarray(v, dtype=np.float32) if isinstance(v, list) else np.float32(v))
             for k, v in ckpt["stats"].items() if k != "input_mode"}
    model = build_model(in_channels=2 + len(stats["scalar_mean"])).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    with np.load(args.test_data) as d:
        R, Z = d["R"], d["Z"]                      # (65,65) meshgrids, physical
        anchor = d["anchor"]                       # (N,2)
        x_coords = d["x_coords"]                   # (N,4): R_lo Z_lo R_up Z_up
        axes_psi = d["axes"]                       # (N,4): R_axis Z_axis psi_bndry psi_axis
        params = d["params"]
    ds = DNFnoDatasetCoils(args.test_data, stats=stats)

    # per-sample rel L2 (same metric as evaluate_dn_fno -> metrics.json)
    rel_l2 = np.empty(len(ds))
    with torch.no_grad():
        for i in range(len(ds)):
            x, y = ds[i]
            pred = model(x[None].to(device))          # (1,1,65,65), keep batch for rel_l2
            rel_l2[i] = float(rel_l2_normalized(pred, y[None].to(device)) * 100)

    order = np.argsort(rel_l2)
    picks = [(int(i), "worst") for i in order[-5:][::-1]] + \
            [(int(i), "best") for i in order[:5]]

    summary = []
    for rank, (i, tag) in enumerate(picks):
        x, y_norm = ds[i]
        with torch.no_grad():
            pred_norm = model(x[None].to(device)).squeeze(0).cpu().numpy()
        psi_true = ds.psi_total[ds.indices[i]].astype(np.float64)          # physical Wb
        psi_pred = pred_norm[0] * float(stats["psi_std"]) + float(stats["psi_mean"])

        xp = np.array([[x_coords[i, 0], x_coords[i, 1]], [x_coords[i, 2], x_coords[i, 3]]])
        anc = anchor[i]
        sep_level = float(axes_psi[i, 2])
        dist_ref = float(np.hypot(anc[0] - REF_ANCHOR[0], anc[1] - REF_ANCHOR[1]))
        d_min_xpt = float(np.hypot(anc[0] - xp[:, 0], anc[1] - xp[:, 1]).min())

        # shared levels over both fields so |truth - pred| is readable
        vmin, vmax = min(psi_true.min(), psi_pred.min()), max(psi_true.max(), psi_pred.max())
        levels = np.linspace(vmin, vmax, 41)

        fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4),
                                 constrained_layout=True, sharex=True, sharey=True)
        for ax, psi, title in ((axes[0], psi_true, "freegs truth"),
                               (axes[1], psi_pred, "FNO prediction (coil input)")):
            cf = ax.contourf(R, Z, psi, levels=levels, cmap="viridis")
            ax.contour(R, Z, psi, levels=[sep_level], colors="white", linewidths=1.4)
            for (r, z) in xp:                                   # X-point constraints
                ax.plot(r, z, "wx", markersize=10, markeredgewidth=2.2)
            ax.plot(*anc, "*", color="magenta", markersize=15, markeredgecolor="black",
                    markeredgewidth=0.8)                        # isoflux anchor
            ax.set_title(title, fontsize=11)
            ax.set_xlabel("R (m)")
            ax.set_ylabel("Z (m)")
        fig.colorbar(cf, ax=axes, shrink=0.9, label="psi_total (Wb)")
        fig.suptitle(
            f"[{rank:02d}] {'WORST' if tag == 'worst' else 'BEST'} #{i}  "
            f"rel L2 = {rel_l2[i]:.4f}%   |   anchor dist (1.5,0) = {dist_ref:.3f} m, "
            f"d_min_Xpt = {d_min_xpt:.3f} m\n"
            f"Ip = {params[i,0]:.2e} A, paxis = {params[i,1]:.0f} Pa, fvac = {params[i,2]:.2f}, "
            f"alpha = ({params[i,3]:.2f}, {params[i,4]:.2f}), "
            f"Xpts = ({xp[0,0]:.3f},{xp[0,1]:.3f}) & ({xp[1,0]:.3f},{xp[1,1]:.3f}), "
            f"anchor = ({anc[0]:.3f},{anc[1]:.3f})",
            fontsize=9)
        out = out_dir / f"rank{rank:02d}_{tag}_idx{i:03d}_relL2{rel_l2[i]:.3f}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"saved {out}")

        summary.append({
            "test_index": i, "rank": rank, "tag": tag, "rel_l2_pct": round(rel_l2[i], 4),
            "anchor": [round(float(anc[0]), 4), round(float(anc[1]), 4)],
            "anchor_dist_ref_m": round(dist_ref, 4), "anchor_dist_xpt_min_m": round(d_min_xpt, 4),
            "x_pts": [round(float(x), 4) for x in x_coords[i].tolist()],
            "params": [round(float(p), 6) for p in params[i].tolist()],
        })

    with open(out_dir / "samples_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nsummary -> {out_dir}/samples_summary.json")


if __name__ == "__main__":
    main()
