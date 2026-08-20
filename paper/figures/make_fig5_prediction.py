"""Fig 5 for the paper, in the style of APS (物理学报) KAN paper Fig. 3.

Best test sample of EACH scheme (2 rows):
  row 1: scheme 1 (single-stage RHS constraint)
  row 2: scheme 2 (two-stage self-consistent)
  columns: (a) psi comparison: colormap of psi_total + solid contours (truth)
               and dashed contours (prediction) + LCFS overlay
           (b) relative error |psi_pred - psi_true| / |psi_true| (full domain,
               NaN only where the truth flux is near zero, where division is
               meaningless)
           (c) PDE residual |Delta* psi_pred + mu0 R J| (full interior grid;
               scheme 1: J = data current density (its frozen RHS);
               scheme 2: J = network's own prediction inside the plasma
               (self-consistency residual) and 0 outside; outside the plasma
               J = 0 in both cases, where the residual reduces to
               |Delta* psi_pred|, a smoothness / harmonicity check)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

FIGDIR = Path(__file__).resolve().parent
ROOT = FIGDIR.parent.parent
EXP = ROOT / "dn_fno_2608" / "experiments"
TEST_NPZ = ROOT / "dn_fno_2608" / "data_v5" / "dn" / "test.npz"

SCHEMES = [
    ("scheme 1 (single-stage)", "exp101_pino_rhs_n500", 1),
    ("scheme 2 (two-stage)", "exp102_pino_twostage_n500", 2),
]

sys.path.insert(0, str(ROOT / "src"))
from gs_pino_dn_fno_2608.evaluate_dn_fno import lap_star
from gs_pino_dn_fno_2608.model_dn_fno import build_model
from gs_pino_fno_phys.data_pino import DNPinoDataset
from gs_pino_fno_phys.losses_pino import denorm_j, denorm_psi

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 200,
})

COIL_RZ = [
    (0.587, 1.55), (0.587, -1.55), (0.700, 1.80), (0.700, -1.80),
    (0.800, 1.95), (0.800, -1.95), (1.150, 1.80), (1.150, -1.80),
    (1.350, 1.55), (1.350, -1.55), (0.340, 0.00),
]
MU0 = 4e-7 * np.pi


def best_sample(ds, expdir, n_out):
    ckpt = torch.load(EXP / expdir / "best.pt", map_location="cpu",
                      weights_only=False)
    stats = ckpt["stats"]
    model = build_model(in_channels=2 + len(stats["scalar_mean"]),
                        out_channels=n_out)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    rows = json.load(open(EXP / expdir / "figures" / "stats_per_sample.json"))
    rl2 = np.array(rows["rel_l2_pct"])
    raw_idx = np.array(rows["indices"])
    sel = int(np.nanargmin(rl2))
    with torch.no_grad():
        pred = model(ds[sel]["x"][None])
    psi_p = denorm_psi(pred[:, 0:1], stats).numpy()[0, 0]
    j_p = None
    if n_out > 1:
        j_p = denorm_j(pred[:, 1:2], stats).numpy()[0, 0]
    return raw_idx[sel], sel, psi_p, j_p, stats, rl2[sel]


def main():
    # dataset must be built with the checkpoint stats (train-pool
    # normalization the models were trained with), otherwise denormalising
    # the network output with the test-pool stats corrupts psi / J values
    ckpt0 = torch.load(EXP / SCHEMES[0][1] / "best.pt", map_location="cpu",
                       weights_only=False)
    ds = DNPinoDataset(str(TEST_NPZ), stats=ckpt0["stats"])
    R, Z = ds.R_phys, ds.Z_phys
    rows = []
    for name, expdir, n_out in SCHEMES:
        j_raw, sel, psi_p, j_p, stats, rl2s = best_sample(ds, expdir, n_out)

        psi_tot_t = ds.psi_total[j_raw]
        psi_tot_p = psi_p + np.einsum("kij,k->ij",
                                      ds.greens[j_raw], ds.coil_currents[j_raw])
        mask = ds.mask[j_raw] > 0.5
        mask_s = gaussian_filter(ds.mask[j_raw].astype(float), sigma=0.9)

        # relative error over the FULL computation domain; only points where
        # the truth flux is near zero (psi crosses zero near the separatrix)
        # are excluded — there pointwise division is meaningless
        safe = np.abs(psi_tot_t) >= 0.02 * np.abs(psi_tot_t).max()
        rel = np.full_like(psi_tot_t, np.nan)
        rel[safe] = np.abs(psi_tot_p[safe] - psi_tot_t[safe]) / np.abs(psi_tot_t[safe])

        # PDE residual |Delta* psi_pred + mu0 R J| (lap_star interior grid),
        # full domain. J = data current density for scheme 1 (its frozen RHS);
        # J = network prediction inside the plasma for scheme 2 (self-
        # consistency residual), 0 outside in both cases. Only the 1-cell FD
        # boundary ring is NaN.
        lap = lap_star(psi_p, R, Z)
        r_c = R[1:-1, 1:-1]
        j_field = ds.j_phys[j_raw] if n_out == 1 else np.where(mask, j_p, 0.0)
        resid = np.full_like(psi_tot_t, np.nan)
        resid[1:-1, 1:-1] = np.abs(lap + MU0 * r_c * j_field[1:-1, 1:-1])

        mre = float(np.nanmean(rel))
        rmse = float(np.sqrt(np.mean((psi_tot_p - psi_tot_t) ** 2)))
        print(f"  {name}: sample #{j_raw} | rel L2 {rl2s:.4f}% | "
              f"MRE {mre*100:.3f}% | RMSE {rmse:.2e} Wb | "
              f"resid mean {np.nanmean(resid):.4f} Wb/m2")
        rows.append(dict(name=name, psi_tot_t=psi_tot_t, psi_tot_p=psi_tot_p,
                         rel=rel, resid=resid, mask_s=mask_s))

    # ---- figure: 2 rows (one scheme each) x 3 columns --------------------
    # compact columns (small figure width, tiny w_pad); each panel keeps its
    # own vertical colorbar on the right, flush against the panel
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 7.8))
    n_lvls = 15
    titles = ["(a) ψ comparison", "(b) Relative error", "(c) PDE residual",
              "(d) ψ comparison", "(e) Relative error", "(f) PDE residual"]

    for row, d in enumerate(rows):
        psi_t, psi_p, rel, resid, mask_s = (d["psi_tot_t"], d["psi_tot_p"],
                                            d["rel"], d["resid"], d["mask_s"])
        lvls = np.linspace(psi_t.min(), psi_t.max(), n_lvls)

        # (a/d) psi comparison: colormap + solid (true) / dashed (pred) contours
        ax = axes[row, 0]
        cf = ax.contourf(R, Z, psi_t, levels=40, cmap="viridis")
        ax.contour(R, Z, psi_t, levels=lvls, colors="k", linewidths=0.6)
        ax.contour(R, Z, psi_p, levels=lvls, colors="w", linewidths=0.7,
                   linestyles="--")
        ax.contour(R, Z, mask_s, levels=[0.5], colors="cyan", linewidths=1.4)
        for r, z in COIL_RZ:
            ax.plot(r, z, "s", ms=4, color="#d62728", mec="k", mew=0.3, zorder=5)
        fig.colorbar(cf, ax=ax, shrink=0.85, pad=0.02, label="ψ (Wb)")
        if row == 0:
            ax.plot([], [], "k-", lw=1.2, label="truth")
            ax.plot([], [], "w--", lw=1.2, label="prediction")
            ax.plot([], [], color="cyan", lw=1.4, label="LCFS (pred)")
            ax.legend(loc="lower left", fontsize=8, framealpha=0.85)
        ax.set_ylabel("Z (m)")
        ax.set_aspect("equal")

        # (b/e) relative error (full domain)
        ax = axes[row, 1]
        im = ax.imshow(rel * 100, extent=[R.min(), R.max(), Z.min(), Z.max()],
                       origin="lower", cmap="magma",
                       vmin=0, vmax=np.nanpercentile(rel * 100, 97))
        fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02, label="rel. error (%)")
        ax.contour(R, Z, mask_s, levels=[0.5], colors="w", linewidths=1.0)
        ax.set_aspect("equal")

        # (c/f) PDE residual (full interior grid; J = 0 outside the plasma)
        ax = axes[row, 2]
        vmax = np.nanpercentile(resid, 97)
        im = ax.imshow(resid, extent=[R.min(), R.max(), Z.min(), Z.max()],
                       origin="lower", cmap="hot",
                       vmin=0, vmax=vmax)
        fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02, label="|Δ*ψ + μ₀RJ| (Wb/m²)")
        ax.contour(R, Z, mask_s, levels=[0.5], colors="cyan", linewidths=1.2)
        ax.set_aspect("equal")

        ax.set_title(titles[3 * row + 0])
        axes[row, 1].set_title(titles[3 * row + 1])
        axes[row, 2].set_title(titles[3 * row + 2])
        # row labels on the left, rotated, tight against the panels
        ax.text(-0.34, 0.5, d["name"], rotation=90,
                transform=ax.transAxes,
                va="center", ha="center", fontsize=10)
    for ax in axes[1]:
        ax.set_xlabel("R (m)")
    fig.tight_layout(w_pad=0.08, h_pad=0.35)
    out = FIGDIR / "fig5_predictions.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Fig 5 saved: {out}")


if __name__ == "__main__":
    main()
