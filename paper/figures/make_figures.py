"""Generate paper figures 1-4 & 6 (fig5 has its own script: make_fig5_prediction.py).

All figures are written next to this script (paper/figures/).

Fig 1: free-boundary problem, 3 subplots — psi_total / psi_plasma / psi_coils
       (data_v5/dn sample; coil-separation motivation)
Fig 2: method evolution (freegs -> PINN -> PINO)
Fig 3: FNO architecture + coil separation
Fig 4: training curves of both schemes (train loss + validation rel L2,
       stage-switch marker for the two-stage scheme)
Fig 6: three-method comparison bars + validation curves
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy.ndimage import gaussian_filter

FIGDIR = Path(__file__).resolve().parent
ROOT = FIGDIR.parent.parent
DATA_V5_DN = ROOT / "dn_fno_2608" / "data_v5" / "dn"
EXP = ROOT / "dn_fno_2608" / "experiments"

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "axes.grid": True,
    "grid.alpha": 0.3,
})


# ---------------------------------------------------------------- helpers
def _box(ax, x, y, w, h, text, fc="#dbe9f6", ec="#2c6fbb", fs=8, lw=1.2, tc="k"):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                       fc=fc, ec=ec, lw=lw, zorder=3)
    ax.add_patch(b)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=tc, zorder=4)


def _arrow(ax, x0, y0, x1, y1, color="#333", lw=1.4, style="-|>", ls="-"):
    a = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style,
                        mutation_scale=14, color=color, lw=lw,
                        linestyle=ls, zorder=2)
    ax.add_patch(a)


# MAST coil centres (freegs machine definition, approx.)
COIL_RZ = [
    (0.587, 1.55), (0.587, -1.55),
    (0.700, 1.80), (0.700, -1.80),
    (0.800, 1.95), (0.800, -1.95),
    (1.150, 1.80), (1.150, -1.80),
    (1.350, 1.55), (1.350, -1.55),
    (0.340, 0.00),
]


# ---------------------------------------------------------------- Fig 1
def fig1_freeboundary():
    """3-subplot: psi_total / psi_plasma / psi_coils + smooth separatrix."""
    d = np.load(DATA_V5_DN / "test.npz")
    idx = 13
    R, Z = d["R"], d["Z"]
    psi_t = d["psi_total"][idx]
    psi_p = d["psi_plasma"][idx]
    psi_c = d["psi_coils"][idx]
    xs = d["x_coords"][idx].reshape(2, 2)
    Raxis, Zaxis = d["axes"][idx][0], d["axes"][idx][1]
    mask_s = gaussian_filter(d["mask"][idx].astype(float), sigma=0.9)

    fig, axes = plt.subplots(1, 3, figsize=(11.6, 4.0))
    titles = [r"$\psi_{\mathrm{total}}$" + "  (plasma + coils)",
              r"$\psi_{\mathrm{plasma}}$",
              r"$\psi_{\mathrm{coils}}$  = Σ I_k G_k (analytic)"]
    for ax, psi, ttl in zip(axes, (psi_t, psi_p, psi_c), titles):
        cf = ax.contourf(R, Z, psi, levels=40, cmap="viridis")
        ax.contour(R, Z, psi, levels=16, colors="k", linewidths=0.35, alpha=0.5)
        ax.contour(R, Z, mask_s, levels=[0.5], colors="w", linewidths=1.6)
        for r, z in COIL_RZ:
            ax.plot(r, z, "s", ms=4, color="#d62728", mec="k", mew=0.3, zorder=5)
        plt.colorbar(cf, ax=ax, shrink=0.88, pad=0.02)
        ax.set_title(ttl, fontsize=9)
        ax.set_xlabel("R (m)")
        ax.set_aspect("equal")
        ax.set_xlim(R.min(), R.max())
        ax.set_ylim(Z.min(), Z.max())
    # geometry markers only on the total-flux panel
    axes[0].plot(xs[:, 0], xs[:, 1], "r*", ms=13, zorder=6, label="X-points")
    axes[0].plot(Raxis, Zaxis, "ko", ms=6, zorder=6, label="magnetic axis")
    axes[0].legend(loc="lower left", fontsize=7, framealpha=0.8)
    axes[0].set_ylabel("Z (m)")
    for ax in axes[1:]:
        ax.set_ylabel("Z (m)")
    fig.suptitle("Free-boundary equilibrium (MAST DN, freegs) — coil separation",
                 fontsize=11)
    fig.tight_layout(w_pad=0.2)
    fig.savefig(FIGDIR / "fig1_freeboundary_problem.png", bbox_inches="tight")
    plt.close(fig)
    print("Fig 1 saved (3 subplots)")


# ---------------------------------------------------------------- Fig 2
def fig2_evolution():
    fig, ax = plt.subplots(figsize=(8.6, 2.7))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis("off")

    _box(ax, 0.15, 1.0, 2.6, 1.1,
         "Classical solver (freegs)\n"
         "Picard iteration + FD\n"
         "von Hagenow Green fn.\n"
         "~seconds per solve",
         fc="#e8d9f0", ec="#7b3fa0")
    _box(ax, 3.75, 1.0, 2.6, 1.1,
         "PINN\n"
         "MLP + autograd residual\n"
         "one solution per training\n"
         "retrain per parameter set",
         fc="#fdeeda", ec="#d07b00")
    _box(ax, 7.3, 1.0, 2.6, 1.1,
         "PINO (this work)\n"
         "FNO operator + GS residual\n"
         "parameter space -> field\n"
         "~2 ms per solve",
         fc="#dcecdd", ec="#2e8b57")

    _arrow(ax, 2.9, 1.55, 3.6, 1.55)
    _arrow(ax, 6.5, 1.55, 7.2, 1.55)

    ax.text(5.0, 2.55, "ML acceleration of GS equilibrium solving",
            ha="center", fontsize=11, fontweight="bold")
    ax.text(0.15, 0.35, "exact, slow", fontsize=8, color="#555")
    ax.text(3.9, 0.35, "physics-embedded, per-case", fontsize=8, color="#555")
    ax.text(7.5, 0.35, "physics-embedded, operator-level", fontsize=8, color="#555")
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig2_evolution.png", bbox_inches="tight")
    plt.close(fig)
    print("Fig 2 saved")


# ---------------------------------------------------------------- Fig 3
def fig3_architecture():
    fig, ax = plt.subplots(figsize=(9.0, 4.4))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5.2)
    ax.axis("off")

    _box(ax, 0.2, 2.0, 2.4, 2.0,
         "Input 18ch @65x65\n"
         "R, Z\n"
         "5 plasma params\n"
         "11 coil currents",
         fc="#dbe9f6", ec="#2c6fbb")
    _box(ax, 3.2, 2.35, 1.6, 1.3, "1x1 conv\nlift -> 64ch", fc="#dbe9f6", ec="#2c6fbb")

    for i in range(4):
        x0 = 5.2 + i * 1.55
        _box(ax, x0, 1.9, 1.3, 2.2,
             f"FNOBlock\n{i+1}\nSpectralConv\n16x16 modes\n+1x1 + GELU",
             fc="#e8f3e8", ec="#2e8b57", fs=7)

    _box(ax, 11.6, 2.35, 1.5, 1.3, "1x1 proj\n-> 1 or 2 ch", fc="#e8f3e8", ec="#2e8b57")

    _box(ax, 8.6, 0.1, 2.0, 1.0,
         "psi_plasma (predicted)", fc="#fdeeda", ec="#d07b00")
    _box(ax, 11.4, 0.1, 2.2, 1.0,
         "psi_total = psi_plasma\n+ sum I_k G_k (exact)", fc="#f6d6d6", ec="#b03030")

    _arrow(ax, 2.6, 3.0, 3.2, 3.0)
    _arrow(ax, 4.8, 3.0, 5.2, 3.0)
    for i in range(3):
        _arrow(ax, 6.5 + i * 1.55, 3.0, 7.15 + i * 1.55, 3.0)
    _arrow(ax, 11.0, 3.0, 11.6, 3.0)
    _arrow(ax, 9.6, 1.05, 9.6, 1.6)
    _arrow(ax, 11.0, 1.05, 11.0, 1.6)
    _arrow(ax, 10.6, 0.6, 11.4, 0.6, color="#b03030")

    ax.text(7.0, 4.7, "Coil-separation design: network predicts only plasma contribution;",
            ha="center", fontsize=10, fontweight="bold")
    ax.text(7.0, 4.25, "coil field (183% of |psi_total| amplitude) added analytically via Green functions",
            ha="center", fontsize=8, color="#555")

    fig.tight_layout()
    fig.savefig(FIGDIR / "fig3_architecture.png", bbox_inches="tight")
    plt.close(fig)
    print("Fig 3 saved")


# ---------------------------------------------------------------- Fig 4
def fig4_training():
    """Both schemes: training loss + validation rel L2, stage switch marker."""
    h1 = json.load(open(EXP / "exp101_pino_rhs_n500" / "history.json"))
    h2 = json.load(open(EXP / "exp102_pino_twostage_n500" / "history.json"))
    ep1, ep2 = h1["epoch"], h2["epoch"]
    stage2 = np.array(h2["stage"])
    switch = int(np.where(stage2 == 2)[0][0]) if (stage2 == 2).any() else None

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))

    ax = axes[0]
    ax.plot(ep1, h1["train_loss"], color="#7fb3d5", lw=1.2,
            label="scheme 1 (single-stage)")
    ax.plot(ep2, h2["train_loss"], color="#2e8b57", lw=1.2,
            label="scheme 2 (two-stage)")
    if switch is not None:
        ax.axvline(ep2[switch], color="r", ls=":", lw=1)
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("training loss (log)")
    ax.legend(fontsize=7)
    ax.set_title("Training loss")

    ax = axes[1]
    ax.plot(ep1, np.array(h1["val_rel_l2"]) * 100, color="#7fb3d5", lw=1.2,
            label="scheme 1 (single-stage)")
    ax.plot(ep2, np.array(h2["val_rel_l2"]) * 100, color="#2e8b57", lw=1.2,
            label="scheme 2 (two-stage)")
    if switch is not None:
        ax.axvline(ep2[switch], color="r", ls=":", lw=1,
                   label="stage switch (e%d)" % ep2[switch])
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation rel L2 (%) (log)")
    ax.legend(fontsize=7)
    ax.set_title("Validation error")

    fig.tight_layout()
    fig.savefig(FIGDIR / "fig4_loss_design.png", bbox_inches="tight")
    plt.close(fig)
    print("Fig 4 saved (two schemes, train+val)")


# ---------------------------------------------------------------- Fig 6
def fig6_comparison():
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))

    # left: bar chart (renamed, no exp codes)
    ax = axes[0]
    labels = ["rel L2\nmean %", "rel L2\nmedian %", "Ip err\n%", "J (core)\n%"]
    baseline = [0.84, 0.66, np.nan, np.nan]
    scheme1 = [0.72, 0.59, np.nan, np.nan]
    scheme2 = [0.80, 0.64, 0.21, 1.36]

    x = np.arange(len(labels))
    w = 0.26
    ax.bar(x - w, baseline, w, label="baseline (data-only)", color="#b0b0b0")
    ax.bar(x, scheme1, w, label="scheme 1 (single-stage)", color="#7fb3d5")
    ax.bar(x + w, scheme2, w, label="scheme 2 (two-stage)", color="#2e8b57")

    for xi, (off, vals) in zip(x, [(-w, baseline), (0, scheme1), (w, scheme2)]):
        for v in vals:
            if not np.isnan(v):
                ax.text(xi + off, v + 0.03, f"{v:.2f}", ha="center", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("error (%)")
    ax.legend(fontsize=7)
    ax.set_title("Test-set comparison (MAST DN, N=500)")
    ax.set_ylim(0, 1.6)

    # right: validation curves
    ax = axes[1]
    for folder, color, name in [
            ("exp011_coil_input_v5/model_b18ch_coils_mix", "#b0b0b0", "baseline"),
            ("exp101_pino_rhs_n500", "#7fb3d5", "scheme 1"),
            ("exp102_pino_twostage_n500", "#2e8b57", "scheme 2")]:
        try:
            h = json.load(open(EXP / folder / "history.json"))
            ep = h["epoch"]
            val = np.array(h["val_rel_l2"]) * 100
            ax.plot(ep, val, color=color, lw=1.2, label=name)
        except Exception as e:
            print(f"  skip {folder}: {e}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation rel L2 (%)")
    ax.set_yscale("log")
    ax.legend(fontsize=7)
    ax.set_title("Validation curves (best-ckpt selection)")

    fig.tight_layout()
    fig.savefig(FIGDIR / "fig6_comparison.png", bbox_inches="tight")
    plt.close(fig)
    print("Fig 6 saved")


if __name__ == "__main__":
    fig1_freeboundary()
    fig2_evolution()
    fig3_architecture()
    fig4_training()
    fig6_comparison()
    print("All figures saved to", FIGDIR)
