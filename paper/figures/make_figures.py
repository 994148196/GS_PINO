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
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

FIGDIR = Path(__file__).resolve().parent
ROOT = FIGDIR.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(FIGDIR))  # fig_geom.py sits next to this script
from gs_pino_dn_fno_2608.evaluate_dn_fno import separatrix_pts
from fig_geom import close_separatrix
DATA_V5_DN = ROOT / "dn_fno_2608" / "data_v5" / "dn"
EXP = ROOT / "dn_fno_2608" / "experiments"

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "figure.dpi": 300,
    "savefig.dpi": 300,
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
    """3-subplot: psi_total / psi_plasma / psi_coils + smooth separatrix.

    Manual layout (no tight_layout): panels keep the MAST data aspect,
    colorbars glued via inset_axes, small inter-panel whitespace.
    """
    d = np.load(DATA_V5_DN / "test.npz")
    idx = 13
    R, Z = d["R"], d["Z"]
    psi_t = d["psi_total"][idx]
    psi_p = d["psi_plasma"][idx]
    psi_c = d["psi_coils"][idx]
    xs = d["x_coords"][idx].reshape(2, 2)
    Raxis, Zaxis = d["axes"][idx][0], d["axes"][idx][1]
    # smooth LCFS: ray-traced separatrix of the true total flux at the
    # boundary level (axes[2]), identical to the experiment evaluation
    # figures — the binarised-mask contour would be pixelated. The X-point
    # cusp gaps are then closed through the X-points (close_separatrix).
    bnd = float(d["axes"][idx][2])
    xa = np.asarray(d["xpts_actual"][idx])[:, :2]
    xpts = xa[np.isfinite(xa[:, 0])]
    sep = close_separatrix(
        separatrix_pts(psi_t, R, Z, bnd, (float(Raxis), float(Zaxis))), xpts)

    # ============================================================
    # 子图间距调整区（图1）：改下面这些数字即可调整间距
    #   PW   面板宽度(in)，高度按数据纵横比自动跟随
    #   CBW  色彩条宽度   CBG  色彩条与面板的间隙
    #   HWS  色彩条与下一面板之间的水平空白（≈2汉字≈0.24in）
    #   LEFT/RIGHT 左右边距   TOP/BOT 上下边距
    # 面板 x 排列：LEFT + n*(PW+CBG+CBW+HWS) + ... + RIGHT
    # ============================================================
    PW = 2.05                       # panel width (in), height follows aspect
    PH = PW * (Z.max() - Z.min()) / (R.max() - R.min())
    CBW, CBG = 0.11, 0.08           # colorbar width / gap after the panel
    HWS = 1.8                     # whitespace between colorbar and next panel
    LEFT, RIGHT = 0.30, 0.30
    TOP, BOT = 0.30, 0.30
    W = LEFT + 3 * PW + 3 * (CBG + CBW) + 2 * HWS + RIGHT
    H = TOP + PH + BOT
    fig = plt.figure(figsize=(W, H))
    axes = []
    for col in range(3):
        x0 = LEFT + col * (PW + CBG + CBW + HWS)
        axes.append(fig.add_axes([x0 / W, BOT / H, PW / W, PH / H]))

    titles = [r"(a) $\psi_{\mathrm{total}}$  (plasma + coils)",
              r"(b) $\psi_{\mathrm{plasma}}$",
              r"(c) $\psi_{\mathrm{coils}}$  = Σ I_k G_k "]
    for ax, psi, ttl in zip(axes, (psi_t, psi_p, psi_c), titles):
        cf = ax.contourf(R, Z, psi, levels=40, cmap="viridis")
        ax.contour(R, Z, psi, levels=16, colors="k", linewidths=0.35, alpha=0.5)
        ax.plot(sep[:, 0], sep[:, 1], "w-", lw=1.8, solid_capstyle="round")
        for r, z in COIL_RZ:
            ax.plot(r, z, "s", ms=4, color="#d62728", mec="k", mew=0.3, zorder=5)
        cax = ax.inset_axes([1.0 + CBG / PW, 0.0, CBW / PW, 1.0],
                            transform=ax.transAxes)
        plt.colorbar(cf, cax=cax)
        ax.set_title(ttl, fontsize=15)
        ax.set_xlabel("R (m)")
        ax.set_aspect("equal")
        ax.set_xlim(R.min(), R.max())
        ax.set_ylim(Z.min(), Z.max())
        ax.set_ylabel("Z (m)")
        ax.tick_params(length=3)
    # geometry markers only on the total-flux panel
    axes[0].plot(xs[:, 0], xs[:, 1], "r*", ms=13, zorder=6, label="X-points")
    axes[0].plot(Raxis, Zaxis, "ko", ms=6, zorder=6, label="magnetic axis")
    axes[0].legend(loc="lower left", fontsize=11, framealpha=0.8)
    fig.savefig(FIGDIR / "fig1_freeboundary_problem.png", bbox_inches="tight")
    plt.close(fig)
    print("Fig 1 saved (3 subplots, no suptitle)")



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
    """FNO pipeline on ONE single row: Input -> 1x1 conv -> FNOBlock x4 ->
    1x1 proj -> psi_plasma -> psi_total (analytic coil superposition, red
    arrow from above). 4 FNO blocks condensed into one "x4" box.
    """
    # xlim/ylim wrap the content tightly: content spans x 0.1-12.1 and
    # y 2.0-4.3, so the canvas is cropped to (0, 12.6) x (1.6, 4.6) to
    # remove the dead right/bottom/top bands
    fig, ax = plt.subplots(figsize=(9.6, 2.8))
    ax.set_xlim(0, 12.6)
    ax.set_ylim(1.6, 4.6)
    ax.axis("off")

    # one row (y_center = 3.0)
    _box(ax, 0.1, 2.0, 2.3, 2.0,
         "Input 18ch\n\n @65x65 R, Z\n"
         "5 plasma params\n"
         "11 coil currents",
         fc="#dbe9f6", ec="#2c6fbb", fs=10)
    _box(ax, 2.9, 2.4, 1.4, 1.2, "1x1 conv\n\n-> 64ch",
         fc="#dbe9f6", ec="#2c6fbb", fs=11)
    _box(ax, 4.8, 2.0, 2.4, 2.0,
         "FNOBlock x4\n\nSpectralConv\n16x16 modes\n+ 1x1 + GELU",
         fc="#e8f3e8", ec="#2e8b57", fs=11)
    _box(ax, 7.7, 2.1, 1.4, 1.8, r"1x1 proj" + "\n\n" + r"->$\psi_{plasma}$," + "\n" + r"$J_{plasma}$",
         fc="#fdeeda", ec="#d07b00", fs=11)
    # psi_total box widened to 12.1 to fill the canvas (was 11.75, leaving
    # a dead band on the right)
    _box(ax, 9.5, 2.3, 2.6, 1.4,
         r"$\psi_{total} = \psi_{plasma}$" + " \n" + r"$+ \sum I_k G_k (exact)$",
         fc="#f6d6d6", ec="#b03030", fs=10)

    _arrow(ax, 2.4, 3.0, 2.9, 3.0)
    _arrow(ax, 4.3, 3.0, 4.8, 3.0)
    _arrow(ax, 7.2, 3.0, 7.7, 3.0)
    _arrow(ax, 9.1, 3.0, 9.55, 3.0)
    # coil field enters psi_total analytically from above
    _arrow(ax, 10.5, 4.2, 10.5, 3.75, color="#b03030", lw=1.6)
    ax.text(10.5, 4.35, "coil field (analytic)", ha="center",
            fontsize=10, color="#b03030")

    # ax.text(7.0, 4.95, "Coil-separation design: network predicts only the plasma contribution;",
    #         ha="center", fontsize=13, fontweight="bold")
    # ax.text(7.0, 4.45, "coil field (183% of |psi_total| amplitude) added analytically via Green functions",
    #         ha="center", fontsize=11, color="#555")

    fig.savefig(FIGDIR / "fig3_architecture.png", bbox_inches="tight")
    plt.close(fig)
    print("Fig 3 saved (single row: Input -> x4 FNO -> psi_total)")


# ---------------------------------------------------------------- Fig 4
def fig4_training():
    """Two panels: (a) training loss of both schemes, (b) validation rel L2
    of both schemes. Each panel keeps its own y-axis — the training loss
    is an MSE in the normalized psi space (no %), the validation curve is
    the physical relative L2 error in % — both on log scale. The two
    quantities have different meanings and are NOT merged onto one axis.
    Stage-switch line on the validation panel.
    """
    h1 = json.load(open(EXP / "exp101_pino_rhs_n500" / "history.json"))
    h2 = json.load(open(EXP / "exp102_pino_twostage_n500" / "history.json"))
    ep1, ep2 = h1["epoch"], h2["epoch"]
    stage2 = np.array(h2["stage"])
    switch = int(np.where(stage2 == 2)[0][0]) if (stage2 == 2).any() else None

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5))
    # ============================================================
    # 子图间距调整区（图4）：
    #   wspace  两面板间空白（比例，越小越挤）
    #   left/right/top/bottom  图内边距（比例）
    # ============================================================
    fig.subplots_adjust(left=0.10, right=0.97, top=0.86, bottom=0.17,
                        wspace=0.3)

    # (a) training loss (normalized-space MSE, log, no %)
    ax = axes[0]
    ax.plot(ep1, h1["train_loss"], color="#2c6fbb", lw=1.2,
            label="scheme 1 (single-stage)")
    ax.plot(ep2, h2["train_loss"], color="#2e8b57", lw=1.2,
            label="scheme 2 (two-stage)")
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("training loss")
    ax.legend(fontsize=12, loc="upper right")
    ax.set_title("(a) Training loss")

    # (b) validation rel L2 (%, log), stage-switch line for scheme 2
    ax = axes[1]
    ax.plot(ep1, np.array(h1["val_rel_l2"]) * 100, color="#2c6fbb", lw=1.2,
            label="scheme 1 (single-stage)")
    ax.plot(ep2, np.array(h2["val_rel_l2"]) * 100, color="#2e8b57", lw=1.2,
            label="scheme 2 (two-stage)")
    if switch is not None:
        ax.axvline(ep2[switch], color="r", ls=":", lw=1,
                   label="stage switch (e%d)" % ep2[switch])
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation rel L2 (%)")
    ax.legend(fontsize=12, loc="upper right")
    ax.set_title("(b) Validation rel L2")

    fig.savefig(FIGDIR / "fig4_loss_design.png", bbox_inches="tight")
    plt.close(fig)
    print("Fig 4 saved (train panel + validation panel)")


# ---------------------------------------------------------------- Fig 6
def fig5_comparison():
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

    # for xi, (off, vals) in zip(x, [(-w, baseline), (0, scheme1), (w, scheme2)]):
    #     for v in vals:
    #         if not np.isnan(v):
    #             ax.text(xi + off, v + 0.03, f"{v:.2f}", ha="center", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("error (%)")
    ax.legend(fontsize=10)
    ax.set_title("(a) Test-set comparison (MAST DN, N=500)")
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
    ax.legend(fontsize=10)
    ax.set_title("(b) Validation curves (best-ckpt selection)")

    fig.tight_layout()
    fig.savefig(FIGDIR / "fig5_comparison.png", bbox_inches="tight")
    plt.close(fig)
    print("Fig 6 saved")


if __name__ == "__main__":
    # fig1_freeboundary()
    # fig2_evolution()
     fig3_architecture()
    # fig4_training()
    # fig5_comparison()
    # print("All figures saved to", FIGDIR)
