"""Fig 5 for the paper, in the style of APS (物理学报) KAN paper Fig. 3.

Best test sample of EACH scheme (2 rows):
  row 1: scheme 1 (single-stage RHS constraint)
  row 2: scheme 2 (two-stage self-consistent)
  columns: (a) psi comparison: colormap of psi_total + solid contours (truth)
               and dashed contours (prediction); LCFS drawn as the smooth
               ray-traced separatrix (cyan = truth, white dashed = predicted),
               truth X-points / magnetic axis marked
           (b) relative error |psi_pred - psi_true| / |psi_true| (full domain,
               NaN only where the truth flux is near zero, where division is
               meaningless)
           (c) PDE residual |Delta* psi_pred + mu0 R J| inside the plasma
               (vacuum is NaN/white): scheme 1 uses the data current density
               (its frozen RHS); scheme 2 uses the network's own prediction
               (self-consistency residual). Outside the plasma J is physically
               zero and the residual would reduce to |Delta* psi_pred|, whose
               magnitude is dominated by high-frequency approximation noise of
               the network's vacuum field — an order of magnitude above the
               in-plasma values — so a shared full-domain scale would hide the
               physics.

Layout is fully manual (no tight_layout): each panel keeps the MAST data
aspect, colorbars are glued to their panel via inset_axes, and the white
space between subplots is ~two CJK characters wide (~0.25 in).
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

FIGDIR = Path(__file__).resolve().parent
ROOT = FIGDIR.parent.parent
EXP = ROOT / "dn_fno_2608" / "experiments"
TEST_NPZ = ROOT / "dn_fno_2608" / "data_v5" / "dn" / "test.npz"

SCHEMES = [
    (" ", "exp101_pino_rhs_n500", 1),
    (" ", "exp102_pino_twostage_n500", 2),
]

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(FIGDIR))  # fig_geom.py sits next to this script
from fig_geom import sample_geoms
from gs_pino_dn_fno_2608.evaluate_dn_fno import lap_star
from gs_pino_dn_fno_2608.model_dn_fno import build_model
from gs_pino_fno_phys.data_pino import DNPinoDataset
from gs_pino_fno_phys.losses_pino import denorm_j, denorm_psi

plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 13,
    "axes.labelsize": 13,
    "figure.dpi": 300,
    "savefig.dpi": 300,
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
        geoms = sample_geoms(ds, j_raw, psi_tot_t, psi_tot_p)

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
        # figure shows the in-plasma residual only: outside the plasma J is
        # physically zero and the residual reduces to |Delta* psi_pred|, whose
        # magnitude is dominated by network approximation noise (above the
        # in-plasma values) — a shared full-domain scale would hide the physics
        resid[~mask] = np.nan

        mre = float(np.nanmean(rel))
        rmse = float(np.sqrt(np.mean((psi_tot_p - psi_tot_t) ** 2)))
        print(f"  {name}: sample #{j_raw} | rel L2 {rl2s:.4f}% | "
              f"MRE {mre*100:.3f}% | RMSE {rmse:.2e} Wb | "
              f"resid mean {np.nanmean(resid):.4f} Wb/m2")
        rows.append(dict(name=name, psi_tot_t=psi_tot_t, psi_tot_p=psi_tot_p,
                         rel=rel, resid=resid, geoms=geoms))

    # shared color scale per quantity across both rows (same physical
    # quantity -> same colormap range; per-panel vmax would make the two
    # schemes look artificially different)
    rel_vmax = max(np.nanpercentile(d["rel"] * 100, 97) for d in rows)
    resid_vmax = max(np.nanpercentile(d["resid"], 97) for d in rows)

    # ---- figure: 2 rows (one scheme each) x 3 columns --------------------
    # manual layout, no tight_layout: panels keep the MAST data aspect
    # (Z span 4.0 m / R span 1.9 m); colorbars glued via inset_axes; the
    # whitespace between panels is ~2 CJK characters (~0.24 in at 9 pt)
    # ============================================================
    # 子图间距调整区（图5）：
    #   PW   面板宽度(in)，高度按数据纵横比自动跟随
    #   CBW  色彩条宽度   CBG  色彩条与面板的间隙
    #   HWS  色彩条与下一面板之间的水平空白（越小左右越挤）
    #   VGAP 两行之间的竖直空白（含上行标题，越小上下越挤）
    #   LEFT/RIGHT 左右边距   TOP/BOT 上下边距
    # 面板 x 排列：LEFT + n*(PW+CBG+CBW+HWS) + ... + RIGHT
    # 面板 y 排列：BOT + row*(PH+VGAP)（row=0 在上）
    # ============================================================
    PW = 1.55                       # panel width (in), height follows aspect
    PH = PW * (Z.max() - Z.min()) / (R.max() - R.min())
    CBW, CBG = 0.09, 0.10           # colorbar width / gap after the panel
    HWS = 1.6                     # whitespace between colorbar and next panel
    LEFT, RIGHT = 0.50, 0.30        # left margin holds the rotated row labels
    VGAP = 0.8                    # vertical band between rows (title + gap)
    TOP, BOT = 0.28, 0.16
    W = LEFT + 3 * PW + 3 * (CBG + CBW) + 2 * HWS + RIGHT
    H = TOP + 2 * PH + VGAP + BOT
    fig = plt.figure(figsize=(W, H))
    axes = np.empty((2, 3), dtype=object)
    for row in range(2):
        for col in range(3):
            x0 = LEFT + col * (PW + CBG + CBW + HWS)
            y0 = BOT + (1 - row) * (PH + VGAP)          # row 0 on top
            axes[row, col] = fig.add_axes([x0 / W, y0 / H, PW / W, PH / H])

    n_lvls = 15
    titles = ["(a) ψ comparison", "(b) Relative error", "(c) PDE residual",
              "(d) ψ comparison", "(e) Relative error", "(f) PDE residual"]

    for row, d in enumerate(rows):
        psi_t, psi_p, rel, resid, g = (d["psi_tot_t"], d["psi_tot_p"],
                                       d["rel"], d["resid"], d["geoms"])
        lvls = np.linspace(psi_t.min(), psi_t.max(), n_lvls)
        sep_t = g["sep_t"]  # NaN rows = rays that missed, line breaks there

        # (a/d) psi comparison: colormap + solid (true) / dashed (pred)
        # contours; smooth ray-traced LCFS + geometry markers
        ax = axes[row, 0]
        cf = ax.contourf(R, Z, psi_t, levels=40, cmap="viridis")
        ax.contour(R, Z, psi_t, levels=lvls, colors="k", linewidths=0.6)
        ax.contour(R, Z, psi_p, levels=lvls, colors="w", linewidths=0.7,
                   linestyles="--")
        ax.plot(sep_t[:, 0], sep_t[:, 1], color="cyan", lw=1.5,
                solid_capstyle="round")
        if g["sep_p"] is not None:
            ax.plot(g["sep_p"][:, 0], g["sep_p"][:, 1], "w--", lw=1.3,
                    solid_capstyle="round")
        for rx, rz in g["xpt_t"]:
            ax.plot(rx, rz, "X", ms=7, color="#d62728", mec="w", mew=0.7,
                    zorder=6)
        ax.plot(g["o_t"][0], g["o_t"][1], "o", ms=6.5, mfc="w", mec="k",
                mew=0.8, zorder=6)
        for r, z in COIL_RZ:
            ax.plot(r, z, "s", ms=4, color="#d62728", mec="k", mew=0.3, zorder=5)
        cax = ax.inset_axes([1.0 + CBG / PW, 0.0, CBW / PW, 1.0],
                            transform=ax.transAxes)
        fig.colorbar(cf, cax=cax, label="ψ (Wb)")
        if row == 0:
            l_t, = ax.plot([], [], "k-", lw=1.2, label="truth")
            l_p, = ax.plot([], [], "w--", lw=1.2, label="prediction")
            lc_t, = ax.plot([], [], color="cyan", lw=1.5, label="LCFS (truth)")
            lc_p, = ax.plot([], [], "w--", lw=1.3, label="LCFS (pred)")
            mk_x, = ax.plot([], [], "X", ms=6, color="#d62728", mec="w",
                            mew=0.7)
            mk_o, = ax.plot([], [], "o", ms=6, mfc="w", mec="k", mew=0.8)
            ax.legend(handles=[l_t, l_p, lc_t, lc_p, (mk_x, mk_o)],
                      labels=["truth", "prediction", "LCFS (truth)",
                              "LCFS (pred)", "X/O-point"],
                      loc="lower left", fontsize=10, framealpha=0.85)
        ax.set_ylabel("Z (m)")
        ax.set_aspect("equal")
        ax.tick_params(length=3)

        # (b/e) relative error (full domain) with truth LCFS overlay
        ax = axes[row, 1]
        # rel/resid arrays are laid out (axis0=R, axis1=Z) while imshow maps
        # axis0 -> y, axis1 -> x; transpose so the physics frame matches
        # extent=[R, Z] and the truth LCFS overlay aligns
        im = ax.imshow(rel.T * 100, extent=[R.min(), R.max(), Z.min(), Z.max()],
                       origin="lower", cmap="magma",
                       vmin=0, vmax=rel_vmax)
        cax = ax.inset_axes([1.0 + CBG / PW, 0.0, CBW / PW, 1.0],
                            transform=ax.transAxes)
        fig.colorbar(im, cax=cax, label="rel. error (%)")
        ax.plot(sep_t[:, 0], sep_t[:, 1], "w-", lw=0.9)
        ax.set_aspect("equal")
        ax.tick_params(length=3)

        # (c/f) PDE residual inside the plasma only (vacuum NaN -> white);
        # outside the plasma J = 0 and the harmonicity check |Delta* psi_pred|
        # would dominate the shared color scale with approximation noise
        ax = axes[row, 2]
        im = ax.imshow(resid.T, extent=[R.min(), R.max(), Z.min(), Z.max()],
                       origin="lower", cmap="hot",
                       vmin=0, vmax=resid_vmax)
        cax = ax.inset_axes([1.0 + CBG / PW, 0.0, CBW / PW, 1.0],
                            transform=ax.transAxes)
        fig.colorbar(im, cax=cax, label="|Δ*ψ + μ₀RJ| (Wb/m²)")
        ax.plot(sep_t[:, 0], sep_t[:, 1], color="cyan", lw=1.0)
        ax.set_aspect("equal")
        ax.tick_params(length=3)

        axes[row, 0].set_title(titles[3 * row + 0])
        axes[row, 1].set_title(titles[3 * row + 1])
        axes[row, 2].set_title(titles[3 * row + 2])
        # row labels on the left, rotated, tight against the panels
        axes[row, 0].text(-0.32, 0.5, d["name"], rotation=90,
                          transform=axes[row, 0].transAxes,
                          va="center", ha="center", fontsize=10)
    for col in range(3):
        axes[1, col].set_xlabel("R (m)")
    out = FIGDIR / "fig6_predictions.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Fig 5 saved: {out}")


if __name__ == "__main__":
    main()
