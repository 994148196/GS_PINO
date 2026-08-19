"""Evaluation + exp011-style visualization for the physics-informed FNO
checkpoints (exp101/102, data_v5/dn).

Per sample:
  - psi_plasma_phys from the network (denormalized)
  - psi_total = psi_plasma_pred + sum_k I_k * G_k  (exact Green-function
    addition from the stored greens / coil_currents — the network never sees
    the coil field)
  - twostage: J_phys from the second output channel

Outputs (--out-dir):
  metrics.json           PINO metrics (rel L2 plasma/total, RMSE, gs residual
                         core/geom, Ip/J) + geometry aggregates
  figures/stats_per_sample.json  exp011-schema per-sample metrics
  figures/fig1_best_worst_psi.png  exp011 style: device (wall/coils) drawn
                         beneath contour fields, white X = separatrix X-point,
                         white circle = magnetic axis, white line = separatrix
                         (psi_total best/worst rows; twostage adds a J row)
  figures/fig2_field_stats.png  rel L2 / RMSE / cumulative / GS residual
                         (+ Ip / J histograms for twostage)
  figures/fig3_geometry_stats.png  X-point / O-point / separatrix errors

Reuses get_machine_geometry/plot_field (visualize_dn_fno) and
geometry_metrics/match_xpoints_and_axis/separatrix_pts (evaluate_dn_fno), so
the figure look matches exp011 figures_dn exactly.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gs_pino_dn_fno_2608 import evaluate_dn_fno
from gs_pino_dn_fno_2608.data_dn_fno import rel_l2_normalized
from gs_pino_dn_fno_2608.evaluate_dn_fno import (
    geometry_metrics,
    match_xpoints_and_axis,
    separatrix_pts,
)
from gs_pino_dn_fno_2608.model_dn_fno import build_model
from gs_pino_dn_fno_2608.visualize_dn_fno import get_machine_geometry, plot_field
from gs_pino_fno_phys.data_pino import DNPinoDataset
from gs_pino_fno_phys.losses_pino import denorm_j, denorm_psi

# exp011 DN-bucket references for fig2 annotation lines
REF_EXP011 = {"rel_l2_dn_pct": 0.84, "rmse_phys_wb": 3.41e-4}


def load_checkpoint(path: str):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    stats = ckpt["stats"]
    mode = ckpt["mode"]
    out_channels = 2 if mode == "twostage" else 1
    model = build_model(in_channels=2 + len(stats["scalar_mean"]),
                        out_channels=out_channels)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, stats, mode, ckpt


def pct(x: np.ndarray) -> dict[str, float]:
    """Percentile summary, NaN-safe (geometry metrics can fail to localize)."""
    return {"mean": float(np.nanmean(x)) * 100, "median": float(np.nanmedian(x)) * 100,
            "p95": float(np.nanpercentile(x, 95)) * 100}


def pct_cm(x: np.ndarray) -> dict[str, float]:
    return {"mean": float(np.nanmean(x)), "median": float(np.nanmedian(x)),
            "p95": float(np.nanpercentile(x, 95))}


def evaluate() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-data", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--fig-dir", default=None,
                    help="figures + stats_per_sample.json (default <out-dir>/figures)")
    ap.add_argument("--max-samples", type=int, default=0, help="0 = all")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--machine", default="mast", choices=["test", "mast", "mastu_simple"],
                    help="device geometry (wall/coils) for fig1; data_v5 = mast")
    ap.add_argument("--title", default="", help="suptitle for fig1")
    args = ap.parse_args()

    model, stats, mode, ckpt = load_checkpoint(args.checkpoint)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model.to(device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = Path(args.fig_dir) if args.fig_dir else out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    title = args.title or f"{Path(out_dir).name} — {mode} (data_v5/dn)"

    ds = DNPinoDataset(args.test_data, stats=stats)
    if args.max_samples > 0:
        ds = DNPinoDataset(args.test_data, stats=stats,
                           indices=np.arange(min(args.max_samples, len(ds))))
    loader = DataLoader(ds, batch_size=16, shuffle=False)
    n = len(ds)

    psi_t_mean, psi_t_std = float(stats["psi_mean"]), float(stats["psi_std"])
    dR, dZ, dA = ds.dR, ds.dZ, ds.dR * ds.dZ
    R_phys, Z_phys = ds.R_phys, ds.Z_phys

    rel_l2_plasma, rel_l2_total, rmse_plasma, rmse_total = [], [], [], []
    gs_pred_core, gs_pred_geom, gs_truth_core, gs_truth_geom = [], [], [], []
    ip_rel_err, j_rel_l2, j_rel_l2_masked = [], [], []
    x_lo, x_up, o_pt, sep_mean, sep_haus, sep_area, n_xpt = [], [], [], [], [], [], []
    cache = {"psi_plasma_true": [], "psi_plasma_pred": [],
             "psi_total_true": [], "psi_total_pred": [],
             "j_true": [], "j_pred": []}

    with torch.no_grad():
        for batch in loader:
            x, y_psi, y_j = (batch["x"].to(device), batch["y_psi"].numpy(),
                             batch["y_j"].numpy())
            mask = batch["mask"].numpy()
            ip_t = batch["ip"].numpy()[:, 0]
            pred = model(x).cpu()
            psi_z = pred[:, 0:1].numpy()
            psi_plasma_pred = denorm_psi(pred[:, 0:1], stats).numpy()[:, 0]
            j_pred = (denorm_j(pred[:, 1:2], stats).numpy()[:, 0]
                      if mode == "twostage" else None)

            for b in range(pred.shape[0]):
                i = len(rel_l2_plasma)          # sequential sample index
                j = ds.indices[i]               # raw row index into the npz arrays
                psi_pl_true = ds.psi_plasma[j]
                psi_tot_true = ds.psi_total[j]
                psi_tot_pred = psi_plasma_pred[b] + np.einsum(
                    "kij,k->ij", ds.greens[j], ds.coil_currents[j])

                rel_l2_plasma.append(float(rel_l2_normalized(
                    torch.from_numpy(psi_z[b:b+1]), torch.from_numpy(y_psi[b:b+1]))))
                rel_l2_total.append(float(rel_l2_normalized(
                    torch.from_numpy(((psi_tot_pred - psi_t_mean) / psi_t_std)[None, None]),
                    torch.from_numpy(((psi_tot_true - psi_t_mean) / psi_t_std)[None, None]))))
                rmse_plasma.append(float(np.sqrt(np.mean((psi_plasma_pred[b] - psi_pl_true) ** 2))))
                rmse_total.append(float(np.sqrt(np.mean((psi_tot_pred - psi_tot_true) ** 2))))

                dp, ff, msk_core = ds.dpdpsi[j], ds.FdFdpsi[j], ds.mask[j]
                bnd = float(ds.axes[j][2])
                msk_geom = psi_tot_true >= bnd
                dst = {"pred_core": gs_pred_core, "pred_geom": gs_pred_geom,
                       "truth_core": gs_truth_core, "truth_geom": gs_truth_geom}
                for key, field in (("pred", psi_plasma_pred[b]),
                                   ("truth", psi_pl_true)):
                    for mkey, msk in (("core", msk_core), ("geom", msk_geom)):
                        dst[f"{key}_{mkey}"].append(
                            evaluate_dn_fno.gs_residual_ratio(
                                field, R_phys, Z_phys, dp, ff, msk))

                # exp011 geometry metrics on the reconstructed psi_total (X
                # points / O point / separatrix grounded on npz ground truth)
                gm = geometry_metrics(psi_tot_pred, psi_tot_true, R_phys, Z_phys,
                                      xpts_true=ds.xpts_actual[j],
                                      o_true=ds.o_point[j],
                                      psi_bndry_true=bnd)
                x_lo.append(gm.get("x_lo_cm", np.nan))
                x_up.append(gm.get("x_up_cm", np.nan))
                o_pt.append(gm.get("o_point_cm", np.nan))
                sep_mean.append(gm.get("sep_mean_cm", np.nan))
                sep_haus.append(gm.get("sep_hausdorff_cm", np.nan))
                sep_area.append(gm.get("sep_area_rel_err_pct", np.nan))
                n_xpt.append(int(gm.get("n_xpt_pred", 0)))

                if mode == "twostage":
                    ip_pred = float((j_pred[b] * mask[b]).sum() * dA)
                    ip_rel_err.append(abs(ip_pred - ip_t[b]) / ip_t[b])
                    # full-grid rel L2 (dominated by the spectral ringing of the
                    # zero-outside-mask target) and mask-only rel L2 (the metric
                    # the masked J loss actually optimized)
                    j_z = pred[:, 1:2][b:b+1]
                    j_t = torch.from_numpy(y_j[b:b+1])
                    msk = torch.from_numpy(mask[b:b+1])
                    j_rel_l2.append(float(rel_l2_normalized(j_z, j_t)))
                    num = torch.linalg.vector_norm((j_z - j_t) * msk)
                    den = torch.linalg.vector_norm(j_t * msk)
                    j_rel_l2_masked.append(float(num / (den + 1e-12)))
                    cache["j_true"].append(ds.j_phys[j])
                    cache["j_pred"].append(j_pred[b])

                cache["psi_plasma_true"].append(psi_pl_true)
                cache["psi_plasma_pred"].append(psi_plasma_pred[b])
                cache["psi_total_true"].append(psi_tot_true)
                cache["psi_total_pred"].append(psi_tot_pred)

    rel_total = np.array(rel_l2_total)
    metrics = {
        "n": n, "mode": mode,
        "rel_l2_plasma_pct": pct(np.array(rel_l2_plasma)),
        "rel_l2_total_pct": pct(rel_total),
        "rmse_phys_Wb_plasma": {"mean": float(np.mean(rmse_plasma)),
                                "median": float(np.median(rmse_plasma))},
        "rmse_phys_Wb_total": {"mean": float(np.mean(rmse_total)),
                               "median": float(np.median(rmse_total))},
        "gs_residual_pred": {"core_mask": float(np.mean(gs_pred_core)),
                             "geom_mask": float(np.mean(gs_pred_geom))},
        "gs_residual_truth": {"core_mask": float(np.mean(gs_truth_core)),
                              "geom_mask": float(np.mean(gs_truth_geom))},
        "geometry": {"x_lo_cm": pct_cm(np.array(x_lo)),
                     "x_up_cm": pct_cm(np.array(x_up)),
                     "o_point_cm": pct_cm(np.array(o_pt)),
                     "sep_mean_cm": pct_cm(np.array(sep_mean)),
                     "sep_hausdorff_cm": pct_cm(np.array(sep_haus)),
                     "sep_area_rel_err_pct": pct(np.array(sep_area)),
                     "n_xpt_fail": int(np.sum(np.array(n_xpt) < 2))},
        "ckpt": {"best_val_rel_l2": ckpt["best_val_rel_l2"],
                 "best_epoch": ckpt["best_epoch"],
                 "switch_epoch": ckpt.get("switch_epoch"),
                 "artifact_stage": ckpt.get("artifact_stage")},
    }
    if mode == "twostage":
        metrics["ip_rel_err_pct"] = pct(np.array(ip_rel_err))
        metrics["j_rel_l2_pct"] = pct(np.array(j_rel_l2))
        metrics["j_rel_l2_mask_pct"] = pct(np.array(j_rel_l2_masked))
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=float)

    # ---- exp011-schema per-sample stats (feeds fig1/2/3 + external reuse) ----
    rows = {
        "schema": 3,
        "rel_l2_pct": [v * 100 for v in rel_l2_total],
        "rmse_phys": rmse_total,
        "gs_pred": gs_pred_geom,
        "gs_true": gs_truth_geom,
        "gs_pred_core": gs_pred_core,
        "gs_true_core": gs_truth_core,
        "x_lo_cm": x_lo, "x_up_cm": x_up, "o_point_cm": o_pt,
        "sep_mean_cm": sep_mean, "sep_hausdorff_cm": sep_haus,
        "sep_area_rel_err_pct": sep_area, "n_xpt_pred": n_xpt,
        "indices": [int(ds.indices[k]) for k in range(n)],
    }
    if mode == "twostage":
        rows["j_rel_l2_mask_pct"] = [v * 100 for v in j_rel_l2_masked]
        rows["ip_rel_err_pct"] = [v * 100 for v in ip_rel_err]
    with open(fig_dir / "stats_per_sample.json", "w") as f:
        json.dump(rows, f, indent=1, default=float)

    # ---- figures (exp011 style) ----
    _plot_fig1(fig_dir / "fig1_best_worst_psi.png", ds, rows, cache, mode,
               R_phys, Z_phys, args.machine, title)
    _plot_fig2(fig_dir / "fig2_field_stats.png", rows, mode)
    _plot_fig3(fig_dir / "fig3_geometry_stats.png", rows, title)
    print(json.dumps(metrics, indent=2, default=float))
    print(f"  stats_per_sample.json + fig1/2/3 -> {fig_dir}")


def _fig1_geoms(psi_true: np.ndarray, psi_pred: np.ndarray,
                R: np.ndarray, Z: np.ndarray,
                xa: np.ndarray, op: np.ndarray, bnd: float):
    """Truth geometry from the dataset fields; prediction paired via
    match_xpoints_and_axis (vacuum saddles excluded), separatrix via
    ray-traced separatrix_pts — identical to exp011's fig1_geoms (data_v4+
    path, which data_v5 has)."""
    xpts_t = [tuple(map(float, row)) for row in xa]
    o_t = (float(op[0]), float(op[1]))
    match = match_xpoints_and_axis(psi_pred, R, Z, xa, bnd, None)
    sep_t = separatrix_pts(psi_true, R, Z, bnd, o_t)
    sep_p = (separatrix_pts(psi_pred, R, Z, match["bnd_p"], match["o_pred"])
             if match["bnd_p"] is not None else None)
    return xpts_t, o_t, sep_t, match["xpt_pred"], match["o_pred"], sep_p


def _plot_fig1(out_path: Path, ds: DNPinoDataset, rows: dict, cache: dict,
               mode: str, R: np.ndarray, Z: np.ndarray,
               machine: str, title: str) -> None:
    from matplotlib.gridspec import GridSpec
    from matplotlib.ticker import MaxNLocator, ScalarFormatter

    def fmt_cbar(cb):
        cb.ax.yaxis.set_major_locator(MaxNLocator(5))
        cb.ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))

    wall_r, wall_z, coils = get_machine_geometry(machine)
    rl2 = np.array(rows["rel_l2_pct"])
    i_best, i_worst = int(np.nanargmin(rl2)), int(np.nanargmax(rl2))
    print(f"  best sample #{rows['indices'][i_best]}: rel L2 {rl2[i_best]:.4f}% | "
          f"worst #{rows['indices'][i_worst]}: {rl2[i_worst]:.4f}%")

    nrows = 3 if mode == "twostage" else 2
    fig = plt.figure(figsize=(19.5, 11 if nrows == 2 else 15.5))
    gs = GridSpec(nrows, 5, width_ratios=[4.2, 4.2, 4.2, 0.35, 0.35],
                  wspace=0.35, hspace=0.35)

    def panel_row(row, idx, tag):
        j = rows["indices"][idx]
        psi_t = cache["psi_total_true"][idx]
        psi_p = cache["psi_total_pred"][idx]
        lvls = np.linspace(psi_t.min(), psi_t.max(), 16)
        xpt_t, o_t, sep_t, xpt_p, o_p, sep_p = _fig1_geoms(
            psi_t, psi_p, R, Z, ds.xpts_actual[j], ds.o_point[j],
            float(ds.axes[j][2]))
        ax_t = fig.add_subplot(gs[row, 0])
        ax_p = fig.add_subplot(gs[row, 1])
        ax_d = fig.add_subplot(gs[row, 2])
        cax_diff, cax_psi = fig.add_subplot(gs[row, 3]), fig.add_subplot(gs[row, 4])

        plot_field(ax_t, R, Z, psi_t, lvls, f"{tag} # {j}  |  freegs truth ψ (Wb)",
                   xpt_t, o_t, sep_t, wall_r, wall_z, coils)
        cf = plot_field(ax_p, R, Z, psi_p, lvls, "FNO prediction ψ (Wb)",
                        xpt_p, o_p, sep_p, wall_r, wall_z, coils)
        cb_psi = fig.colorbar(cf, cax=cax_psi, label="ψ (Wb)")
        fmt_cbar(cb_psi)

        diff = np.abs(psi_p - psi_t)
        ax_d.set_aspect("equal")
        im = ax_d.imshow(diff, extent=[R.min(), R.max(), Z.min(), Z.max()],
                         origin="lower", cmap="magma")
        for r, z, _ in coils:
            ax_d.plot(r, z, "s", ms=6, color="#b30", mec="w", mew=0.6)
        if wall_r is not None:
            ax_d.plot(np.append(wall_r, wall_r[0]), np.append(wall_z, wall_z[0]),
                      "k-", lw=1.8)
        for pt in xpt_t:
            r, z, _ = pt
            ax_d.plot(r, z, "wx", markersize=9, markeredgewidth=2)
        cb_diff = fig.colorbar(im, cax=cax_diff, label="|Δψ| (Wb)")
        fmt_cbar(cb_diff)
        ax_d.set_title(f"|ψ_pred − ψ_true|  (rel L2 {rl2[idx]:.4f}%, "
                       f"RMSE {rows['rmse_phys'][idx]:.2e} Wb)", fontsize=10)
        ax_d.set_xlabel("R (m)")
        ax_d.set_ylabel("Z (m)")

    for row, idx, tag in ((0, i_best, "BEST"), (1, i_worst, "WORST")):
        panel_row(row, idx, tag)
    if mode == "twostage":
        # J row (same best/worst samples): no geometry markers — J is
        # plasma-only; machine overlay still drawn for spatial reference
        for row, idx, tag in ((2, i_best, "BEST"), (2, i_worst, "WORST")):
            j_t = cache["j_true"][idx]
            j_p = cache["j_pred"][idx]
            lvls = np.linspace(j_t.min(), j_t.max(), 16)
            ax_t = fig.add_subplot(gs[row, 0])
            ax_p = fig.add_subplot(gs[row, 1])
            ax_d = fig.add_subplot(gs[row, 2])
            cax_diff, cax_psi = fig.add_subplot(gs[row, 3]), fig.add_subplot(gs[row, 4])
            plot_field(ax_t, R, Z, j_t, lvls, f"{tag} # {rows['indices'][idx]}  |  J data (A/m²)",
                       None, None, None, wall_r, wall_z, coils)
            cf = plot_field(ax_p, R, Z, j_p, lvls, "FNO prediction J (A/m²)",
                            None, None, None, wall_r, wall_z, coils)
            cb = fig.colorbar(cf, cax=cax_psi, label="J (A/m²)")
            fmt_cbar(cb)
            diff = np.abs(j_p - j_t)
            ax_d.set_aspect("equal")
            im = ax_d.imshow(diff, extent=[R.min(), R.max(), Z.min(), Z.max()],
                             origin="lower", cmap="magma")
            for r, z, _ in coils:
                ax_d.plot(r, z, "s", ms=6, color="#b30", mec="w", mew=0.6)
            if wall_r is not None:
                ax_d.plot(np.append(wall_r, wall_r[0]), np.append(wall_z, wall_z[0]),
                          "k-", lw=1.8)
            cb = fig.colorbar(im, cax=cax_diff, label="|ΔJ| (A/m²)")
            fmt_cbar(cb)
            ax_d.set_title(f"|J_pred − J_true|  (mask rel L2 "
                           f"{rows['j_rel_l2_mask_pct'][idx]:.4f}%)", fontsize=10)
            ax_d.set_xlabel("R (m)")
            ax_d.set_ylabel("Z (m)")

    fig.suptitle(f"{title} — best & worst test samples\n"
                 "white X = separatrix X-point, white circle = magnetic axis, white line = separatrix; "
                 "red squares = coils, black line = wall (aspect equal: R and Z on the same scale). "
                 "left colorbar = |Δ|, right colorbar = shared field scale (all rows)",
                 fontsize=11)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved fig1_best_worst_psi.png")


def _plot_fig2(out_path: Path, rows: dict, mode: str) -> None:
    ncols = 3 if mode == "twostage" else 2
    fig, axes = plt.subplots(2, ncols, figsize=(6.4 * ncols, 9))
    rl2 = np.array(rows["rel_l2_pct"])
    rmse = np.array(rows["rmse_phys"])
    gs_p, gs_t = np.array(rows["gs_pred"]), np.array(rows["gs_true"])

    ax = axes[0, 0]
    ax.hist(rl2, bins=40, color="steelblue", alpha=0.8)
    ax.axvline(REF_EXP011["rel_l2_dn_pct"], color="red", ls="--",
               label=f"exp011 DN 0.84%")
    ax.axvline(np.nanmean(rl2), color="green", ls="-",
               label=f"ours mean {np.nanmean(rl2):.3f}%")
    ax.set_xlabel("rel L2 (%)")
    ax.set_ylabel("count")
    ax.legend(fontsize=8)
    ax.set_title("Test rel L2 (psi_total, z domain)")

    ax = axes[0, 1]
    ax.hist(rmse, bins=40, color="darkorange", alpha=0.8)
    ax.axvline(REF_EXP011["rmse_phys_wb"], color="red", ls="--",
               label=f"exp011 DN {REF_EXP011['rmse_phys_wb']:.2e}")
    ax.axvline(np.nanmean(rmse), color="green", ls="-",
               label=f"ours mean {np.nanmean(rmse):.2e}")
    ax.set_xlabel("physical RMSE (Wb)")
    ax.set_ylabel("count")
    ax.legend(fontsize=8)
    ax.set_title("Physical RMSE (psi_total)")

    ax = axes[1, 0]
    srt = np.sort(rl2)
    ax.plot(srt, np.arange(1, len(srt) + 1) / len(srt), color="steelblue")
    ax.axvline(1.0, color="orange", ls=":", label="1%")
    ax.axvline(REF_EXP011["rel_l2_dn_pct"], color="red", ls="--",
               label="exp011 DN 0.84%")
    ax.set_xlabel("rel L2 (%)")
    ax.set_ylabel("cumulative fraction")
    ax.legend(fontsize=8)
    ax.set_title(f"cumulative rel L2 (<1%: {(rl2 < 1.0).mean()*100:.1f}% of samples)")

    ax = axes[1, 1]
    ax.hist(gs_t, bins=30, color="gray", alpha=0.7, label="freegs truth")
    ax.hist(gs_p, bins=30, color="steelblue", alpha=0.7, label="FNO pred")
    ax.set_xlabel("normalized GS residual (geom mask)")
    ax.set_ylabel("count")
    ax.legend(fontsize=8)
    ax.set_title(f"GS residual  pred {np.nanmean(gs_p):.3f} / "
                 f"truth {np.nanmean(gs_t):.3f}")

    if mode == "twostage":
        ax = axes[0, 2]
        ip = np.array(rows["ip_rel_err_pct"])
        ax.hist(ip, bins=40, color="mediumseagreen", alpha=0.8)
        ax.axvline(np.nanmean(ip), color="green", ls="-",
                   label=f"mean {np.nanmean(ip):.3f}%")
        ax.set_xlabel("Ip rel. error (%)")
        ax.set_ylabel("count")
        ax.legend(fontsize=8)
        ax.set_title("Ip constraint (sum J dA vs target)")

        ax = axes[1, 2]
        jm = np.array(rows["j_rel_l2_mask_pct"])
        ax.hist(jm, bins=40, color="plum", alpha=0.8)
        ax.axvline(np.nanmean(jm), color="green", ls="-",
                   label=f"mean {np.nanmean(jm):.3f}%")
        ax.set_xlabel("J rel L2, mask-in (%)")
        ax.set_ylabel("count")
        ax.legend(fontsize=8)
        ax.set_title("J prediction error (inside core mask)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved fig2_field_stats.png")


def _plot_fig3(out_path: Path, rows: dict, title: str) -> None:
    """Same layout as exp011 fig3: X-point scatter + separatrix/O-point/area
    error histograms (geometry_metrics on the reconstructed psi_total)."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    x_up = np.array(rows["x_up_cm"])
    x_lo = np.array(rows["x_lo_cm"])
    rl2 = np.array(rows["rel_l2_pct"])
    n_eval = len(rl2)

    ax = axes[0, 0]
    sc = ax.scatter(x_up, x_lo, c=rl2, cmap="viridis", s=18, alpha=0.8)
    all_x = np.concatenate([x_up, x_lo])
    n_xok = int(np.isfinite(all_x).sum())
    lim = 1.3 * max(1.0, float(np.nanpercentile(all_x, 95)))
    ax.plot([0, lim], [0, lim], "k--", lw=0.8)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("upper X-point error (cm)")
    ax.set_ylabel("lower X-point error (cm)")
    fig.colorbar(sc, ax=ax, label="rel L2 (%)")
    ax.set_title(f"X-point localization errors (n={n_xok} localizable / {n_eval})")

    for pos, key, label in (
            ((0, 1), "sep_mean_cm", "separatrix mean distance"),
            ((1, 0), "o_point_cm", "O-point error"),
            ((1, 1), "sep_area_rel_err_pct", "separatrix area rel. error")):
        ax = axes[pos]
        vals = np.array(rows[key])
        ax.hist(vals, bins=40, color="mediumseagreen", alpha=0.8)
        ax.axvline(np.nanmean(vals), color="green", ls="-",
                   label=f"ours mean {np.nanmean(vals):.4f}")
        ax.set_xlabel("cm" if "cm" in key else "%")
        ax.set_ylabel("count")
        ax.legend(fontsize=8)
        ax.set_title(label)
    fig.suptitle(f"Geometry diagnostics ({title}, n={n_eval} test samples)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved fig3_geometry_stats.png")


if __name__ == "__main__":
    evaluate()
