"""Visualization for the DN FNO reproduction (best model N=5000 seed=1).

Outputs (all into --out-dir, default dn_fno_2608/outputs/report/figures/):
  fig1_best_worst_psi.png  true / predicted / |diff| fields for the best and
                            worst test samples (by normalized rel L2); the two
                            psi panels share one colorbar (same levels) so the
                            error magnitude is readable against the field scale
  fig2_field_stats.png     rel L2 histogram + cumulative, RMSE histogram,
                            GS residual histograms (pred vs freegs truth)
  fig3_geometry_stats.png  X-point error scatter, separatrix mean error hist,
                            O-point error hist, separatrix area rel err hist
  stats_per_sample.json    per-sample metrics (cached; reused on re-runs so
                            re-plotting is instant)

Usage:
  python -m gs_pino_dn_fno_2608.visualize_dn_fno --test-data dn_fno_2608/data/test.npz \
      --checkpoint dn_fno_2608/outputs/fno_n5000_s1/best.pt \
      --out-dir dn_fno_2608/outputs/report/figures
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

from gs_pino_dn_fno_2608.data_dn_fno import DNFnoDataset
from gs_pino_dn_fno_2608.data_dn_fno_coils import DNFnoDatasetCoils
from gs_pino_dn_fno_2608.model_dn_fno import build_model
from gs_pino_dn_fno_2608.evaluate_dn_fno import (
    gs_residual_ratio, geometry_metrics, match_xpoints_and_axis,
    separatrix_pts, separatrix_points)

from freegs import critical

# paper reference numbers for annotation
PAPER = {
    "rel_l2_mean_pct": 0.061, "rel_l2_best_pct": 0.052, "rmse_phys": 1.79e-5,
    "sep_mean_cm": 0.072, "x_lo_cm": 0.161, "x_up_cm": 0.112, "o_point_cm": 0.031,
    "gs_residual": 2.29,
}


def get_machine_geometry(machine_name: str | None):
    """(wall_r, wall_z, coils) for fig1 overlays; coils = [(R, Z, I), ...].

    MAST is wall-less (wall None) with 11 coils; TestTokamak has a wall.
    """
    if machine_name is None:
        return None, None, []
    from freegs import machine as fg_machine
    if machine_name == "mast":
        m = fg_machine.MAST()
    elif machine_name == "mastu_simple":
        m = fg_machine.MASTU_simple()   # data_v6: real vessel wall (R 0.244-2.0)
    else:
        m = fg_machine.TestTokamak()
    wall_r, wall_z = None, None
    if m.wall is not None:
        wall_r, wall_z = np.asarray(m.wall.R), np.asarray(m.wall.Z)
    coils = []
    for entry in m.coils:
        name, coil = entry[0], entry[1]
        if hasattr(coil, "R"):        # PF coil
            coils.append((float(coil.R), float(coil.Z), float(coil.current)))
        elif hasattr(coil, "Rs"):     # solenoid: axial stack at Rs
            coils.append((float(coil.Rs), 0.0, float(coil.current)))
        elif hasattr(coil, "coils"):  # Circuit (MASTU_simple): paired coils,
            c0 = coil.coils[0][1]     # same R, opposite Z — plot the upper one
            coils.append((float(c0.R), float(c0.Z), float(coil.current)))
    return wall_r, wall_z, coils


def plot_field(ax, R, Z, psi, levels, title, xpt=None, opt=None, sep_pts=None,
               wall_r=None, wall_z=None, coils=None):
    """True/|diff| panels: device (wall+coils) drawn beneath, X-points as white
    crosses, magnetic axis as white circle, separatrix as a white line. R/Z are
    kept on the same scale (aspect equal) so the device is not distorted."""
    ax.set_aspect("equal")
    cf = ax.contourf(R, Z, psi, levels=levels, cmap="viridis")
    if coils:
        for r, z, i in coils:
            ax.plot(r, z, "s", ms=6, color="#b30", mec="w", mew=0.6)
            ax.text(r + 0.03, z + 0.03, f"{i*1e-3:+.1f}", fontsize=5, color="0.25")
    if wall_r is not None:
        ax.plot(np.append(wall_r, wall_r[0]), np.append(wall_z, wall_z[0]),
                "k-", lw=1.8)
    if sep_pts is not None and len(sep_pts):
        ax.plot(np.append(sep_pts[:, 0], sep_pts[0, 0]),
                np.append(sep_pts[:, 1], sep_pts[0, 1]), "w-", lw=1.3)
    if xpt:
        for pt in xpt:
            if pt is None:
                continue
            r, z, _ = pt
            ax.plot(r, z, "wx", markersize=9, markeredgewidth=2)
    if opt is not None:
        ax.plot(opt[0], opt[1], "wo", markersize=6, markeredgewidth=1.5)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("R (m)")
    ax.set_ylabel("Z (m)")
    return cf


def predict_pair(model, ds, idx, device, stats):
    """psi_pred / psi_true (physical, Wb) for one sample (2 small forwards)."""
    x, _ = ds[idx]
    with torch.no_grad():
        pred_norm = model(x[None].to(device)).squeeze(0).cpu().numpy()
    psi_pred = pred_norm[0] * float(stats["psi_std"]) + float(stats["psi_mean"])
    psi_true = ds.psi_total[ds.indices[idx]]
    return psi_pred, psi_true


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-data", default="dn_fno_2608/data/test.npz")
    ap.add_argument("--checkpoint", default="dn_fno_2608/outputs/fno_n5000_s1/best.pt")
    ap.add_argument("--out-dir", default="dn_fno_2608/outputs/report/figures")
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--machine", default=None, choices=["test", "mast", "mastu_simple"],
                    help="device geometry (wall/coils) for fig1; data_v4/v5 use "
                         "'test'/'mast' respectively (npz has no machine field)")
    ap.add_argument("--title", default="arXiv:2608.05555 reproduction — N=5000 FNO",
                    help="suptitle for fig1 (dataset/experiment description)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    # input_mode is a string marker, config_input a bool — neither is a stat
    # (np.float32(str) throws; np.float32(bool) silently pollutes the scalars)
    stats = {k: (np.asarray(v, dtype=np.float32) if isinstance(v, list) else np.float32(v))
             for k, v in ckpt["stats"].items() if k not in ("input_mode", "config_input")}

    with np.load(args.test_data) as d:
        R, Z = d["R"], d["Z"]
    # dataset follows the checkpoint's input_mode (coils checkpoints use the
    # coil-variant dataset; anchor checkpoints append the anchor channels)
    input_mode = ckpt.get("input_mode", "xpoints")
    if input_mode == "coils":
        # exp012 trained with --no-config-channel (21ch): dataset must match
        # the ckpt stats or scalar normalization misaligns (22ch vs 19 stats)
        use_config = not ckpt.get("no_config_channel", False)
        ds = DNFnoDatasetCoils(args.test_data, stats=stats, use_config=use_config)
    else:
        ds = DNFnoDataset(args.test_data, stats=stats, use_anchor=(input_mode == "xa"),
                          use_config=bool(ckpt.get("config_input", False)))
    n_full = min(len(ds), args.max_samples) if args.max_samples else len(ds)

    # input channels inferred from the checkpoint stats (9 baseline / 11 data_v2)
    model = build_model(in_channels=2 + len(stats["scalar_mean"])).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # ---- per-sample metrics (cached in stats_per_sample.json) ----
    # schema: bump when the metric definition changes (v2 = truth-anchored
    # geometry with vacuum-saddle exclusion / ray-traced separatrix)
    SCHEMA = 2
    stats_path = out_dir / "stats_per_sample.json"
    if stats_path.exists():
        with open(stats_path) as f:
            cached = json.load(f)
        if cached.get("schema") == SCHEMA:
            rows = {k: np.array(v) for k, v in cached.items() if k != "schema"}
            n_eval = len(rows["rel_l2_pct"])
            print(f"reusing cached per-sample stats (schema {SCHEMA}): {stats_path}")
            print(f"  ({n_eval} samples)")
        else:
            rows = None
            print(f"stale cache (schema mismatch), recomputing: {stats_path}")
    else:
        rows = None
    if rows is None:
        print(f"evaluating {n_full} test samples (checkpoint {args.checkpoint})")
        rows = {k: [] for k in
                ["rel_l2_pct", "rmse_phys", "gs_pred", "gs_true",
                 "x_lo_cm", "x_up_cm", "o_point_cm", "sep_mean_cm",
                 "sep_hausdorff_cm", "sep_area_rel_err_pct", "n_xpt_pred"]}
        with torch.no_grad():
            for i in range(n_full):
                x, y_norm = ds[i]
                pred_norm = model(x[None].to(device)).squeeze(0).cpu().numpy()
                y_norm = y_norm.numpy()
                rel_l2 = float(np.linalg.norm(pred_norm - y_norm) / (np.linalg.norm(y_norm) + 1e-12))
                rows["rel_l2_pct"].append(rel_l2 * 100.0)

                psi_pred = pred_norm[0] * float(stats["psi_std"]) + float(stats["psi_mean"])
                psi_true = ds.psi_total[ds.indices[i]]
                rows["rmse_phys"].append(float(np.sqrt(np.mean((psi_pred - psi_true) ** 2))))

                mask_i = (psi_true >= ds.axes[ds.indices[i]][2]).astype(np.float32)
                rows["gs_pred"].append(gs_residual_ratio(
                    psi_pred, R, Z, ds.dpdpsi[ds.indices[i]], ds.FdFdpsi[ds.indices[i]], mask_i))
                rows["gs_true"].append(gs_residual_ratio(
                    psi_true, R, Z, ds.dpdpsi[ds.indices[i]], ds.FdFdpsi[ds.indices[i]], mask_i))

                j_i = ds.indices[i]
                xa = getattr(ds, "xpts_actual", None)   # absent on coil datasets
                op = getattr(ds, "o_point", None)
                gm = geometry_metrics(
                    psi_pred, psi_true, R, Z,
                    xpts_true=xa[j_i] if xa is not None else None,
                    o_true=op[j_i] if op is not None else None,
                    psi_bndry_true=float(ds.axes[j_i][2]) if ds.axes is not None else None,
                    anchor=ds.anchor[j_i] if ds.anchor is not None else None)
                rows["n_xpt_pred"].append(gm.get("n_xpt_pred", 0))
                for k in rows:
                    if k in gm:
                        rows[k].append(gm[k])
                for k, v in list(rows.items()):
                    if len(v) < i + 1:
                        v.append(float("nan"))
                if (i + 1) % 100 == 0:
                    print(f"  {i + 1}/{n_full}")
        rows = {k: np.array(v) for k, v in rows.items()}
        with open(stats_path, "w") as f:
            json.dump({"schema": SCHEMA, **{k: v.tolist() for k, v in rows.items()}},
                      f, indent=1)
        n_eval = n_full

    n_fail = int((rows["n_xpt_pred"] < 2).sum())
    print(f"  find_critical failures: {n_fail}/{n_eval}")

    # ---- fig 1: best / worst psi comparison ----
    from matplotlib.gridspec import GridSpec
    from matplotlib.ticker import MaxNLocator, ScalarFormatter

    wall_r, wall_z, coils = get_machine_geometry(args.machine)
    i_best = int(np.nanargmin(rows["rel_l2_pct"]))
    i_worst = int(np.nanargmax(rows["rel_l2_pct"]))
    print(f"  best sample #{i_best}: rel L2 {rows['rel_l2_pct'][i_best]:.4f}% | "
          f"worst #{i_worst}: {rows['rel_l2_pct'][i_worst]:.4f}%")

    def fmt_cbar(cb):
        cb.ax.yaxis.set_major_locator(MaxNLocator(5))
        cb.ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))

    def fig1_geoms(psi_pred, psi_true, j):
        """(xpt_t, o_t, sep_t, xpt_p, o_p, sep_p) — separatrix X-points, magnetic
        axis and separatrix curve, grounded on the dataset's true geometry
        (data_v4+) with the prediction paired via match_xpoints_and_axis;
        paper data/ falls back to find_critical + closed-contour separatrix."""
        xa = getattr(ds, "xpts_actual", None)
        op = getattr(ds, "o_point", None)
        if xa is not None:
            xpts_t = [tuple(map(float, row)) for row in xa[j]]
            o_t = (float(op[j][0]), float(op[j][1]))
            bnd_t = float(ds.axes[j][2])
            anc = ds.anchor[j] if ds.anchor is not None else None
            match = match_xpoints_and_axis(psi_pred, R, Z, xa[j], bnd_t, anc)
            sep_t = separatrix_pts(psi_true, R, Z, bnd_t, o_t)
            sep_p = separatrix_pts(psi_pred, R, Z, match["bnd_p"], match["o_pred"]) \
                if match["bnd_p"] is not None else None
            return xpts_t, o_t, sep_t, match["xpt_pred"], match["o_pred"], sep_p
        opt_t, xpt_t = critical.find_critical(R, Z, psi_true)
        opt_p, xpt_p = critical.find_critical(R, Z, psi_pred)
        lvl_t = 0.5 * (xpt_t[0][2] + xpt_t[1][2]) if len(xpt_t) >= 2 else None
        lvl_p = 0.5 * (xpt_p[0][2] + xpt_p[1][2]) if len(xpt_p) >= 2 else None
        sep_t = separatrix_points(psi_true, R, Z, lvl_t) if lvl_t else None
        sep_p = separatrix_points(psi_pred, R, Z, lvl_p) if lvl_p else None
        return xpt_t, opt_t, sep_t, xpt_p, opt_p, sep_p

    fig = plt.figure(figsize=(19.5, 11))
    gs = GridSpec(2, 5, width_ratios=[4.2, 4.2, 4.2, 0.35, 0.35],
                  wspace=0.35, hspace=0.35)
    for row, idx, tag in ((0, i_best, "BEST"), (1, i_worst, "WORST")):
        ax_t = fig.add_subplot(gs[row, 0])
        ax_p = fig.add_subplot(gs[row, 1])
        ax_d = fig.add_subplot(gs[row, 2])
        cax_diff = fig.add_subplot(gs[row, 3])
        cax_psi = fig.add_subplot(gs[row, 4])

        psi_t, psi_p = predict_pair(model, ds, idx, device, stats)
        lvls = np.linspace(psi_t.min(), psi_t.max(), 16)
        xpt_t, o_t, sep_t, xpt_p, o_p, sep_p = fig1_geoms(psi_p, psi_t, ds.indices[idx])

        plot_field(ax_t, R, Z, psi_t, lvls, f"{tag} # {idx}  |  freegs truth ψ (Wb)",
                   xpt_t, o_t, sep_t, wall_r, wall_z, coils)
        cf = plot_field(ax_p, R, Z, psi_p, lvls, f"FNO prediction ψ (Wb)",
                        xpt_p, o_p, sep_p, wall_r, wall_z, coils)
        # both psi panels share one colorbar, placed in its own column on the
        # right (never overlapping the panels)
        cb_psi = fig.colorbar(cf, cax=cax_psi, label="ψ (Wb)")
        fmt_cbar(cb_psi)

        diff = np.abs(psi_p - psi_t)
        ax_d.set_aspect("equal")
        im = ax_d.imshow(diff, extent=[R.min(), R.max(), Z.min(), Z.max()],
                         origin="lower", cmap="magma")
        if coils:
            for r, z, _ in coils:
                ax_d.plot(r, z, "s", ms=6, color="#b30", mec="w", mew=0.6)
        if wall_r is not None:
            ax_d.plot(np.append(wall_r, wall_r[0]), np.append(wall_z, wall_z[0]),
                      "k-", lw=1.8)
        if xpt_t:
            for r, z, _ in xpt_t:
                ax_d.plot(r, z, "wx", markersize=9, markeredgewidth=2)
        cb_diff = fig.colorbar(im, cax=cax_diff, label="|Δψ| (Wb)")
        fmt_cbar(cb_diff)
        ax_d.set_title(
            f"|ψ_pred − ψ_true|  (rel L2 {rows['rel_l2_pct'][idx]:.4f}%, "
            f"RMSE {rows['rmse_phys'][idx]:.2e} Wb)", fontsize=10)
        ax_d.set_xlabel("R (m)")
        ax_d.set_ylabel("Z (m)")
    fig.suptitle(f"{args.title} — best & worst test samples\n"
                 "white X = separatrix X-point, white circle = magnetic axis, white line = separatrix; "
                 "red squares = coils, black line = wall (aspect equal: R and Z on the same scale). "
                 "left colorbar = |Δψ|, right colorbar = shared ψ scale (both rows)",
                 fontsize=11)
    fig.savefig(out_dir / "fig1_best_worst_psi.png", dpi=150)
    plt.close(fig)
    print(f"  saved fig1_best_worst_psi.png")

    # ---- fig 2: field-level statistics ----
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    ax = axes[0, 0]
    ax.hist(rows["rel_l2_pct"], bins=40, color="steelblue", alpha=0.8)
    ax.axvline(PAPER["rel_l2_mean_pct"], color="red", ls="--", label="paper mean 0.061%")
    ax.axvline(np.nanmean(rows["rel_l2_pct"]), color="green", ls="-",
               label=f"ours mean {np.nanmean(rows['rel_l2_pct']):.3f}%")
    ax.axvline(0.12, color="orange", ls=":", label="0.12% threshold (paper >95% below)")
    ax.set_xlabel("rel L2 (%)")
    ax.set_ylabel("count")
    ax.legend(fontsize=8)
    ax.set_title(f"Test rel L2 (n={n_eval})")

    ax = axes[0, 1]
    ax.hist(rows["rmse_phys"], bins=40, color="darkorange", alpha=0.8)
    ax.axvline(PAPER["rmse_phys"], color="red", ls="--", label="paper mean 1.79e-5")
    ax.axvline(np.nanmean(rows["rmse_phys"]), color="green", ls="-",
               label=f"ours mean {np.nanmean(rows['rmse_phys']):.2e}")
    ax.set_xlabel("physical RMSE (Wb)")
    ax.set_ylabel("count")
    ax.legend(fontsize=8)
    ax.set_title("Physical RMSE")

    ax = axes[1, 0]
    srt = np.sort(rows["rel_l2_pct"])
    ax.plot(srt, np.arange(1, len(srt) + 1) / len(srt), color="steelblue")
    ax.axvline(PAPER["rel_l2_mean_pct"], color="red", ls="--", label="paper mean")
    ax.axvline(0.12, color="orange", ls=":", label="0.12%")
    ax.set_xlabel("rel L2 (%)")
    ax.set_ylabel("cumulative fraction")
    ax.legend(fontsize=8)
    ax.set_title(f"cumulative rel L2 (<0.12%: "
                 f"{(rows['rel_l2_pct'] < 0.12).mean()*100:.1f}% of samples)")

    ax = axes[1, 1]
    ax.hist(rows["gs_true"], bins=30, color="gray", alpha=0.7, label="freegs truth")
    ax.hist(rows["gs_pred"], bins=30, color="steelblue", alpha=0.7, label="FNO pred")
    ax.axvline(PAPER["gs_residual"], color="red", ls="--", label="paper 2.29")
    ax.set_xlabel("normalized GS residual")
    ax.set_ylabel("count")
    ax.legend(fontsize=8)
    ax.set_title(f"GS residual  pred {np.nanmean(rows['gs_pred']):.3f} / "
                 f"truth {np.nanmean(rows['gs_true']):.3f}")
    fig.tight_layout()
    fig.savefig(out_dir / "fig2_field_stats.png", dpi=150)
    plt.close(fig)
    print(f"  saved fig2_field_stats.png")

    # ---- fig 3: geometry statistics ----
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    ax = axes[0, 0]
    # SN samples have no upper X-point (x_up = NaN) and are skipped by scatter;
    # axis limit from P95 so a few non-localizable samples do not stretch it
    sc = ax.scatter(rows["x_up_cm"], rows["x_lo_cm"], c=rows["rel_l2_pct"],
                    cmap="viridis", s=18, alpha=0.8)
    all_x = np.concatenate([rows["x_up_cm"], rows["x_lo_cm"]])
    n_xok = int(np.isfinite(all_x).sum())
    lim = 1.3 * max(1.0, float(np.nanpercentile(all_x, 95)))
    ax.plot([0, lim], [0, lim], "k--", lw=0.8)
    ax.axvline(PAPER["x_up_cm"], color="red", ls=":", label="paper mean")
    ax.axhline(PAPER["x_lo_cm"], color="red", ls=":")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("upper X-point error (cm)")
    ax.set_ylabel("lower X-point error (cm)")
    fig.colorbar(sc, ax=ax, label="rel L2 (%)")
    ax.legend(fontsize=8)
    ax.set_title(f"X-point localization errors (n={n_xok} localizable / {n_eval})")

    for pos, key, title, ref in (
            ((0, 1), "sep_mean_cm", "separatrix mean distance", PAPER["sep_mean_cm"]),
            ((1, 0), "o_point_cm", "O-point error", PAPER["o_point_cm"]),
            ((1, 1), "sep_area_rel_err_pct", "separatrix area rel. error", 0.465)):
        ax = axes[pos]
        d = np.asarray(rows[key], float)
        d = d[np.isfinite(d)]  # limiter 桶几何指标全 NaN (无分离面) -> 跳过
        if len(d) < 2:
            ax.text(0.5, 0.5, f"no finite data ({len(d)} pts)",
                    ha="center", va="center", transform=ax.transAxes)
        else:
            ax.hist(d, bins=40, color="mediumseagreen", alpha=0.8)
            ax.axvline(ref, color="red", ls="--", label=f"paper {ref}")
            ax.axvline(np.mean(d), color="green", ls="-",
                       label=f"ours {np.mean(d):.4f}")
        ax.set_xlabel("cm" if "cm" in key else "%")
        ax.set_ylabel("count")
        ax.legend(fontsize=8)
        ax.set_title(title)
    fig.suptitle(f"Geometry diagnostics ({args.title}, n={n_eval} test samples)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "fig3_geometry_stats.png", dpi=150)
    plt.close(fig)
    print(f"  saved fig3_geometry_stats.png")

    print(f"\nall figures saved -> {out_dir}")


if __name__ == "__main__":
    main()
