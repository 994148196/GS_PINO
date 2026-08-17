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
from gs_pino_dn_fno_2608.evaluate_dn_fno import gs_residual_ratio, geometry_metrics

from freegs import critical

# paper reference numbers for annotation
PAPER = {
    "rel_l2_mean_pct": 0.061, "rel_l2_best_pct": 0.052, "rmse_phys": 1.79e-5,
    "sep_mean_cm": 0.072, "x_lo_cm": 0.161, "x_up_cm": 0.112, "o_point_cm": 0.031,
    "gs_residual": 2.29,
}


def plot_field(ax, R, Z, psi, levels, title, xpt=None, opt=None, sep_level=None):
    cf = ax.contourf(R, Z, psi, levels=levels, cmap="viridis")
    if sep_level is not None:
        ax.contour(R, Z, psi, levels=[sep_level], colors="white", linewidths=1.2)
    if xpt:
        for r, z, _ in xpt:
            ax.plot(r, z, "wx", markersize=9, markeredgewidth=2)
    if opt:
        r, z, _ = opt[0]
        ax.plot(r, z, "wo", markersize=6, markeredgewidth=1.5)
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
        ds = DNFnoDatasetCoils(args.test_data, stats=stats)
    else:
        ds = DNFnoDataset(args.test_data, stats=stats, use_anchor=(input_mode == "xa"),
                          use_config=bool(ckpt.get("config_input", False)))
    n_full = min(len(ds), args.max_samples) if args.max_samples else len(ds)

    # input channels inferred from the checkpoint stats (9 baseline / 11 data_v2)
    model = build_model(in_channels=2 + len(stats["scalar_mean"])).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # ---- per-sample metrics (cached in stats_per_sample.json) ----
    stats_path = out_dir / "stats_per_sample.json"
    if stats_path.exists():
        print(f"reusing cached per-sample stats: {stats_path}")
        with open(stats_path) as f:
            rows = {k: np.array(v) for k, v in json.load(f).items()}
        n_eval = len(rows["rel_l2_pct"])
        print(f"  ({n_eval} samples)")
    else:
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

                gm = geometry_metrics(psi_pred, psi_true, R, Z)
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
            json.dump({k: v.tolist() for k, v in rows.items()}, f, indent=1)
        n_eval = n_full

    n_fail = int((rows["n_xpt_pred"] < 2).sum())
    print(f"  find_critical failures: {n_fail}/{n_eval}")

    # ---- fig 1: best / worst psi comparison ----
    from matplotlib.gridspec import GridSpec
    from matplotlib.ticker import MaxNLocator, ScalarFormatter

    i_best = int(np.nanargmin(rows["rel_l2_pct"]))
    i_worst = int(np.nanargmax(rows["rel_l2_pct"]))
    print(f"  best sample #{i_best}: rel L2 {rows['rel_l2_pct'][i_best]:.4f}% | "
          f"worst #{i_worst}: {rows['rel_l2_pct'][i_worst]:.4f}%")

    def fmt_cbar(cb):
        cb.ax.yaxis.set_major_locator(MaxNLocator(5))
        cb.ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))

    fig = plt.figure(figsize=(19.5, 11))
    gs = GridSpec(2, 5, width_ratios=[4.2, 4.2, 4.2, 0.35, 0.35],
                  wspace=0.30, hspace=0.35)
    for row, idx, tag in ((0, i_best, "BEST"), (1, i_worst, "WORST")):
        ax_t = fig.add_subplot(gs[row, 0])
        ax_p = fig.add_subplot(gs[row, 1])
        ax_d = fig.add_subplot(gs[row, 2])
        cax_diff = fig.add_subplot(gs[row, 3])
        cax_psi = fig.add_subplot(gs[row, 4])

        psi_t, psi_p = predict_pair(model, ds, idx, device, stats)
        lvls = np.linspace(psi_t.min(), psi_t.max(), 16)
        try:
            opt_t, xpt_t = critical.find_critical(R, Z, psi_t)
            sep_t = 0.5 * (xpt_t[0][2] + xpt_t[1][2]) if len(xpt_t) >= 2 else None
        except Exception:
            opt_t, xpt_t, sep_t = None, None, None
        try:
            opt_p, xpt_p = critical.find_critical(R, Z, psi_p)
            sep_p = 0.5 * (xpt_p[0][2] + xpt_p[1][2]) if len(xpt_p) >= 2 else None
        except Exception:
            opt_p, xpt_p, sep_p = None, None, None

        plot_field(ax_t, R, Z, psi_t, lvls, f"{tag} # {idx}  |  freegs truth ψ (Wb)",
                   xpt_t, opt_t, sep_t)
        cf = plot_field(ax_p, R, Z, psi_p, lvls, f"FNO prediction ψ (Wb)",
                        xpt_p, opt_p, sep_p)
        # both psi panels share one colorbar, placed in its own column on the
        # right (never overlapping the panels)
        cb_psi = fig.colorbar(cf, cax=cax_psi, label="ψ (Wb)")
        fmt_cbar(cb_psi)

        diff = np.abs(psi_p - psi_t)
        im = ax_d.imshow(diff, extent=[R.min(), R.max(), Z.min(), Z.max()],
                         origin="lower", cmap="magma")
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
                 "white X = X-points, white circle = O-point, white contour = separatrix; "
                 "left colorbar = |Δψ|, right colorbar = shared ψ scale (both rows)",
                 fontsize=12)
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
    sc = ax.scatter(rows["x_up_cm"], rows["x_lo_cm"], c=rows["rel_l2_pct"],
                    cmap="viridis", s=18, alpha=0.8)
    lim = np.nanmax([np.nanmax(rows["x_up_cm"]), np.nanmax(rows["x_lo_cm"])]) * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=0.8)
    ax.axvline(PAPER["x_up_cm"], color="red", ls=":", label="paper mean")
    ax.axhline(PAPER["x_lo_cm"], color="red", ls=":")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("upper X-point error (cm)")
    ax.set_ylabel("lower X-point error (cm)")
    fig.colorbar(sc, ax=ax, label="rel L2 (%)")
    ax.legend(fontsize=8)
    ax.set_title("X-point localization errors (per sample)")

    for pos, key, title, ref in (
            ((0, 1), "sep_mean_cm", "separatrix mean distance", PAPER["sep_mean_cm"]),
            ((1, 0), "o_point_cm", "O-point error", PAPER["o_point_cm"]),
            ((1, 1), "sep_area_rel_err_pct", "separatrix area rel. error", 0.465)):
        ax = axes[pos]
        ax.hist(rows[key], bins=40, color="mediumseagreen", alpha=0.8)
        ax.axvline(ref, color="red", ls="--", label=f"paper {ref}")
        ax.axvline(np.nanmean(rows[key]), color="green", ls="-",
                   label=f"ours {np.nanmean(rows[key]):.4f}")
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
