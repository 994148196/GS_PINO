"""KAN visualization (mirrors visualize_dn_fno.py; paper aps.75.20260331).

Outputs into --out-dir:
  fig1_best_worst_psi.png   best/worst rel-L2 sample: truth / KAN pred / |diff|
  fig2_field_stats.png      rel L2, RMSE, R^2, GS-residual histograms
  fig3_geometry_stats.png   X-point/O-point/separatrix error scatter+hist
  stats_per_sample.json     per-sample metrics (cached, schema-versioned)

Usage:
  "$PY" -u -m gs_pino_kan_2608.visualize_kan \
    --test-data dn_fno_2608/data_v5/sn/test.npz \
    --checkpoint <out>/best.pt --machine mast --out-dir <out>/figures_sn
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from gs_pino_dn_fno_2608.evaluate_dn_fno import (geometry_metrics,
                                                 gs_residual_ratio,
                                                 match_xpoints_and_axis,
                                                 separatrix_pts)
from gs_pino_dn_fno_2608.visualize_dn_fno import (get_machine_geometry,
                                                  plot_field)
from gs_pino_kan_2608.data_kan import PointKANDataset
from gs_pino_kan_2608.model_kan import build_model

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# paper (aps.75.20260331) reference numbers for fig2/fig3 reference lines
PAPER = {"rel_l2_kan1_pct": 0.631, "rel_l2_kan2_pct": 0.912,
         "rel_l2_kan3_pct": 1.022, "r_squared": 0.9953}


def predict_pair(model, ds, i, device, stats):
    """psi_pred / psi_true (physical, Wb) for sample row i (full grid)."""
    b = ds.full_grid(i)
    with torch.no_grad():
        out = model(b["x"].to(device))
    psi_z = out[:, 0].cpu().numpy().reshape(65, 65)
    psi_pred = psi_z * float(stats["psi_std"]) + float(stats["psi_mean"])
    return psi_pred, ds.psi[i]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-data", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--machine", default=None, choices=["test", "mast", "mastu_simple"])
    ap.add_argument("--title", default="KAN GS free-boundary (aps.75.20260331) — 19ch")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else ("cpu" if args.device == "auto" else args.device))

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    stats = {k: (np.asarray(v, dtype=np.float32) if isinstance(v, list) else np.float32(v))
             for k, v in ckpt["stats"].items()}
    ds = PointKANDataset(args.test_data, stats=stats)
    ds.load_truth_fields()
    n_full = min(len(ds), args.max_samples) if args.max_samples else len(ds)
    with np.load(args.test_data) as d:
        R, Z = d["R"], d["Z"]
    model = build_model(**ckpt.get("arch", {})).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # ---- per-sample metrics (cached, schema-versioned) ----
    SCHEMA = 1
    stats_path = out_dir / "stats_per_sample.json"
    if stats_path.exists():
        with open(stats_path) as f:
            cached = json.load(f)
        rows = {k: np.array(v) for k, v in cached.items() if k != "schema"} \
            if cached.get("schema") == SCHEMA else None
        if rows is None:
            print(f"stale cache (schema mismatch), recomputing: {stats_path}")
    else:
        rows = None
    if rows is None:
        print(f"evaluating {n_full} test samples (checkpoint {args.checkpoint})")
        rows = {k: [] for k in
                ["rel_l2_pct", "rmse_phys", "r_squared", "j_self_check_pct",
                 "gs_pred", "gs_true", "x_lo_cm", "x_up_cm", "o_point_cm",
                 "sep_mean_cm", "sep_hausdorff_cm", "sep_area_rel_err_pct",
                 "n_xpt_pred"]}
        dA = float(R[1, 0] - R[0, 0]) * float(Z[0, 1] - Z[0, 0])
        with torch.no_grad():
            for i in range(n_full):
                b = ds.full_grid(i)
                out = model(b["x"].to(device))
                psi_z = out[:, 0].cpu().numpy().reshape(65, 65)
                j_z = out[:, 1].cpu().numpy().reshape(65, 65)
                y_norm = b["y_psi"].numpy().reshape(65, 65)
                rel_l2 = float(np.linalg.norm(psi_z - y_norm) /
                               (np.linalg.norm(y_norm) + 1e-12))
                rows["rel_l2_pct"].append(rel_l2 * 100.0)

                psi_pred = psi_z * float(stats["psi_std"]) + float(stats["psi_mean"])
                psi_true = ds.psi[i]
                rows["rmse_phys"].append(
                    float(np.sqrt(np.mean((psi_pred - psi_true) ** 2))))
                denom = np.sum((psi_true - psi_true.mean()) ** 2) + 1e-30
                rows["r_squared"].append(
                    1.0 - float(np.sum((psi_pred - psi_true) ** 2)) / denom)
                j_phys = j_z * float(stats["j_std"]) + float(stats["j_mean"])
                est = float(np.sum(j_phys) * dA)
                rows["j_self_check_pct"].append(
                    abs(est - ds.ip[i]) / (abs(ds.ip[i]) + 1e-30) * 100.0)

                mask_i = (psi_true >= ds.axes[i][2]).astype(np.float32)
                rows["gs_pred"].append(gs_residual_ratio(
                    psi_pred, R, Z, ds.dpdpsi[i], ds.FdFdpsi[i], mask_i))
                rows["gs_true"].append(gs_residual_ratio(
                    psi_true, R, Z, ds.dpdpsi[i], ds.FdFdpsi[i], mask_i))

                gm = geometry_metrics(
                    psi_pred, psi_true, R, Z,
                    xpts_true=ds.xpts_actual[i] if ds.xpts_actual is not None else None,
                    o_true=ds.o_point[i], psi_bndry_true=float(ds.axes[i][2]),
                    anchor=None)
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
    else:
        n_eval = len(rows["rel_l2_pct"])
        print(f"reusing cached per-sample stats (schema {SCHEMA}): {stats_path}")

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
        xa = ds.xpts_actual
        if xa is not None:
            xpts_t = [tuple(map(float, row)) for row in xa[j]]
            o_t = (float(ds.o_point[j][0]), float(ds.o_point[j][1]))
            bnd_t = float(ds.axes[j][2])
            match = match_xpoints_and_axis(psi_pred, R, Z, xa[j], bnd_t, None)
            sep_t = separatrix_pts(psi_true, R, Z, bnd_t, o_t)
            sep_p = separatrix_pts(psi_pred, R, Z, match["bnd_p"], match["o_pred"]) \
                if match["bnd_p"] is not None else None
            return xpts_t, o_t, sep_t, match["xpt_pred"], match["o_pred"], sep_p
        return None, None, None, None, None, None

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
        xpt_t, o_t, sep_t, xpt_p, o_p, sep_p = fig1_geoms(psi_p, psi_t, idx)

        plot_field(ax_t, R, Z, psi_t, lvls, f"{tag} # {idx}  |  freegs truth ψ (Wb)",
                   xpt_t, o_t, sep_t, wall_r, wall_z, coils)
        cf = plot_field(ax_p, R, Z, psi_p, lvls, "KAN prediction ψ (Wb)",
                        xpt_p, o_p, sep_p, wall_r, wall_z, coils)
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
        ax_d.set_title(f"{tag} # {idx}  |  |Δψ| (Wb)", fontsize=10)
        ax_d.set_xlabel("R (m)")
        ax_d.set_ylabel("Z (m)")
        cb_diff = fig.colorbar(im, cax=cax_diff)
        fmt_cbar(cb_diff)
    fig.suptitle(args.title, fontsize=13)
    fig.savefig(out_dir / "fig1_best_worst_psi.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  saved fig1_best_worst_psi.png")

    # ---- fig 2: field-level statistics ----
    fig, axes = plt.subplots(1, 4, figsize=(19.5, 4.6))
    rl = rows["rel_l2_pct"]
    axes[0].hist(rl, bins=40)
    axes[0].axvline(PAPER["rel_l2_kan2_pct"], color="red", ls="--",
                    label="paper KAN-2 0.912%")
    axes[0].axvline(PAPER["rel_l2_kan1_pct"], color="tab:green", ls="--",
                    label="paper KAN-1 0.631%")
    axes[0].set_title(f"rel L2 (%) — mean {np.nanmean(rl):.3f}")
    axes[0].legend(fontsize=8)

    rm = rows["rmse_phys"]
    axes[1].hist(rm, bins=40)
    axes[1].set_title(f"RMSE ψ (Wb) — mean {np.nanmean(rm):.2e}")

    r2 = rows["r_squared"]
    axes[2].hist(r2, bins=40)
    axes[2].axvline(PAPER["r_squared"], color="red", ls="--",
                    label="paper R² 0.9953")
    axes[2].set_title(f"R² — mean {np.nanmean(r2):.4f}")
    axes[2].legend(fontsize=8)

    gr = np.array(rows["gs_pred"]) / (np.array(rows["gs_true"]) + 1e-30)
    axes[3].hist(gr, bins=40, range=(0, 3))
    axes[3].axvline(1.0, color="red", ls="--", label="= truth")
    axes[3].set_title(f"GS residual ratio pred/true — mean {np.nanmean(gr):.3f}")
    axes[3].legend(fontsize=8)
    for ax in axes:
        ax.set_xlabel(ax.get_title().split("—")[0].strip())
    fig.suptitle(args.title, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "fig2_field_stats.png", dpi=150)
    plt.close(fig)
    print("  saved fig2_field_stats.png")

    # ---- fig 3: geometry statistics ----
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 9))
    geo_pairs = [
        ("x_lo_cm", "lower X-point err (cm)", axes[0, 0]),
        ("x_up_cm", "upper X-point err (cm)", axes[0, 1]),
        ("o_point_cm", "O-point err (cm)", axes[0, 2]),
        ("sep_mean_cm", "separatrix mean dist (cm)", axes[1, 0]),
        ("sep_hausdorff_cm", "separatrix Hausdorff (cm)", axes[1, 1]),
        ("sep_area_rel_err_pct", "separatrix area rel err (%)", axes[1, 2]),
    ]
    for key, label, ax in geo_pairs:
        v = rows[key]
        vn = v[~np.isnan(v)]
        ax.hist(vn, bins=30)
        if vn.size:
            ax.axvline(np.nanmean(v), color="red", ls="--",
                       label=f"mean {np.nanmean(v):.3f}")
            ax.legend(fontsize=8)
        ax.set_title(label, fontsize=10)
    fig.suptitle(args.title, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "fig3_geometry_stats.png", dpi=150)
    plt.close(fig)
    print("  saved fig3_geometry_stats.png")
    print(f"\nall figures saved -> {out_dir}")


if __name__ == "__main__":
    main()
