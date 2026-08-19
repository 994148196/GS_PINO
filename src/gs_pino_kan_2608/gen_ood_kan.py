"""M3: OOD dataset generation for the KAN extrapolation test (paper Sec. 2.2).

Reuses the FNO freegs generator (`gs_pino_dn_fno_2608.generate_dn_dataset`)
by pure import + runtime registry override — the FNO package itself is never
edited. The OOD parameter box is anchored on the ACTUAL training subset the
KAN saw (nested 500 rows, same indices as train_kan):

    Ip, paxis  x [0.8, 1.25]      (both directions)
    fvac       +/- 0.15
    alpha_m/n  +/- 0.3
    X-point jitter x 1.5          (R 0.06 -> 0.09, Z 0.10 -> 0.15)

Coil currents are not sampled directly — they come out of the freegs solve
for the OOD equilibrium (hence automatically OOD as well). The 16D
(params + 11 coils) in_hull filter drops samples that ended up inside the
training envelope.

Workflow (per config): --probe (acceptance rate) -> generate chunks
(resumable) -> filter against training hull -> save data_ood/{config}/test.npz
+ ood_box.json + hull.json.

Usage:
  "$PY" -u -m gs_pino_kan_2608.gen_ood_kan \
    --config dn --train-data dn_fno_2608/data_v5/dn/train.npz,dn_fno_2608/data_v5/sn/train.npz \
    --out-dir dn_fno_2608/kan/data_ood --n-samples 600 --probe-only
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import re
import sys
from pathlib import Path

import numpy as np

import gs_pino_dn_fno_2608.generate_dn_dataset as G
from gs_pino_dn_fno_2608.data_dn_fno import nested_train_indices
from gs_pino_kan_2608.data_kan import PointKANDataset
from gs_pino_kan_2608.extrapolate_kan import build_hull, in_hull

# data_v5 nominal (MAST, probe-calibrated; see run_generate_v5.sh)
V5_XPT_JITTER_R, V5_XPT_JITTER_Z = 0.06, 0.10
OOD_SCALE = {"Ip": (0.8, 1.25), "paxis": (0.8, 1.25), "fvac": (-0.15, +0.15),
             "alpha_m": (-0.3, +0.3), "alpha_n": (-0.3, +0.3)}
MIN_ACCEPT = 0.20        # probe acceptance floor (plan: <20% -> tighten)


def train_rows(train_data: str, n_train: int, seed: int):
    """(params (N,5), coils (N,11)) of the nested training subset (KAN parity)."""
    paths = [p.strip() for p in train_data.split(",")]
    ds = PointKANDataset(paths)
    idx = nested_train_indices(ds.n_full, n_train, seed)
    params = np.concatenate([np.load(p)["params"] for p in paths])[idx]
    coils = np.concatenate([np.load(p)["coil_currents"] for p in paths])[idx]
    return params, coils


def build_ood_ranges(params: np.ndarray) -> dict:
    """OOD param box anchored on the actual train min/max (per config)."""
    names = ["Ip", "paxis", "fvac", "alpha_m", "alpha_n"]  # data_v5 params order
    ranges = {}
    for j, name in enumerate(names):
        lo, hi = float(params[:, j].min()), float(params[:, j].max())
        dl, dh = OOD_SCALE[name]
        if name in ("Ip", "paxis"):
            ranges[name] = (lo * dl, hi * dh)
        else:  # fvac / alpha: absolute offsets
            ranges[name] = (lo + dl, hi + dh)
    return ranges


def _run_generate(cfg_kwargs: dict, n: int, chunk_size: int, n_jobs: int,
                  ood_ranges: dict, alpha_ranges: dict, xpt_jitter: float,
                  xpt_jitter_z: float, machine: str, config: str):
    """generate() with registry overrides; returns the captured stdout text."""
    spec = G.CONFIG_SPECS[(machine, config)]
    orig_pr = spec.get("param_ranges")
    orig_al = G.ALPHA_RANGES
    spec["param_ranges"] = ood_ranges
    G.ALPHA_RANGES = alpha_ranges
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            G.generate(**cfg_kwargs, n_samples=n, chunk_size=chunk_size,
                       n_jobs=n_jobs, alpha_sampling=True,
                       xpt_jitter=xpt_jitter, xpt_jitter_z=xpt_jitter_z,
                       machine=machine, config=config)
    finally:
        spec["param_ranges"] = orig_pr
        G.ALPHA_RANGES = orig_al
    return buf.getvalue()


def acceptance(text: str) -> tuple[int, int]:
    m = re.findall(r"(\d+)/(\d+) accepted", text)
    if not m:
        raise RuntimeError(f"could not parse acceptance from output:\n{text[-500:]}")
    a, s = m[-1]
    return int(a), int(s)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=["dn", "sn"], required=True)
    ap.add_argument("--train-data", required=True,
                    help="dn+sn train npz (envelope + hull + nested indices)")
    ap.add_argument("--out-dir", default="dn_fno_2608/kan/data_ood")
    ap.add_argument("--n-train", type=int, default=500)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--n-samples", type=int, default=600)
    ap.add_argument("--chunk-size", type=int, default=200)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--probe-n", type=int, default=80)
    ap.add_argument("--min-keep", type=int, default=200)
    ap.add_argument("--hull-mode", choices=["ellipsoid", "convex", "box"],
                    default="ellipsoid",
                    help="training envelope (convex: qhull needs k<=10 on this data)")
    ap.add_argument("--probe-only", action="store_true")
    ap.add_argument("--skip-probe", action="store_true",
                    help="chunks already validated — don't re-probe")
    ap.add_argument("--skip-generate", action="store_true",
                    help="chunk files already on disk — merge + filter only")
    ap.add_argument("--max-retries", type=int, default=5)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    (out_dir / args.config).mkdir(parents=True, exist_ok=True)

    params, coils = train_rows(args.train_data, args.n_train, args.seed)
    ood_ranges = build_ood_ranges(params)
    alpha_ranges = {"alpha_m": ood_ranges["alpha_m"], "alpha_n": ood_ranges["alpha_n"]}
    xj, xjz = V5_XPT_JITTER_R * 1.5, V5_XPT_JITTER_Z * 1.5
    print(f"== OOD generation | config={args.config} | machine=mast ==")
    for j, k in enumerate(("Ip", "paxis", "fvac", "alpha_m", "alpha_n")):
        print(f"    {k:8s} train [{params[:, j].min():.4g}, {params[:, j].max():.4g}] "
              f"-> OOD {ood_ranges[k]}")
    print(f"  xpt jitter x1.5: R +-{xj:.3f} / Z +-{xjz:.3f} m")

    # ---- acceptance probe ----
    # save_constraint_diag=True matches data_v5 (the wall-contact block in
    # _acceptance_checks is v4_active-gated; without it wall_contact stays
    # unbound on MAST and every solve raises UnboundLocalError)
    cfg = dict(out_dir=str(out_dir / "tmp" / args.config), split="test",
               seed=args.seed, max_retries=args.max_retries,
               save_constraint_diag=True)
    # probe writes to its OWN chunk dir so it never collides with the full
    # run's chunk_000 (a partial 80-row chunk_000 would otherwise truncate
    # the first generation chunk via the resume logic)
    cfg_probe = {**cfg, "out_dir": str(out_dir / "tmp" / args.config / "probe")}
    if args.skip_probe:
        print("  probe skipped (--skip-probe)")
    else:
        text = _run_generate(cfg_probe, args.probe_n, args.probe_n, args.n_jobs,
                             ood_ranges, alpha_ranges, xj, xjz, "mast", args.config)
        a, s = acceptance(text)
        rate = a / max(s, 1)
        print(f"  probe: {a}/{s} accepted ({rate*100:.1f}%)")
        if rate < MIN_ACCEPT:
            raise SystemExit(
                f"  probe acceptance {rate*100:.1f}% < {MIN_ACCEPT*100:.0f}% — "
                f"widen the box or drop acceptance constraints (plan: tighten 0.5x)")
        if args.probe_only:
            print("  probe OK; run without --probe-only to generate.")
            return

    # ---- full generation (chunks are resumable) ----
    a = None
    if args.skip_generate:
        print("  generation skipped (--skip-generate); merging existing chunks")
    else:
        text = _run_generate(cfg, args.n_samples, args.chunk_size, args.n_jobs,
                             ood_ranges, alpha_ranges, xj, xjz, "mast", args.config)
        a, s = acceptance(text)
        print(f"  generated: {a}/{s} accepted")
        if a == 0:
            raise SystemExit("  0 accepted — aborting")

    # ---- in_hull filter against the 16D training envelope ----
    hull = build_hull(np.hstack([params, coils]), args.hull_mode)
    print(f"  hull: mode={hull['mode']} k={hull.get('k', 16)} "
          f"thr={hull.get('thr', '')} facets={hull.get('n_facets', '')}")
    with open(out_dir / "hull.json", "w") as f:
        json.dump(hull, f, indent=1)

    chunk_dir = Path(cfg["out_dir"]) / "test"
    rows, x16 = [], []
    for ci, cp in enumerate(sorted(chunk_dir.glob("chunk_*.npz"))):
        with np.load(cp) as d:
            n = d["params"].shape[0]
            x16.append(np.hstack([d["params"], d["coil_currents"]]))
            rows.append({k: d[k] for k in d.keys() if k not in ("R", "Z")})
    x16 = np.vstack(x16)
    keep_flat = ~in_hull(x16, hull)
    keep = []                                   # per-chunk masks (chunk_000 can
    i = 0                                       # be the probe's partial 80-row)
    for cp in sorted(chunk_dir.glob("chunk_*.npz")):
        with np.load(cp) as d:
            n = d["params"].shape[0]
        keep.append(keep_flat[i:i + n])
        i += n
    print(f"  in_hull filter: {int(keep_flat.sum())}/{len(keep_flat)} kept OOD "
          f"({int((~keep_flat).sum())} dropped as in-distribution)")
    if int(keep_flat.sum()) < args.min_keep:
        print(f"  WARNING: kept {int(keep_flat.sum())} < --min-keep {args.min_keep} — "
              f"regenerate with larger --n-samples")

    merged = {k: np.concatenate([r[k][m] for r, m in zip(rows, keep)])
              for k in rows[0]}
    with np.load(sorted(chunk_dir.glob("chunk_*.npz"))[0]) as d0:
        merged["R"], merged["Z"] = d0["R"], d0["Z"]
    out_npz = out_dir / args.config / "test.npz"
    np.savez(out_npz, **merged)
    box_file = out_dir / f"ood_box_{args.config}.json"
    with open(box_file, "w") as f:
        json.dump({"config": args.config, "n_train": args.n_train, "seed": args.seed,
                   "n_generated": (a if a is not None else "skipped"),
                   "n_kept_ood": int(keep_flat.sum()),
                   "xpt_jitter_r": xj, "xpt_jitter_z": xjz,
                   "ranges": {k: ood_ranges[k] for k in ood_ranges},
                   "train_minmax": {k: [float(params[:, j].min()),
                                        float(params[:, j].max())]
                                    for j, k in
                                    enumerate(("Ip", "paxis", "fvac",
                                               "alpha_m", "alpha_n"))}},
                  f, indent=1)
    print(f"  saved -> {out_npz} (rows {merged['params'].shape[0]})")
    print(f"  saved -> {box_file}, hull.json")


if __name__ == "__main__":
    main()
