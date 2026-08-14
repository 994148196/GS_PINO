"""Anchor-bucket analysis for data_v3 experiments (exp004/exp005).

For each checkpoint, evaluate the test set per-sample rel L2 and bucket the
errors by properties of the isoflux anchor (distance from the default (1.5, 0.0),
quadrant, distance to nearest X-point). Intended read: with anchor-only input
missing, errors grow with anchor excursion (A 11ch), while full-info (A' 13ch)
and coil (B 11ch) inputs stay flat — quantifying the one-to-many gap.

Usage:
  python -m gs_pino_dn_fno_2608.analyze_anchor_buckets \
      --test-data dn_fno_2608/data_v3/test.npz \
      --checkpoint <name>=<path.best.pt> [--checkpoint ...] \
      --out-dir dn_fno_2608/experiments/exp004_xpoints_anchor_v3/analysis
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from gs_pino_dn_fno_2608.data_dn_fno import DNFnoDataset
from gs_pino_dn_fno_2608.data_dn_fno_coils import DNFnoDatasetCoils
from gs_pino_dn_fno_2608.model_dn_fno import build_model

REF_ANCHOR = (1.5, 0.0)


def _dataset_and_channels(test_data: str, ckpt: dict) -> tuple:
    stats = {k: (np.asarray(v, dtype=np.float32) if isinstance(v, list) else np.float32(v))
             for k, v in ckpt["stats"].items() if k != "input_mode"}
    mode = ckpt.get("input_mode", "xpoints")
    in_ch = 2 + len(stats["scalar_mean"])
    if mode == "coils":
        return DNFnoDatasetCoils(test_data, stats=stats), in_ch
    return DNFnoDataset(test_data, stats=stats, use_anchor=(mode == "xa")), in_ch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-data", required=True)
    ap.add_argument("--checkpoint", action="append", required=True,
                    help="name=path.best.pt (repeatable)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with np.load(args.test_data) as d:
        anchor = d["anchor"]            # (N, 2)
        x_coords = d["x_coords"]        # (N, 4)

    dist_ref = np.hypot(anchor[:, 0] - REF_ANCHOR[0], anchor[:, 1] - REF_ANCHOR[1])
    xpts = np.stack([x_coords[:, 0:2], x_coords[:, 2:4]], axis=1)  # (N, 2, 2)
    d_min = np.array([np.hypot(a[0] - x[:, 0], a[1] - x[:, 1]).min() for a, x in zip(anchor, xpts)])

    def bucket_of(v: float, edges: list[float]) -> int:
        for i, e in enumerate(edges):
            if v < e:
                return i
        return len(edges)

    quad = np.where(anchor[:, 0] >= REF_ANCHOR[0], 1, 0) * 2 + np.where(anchor[:, 1] >= 0, 1, 0)
    labels = {
        "dist_ref": ["<0.2m", "0.2-0.4m", ">=0.4m"],
        "quadrant": ["R<1.5,Z<0", "R<1.5,Z>=0", "R>=1.5,Z<0", "R>=1.5,Z>=0"],
        "d_min_xpt": ["<0.3m", "0.3-0.6m", ">=0.6m"],
    }
    bkt = {
        "dist_ref": [bucket_of(v, [0.2, 0.4]) for v in dist_ref],
        "quadrant": [int(q) for q in quad],
        "d_min_xpt": [bucket_of(v, [0.3, 0.6]) for v in d_min],
    }

    results: dict[str, dict] = {}
    for spec in args.checkpoint:
        name, path = spec.split("=", 1)
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        ds, in_ch = _dataset_and_channels(args.test_data, ckpt)
        model = build_model(in_channels=in_ch).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        rel_l2 = np.empty(len(ds))
        with torch.no_grad():
            for i in range(len(ds)):
                x, y = ds[i]
                pred = model(x[None].to(device)).squeeze(0).cpu().numpy()
                rel_l2[i] = float(np.linalg.norm(pred - y.numpy()) / (np.linalg.norm(y.numpy()) + 1e-12))

        results[name] = {"input_mode": ckpt.get("input_mode", "xpoints"),
                         "in_channels": in_ch,
                         "overall_rel_l2_pct": float(rel_l2.mean() * 100)}
        for key, tag in labels.items():
            bm = np.array(bkt[key])   # (N,) int bucket ids; np-compare needed (list==int is scalar)
            table = []
            for b in range(len(tag)):
                m = rel_l2[bm == b]
                if len(m) == 0:
                    table.append({"bucket": tag[b], "count": 0})
                    continue
                table.append({"bucket": tag[b], "count": int(len(m)),
                              "rel_l2_pct_mean": float(m.mean() * 100),
                              "rel_l2_pct_median": float(np.median(m) * 100),
                              "rel_l2_pct_p95": float(np.percentile(m, 95) * 100)})
            results[name][f"buckets_{key}"] = table

    with open(out_dir / "anchor_buckets.json", "w") as f:
        json.dump(results, f, indent=2)

    # terminal table
    print(f"\n{'='*78}")
    for name, r in results.items():
        print(f"\n  {name} [{r['input_mode']}, {r['in_channels']}ch] | overall rel L2 {r['overall_rel_l2_pct']:.4f}%")
        for key, tag in labels.items():
            print(f"    bucket by {key}:")
            for t in r[f"buckets_{key}"]:
                if t["count"] == 0:
                    print(f"      {t['bucket']:10s}: n=0")
                else:
                    print(f"      {t['bucket']:10s}: n={t['count']:4d}  "
                          f"mean {t['rel_l2_pct_mean']:.4f}%  "
                          f"med {t['rel_l2_pct_median']:.4f}%  p95 {t['rel_l2_pct_p95']:.4f}%")
    print(f"\n  saved -> {out_dir}/anchor_buckets.json")


if __name__ == "__main__":
    main()
