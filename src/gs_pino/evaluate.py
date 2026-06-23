from __future__ import annotations

import argparse, json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from .data import GSDataset, Normalization
from .losses import masked_mse
from .models import UFNO2d


def rel_l2(pred, y, mask):
    num = (((pred - y) ** 2) * mask).sum(dim=(1,2,3)).sqrt()
    den = ((y ** 2 * mask).sum(dim=(1,2,3))).sqrt().clamp_min(1e-8)
    return num / den


def plot_case(path, R, Z, true, pred, mask):
    err = np.where(mask > 0, pred - true, np.nan)
    true = np.where(mask > 0, true, np.nan); pred = np.where(mask > 0, pred, np.nan)
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.5), constrained_layout=True)
    for a, data, title in zip(ax, [true, pred, err], ["true psi_bar", "pred psi_bar", "error"]):
        im = a.contourf(R, Z, data, levels=40)
        a.contour(R, Z, mask, levels=[0.5], colors="k", linewidths=1)
        a.set_aspect("equal"); a.set_title(title); fig.colorbar(im, ax=a)
    fig.savefig(path, dpi=180); plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/gs_fixed_boundary.npz")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-dir", default="outputs/eval")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-plots", type=int, default=6)
    args = p.parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    idx = ckpt.get("test_indices")
    ds = GSDataset(args.data, idx, Normalization(ckpt["param_mean"], ckpt["param_std"]))
    loader = DataLoader(ds, batch_size=args.batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_args = ckpt["args"]
    model = UFNO2d(ds[0][0].shape[0], model_args["modes1"], model_args["modes2"], model_args["width"], model_args["layers"]).to(device)
    model.load_state_dict(ckpt["model"]); model.eval()
    mse_vals = []; rel_vals = []
    with torch.no_grad():
        for x, y, mask, sdf, params in loader:
            pred = model(x.to(device)).cpu()
            mse_vals.append(float(masked_mse(pred, y, mask)))
            rel_vals.extend(rel_l2(pred, y, mask).numpy().tolist())
    metrics = {"masked_mse": float(np.mean(mse_vals)), "relative_l2_mean": float(np.mean(rel_vals)), "relative_l2_p95": float(np.percentile(rel_vals, 95)), "n_cases": len(ds)}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    raw = np.load(args.data)
    for j in range(min(args.max_plots, len(ds))):
        i = int(ds.indices[j]); x, y, mask, *_ = ds[j]
        with torch.no_grad(): pred = model(x[None].to(device)).cpu().numpy()[0,0]
        plot_case(out / f"case_{j:03d}.png", raw["R"][i], raw["Z"][i], y.numpy()[0], pred, mask.numpy()[0])

if __name__ == "__main__":
    main()
