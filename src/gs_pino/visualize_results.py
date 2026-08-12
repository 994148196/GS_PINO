"""Visualization script for free-boundary GS PINO surrogate results.

Creates comparison plots showing:
1. Predicted vs True psi_total contour maps
2. Error maps (absolute and relative)
3. Line cuts at different radii
4. Scatter plots of predicted vs true values
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import torch

from .data_freegs import FreeBndDataset


def load_test_data(checkpoint_path: str, data_path: str):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    test_idx = checkpoint["test_indices"]
    param_norm = type("obj", (), {
        "mean": checkpoint["param_mean"],
        "std": checkpoint["param_std"],
        "apply": lambda self, x: (x - self.mean) / (self.std + 1e-7),
    })()
    
    test_ds = FreeBndDataset(data_path, test_idx, param_norm)
    
    preds = []
    targets = []
    masks = []
    metas = []
    
    for i in range(len(test_ds)):
        measures, R, Z, psi_total, mask, psi_plasma, psi_coils, rhs, meta = test_ds[i]
        metas.append(meta)
    
    pred_file = Path(checkpoint_path).parent / "test_predictions.pt"
    pred_data = torch.load(pred_file, map_location="cpu", weights_only=False)

    # 新版 evaluate_freegs 保存的键：preds / preds_total / targets_plasma / targets_total / masks / interior_masks
    # 旧版只有 preds / targets / masks。优先使用总通量（psi_total）做对比。
    preds = pred_data.get("preds_total", pred_data["preds"])
    if "targets_total" in pred_data:
        targets = pred_data["targets_total"]
    else:
        targets = pred_data["targets"]
    masks = pred_data["masks"]

    return preds, targets, masks, metas, test_ds


def plot_contour_comparison(pred, target, mask, R, Z, idx, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    
    vmin = min(pred.min(), target.min())
    vmax = max(pred.max(), target.max())
    
    im0 = axes[0].contourf(R, Z, target, levels=20, vmin=vmin, vmax=vmax, cmap="viridis")
    axes[0].set_title(f"True psi_total (Sample {idx})")
    axes[0].set_xlabel("R (m)")
    axes[0].set_ylabel("Z (m)")
    axes[0].set_aspect("equal")
    plt.colorbar(im0, ax=axes[0])
    
    im1 = axes[1].contourf(R, Z, pred, levels=20, vmin=vmin, vmax=vmax, cmap="viridis")
    axes[1].set_title(f"Predicted psi_total (Sample {idx})")
    axes[1].set_xlabel("R (m)")
    axes[1].set_aspect("equal")
    plt.colorbar(im1, ax=axes[1])
    
    error = pred - target
    vmax_err = max(abs(error.min()), abs(error.max()))
    norm = TwoSlopeNorm(vcenter=0, vmin=-vmax_err, vmax=vmax_err)
    im2 = axes[2].contourf(R, Z, error, levels=20, norm=norm, cmap="RdBu_r")
    axes[2].set_title(f"Error (Pred - True)")
    axes[2].set_xlabel("R (m)")
    axes[2].set_aspect("equal")
    plt.colorbar(im2, ax=axes[2])
    
    fig.savefig(output_dir / f"contour_{idx:03d}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_error_maps(pred, target, mask, R, Z, idx, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    
    abs_error = np.abs(pred - target)
    im0 = axes[0].contourf(R, Z, abs_error, levels=20, cmap="Reds")
    axes[0].set_title(f"Absolute Error (Sample {idx})")
    axes[0].set_xlabel("R (m)")
    axes[0].set_ylabel("Z (m)")
    axes[0].set_aspect("equal")
    plt.colorbar(im0, ax=axes[0])
    
    rel_error = np.abs(pred - target) / (np.abs(target) + 1e-6)
    im1 = axes[1].contourf(R, Z, rel_error, levels=20, vmin=0, vmax=0.5, cmap="Oranges")
    axes[1].set_title(f"Relative Error (Sample {idx})")
    axes[1].set_xlabel("R (m)")
    axes[1].set_aspect("equal")
    plt.colorbar(im1, ax=axes[1])
    
    fig.savefig(output_dir / f"error_{idx:03d}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_line_cuts(pred, target, mask, R, Z, idx, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    
    z_center = Z[0, Z.shape[1] // 2]
    z_idx = np.argmin(np.abs(Z[0, :] - z_center))
    
    r_vals = R[:, 0]
    axes[0].plot(r_vals, target[:, z_idx], "b-", label="True")
    axes[0].plot(r_vals, pred[:, z_idx], "r--", label="Predicted")
    if np.any(mask[:, z_idx] > 0.5):
        axes[0].axvline(x=r_vals[mask[:, z_idx] > 0.5].min(), 
                        color="g", linestyle=":", label="Plasma boundary")
        axes[0].axvline(x=r_vals[mask[:, z_idx] > 0.5].max(), 
                        color="g", linestyle=":")
    axes[0].set_title(f"Line Cut at Z={z_center:.3f} (Sample {idx})")
    axes[0].set_xlabel("R (m)")
    axes[0].set_ylabel("psi_total")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    r_center = R[R.shape[0] // 2, 0]
    r_idx = np.argmin(np.abs(R[:, 0] - r_center))
    
    z_vals = Z[0, :]
    axes[1].plot(z_vals, target[r_idx, :], "b-", label="True")
    axes[1].plot(z_vals, pred[r_idx, :], "r--", label="Predicted")
    if np.any(mask[r_idx, :] > 0.5):
        axes[1].axvline(x=z_vals[mask[r_idx, :] > 0.5].min(), 
                        color="g", linestyle=":", label="Plasma boundary")
        axes[1].axvline(x=z_vals[mask[r_idx, :] > 0.5].max(), 
                        color="g", linestyle=":")
    axes[1].set_title(f"Line Cut at R={r_center:.3f} (Sample {idx})")
    axes[1].set_xlabel("Z (m)")
    axes[1].set_ylabel("psi_total")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    fig.savefig(output_dir / f"linecuts_{idx:03d}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_scatter_all(preds, targets, masks, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    
    all_pred = preds.flatten()
    all_target = targets.flatten()
    all_mask = masks.flatten()
    
    axes[0].scatter(all_target, all_pred, alpha=0.1, s=1)
    axes[0].plot([all_target.min(), all_target.max()], [all_target.min(), all_target.max()], "r--")
    axes[0].set_title("Predicted vs True (All Points)")
    axes[0].set_xlabel("True psi_total")
    axes[0].set_ylabel("Predicted psi_total")
    axes[0].set_aspect("equal")
    axes[0].grid(alpha=0.3)
    
    plasma_idx = all_mask > 0.5
    axes[1].scatter(all_target[plasma_idx], all_pred[plasma_idx], alpha=0.1, s=1)
    axes[1].plot([all_target[plasma_idx].min(), all_target[plasma_idx].max()], 
                 [all_target[plasma_idx].min(), all_target[plasma_idx].max()], "r--")
    axes[1].set_title("Predicted vs True (Plasma Region Only)")
    axes[1].set_xlabel("True psi_total")
    axes[1].set_ylabel("Predicted psi_total")
    axes[1].set_aspect("equal")
    axes[1].grid(alpha=0.3)
    
    fig.savefig(output_dir / "scatter_all.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_error_histogram(preds, targets, masks, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    
    error = preds - targets
    rel_error = np.abs(error) / (np.abs(targets) + 1e-6)
    
    axes[0].hist(error.flatten(), bins=50, alpha=0.7, edgecolor="black")
    axes[0].set_title("Distribution of Absolute Errors")
    axes[0].set_xlabel("Error (Pred - True)")
    axes[0].set_ylabel("Count")
    axes[0].grid(alpha=0.3)
    
    axes[1].hist(rel_error.flatten(), bins=50, alpha=0.7, edgecolor="black", range=(0, 0.5))
    axes[1].set_title("Distribution of Relative Errors (0-0.5)")
    axes[1].set_xlabel("Relative Error")
    axes[1].set_ylabel("Count")
    axes[1].grid(alpha=0.3)
    
    fig.savefig(output_dir / "error_histogram.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to best.pt checkpoint.")
    parser.add_argument("--data", default="data/freegs_merged_1500.npz", help="Free-boundary dataset path.")
    parser.add_argument("--num-samples", type=int, default=5, help="Number of samples to visualize.")
    args = parser.parse_args()
    
    output_dir = Path(args.checkpoint).parent / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    preds, targets, masks, metas, test_ds = load_test_data(args.checkpoint, args.data)
    
    n_samples = min(args.num_samples, len(preds))
    
    for i in range(n_samples):
        R = metas[i]["R"].numpy()
        Z = metas[i]["Z"].numpy()
        pred = preds[i]
        target = targets[i]
        mask = masks[i]
        
        plot_contour_comparison(pred, target, mask, R, Z, i, output_dir)
        plot_error_maps(pred, target, mask, R, Z, i, output_dir)
        plot_line_cuts(pred, target, mask, R, Z, i, output_dir)
        print(f"  Generated plots for sample {i+1}/{n_samples}")
    
    plot_scatter_all(preds, targets, masks, output_dir)
    plot_error_histogram(preds, targets, masks, output_dir)
    
    print(f"\nAll visualizations saved to: {output_dir}")


if __name__ == "__main__":
    main()
