"""Visualization CLI for the free-boundary GS PINO surrogate.

Generates plots:
  - psi_total comparison (predicted vs ground truth)
  - psi_plasma comparison (predicted vs ground truth)
  - Error heatmap
  - Plasma region zoom
  - LCFS contour comparison
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_psi_comparison(R, Z, pred, target, mask, idx, output_dir, title_prefix="psi_total"):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    
    vmin = min(pred.min(), target.min())
    vmax = max(pred.max(), target.max())
    
    im0 = axes[0].pcolormesh(R, Z, target, cmap="viridis", vmin=vmin, vmax=vmax, shading="gouraud")
    axes[0].contour(R, Z, target, levels=[0.0], colors="white", linewidths=1.5)
    axes[0].set_title(f"Ground Truth {title_prefix} (Sample {idx})")
    axes[0].set_xlabel("R (m)")
    axes[0].set_ylabel("Z (m)")
    axes[0].axis("equal")
    
    im1 = axes[1].pcolormesh(R, Z, pred, cmap="viridis", vmin=vmin, vmax=vmax, shading="gouraud")
    axes[1].contour(R, Z, pred, levels=[0.0], colors="white", linewidths=1.5)
    axes[1].set_title(f"Predicted {title_prefix}")
    axes[1].set_xlabel("R (m)")
    axes[1].axis("equal")
    
    diff = pred - target
    im2 = axes[2].pcolormesh(R, Z, diff, cmap="RdBu_r", shading="gouraud")
    axes[2].contour(R, Z, mask, levels=[0.5], colors="black", linewidths=1.0)
    axes[2].set_title(f"Error (Pred - True)")
    axes[2].set_xlabel("R (m)")
    axes[2].axis("equal")
    
    fig.colorbar(im0, ax=axes[0], label=f"ψ_{title_prefix} (Wb/m)")
    fig.colorbar(im1, ax=axes[1], label=f"ψ_{title_prefix} (Wb/m)")
    fig.colorbar(im2, ax=axes[2], label="Error (Wb/m)")
    
    fig.savefig(output_dir / f"{title_prefix}_comparison_{idx}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_error_heatmap(R, Z, pred, target, mask, idx, output_dir, title_prefix="psi_total"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    
    abs_error = np.abs(pred - target)
    im0 = axes[0].pcolormesh(R, Z, abs_error, cmap="plasma", shading="gouraud")
    axes[0].contour(R, Z, mask, levels=[0.5], colors="white", linewidths=1.5)
    axes[0].set_title(f"{title_prefix} Absolute Error (Sample {idx})")
    axes[0].set_xlabel("R (m)")
    axes[0].set_ylabel("Z (m)")
    axes[0].axis("equal")
    
    rel_error = np.abs(pred - target) / (np.abs(target) + 1e-10)
    im1 = axes[1].pcolormesh(R, Z, rel_error, cmap="plasma", vmin=0, vmax=0.1, shading="gouraud")
    axes[1].contour(R, Z, mask, levels=[0.5], colors="white", linewidths=1.5)
    axes[1].set_title(f"{title_prefix} Relative Error")
    axes[1].set_xlabel("R (m)")
    axes[1].axis("equal")
    
    fig.colorbar(im0, ax=axes[0], label="|Pred - True| (Wb/m)")
    fig.colorbar(im1, ax=axes[1], label="|Pred - True| / |True|")
    
    fig.savefig(output_dir / f"{title_prefix}_error_heatmap_{idx}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_lcfs_comparison(R, Z, pred, target, mask, idx, output_dir, title_prefix="psi_total"):
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    
    ax.pcolormesh(R, Z, target, cmap="viridis", alpha=0.3, shading="gouraud")
    
    ax.contour(R, Z, target, levels=[0.0], colors="blue", linewidths=2.0, label="Ground Truth LCFS")
    ax.contour(R, Z, pred, levels=[0.0], colors="red", linewidths=2.0, linestyles="--", label="Predicted LCFS")
    
    ax.contour(R, Z, mask, levels=[0.5], colors="black", linewidths=1.0, linestyles=":", label="Plasma Boundary")
    
    ax.set_title(f"{title_prefix} LCFS Comparison (Sample {idx})")
    ax.set_xlabel("R (m)")
    ax.set_ylabel("Z (m)")
    ax.axis("equal")
    ax.legend()
    
    fig.savefig(output_dir / f"{title_prefix}_lcfs_comparison_{idx}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_plasma_zoom(R, Z, pred, target, mask, idx, output_dir, title_prefix="psi_total"):
    plasma_mask = mask > 0.5
    if plasma_mask.any():
        r_min, r_max = R[plasma_mask].min(), R[plasma_mask].max()
        z_min, z_max = Z[plasma_mask].min(), Z[plasma_mask].max()
        
        padding = 0.1
        r_min -= padding
        r_max += padding
        z_min -= padding
        z_max += padding
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    
    im0 = axes[0].pcolormesh(R, Z, target, cmap="viridis", shading="gouraud")
    axes[0].contour(R, Z, target, levels=[0.0], colors="white", linewidths=1.5)
    axes[0].set_xlim(r_min, r_max)
    axes[0].set_ylim(z_min, z_max)
    axes[0].set_title(f"Ground Truth {title_prefix} (Zoomed)")
    axes[0].set_xlabel("R (m)")
    axes[0].set_ylabel("Z (m)")
    axes[0].axis("equal")
    
    im1 = axes[1].pcolormesh(R, Z, pred, cmap="viridis", shading="gouraud")
    axes[1].contour(R, Z, pred, levels=[0.0], colors="white", linewidths=1.5)
    axes[1].set_xlim(r_min, r_max)
    axes[1].set_ylim(z_min, z_max)
    axes[1].set_title(f"Predicted {title_prefix} (Zoomed)")
    axes[1].set_xlabel("R (m)")
    axes[1].axis("equal")
    
    fig.colorbar(im0, ax=axes[0], label=f"ψ_{title_prefix} (Wb/m)")
    fig.colorbar(im1, ax=axes[1], label=f"ψ_{title_prefix} (Wb/m)")
    
    fig.savefig(output_dir / f"{title_prefix}_plasma_zoom_{idx}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, help="Path to test_predictions.pt.")
    parser.add_argument("--data", default="data/gs_free_boundary.npz", help="Free-boundary dataset path.")
    parser.add_argument("--num-samples", type=int, default=5, help="Number of samples to visualize.")
    parser.add_argument("--output-dir", help="Output directory (default: same as predictions).")
    args = parser.parse_args()

    preds_data = torch.load(args.predictions, map_location="cpu", weights_only=False)
    
    if args.output_dir is None:
        output_dir = Path(args.predictions).parent
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = np.load(args.data)
    
    indices = preds_data["indices"]
    n_vis = min(args.num_samples, len(indices))
    
    predict_plasma = "targets_plasma" in preds_data and "preds_total" in preds_data
    
    print(f"\n{'='*60}")
    print(f"  Visualizing {n_vis} test samples")
    print(f"  Predict plasma mode: {predict_plasma}")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}")

    for i in range(n_vis):
        idx = int(indices[i])
        
        base_R = raw["R"][idx]
        base_Z = raw["Z"][idx]
        
        if predict_plasma:
            pred_plasma = preds_data["preds"][i]
            nr, nz = pred_plasma.shape
        else:
            pred = preds_data["preds"][i]
            nr, nz = pred.shape
        
        rr = np.linspace(base_R[:, 0].min(), base_R[:, 0].max(), nr)
        zz = np.linspace(base_Z[0, :].min(), base_Z[0, :].max(), nz)
        R, Z = np.meshgrid(rr, zz, indexing='ij')
        
        mask = preds_data["masks"][i]
        interior_mask = preds_data.get("interior_masks", None)
        if interior_mask is not None:
            interior_mask = interior_mask[i]
        
        print(f"  Sample {idx}...")
        
        if interior_mask is not None:
            if predict_plasma:
                pred_plasma = pred_plasma * interior_mask
                pred_total = preds_data["preds_total"][i] * interior_mask
            else:
                pred = pred * interior_mask
        
        if predict_plasma:
            target_plasma = preds_data["targets_plasma"][i]
            pred_total = preds_data["preds_total"][i] if interior_mask is None else pred_total
            target_total = preds_data["targets_total"][i]
            
            if interior_mask is not None:
                pred_plasma = pred_plasma * interior_mask
                pred_total = pred_total * interior_mask
            
            plot_psi_comparison(R, Z, pred_plasma, target_plasma, mask, idx, output_dir, title_prefix="psi_plasma")
            plot_error_heatmap(R, Z, pred_plasma, target_plasma, mask, idx, output_dir, title_prefix="psi_plasma")
            plot_lcfs_comparison(R, Z, pred_plasma, target_plasma, mask, idx, output_dir, title_prefix="psi_plasma")
            plot_plasma_zoom(R, Z, pred_plasma, target_plasma, mask, idx, output_dir, title_prefix="psi_plasma")
            
            plot_psi_comparison(R, Z, pred_total, target_total, mask, idx, output_dir, title_prefix="psi_total")
            plot_error_heatmap(R, Z, pred_total, target_total, mask, idx, output_dir, title_prefix="psi_total")
            plot_lcfs_comparison(R, Z, pred_total, target_total, mask, idx, output_dir, title_prefix="psi_total")
            plot_plasma_zoom(R, Z, pred_total, target_total, mask, idx, output_dir, title_prefix="psi_total")
        else:
            pred = preds_data["preds"][i]
            if interior_mask is not None:
                pred = pred * interior_mask
            target = preds_data["targets_total"][i]
            
            plot_psi_comparison(R, Z, pred, target, mask, idx, output_dir)
            plot_error_heatmap(R, Z, pred, target, mask, idx, output_dir)
            plot_lcfs_comparison(R, Z, pred, target, mask, idx, output_dir)
            plot_plasma_zoom(R, Z, pred, target, mask, idx, output_dir)

    print(f"\n  Visualization complete.")


if __name__ == "__main__":
    main()
