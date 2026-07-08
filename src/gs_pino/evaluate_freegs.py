"""Evaluation CLI for the free-boundary GS PINO surrogate.

Following PlaNet-equil architecture:
- Input: measures (scalar parameters), R (grid), Z (grid)
- Output: psi_total (total poloidal flux) over entire computational domain

Computes test-set metrics including:
  - Global MSE (entire domain)
  - Plasma-region MSE
  - Relative L2 error (mean/median/P95/max)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data_freegs import FreeBndDataset
from .losses_freegs import global_mse, plasma_mse
from .models import PlaNetCore


def compute_relative_l2(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> dict[str, float]:
    diff = pred - target
    
    if mask is not None:
        diff = diff * mask
        norm = (target * mask).norm(dim=(-2, -1))
    else:
        norm = target.norm(dim=(-2, -1))
    
    rel_l2 = (diff.norm(dim=(-2, -1)) / (norm + 1e-10)).cpu().numpy()
    
    return {
        "mean": float(np.mean(rel_l2)),
        "median": float(np.median(rel_l2)),
        "p95": float(np.percentile(rel_l2, 95)),
        "max": float(np.max(rel_l2)),
    }


def compute_axis_error(pred: torch.Tensor, meta_list: list[dict]) -> dict[str, float]:
    axis_errors = []
    for i, meta in enumerate(meta_list):
        R_axis = meta["R_axis"]
        Z_axis = meta["Z_axis"]
        psi_axis_true = meta["psi_axis"]
        
        R = meta["R"].numpy()
        Z = meta["Z"].numpy()
        
        idx_r = np.argmin(np.abs(R[:, 0] - R_axis))
        idx_z = np.argmin(np.abs(Z[0, :] - Z_axis))
        
        psi_axis_pred = float(pred[i, idx_r, idx_z])
        axis_errors.append(np.abs(psi_axis_pred - psi_axis_true) / np.abs(psi_axis_true))
    
    return {
        "mean": float(np.mean(axis_errors)),
        "median": float(np.median(axis_errors)),
        "p95": float(np.percentile(axis_errors, 95)),
        "max": float(np.max(axis_errors)),
    }


def compute_ip_error(pred: torch.Tensor, targets: torch.Tensor, meta_list: list[dict]) -> dict[str, float]:
    ip_errors = []
    for i, meta in enumerate(meta_list):
        psi_bndry = meta["psi_bndry"]
        psi_axis = meta["psi_axis"]
        L = meta["L"]
        
        dpsi_dR_pred = torch.gradient(pred[i], dim=0)[0]
        dpsi_dZ_pred = torch.gradient(pred[i], dim=1)[0]
        
        bphi_sq_pred = (dpsi_dR_pred**2 + dpsi_dZ_pred**2) / (2 * np.pi)
        
        psi_plasma_norm_pred = (pred[i] - psi_bndry) / (psi_axis - psi_bndry)
        Ip_pred = float(torch.sum(bphi_sq_pred * psi_plasma_norm_pred))
        
        Ip_true = float(meta["Ip"])
        
        ip_errors.append(np.abs(Ip_pred - Ip_true) / np.abs(Ip_true))
    
    return {
        "mean": float(np.mean(ip_errors)),
        "median": float(np.median(ip_errors)),
        "p95": float(np.percentile(ip_errors, 95)),
        "max": float(np.max(ip_errors)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to best.pt checkpoint.")
    parser.add_argument("--data", default="data/gs_free_boundary.npz", help="Free-boundary dataset path.")
    parser.add_argument("--batch-size", type=int, default=8, help="Evaluation batch size.")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    params = checkpoint["args"]

    print(f"\n{'='*60}")
    print(f"  PlaNet-equil Architecture Evaluation")
    print(f"{'='*60}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Dataset: {args.data}")
    print(f"  Model: hidden_dim={params['hidden_dim']}")
    print(f"{'='*60}")

    test_idx = checkpoint["test_indices"]
    param_norm = type("obj", (), {
        "mean": checkpoint["param_mean"],
        "std": checkpoint["param_std"],
        "apply": lambda self, x: (x - self.mean) / (self.std + 1e-7),
    })()
    param_norm.__class__.__name__ = "obj"

    test_ds = FreeBndDataset(args.data, test_idx, param_norm)
    print(f"  Test samples: {len(test_ds)}")

    def _collate(batch):
        measures, R, Z, psi_total, mask, psi_plasma, psi_coils, rhs, meta = zip(*batch)
        return (
            torch.stack(measures),
            torch.stack(R),
            torch.stack(Z),
            torch.stack(psi_total),
            torch.stack(mask),
            torch.stack(psi_plasma),
            torch.stack(psi_coils),
            list(meta),
        )

    test_loader = DataLoader(test_ds, batch_size=args.batch_size, collate_fn=_collate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    n_measures = checkpoint["n_measures"]
    nr = checkpoint["nr"]
    nz = checkpoint["nz"]
    model = PlaNetCore(n_measures=n_measures, hidden_dim=params["hidden_dim"], nr=nr, nz=nz).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    all_preds = []
    all_targets = []
    all_masks = []
    
    total_global_mse = 0.0
    total_plasma_mse = 0.0
    n_samples = 0

    with torch.no_grad():
        for measures, R, Z, psi_total, mask, psi_plasma, psi_coils, meta_list in tqdm(test_loader, desc="Evaluating"):
            measures = measures.to(device)
            R = R.to(device)
            Z = Z.to(device)
            psi_total = psi_total.to(device)
            mask = mask.to(device)

            pred = model((measures, R, Z))

            all_preds.append(pred.cpu())
            all_targets.append(psi_total.cpu())
            all_masks.append(mask.cpu())

            loss_global = global_mse(pred, psi_total)
            loss_plasma = plasma_mse(pred, psi_total, mask)
            
            batch_size = measures.shape[0]
            total_global_mse += float(loss_global) * batch_size
            total_plasma_mse += float(loss_plasma) * batch_size
            n_samples += batch_size

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    all_masks = torch.cat(all_masks)

    global_rel_l2 = compute_relative_l2(all_preds, all_targets)
    plasma_rel_l2 = compute_relative_l2(all_preds, all_targets, all_masks)

    print(f"\n{'='*60}")
    print(f"  Test Set Metrics")
    print(f"{'='*60}")
    print(f"\n  [Data Loss]")
    print(f"    Global MSE: {total_global_mse / n_samples:.6f}")
    print(f"    Plasma MSE: {total_plasma_mse / n_samples:.6f}")
    
    print(f"\n  [Relative L2 Error - Global]")
    print(f"    Mean: {global_rel_l2['mean']:.4f}")
    print(f"    Median: {global_rel_l2['median']:.4f}")
    print(f"    P95: {global_rel_l2['p95']:.4f}")
    print(f"    Max: {global_rel_l2['max']:.4f}")
    
    print(f"\n  [Relative L2 Error - Plasma Region]")
    print(f"    Mean: {plasma_rel_l2['mean']:.4f}")
    print(f"    Median: {plasma_rel_l2['median']:.4f}")
    print(f"    P95: {plasma_rel_l2['p95']:.4f}")
    print(f"    Max: {plasma_rel_l2['max']:.4f}")

    output_dir = Path(args.checkpoint).parent
    metrics = {
        "global_mse": total_global_mse / n_samples,
        "plasma_mse": total_plasma_mse / n_samples,
        "global_rel_l2": global_rel_l2,
        "plasma_rel_l2": plasma_rel_l2,
        "n_samples": n_samples,
    }

    (output_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\n  Metrics saved to: {output_dir / 'test_metrics.json'}")

    torch.save({
        "preds": all_preds.numpy(),
        "targets": all_targets.numpy(),
        "masks": all_masks.numpy(),
        "indices": test_idx,
    }, output_dir / "test_predictions.pt")
    print(f"  Predictions saved to: {output_dir / 'test_predictions.pt'}")


if __name__ == "__main__":
    main()