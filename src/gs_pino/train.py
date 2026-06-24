"""Training CLI for the masked U-FNO/PINO fixed-boundary GS surrogate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import trange

from .data import GSDataset, split_indices
from .losses import boundary_band_loss, gs_residual_loss, masked_mse
from .models import UFNO2d


def run_epoch(model: UFNO2d, loader: DataLoader, opt: torch.optim.Optimizer | None, device: torch.device, pde_weight: float, bc_weight: float) -> float:
    """Run one train or validation epoch and return average loss."""
    # `opt is None` means validation mode; otherwise gradients are enabled.
    train = opt is not None
    model.train(train)
    total = 0.0

    with torch.set_grad_enabled(train):
        for x, y, mask, sdf, _params in loader:
            # Move tensors to the selected accelerator/CPU.
            x = x.to(device)
            y = y.to(device)
            mask = mask.to(device)
            sdf = sdf.to(device)

            # Predict normalized flux over the whole rectangular grid.
            pred = model(x)

            # Supervised loss is restricted to physically meaningful LCFS interior.
            loss_data = masked_mse(pred, y, mask)

            # PDE proxy is evaluated away from LCFS so finite-difference stencils stay inside.
            interior = mask * (sdf < -0.04).float()
            loss = loss_data + bc_weight * boundary_band_loss(pred, sdf) + pde_weight * gs_residual_loss(pred, x[:, 0:1], interior)

            # Standard optimizer update for training epochs only.
            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()

            # Accumulate sample-weighted average for stable epoch reporting.
            total += float(loss.detach()) * x.shape[0]
    return total / len(loader.dataset)


def main() -> None:
    """Parse CLI arguments, train U-FNO, and save best checkpoint/history."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/gs_fixed_boundary.npz", help="Dataset produced by gs_pino.generate_dataset.")
    parser.add_argument("--output-dir", default="outputs/run", help="Directory for best.pt and history.json.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=4, help="Training batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="AdamW learning rate.")
    parser.add_argument("--modes1", type=int, default=16, help="Fourier modes in R dimension.")
    parser.add_argument("--modes2", type=int, default=16, help="Fourier modes in Z/rFFT dimension.")
    parser.add_argument("--width", type=int, default=32, help="Latent channel width.")
    parser.add_argument("--layers", type=int, default=4, help="Number of U-FNO blocks.")
    parser.add_argument("--pde-weight", type=float, default=0.01, help="Weight for interior elliptic regularizer.")
    parser.add_argument("--bc-weight", type=float, default=0.05, help="Weight for LCFS boundary-band loss.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for split/model initialization.")
    args = parser.parse_args()

    # Seed numpy and torch for reproducible splits and initialization.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Prepare output directory and deterministic train/val/test indices.
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    n_samples = np.load(args.data)["params"].shape[0]
    train_idx, val_idx, test_idx = split_indices(n_samples, 0.15, 0.15, args.seed)

    # Train dataset owns normalization; validation reuses the same statistics.
    train_ds = GSDataset(args.data, train_idx)
    val_ds = GSDataset(args.data, val_idx, train_ds.param_norm)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    # Build model after reading one sample so channel count stays data-driven.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_channels = train_ds[0][0].shape[0]
    model = UFNO2d(in_channels, args.modes1, args.modes2, args.width, args.layers).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-6)

    # Track validation loss and keep only the best checkpoint.
    best = float("inf")
    history: list[dict[str, float | int]] = []
    for epoch in trange(args.epochs, desc="Training"):
        train_loss = run_epoch(model, train_loader, opt, device, args.pde_weight, args.bc_weight)
        val_loss = run_epoch(model, val_loader, None, device, args.pde_weight, args.bc_weight) if len(val_ds) else train_loss
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})

        # Save all information needed to rebuild the model and test split later.
        if val_loss < best:
            best = val_loss
            torch.save(
                {
                    "model": model.state_dict(),
                    "args": vars(args),
                    "param_mean": train_ds.param_norm.mean,
                    "param_std": train_ds.param_norm.std,
                    "test_indices": test_idx,
                },
                output_dir / "best.pt",
            )

    # Save a lightweight JSON training curve for plotting/debugging.
    (output_dir / "history.json").write_text(json.dumps(history, indent=2))


if __name__ == "__main__":
    main()
