from __future__ import annotations

import argparse, json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import trange

from .data import GSDataset, split_indices
from .losses import boundary_band_loss, gs_residual_loss, masked_mse
from .models import UFNO2d


def run_epoch(model, loader, opt, device, pde_weight, bc_weight):
    train = opt is not None
    model.train(train)
    total = 0.0
    with torch.set_grad_enabled(train):
        for x, y, mask, sdf, _ in loader:
            x, y, mask, sdf = x.to(device), y.to(device), mask.to(device), sdf.to(device)
            pred = model(x)
            loss_data = masked_mse(pred, y, mask)
            interior = (mask * (sdf < -0.04).float())
            loss = loss_data + bc_weight * boundary_band_loss(pred, sdf) + pde_weight * gs_residual_loss(pred, x[:, 0:1], interior)
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
            total += float(loss.detach()) * x.shape[0]
    return total / len(loader.dataset)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/gs_fixed_boundary.npz")
    p.add_argument("--output-dir", default="outputs/run")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--modes1", type=int, default=16)
    p.add_argument("--modes2", type=int, default=16)
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--pde-weight", type=float, default=0.01)
    p.add_argument("--bc-weight", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    n = np.load(args.data)["params"].shape[0]
    tr, va, te = split_indices(n, 0.15, 0.15, args.seed)
    train_ds = GSDataset(args.data, tr)
    val_ds = GSDataset(args.data, va, train_ds.param_norm)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_channels = train_ds[0][0].shape[0]
    model = UFNO2d(in_channels, args.modes1, args.modes2, args.width, args.layers).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-6)
    best = float("inf")
    history = []
    for epoch in trange(args.epochs, desc="Training"):
        tr_loss = run_epoch(model, train_loader, opt, device, args.pde_weight, args.bc_weight)
        va_loss = run_epoch(model, val_loader, None, device, args.pde_weight, args.bc_weight) if len(val_ds) else tr_loss
        history.append({"epoch": epoch + 1, "train_loss": tr_loss, "val_loss": va_loss})
        if va_loss < best:
            best = va_loss
            torch.save({"model": model.state_dict(), "args": vars(args), "param_mean": train_ds.param_norm.mean, "param_std": train_ds.param_norm.std, "test_indices": te}, out / "best.pt")
    (out / "history.json").write_text(json.dumps(history, indent=2))

if __name__ == "__main__":
    main()
