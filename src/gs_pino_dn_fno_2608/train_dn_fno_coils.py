"""Training CLI for the coil-current-input DN FNO experiment (exp001).

Replaces the paper's 4 X-point coordinate channels with the 4 control-coil
currents (P1L/P1U/P2L/P2U). Everything else is identical to the baseline
(train_dn_fno.py): pure MSE, AdamW(1e-3, 1e-4), ReduceLROnPlateau(20, 0.5,
min 1e-5), early stop 75, batch 16, 800 epochs, nested train subsets.

The baseline training module is NOT modified; shared helpers (set_seed /
evaluate) are imported from it, the dataset comes from data_dn_fno_coils.

Usage (smoke):
  python -m gs_pino_dn_fno_2608.train_dn_fno_coils --train-data dn_fno_2608/data/train.npz \
      --val-data dn_fno_2608/data/val.npz --n-train 100 --seed 1 \
      --epochs 30 --out-dir dn_fno_2608/experiments/exp001_coil_input/smoke
Usage (full):
  python -m gs_pino_dn_fno_2608.train_dn_fno_coils --train-data dn_fno_2608/data/train.npz \
      --val-data dn_fno_2608/data/val.npz --n-train 500 --seed 1 \
      --out-dir dn_fno_2608/experiments/exp001_coil_input
Usage (data_v2, 11 channels):
  python -m gs_pino_dn_fno_2608.train_dn_fno_coils --train-data dn_fno_2608/data_v2/train.npz \
      --val-data dn_fno_2608/data_v2/val.npz --n-train 500 --seed 1 \
      --out-dir dn_fno_2608/experiments/exp003_coil_input_v2
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from gs_pino_dn_fno_2608.data_dn_fno_coils import DNFnoDatasetCoils, compute_stats_coils
from gs_pino_dn_fno_2608.data_dn_fno import nested_train_indices
from gs_pino_dn_fno_2608.train_dn_fno import set_seed, evaluate
from gs_pino_dn_fno_2608.model_dn_fno import build_model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-data", required=True)
    ap.add_argument("--val-data", required=True)
    ap.add_argument("--n-train", type=int, default=500)
    ap.add_argument("--perm-seed", type=int, default=12345)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=800)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--lr-patience", type=int, default=20)
    ap.add_argument("--lr-factor", type=float, default=0.5)
    ap.add_argument("--min-lr", type=float, default=1e-5)
    ap.add_argument("--patience", type=int, default=75)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- data (stats from the FULL train pool, coil currents z-scored) ----
    with np.load(args.train_data) as d:
        stats = compute_stats_coils({
            "params": d["params"], "coil_currents": d["coil_currents"],
            "psi_total": d["psi_total"]})
    train_idx = nested_train_indices(
        len(np.load(args.train_data)["psi_total"]), args.n_train, args.perm_seed)

    train_ds = DNFnoDatasetCoils(args.train_data, stats=stats, indices=train_idx)
    val_ds = DNFnoDatasetCoils(args.val_data, stats=stats)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    # ---- model / optimizer / scheduler (unchanged architecture; channels
    # inferred from stats: 9 on data/, 11 on data_v2) ----
    model = build_model(in_channels=2 + len(stats["scalar_mean"])).to(device)
    n_params = model.count_params()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=args.lr_factor, patience=args.lr_patience,
        min_lr=args.min_lr)
    loss_fn = torch.nn.MSELoss()

    n_scalars = len(stats["scalar_mean"])
    print(f"\n{'='*70}")
    print(f"  DN-FNO coil-input experiment | n_train={args.n_train} | seed={args.seed}")
    print(f"  input scalars: params({n_scalars-4}) + coil currents(4)  "
          f"[baseline: params({n_scalars-4}) + X-points(4)]")
    print(f"  model params: {n_params} (paper: 4,770,241)")
    print(f"  lr={args.lr}, wd={args.weight_decay}, batch={args.batch_size}, "
          f"lr patience={args.lr_patience} (x{args.lr_factor}, min {args.min_lr}), "
          f"early-stop patience={args.patience}")
    print(f"  train: {len(train_ds)} samples | val: {len(val_ds)} | out: {out_dir}")
    print(f"{'='*70}\n")

    history = {"epoch": [], "train_loss": [], "val_rel_l2": [], "val_mse": [], "lr": []}
    best_val, best_epoch, best_state, stale = float("inf"), -1, None, 0
    t_start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss, n_batches = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(x)
            n_batches += len(x)
        train_loss = epoch_loss / n_batches

        val_l2, val_mse = evaluate(model, val_loader, device)
        scheduler.step(val_l2)
        lr = optimizer.param_groups[0]["lr"]

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_rel_l2"].append(val_l2)
        history["val_mse"].append(val_mse)
        history["lr"].append(lr)

        if val_l2 < best_val:
            best_val, best_epoch, stale = val_l2, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1

        if epoch % 10 == 0 or stale == 0:
            print(f"  epoch {epoch:4d} | train_loss {train_loss:.4e} | "
                  f"val_rel_l2 {val_l2*100:.4f}% | val_mse {val_mse:.4e} | lr {lr:.2e}")

        if stale >= args.patience:
            print(f"\n  early stop at epoch {epoch} (no improvement for {args.patience} epochs)")
            break

    model.load_state_dict(best_state)

    print(f"\n  best val rel L2: {best_val*100:.4f}% at epoch {best_epoch} "
          f"(baseline N=500: 0.2253% @ 756)")
    print(f"  total training time: {(time.perf_counter()-t_start)/60:.1f} min")

    torch.save({
        "model_state": best_state,
        "best_epoch": best_epoch,
        "best_val_rel_l2": best_val,
        "n_params": n_params,
        "n_train": args.n_train,
        "seed": args.seed,
        "input_mode": "coils",
        "stats": {k: (v.tolist() if hasattr(v, "tolist") else v)
                  for k, v in stats.items()},
    }, out_dir / "best.pt")
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    with open(out_dir / "args.json", "w") as f:
        json.dump(vars(args) | {"n_params": n_params, "device": str(device)}, f, indent=2)
    print(f"  saved -> {out_dir}/best.pt, history.json, args.json")


if __name__ == "__main__":
    main()
