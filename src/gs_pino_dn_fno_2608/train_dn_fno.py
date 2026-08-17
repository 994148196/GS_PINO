"""Training CLI for the arXiv:2608.05555 double-null FNO surrogate.

Paper training config (Sec. II.C):
  - loss: pure MSE on the z-scored psi field (no PDE/geometry terms)
  - optimizer: AdamW, lr=1e-3, weight_decay=1e-4
  - scheduler: ReduceLROnPlateau on validation rel L2, patience 20, factor 0.5,
    min_lr=1e-5
  - early stopping: patience 75 (validation rel L2), best weights restored
  - batch size 16, 4 dataloader workers, 3 independent init seeds per N

Usage (smoke):
  python -m gs_pino_dn_fno_2608.train_dn_fno --train-data dn_fno_2608/data/train.npz \
      --val-data dn_fno_2608/data/val.npz --n-train 500 --seed 1 \
      --epochs 30 --out-dir dn_fno_2608/outputs/smoke
Usage (full):
  python -m gs_pino_dn_fno_2608.train_dn_fno --train-data dn_fno_2608/data/train.npz \
      --val-data dn_fno_2608/data/val.npz --n-train 5000 --seed 3 \
      --out-dir dn_fno_2608/outputs/fno_n5000_s3
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from gs_pino_dn_fno_2608.data_dn_fno import DNFnoDataset, compute_stats, nested_train_indices, rel_l2_normalized
from gs_pino_dn_fno_2608.data_dn_fno_coils import DNFnoDatasetCoils
from gs_pino_dn_fno_2608.model_dn_fno import build_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    """Validation rel L2 (normalized domain, paper Eq. 7) + MSE loss."""
    model.eval()
    total_l2, total_mse, n = 0.0, 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            total_l2 += rel_l2_normalized(pred, y).item() * len(x)
            total_mse += torch.nn.functional.mse_loss(pred, y).item() * len(x)
            n += len(x)
    return total_l2 / n, total_mse / n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-data", required=True)
    ap.add_argument("--val-data", required=True)
    ap.add_argument("--n-train", type=int, default=5000,
                    help="scaling-study subset size (nested permutation, seed 12345)")
    ap.add_argument("--perm-seed", type=int, default=12345)
    ap.add_argument("--seed", type=int, default=1, help="init seed (paper: 1/2/3)")
    ap.add_argument("--epochs", type=int, default=800)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--lr-patience", type=int, default=20)
    ap.add_argument("--lr-factor", type=float, default=0.5)
    ap.add_argument("--min-lr", type=float, default=1e-5)
    ap.add_argument("--patience", type=int, default=75, help="early-stop patience (epochs)")
    ap.add_argument("--workers", type=int, default=0,
                    help="dataloader workers (paper: 4; 0 on Windows: data is in-memory "
                         "and spawn workers deadlock under CPU contention)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--input-mode", choices=["xpoints", "xa", "coils"], default="xpoints",
                    help="'xa' (data_v3): append the 2 sampled isoflux-anchor "
                         "coordinates -> 13 channels; 'coils' (data_v5 exp011): "
                         "control-coil currents instead of X-point/anchor "
                         "coordinates -> 18 channels on MAST (R, Z + 5 params + "
                         "11 coil currents); default 'xpoints' keeps 9/11-channel "
                         "behavior")
    ap.add_argument("--config-input", action="store_true",
                    help="data_v5 mixed-config training: append the 1-channel "
                         "config code (0=DN, 1=SN) to the scalars (requires "
                         "'config' in the dataset; exp008)")
    args = ap.parse_args()

    use_anchor = args.input_mode == "xa"
    use_config = args.config_input
    use_coils = args.input_mode == "coils"

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- data ----
    # --train-data/--val-data accept comma-separated file lists (data_v5
    # mixed-config: dn.npz,sn.npz); normalization stats come from the FULL
    # concatenated train pool (kept identical across all scaling-study N so
    # input representations are comparable)
    if use_coils:  # exp011: coil-current scalars, no X-point/anchor/config
        train_ds = DNFnoDatasetCoils(args.train_data, stats=None)
        stats = train_ds.stats
        train_idx = nested_train_indices(train_ds.n_full, args.n_train, args.perm_seed)
        train_ds = DNFnoDatasetCoils(args.train_data, stats=stats, indices=train_idx)
        val_ds = DNFnoDatasetCoils(args.val_data, stats=stats)
    else:
        train_ds = DNFnoDataset(args.train_data, stats=None, use_anchor=use_anchor,
                                use_config=use_config)
        stats = train_ds.stats
        train_idx = nested_train_indices(train_ds.n_full, args.n_train, args.perm_seed)

        train_ds = DNFnoDataset(args.train_data, stats=stats, indices=train_idx,
                                use_anchor=use_anchor, use_config=use_config)
        val_ds = DNFnoDataset(args.val_data, stats=stats, use_anchor=use_anchor,
                              use_config=use_config)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    # ---- model / optimizer / scheduler ----
    # input channels inferred from the scalar stats: R, Z + scalars
    # (7 scalars -> 9 channels paper baseline; 9 scalars -> 11 channels data_v2)
    model = build_model(in_channels=2 + len(stats["scalar_mean"])).to(device)
    n_params = model.count_params()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=args.lr_factor, patience=args.lr_patience,
        min_lr=args.min_lr)
    loss_fn = torch.nn.MSELoss()

    print(f"\n{'='*70}")
    print(f"  DN-FNO training | n_train={args.n_train} | seed={args.seed} | device={device}")
    print(f"  input mode: {args.input_mode} | scalars: {len(stats['scalar_mean'])} "
          f"| in_channels: {2 + len(stats['scalar_mean'])}")
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

    # restore best
    model.load_state_dict(best_state)

    print(f"\n  best val rel L2: {best_val*100:.4f}% at epoch {best_epoch} "
          f"(paper N=5000 seed3: 310)")
    print(f"  total training time: {(time.perf_counter()-t_start)/60:.1f} min")

    # ---- save ----
    torch.save({
        "model_state": best_state,
        "best_epoch": best_epoch,
        "best_val_rel_l2": best_val,
        "n_params": n_params,
        "n_train": args.n_train,
        "seed": args.seed,
        "input_mode": args.input_mode,
        "config_input": args.config_input,
        "stats": {k: (v.tolist() if hasattr(v, "tolist") else float(v))
                  for k, v in stats.items() if k not in ("input_mode", "config_input")},
    }, out_dir / "best.pt")
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    with open(out_dir / "args.json", "w") as f:
        json.dump(vars(args) | {"n_params": n_params, "device": str(device)}, f, indent=2)
    print(f"  saved -> {out_dir}/best.pt, history.json, args.json")


if __name__ == "__main__":
    main()
