"""Training CLI for the free-boundary GS PINO surrogate.

Following PlaNet-equil architecture:
- Input: measures (scalar parameters), R (grid), Z (grid)
- Output: psi_total (total poloidal flux) over entire computational domain

The network predicts psi_total directly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import trange

from .data_freegs import FreeBndDataset, split_indices, build_ufno_input
from .losses_freegs import PlaNetLoss, compute_grad_shafranov_kernels, global_mse
from .models import PlaNetCore, UFNO2d_v2


def _stack_metadata(meta_list: list[dict]) -> dict[str, torch.Tensor]:
    keys = meta_list[0].keys()
    result = {}
    for k in keys:
        vals = [m[k] for m in meta_list]
        if isinstance(vals[0], torch.Tensor):
            result[k] = torch.stack(vals)
        else:
            result[k] = torch.tensor(vals, dtype=torch.float32)
    return result


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
        torch.stack(rhs),
        list(meta),
    )


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    opt: torch.optim.Optimizer | None,
    device: torch.device,
    loss_module: PlaNetLoss,
    clip_grad: float = 0.0,
    amp: bool = False,
    accum_steps: int = 1,
    pde_scale: float = 1.0,
    axis_scale: float = 1.0,
    ip_scale: float = 1.0,
    curvature_scale: float = 1.0,
    model_type: str = "planet",
) -> dict[str, float]:
    train = opt is not None
    model.train(train)
    total_mse = 0.0
    total_pde = 0.0
    total_axis = 0.0
    total_ip = 0.0
    total_curvature = 0.0
    total_normalized_error = 0.0
    n_samples = 0

    scaler = torch.amp.GradScaler(device.type, enabled=(train and amp))
    ctx = torch.amp.autocast(device.type, enabled=amp)

    if train:
        opt.zero_grad()

    with torch.set_grad_enabled(train):
        for step, (measures, R, Z, psi_total, mask, psi_plasma, psi_coils, rhs, meta_list) in enumerate(loader):
            measures = measures.to(device)
            R = R.to(device)
            Z = Z.to(device)
            psi_total = psi_total.to(device)
            mask = mask.to(device)
            psi_coils = psi_coils.to(device)
            rhs = rhs.to(device)

            meta = _stack_metadata(meta_list)
            meta = {k: v.to(device) for k, v in meta.items()}

            L_ker, Df_ker = compute_grad_shafranov_kernels(R, Z)

            with ctx:
                if model_type == "planet":
                    pred = model((measures, R, Z))
                else:
                    ufno_input = build_ufno_input(measures, R, Z, psi_coils)
                    pred = model(ufno_input).squeeze(1)

                original_scale_pde = loss_module.scale_pde
                original_scale_axis = loss_module.scale_axis
                original_scale_ip = loss_module.scale_ip
                original_scale_curvature = loss_module.scale_curvature
                
                loss_module.scale_pde = original_scale_pde * pde_scale
                loss_module.scale_axis = original_scale_axis * axis_scale
                loss_module.scale_ip = original_scale_ip * ip_scale
                loss_module.scale_curvature = original_scale_curvature * curvature_scale

                loss = loss_module(
                    pred=pred,
                    target=psi_total,
                    rhs=rhs,
                    Laplace_kernel=L_ker,
                    Df_dr_kernel=Df_ker,
                    RR=R,
                    ZZ=Z,
                    psi_coils=psi_coils,
                    mask=mask,
                    meta=meta,
                )

                loss_module.scale_pde = original_scale_pde
                loss_module.scale_axis = original_scale_axis
                loss_module.scale_ip = original_scale_ip
                loss_module.scale_curvature = original_scale_curvature

            if train:
                if amp:
                    scaler.scale(loss / accum_steps).backward()
                else:
                    (loss / accum_steps).backward()

                if (step + 1) % accum_steps == 0:
                    if amp:
                        if clip_grad > 0:
                            scaler.unscale_(opt)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                        scaler.step(opt)
                        scaler.update()
                    else:
                        if clip_grad > 0:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                        opt.step()
                    opt.zero_grad()

            batch_size = measures.shape[0]
            total_mse += float(loss_module.log_dict.get("mse_loss", 0)) * batch_size
            total_pde += float(loss_module.log_dict.get("pde_loss", 0)) * batch_size
            total_axis += float(loss_module.log_dict.get("axis_loss", 0)) * batch_size
            total_ip += float(loss_module.log_dict.get("ip_loss", 0)) * batch_size
            total_curvature += float(loss_module.log_dict.get("curvature_loss", 0)) * batch_size
            
            target_max = psi_total.max(dim=1)[0].max(dim=1)[0]
            target_min = psi_total.min(dim=1)[0].min(dim=1)[0]
            target_range = target_max - target_min
            target_range = target_range.clamp(min=1e-6).mean().item()
            normalized_error = float(loss_module.log_dict.get("mse_loss", 0)) ** 0.5 / target_range
            total_normalized_error += normalized_error * batch_size
            
            n_samples += batch_size

    mse_val = total_mse / n_samples
    return {
        "rmse": mse_val ** 0.5,
        "pde": total_pde / n_samples,
        "axis": total_axis / n_samples,
        "ip": total_ip / n_samples,
        "curvature": total_curvature / n_samples,
        "normalized_error": total_normalized_error / n_samples,
        "total": (total_mse + total_pde + total_axis + total_ip + total_curvature) / n_samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/freegs_merged_1500.npz", help="Free-boundary dataset path.")
    parser.add_argument("--output-dir", default="outputs/freegs_planet", help="Directory for best.pt and history.json.")
    parser.add_argument("--epochs", type=int, default=300, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=4, help="Training batch size.")
    parser.add_argument("--lr", type=float, default=5e-4, help="AdamW learning rate.")
    parser.add_argument("--model", type=str, default="planet", choices=["planet", "ufno"], help="Model architecture: planet (PlaNetCore) or ufno (U-FNO v2).")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden dimension for PlaNetCore.")
    parser.add_argument("--width", type=int, default=128, help="Width dimension for U-FNO.")
    parser.add_argument("--layers", type=int, default=6, help="Number of layers for U-FNO.")
    parser.add_argument("--modes1", type=int, default=32, help="Number of Fourier modes in R direction.")
    parser.add_argument("--modes2", type=int, default=32, help="Number of Fourier modes in Z direction.")
    parser.add_argument("--scale-mse", type=float, default=1.0, help="Weight for MSE loss.")
    parser.add_argument("--scale-pde", type=float, default=0.01, help="Weight for PDE loss.")
    parser.add_argument("--scale-axis", type=float, default=0.01, help="Weight for axis constraint loss.")
    parser.add_argument("--scale-ip", type=float, default=0.001, help="Weight for Ip constraint loss.")
    parser.add_argument("--scale-curvature", type=float, default=0.5, help="Weight for curvature (smoothness) loss.")
    parser.add_argument("--clip-grad", type=float, default=1.0, help="Gradient clipping max norm.")
    parser.add_argument("--accum-steps", type=int, default=2, help="Gradient accumulation steps.")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True, help="Enable mixed precision.")
    parser.add_argument("--warmup-epochs", type=int, default=10, help="Number of warmup epochs.")
    parser.add_argument("--patience", type=int, default=80, help="Early stopping patience.")
    parser.add_argument("--min-epochs", type=int, default=150, help="Minimum training epochs before early stopping.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    n_samples = np.load(args.data)["params"].shape[0]
    train_idx, val_idx, test_idx = split_indices(n_samples, 0.15, 0.15, args.seed)

    print(f"\n{'='*60}")
    print(f"  PlaNet-equil Architecture Training")
    print(f"{'='*60}")
    print(f"  Dataset: {n_samples} samples")
    print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    print(f"  Model: {args.model}")
    if args.model == "planet":
        print(f"    hidden_dim={args.hidden_dim}")
    else:
        print(f"    width={args.width}, layers={args.layers}, modes1={args.modes1}, modes2={args.modes2}")
    print(f"  Loss weights: mse={args.scale_mse}, pde={args.scale_pde}, axis={args.scale_axis}, ip={args.scale_ip}, curvature={args.scale_curvature}")
    print(f"{'='*60}\n")

    train_ds = FreeBndDataset(args.data, train_idx)
    val_ds = FreeBndDataset(args.data, val_idx, train_ds.param_norm)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=_collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, collate_fn=_collate)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"  Device: {device}")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = torch.device("cpu")
        print(f"  Device: {device} (CUDA not available, using CPU)")
        args.amp = False
    
    if args.model == "ufno":
        args.amp = False
        print("  AMP disabled for U-FNO (complex number support)")
    n_measures = train_ds.n_measures
    nr, nz = train_ds[0][1].shape
    print(f"  Input measures: {n_measures}, Grid size: {nr}x{nz}")
    
    if args.model == "planet":
        model = PlaNetCore(n_measures=n_measures, hidden_dim=args.hidden_dim, nr=nr, nz=nz).to(device)
    else:
        model = UFNO2d_v2(in_channels=12, modes1=args.modes1, modes2=args.modes2, width=args.width, layers=args.layers).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {total_params:,}")
    
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs - args.warmup_epochs)

    loss_module = PlaNetLoss(
        is_physics_informed=True, 
        scale_mse=args.scale_mse, 
        scale_pde=args.scale_pde,
        scale_axis=args.scale_axis,
        scale_ip=args.scale_ip,
        scale_curvature=args.scale_curvature,
    )

    print(f"  Optimizer: AdamW, lr={args.lr}, weight_decay=1e-6")
    print(f"  Scheduler: CosineAnnealingLR (after {args.warmup_epochs} epochs warmup)")
    print(f"  Gradient accumulation: {args.accum_steps} steps (effective batch = {args.batch_size * args.accum_steps})")
    if args.clip_grad > 0:
        print(f"  Gradient clipping: max_norm={args.clip_grad}")
    if args.patience > 0:
        print(f"  Early stopping: patience={args.patience}, min_epochs={args.min_epochs}")
    print()

    best = float("inf")
    history: list[dict] = []
    pbar = trange(args.epochs, desc="Training")
    patience_counter = 0

    mse_only_epochs = 100
    pde_ramp_epochs = 150
    axis_start_epoch = 80
    ip_start_epoch = 100
    curvature_start_epoch = 120

    for epoch in pbar:
        if epoch < args.warmup_epochs:
            warmup_factor = (epoch + 1) / args.warmup_epochs
            for param_group in opt.param_groups:
                param_group["lr"] = args.lr * warmup_factor
            current_lr = args.lr * warmup_factor
        else:
            current_lr = scheduler.get_last_lr()[0]

        if epoch < mse_only_epochs:
            pde_scale = 0.0
        elif epoch < mse_only_epochs + pde_ramp_epochs:
            pde_scale = (epoch - mse_only_epochs) / pde_ramp_epochs
        else:
            pde_scale = 1.0

        axis_scale = 0.0 if epoch < axis_start_epoch else 1.0
        ip_scale = 0.0 if epoch < ip_start_epoch else 1.0
        curvature_scale = 0.0 if epoch < curvature_start_epoch else 1.0

        train_losses = run_epoch(
            model, train_loader, opt, device, loss_module, 
            clip_grad=args.clip_grad, amp=args.amp, accum_steps=args.accum_steps,
            pde_scale=pde_scale, axis_scale=axis_scale, ip_scale=ip_scale,
            curvature_scale=curvature_scale,
            model_type=args.model
        )
        val_losses = run_epoch(
            model, val_loader, None, device, loss_module,
            pde_scale=pde_scale, axis_scale=axis_scale, ip_scale=ip_scale,
            curvature_scale=curvature_scale,
            model_type=args.model
        ) if len(val_ds) else train_losses

        if epoch >= args.warmup_epochs:
            scheduler.step()

        history.append({
            "epoch": epoch + 1,
            "lr": current_lr,
            "pde_scale": pde_scale,
            "axis_scale": axis_scale,
            "ip_scale": ip_scale,
            "curvature_scale": curvature_scale,
            "train_rmse": train_losses["rmse"],
            "train_pde": train_losses["pde"],
            "train_axis": train_losses["axis"],
            "train_ip": train_losses["ip"],
            "train_curvature": train_losses["curvature"],
            "train_normalized_error": train_losses["normalized_error"],
            "train_total": train_losses["total"],
            "val_rmse": val_losses["rmse"],
            "val_pde": val_losses["pde"],
            "val_axis": val_losses["axis"],
            "val_ip": val_losses["ip"],
            "val_curvature": val_losses["curvature"],
            "val_normalized_error": val_losses["normalized_error"],
            "val_total": val_losses["total"],
        })

        pbar.set_postfix(
            rmse=f"{train_losses['rmse']:.5f}",
            norm=f"{train_losses['normalized_error']:.3f}",
            pde=f"{train_losses['pde']:.6f}",
            axis=f"{train_losses['axis']:.6f}",
            ip=f"{train_losses['ip']:.6f}",
            curv=f"{train_losses['curvature']:.6f}",
            val_rmse=f"{val_losses['rmse']:.5f}",
            val_norm=f"{val_losses['normalized_error']:.3f}",
            val=f"{val_losses['total']:.6f}",
        )

        if val_losses["total"] < best:
            best = val_losses["total"]
            patience_counter = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "args": vars(args),
                    "param_mean": train_ds.param_norm.mean,
                    "param_std": train_ds.param_norm.std,
                    "n_measures": n_measures,
                    "nr": nr,
                    "nz": nz,
                    "test_indices": test_idx,
                },
                output_dir / "best.pt",
            )
        else:
            patience_counter += 1
            if args.patience > 0 and epoch >= args.min_epochs and patience_counter >= args.patience:
                print(f"\n  Early stopping triggered after {patience_counter} epochs without improvement.")
                break

    (output_dir / "history.json").write_text(json.dumps(history, indent=2))
    plot_training_history(history, output_dir)
    print(f"\n  Training complete. Best val loss: {best:.6f}")
    print(f"  Final epoch: {len(history)}")
    print(f"  Saved to: {output_dir}")


def plot_training_history(history: list[dict], output_dir: Path) -> None:
    epochs = [h["epoch"] for h in history]

    fig, axes = plt.subplots(2, 4, figsize=(24, 10), constrained_layout=True)

    ax = axes[0, 0]
    ax.plot(epochs, [h["train_total"] for h in history], "b-", label="Train")
    ax.plot(epochs, [h["val_total"] for h in history], "r--", label="Val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Total Loss")
    ax.set_title("Total Loss (Linear Scale)")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(epochs, [h["train_total"] for h in history], "b-", label="Train")
    ax.plot(epochs, [h["val_total"] for h in history], "r--", label="Val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Total Loss")
    ax.set_title("Total Loss (Log Scale)")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[0, 2]
    ax.plot(epochs, [h["train_rmse"] for h in history], "b-", label="Train")
    ax.plot(epochs, [h["val_rmse"] for h in history], "r--", label="Val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("RMSE")
    ax.set_title("Data Loss (RMSE)")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[0, 3]
    ax.plot(epochs, [h["train_pde"] for h in history], "b-", label="Train")
    ax.plot(epochs, [h["val_pde"] for h in history], "r--", label="Val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("PDE Loss")
    ax.set_title("PDE Residual Loss")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(epochs, [h["train_pde"] for h in history], "b-", label="Train")
    ax.plot(epochs, [h["val_pde"] for h in history], "r--", label="Val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("PDE Loss")
    ax.set_title("PDE Loss (Log Scale)")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.plot(epochs, [h["train_axis"] for h in history], "b-", label="Train")
    ax.plot(epochs, [h["val_axis"] for h in history], "r--", label="Val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Axis Loss")
    ax.set_title("Magnetic Axis Constraint Loss")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1, 2]
    ax.plot(epochs, [h["train_ip"] for h in history], "b-", label="Train")
    ax.plot(epochs, [h["val_ip"] for h in history], "r--", label="Val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Ip Loss")
    ax.set_title("Plasma Current Constraint Loss")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1, 3]
    ax.plot(epochs, [h.get("train_curvature", 0) for h in history], "b-", label="Train")
    ax.plot(epochs, [h.get("val_curvature", 0) for h in history], "r--", label="Val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Curvature Loss")
    ax.set_title("Curvature (Smoothness) Loss")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.savefig(output_dir / "training_curves.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()