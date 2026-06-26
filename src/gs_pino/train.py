"""Training CLI for the masked U-FNO/PINO fixed-boundary GS surrogate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import trange

from .data import GSDataset, split_indices
from .losses import (
    axis_constraint_loss,
    boundary_band_loss,
    gs_residual_loss,
    ip_constraint_loss,
    masked_mse,
)
from .models import UFNO2d


def _stack_metadata(meta_list: list[dict]) -> dict[str, torch.Tensor]:
    """Stack a list of per-sample metadata dicts into batched tensors."""
    keys = meta_list[0].keys()
    result = {}
    for k in keys:
        vals = [m[k] for m in meta_list]
        if isinstance(vals[0], torch.Tensor):
            result[k] = torch.stack(vals)
        else:
            result[k] = torch.tensor(vals, dtype=torch.float32)
    return result


def run_epoch(
    model: UFNO2d,
    loader: DataLoader,
    opt: torch.optim.Optimizer | None,
    device: torch.device,
    pde_weight: float,
    bc_weight: float,
    ip_weight: float = 0.0,
    betap_weight: float = 0.0,
    clip_grad: float = 0.0,
    axis_weight: float = 0.0,
    amp: bool = False,
    accum_steps: int = 1,
) -> dict[str, float]:
    """Run one train or validation epoch and return average losses for each component."""
    train = opt is not None
    model.train(train)
    total_data = 0.0
    total_bc = 0.0
    total_pde = 0.0
    total_ip = 0.0
    total_axis = 0.0
    n_samples = 0

    scaler = torch.amp.GradScaler(device.type, enabled=(train and amp))
    ctx = torch.amp.autocast(device.type, enabled=amp)

    # Gradient accumulation state
    if train:
        opt.zero_grad()

    with torch.set_grad_enabled(train):
        for step, (x, y, mask, sdf, params, meta_list) in enumerate(loader):
            x = x.to(device)
            y = y.to(device)
            mask = mask.to(device)
            sdf = sdf.to(device)

            # Stack per-sample metadata into batched tensors
            meta = _stack_metadata(meta_list)
            meta = {k: v.to(device) for k, v in meta.items()}

            with ctx:
                # Predict normalized flux over the whole rectangular grid.
                pred = model(x)

                # ---- 监督损失 ----
                loss_data = masked_mse(pred, y, mask)

                # ---- 边界条件 ----
                loss_bc = bc_weight * boundary_band_loss(pred, sdf)

                # ---- PDE 残差 (真实 GS 方程) ----
                loss_pde = torch.tensor(0.0, device=device)
                if pde_weight > 0:
                    loss_pde = pde_weight * gs_residual_loss(
                        pred,
                        R=meta["R"],
                        Z=meta["Z"],
                        mask=mask,
                        L=meta["profile_params"][:, 0],
                        Beta0=meta["profile_params"][:, 1],
                        R0=meta["R0"],
                        alpha_m=meta["alpha_m"],
                        alpha_n=meta["alpha_n"],
                        psi_axis=meta["psi_axis"],
                        psi_lcfs=meta["psi_lcfs"],
                    )

                # ---- Ip 积分约束 (可选) ----
                loss_ip = torch.tensor(0.0, device=device)
                if ip_weight > 0:
                    loss_ip = ip_weight * ip_constraint_loss(
                        pred,
                        R=meta["R"],
                        Z=meta["Z"],
                        mask=mask,
                        L=meta["profile_params"][:, 0],
                        Beta0=meta["profile_params"][:, 1],
                        R0=meta["R0"],
                        alpha_m=meta["alpha_m"],
                        alpha_n=meta["alpha_n"],
                        Ip_target=params[:, 4].to(device),
                    )

                # ---- 磁轴约束 (psi_bar(R_axis, Z_axis) = 1) ----
                loss_axis = torch.tensor(0.0, device=device)
                if axis_weight > 0:
                    loss_axis = axis_weight * axis_constraint_loss(
                        pred, R=meta["R"], Z=meta["Z"],
                        R_axis=meta["R_axis"], Z_axis=meta["Z_axis"],
                    )

                loss = loss_data + loss_bc + loss_pde + loss_ip + loss_axis

            if train:
                # 梯度累积：除 accum_steps 使有效 batch 大小 = batch_size × accum_steps
                (loss / accum_steps).backward()

                # 每 accum_steps 步更新一次参数
                if (step + 1) % accum_steps == 0:
                    if clip_grad > 0:
                        scaler.unscale_(opt)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad()

            batch_size = x.shape[0]
            total_data += float(loss_data.detach()) * batch_size
            total_bc += float(loss_bc.detach()) * batch_size
            total_pde += float(loss_pde.detach()) * batch_size
            total_ip += float(loss_ip.detach()) * batch_size
            total_axis += float(loss_axis.detach()) * batch_size
            n_samples += batch_size

    return {
        "data": total_data / n_samples,
        "bc": total_bc / n_samples,
        "pde": total_pde / n_samples,
        "ip": total_ip / n_samples,
        "axis": total_axis / n_samples,
        "total": (total_data + total_bc + total_pde + total_ip + total_axis) / n_samples,
    }


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
    parser.add_argument("--pde-weight", type=float, default=0.01, help="Weight for GS PDE residual loss.")
    parser.add_argument("--bc-weight", type=float, default=0.05, help="Weight for LCFS boundary-band loss.")
    parser.add_argument("--ip-weight", type=float, default=0.0, help="Weight for Ip integral constraint (optional).")
    parser.add_argument("--axis-weight", type=float, default=0.01, help="Weight for magnetic axis constraint.")
    parser.add_argument("--clip-grad", type=float, default=1.0, help="Gradient clipping max norm (0 = disable).")
    parser.add_argument("--accum-steps", type=int, default=2, help="Gradient accumulation steps (effective batch = batch_size × accum_steps).")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False, help="Enable mixed precision training (requires power-of-2 grid sizes).")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for split/model initialization.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    n_samples = np.load(args.data)["params"].shape[0]
    train_idx, val_idx, test_idx = split_indices(n_samples, 0.15, 0.15, args.seed)

    print(f"\n{'='*60}")
    print(f"  Dataset: {n_samples} samples")
    print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    print(f"  Model: width={args.width}, modes={args.modes1}/{args.modes2}, layers={args.layers}")
    print(f"  Loss weights: data=1.0, bc={args.bc_weight}, pde={args.pde_weight}, ip={args.ip_weight}, axis={args.axis_weight}")
    print(f"  Mixed precision: {args.amp}")
    print(f"{'='*60}\n")

    train_ds = GSDataset(args.data, train_idx)
    val_ds = GSDataset(args.data, val_idx, train_ds.param_norm)

    # DataLoader must return per-sample metadata (not a single collated tensor).
    # We use a custom collate that preserves the metadata list-of-dicts.
    def _collate(batch):
        x, y, mask, sdf, params, meta = zip(*batch)
        return (
            torch.stack([torch.as_tensor(item) for item in x]),
            torch.stack([torch.as_tensor(item) for item in y]),
            torch.stack([torch.as_tensor(item) for item in mask]),
            torch.stack([torch.as_tensor(item) for item in sdf]),
            torch.stack([torch.as_tensor(item) for item in params]),
            list(meta),  # keep as list of dicts
        )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=_collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, collate_fn=_collate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    in_channels = train_ds[0][0].shape[0]
    model = UFNO2d(in_channels, args.modes1, args.modes2, args.width, args.layers).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    print(f"  Optimizer: AdamW, lr={args.lr}, weight_decay=1e-6")
    print(f"  Scheduler: CosineAnnealingLR, T_max={args.epochs}")
    print(f"  Gradient accumulation: {args.accum_steps} steps (effective batch = {args.batch_size * args.accum_steps})")
    if args.clip_grad > 0:
        print(f"  Gradient clipping: max_norm={args.clip_grad}")
    print()

    best = float("inf")
    history: list[dict] = []
    pbar = trange(args.epochs, desc="Training")
    for epoch in pbar:
        train_losses = run_epoch(model, train_loader, opt, device, args.pde_weight, args.bc_weight, args.ip_weight, clip_grad=args.clip_grad, axis_weight=args.axis_weight, amp=args.amp, accum_steps=args.accum_steps)
        val_losses = run_epoch(model, val_loader, None, device, args.pde_weight, args.bc_weight, args.ip_weight, axis_weight=args.axis_weight) if len(val_ds) else train_losses

        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        # 记录历史
        history.append({
            "epoch": epoch + 1,
            "lr": current_lr,
            "train_data": train_losses["data"],
            "train_bc": train_losses["bc"],
            "train_pde": train_losses["pde"],
            "train_ip": train_losses["ip"],
            "train_axis": train_losses["axis"],
            "train_total": train_losses["total"],
            "val_data": val_losses["data"],
            "val_bc": val_losses["bc"],
            "val_pde": val_losses["pde"],
            "val_ip": val_losses["ip"],
            "val_axis": val_losses["axis"],
            "val_total": val_losses["total"],
        })

        # 更新进度条显示
        pbar.set_postfix(
            data=f"{train_losses['data']:.4f}",
            bc=f"{train_losses['bc']:.4f}",
            pde=f"{train_losses['pde']:.4f}",
            val=f"{val_losses['total']:.4f}",
        )

        if val_losses["total"] < best:
            best = val_losses["total"]
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

    # 保存历史
    (output_dir / "history.json").write_text(json.dumps(history, indent=2))

    # 生成训练曲线图
    plot_training_history(history, output_dir)
    print(f"\n  Training complete. Best val loss: {best:.6f}")
    print(f"  Saved to: {output_dir}")


def plot_training_history(history: list[dict], output_dir: Path) -> None:
    """Generate training loss curves plot."""
    epochs = [h["epoch"] for h in history]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)

    # 总损失
    ax = axes[0, 0]
    ax.plot(epochs, [h["train_total"] for h in history], "b-", label="Train")
    ax.plot(epochs, [h["val_total"] for h in history], "r--", label="Val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Total Loss")
    ax.set_title("Total Loss")
    ax.legend()
    ax.grid(alpha=0.3)

    # 数据损失
    ax = axes[0, 1]
    ax.plot(epochs, [h["train_data"] for h in history], "b-", label="Train")
    ax.plot(epochs, [h["val_data"] for h in history], "r--", label="Val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Data Loss (MSE)")
    ax.set_title("Data Loss (masked MSE)")
    ax.legend()
    ax.grid(alpha=0.3)

    # PDE 残差损失
    ax = axes[0, 2]
    ax.plot(epochs, [h["train_pde"] for h in history], "b-", label="Train")
    ax.plot(epochs, [h["val_pde"] for h in history], "r--", label="Val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("PDE Loss")
    ax.set_title("PDE Residual Loss (GS equation)")
    ax.legend()
    ax.grid(alpha=0.3)

    # 边界损失
    ax = axes[1, 0]
    ax.plot(epochs, [h["train_bc"] for h in history], "b-", label="Train")
    ax.plot(epochs, [h["val_bc"] for h in history], "r--", label="Val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("BC Loss")
    ax.set_title("Boundary Condition Loss")
    ax.legend()
    ax.grid(alpha=0.3)

    # 学习率
    ax = axes[1, 1]
    ax.plot(epochs, [h.get("lr", 1e-3) for h in history], "g-")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate (Cosine Annealing)")
    ax.grid(alpha=0.3)

    # 右侧空白
    axes[1, 2].axis("off")

    fig.savefig(output_dir / "training_curves.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
