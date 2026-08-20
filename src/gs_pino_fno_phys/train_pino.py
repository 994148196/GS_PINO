"""Training CLI for physics-informed FNO (exp101/102, data_v5/dn).

Two modes (shared data/model framework, loss-configuration difference only):
  - rhs (exp101, single stage): supervised psi_plasma + PDE residual from the
    DATASET RHS  L = MSE(psi_z) + w_pde * ||Delta* psi_phys + mu0 R J_data||^2
  - twostage (exp102): stage-1 supervised psi_plasma + J (2 output channels);
    stage-2 additionally the SELF-CONSISTENT residual (network J) and the Ip
    constraint. Stage 2 starts automatically once the stage-1 validation
    rel L2 (psi_plasma) drops below --stage1-threshold (default 3%), with
    --stage1-max-epochs as a fallback.

Inherits all hyper-parameters of train_dn_fno.py (AdamW lr 1e-3 wd 1e-4,
ReduceLROnPlateau, early stop patience 75, batch 16, nested N subsets).

Usage (smoke):
  python -m gs_pino_fno_phys.train_pino --mode rhs \
      --train-data dn_fno_2608/data_v5/dn/train.npz --val-data dn_fno_2608/data_v5/dn/val.npz \
      --n-train 500 --seed 1 --epochs 30 --out-dir dn_fno_2608/experiments/smoke
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from gs_pino_dn_fno_2608.data_dn_fno import nested_train_indices, rel_l2_normalized
from gs_pino_dn_fno_2608.model_dn_fno import build_model
from gs_pino_fno_phys.data_pino import DNPinoDataset
from gs_pino_fno_phys.losses_pino import (
    denorm_j,
    denorm_psi,
    ip_constraint,
    masked_mse,
    pde_residual_rhs,
    pde_residual_self,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def erode_mask(mask: torch.Tensor, k: int) -> torch.Tensor:
    """Binary erosion by k layers (3x3 avg-pool < 1 -> boundary cell)."""
    m = mask.float()
    for _ in range(k):
        m = F.avg_pool2d(m, kernel_size=3, stride=1, padding=1)
        m = (m > 0.999).float()
    return m


def validate(model: torch.nn.Module, loader: DataLoader, device: torch.device,
             stats: dict, R_c: torch.Tensor, dR: float, dZ: float,
             dA: float, erode_k: int) -> dict[str, float]:
    """Validation: rel L2 (psi_plasma z domain) + MSE + physics components."""
    model.eval()
    acc = {"n": 0, "rel_l2": 0.0, "mse": 0.0, "l_pde": 0.0, "l_ip": 0.0}
    with torch.no_grad():
        for batch in loader:
            x, y_psi = batch["x"].to(device), batch["y_psi"].to(device)
            mask_b = batch["mask"].to(device)
            pred = model(x)
            psi_z = pred[:, 0:1]
            psi_phys = denorm_psi(psi_z, stats)
            mask_i = erode_mask(mask_b, erode_k)[..., 1:-1, 1:-1]
            b = len(x)
            acc["rel_l2"] += rel_l2_normalized(psi_z, y_psi).item() * b
            acc["mse"] += F.mse_loss(psi_z, y_psi).item() * b
            if pred.shape[1] > 1:  # twostage: physics components on val
                j_phys = denorm_j(pred[:, 1:2], stats)
                acc["l_pde"] += pde_residual_self(
                    psi_phys, j_phys, R_c, dR, dZ, mask_i,
                    stats["pde_scale"]).item() * b
                acc["l_ip"] += ip_constraint(
                    j_phys, mask_b, dA, batch["ip"].to(device),
                    stats["ip_scale"]).item() * b
            acc["n"] += b
    return {k: (v / acc["n"] if k != "n" else int(acc["n"])) for k, v in acc.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["rhs", "twostage"], default="rhs",
                    help="rhs: single-stage PDE residual from dataset RHS; "
                         "twostage: stage-1 supervised psi+J, stage-2 adds "
                         "self-consistent PDE + Ip")
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
    ap.add_argument("--patience", type=int, default=75,
                    help="early-stop patience, counted from stage 2 only")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--phys-weight", type=float, default=0.1, help="w_pde")
    ap.add_argument("--ip-weight", type=float, default=1.0, help="w_ip (twostage stage 2)")
    ap.add_argument("--j-weight", type=float, default=1.0, help="w_j (twostage)")
    ap.add_argument("--stage1-threshold", type=float, default=0.03,
                    help="stage-1 val rel L2 below which stage 2 starts")
    ap.add_argument("--stage1-max-epochs", type=int, default=300,
                    help="fallback: force stage 2 after this many epochs")
    ap.add_argument("--stage2-ramp-epochs", type=int, default=30,
                    help="warm-up: w_pde/w_ip ramp 0->target linearly over the "
                         "first N stage-2 epochs (prevents the PDE gradient "
                         "from destroying the stage-1 psi fit when the "
                         "psi<->J self-consistency is not yet learned)")
    ap.add_argument("--pde-mask-erode", type=int, default=0,
                    help="erode the PDE mask by k cells (boundary-noise mitigation)")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- data (stats from the FULL train pool, nested N subset) ----
    train_ds = DNPinoDataset(args.train_data, stats=None)
    stats = train_ds.stats
    train_idx = nested_train_indices(train_ds.n_full, args.n_train, args.perm_seed)
    train_ds = DNPinoDataset(args.train_data, stats=stats, indices=train_idx)
    val_ds = DNPinoDataset(args.val_data, stats=stats)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    # ---- model (psi_plasma 1ch / psi_plasma+J 2ch) / optimizer / scheduler ----
    out_channels = 2 if args.mode == "twostage" else 1
    model = build_model(in_channels=2 + len(stats["scalar_mean"]),
                        out_channels=out_channels).to(device)
    n_params = model.count_params()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=args.lr_factor, patience=args.lr_patience,
        min_lr=args.min_lr)

    # ---- physical grid (shared across samples) ----
    R_c = torch.tensor(train_ds.R_phys[1:-1, 1:-1], device=device)
    dR, dZ, dA = train_ds.dR, train_ds.dZ, train_ds.dR * train_ds.dZ
    pde_scale, ip_scale = stats["pde_scale"], stats["ip_scale"]
    w_pde, w_ip, w_j = args.phys_weight, args.ip_weight, args.j_weight

    print(f"\n{'='*70}")
    print(f"  FNO+physics training | mode={args.mode} | n_train={args.n_train} "
          f"| seed={args.seed} | device={device}")
    print(f"  input channels: {2 + len(stats['scalar_mean'])} | output: {out_channels}")
    print(f"  model params: {n_params} | w_pde={w_pde} w_ip={w_ip} w_j={w_j} "
          f"(pde_scale={pde_scale:.3f}, ip_scale={ip_scale:.3g})")
    if args.mode == "twostage":
        print(f"  stage-1 -> stage-2: val rel L2 < {args.stage1_threshold*100:.2f}% "
              f"or epoch >= {args.stage1_max_epochs} "
              f"(physics weights warm up over {args.stage2_ramp_epochs} epochs)")
    print(f"  train: {len(train_ds)} | val: {len(val_ds)} | out: {out_dir}")
    print(f"{'='*70}\n")

    history = {"epoch": [], "stage": [], "ramp": [], "train_loss": [], "l_psi": [],
               "l_j": [], "l_pde": [], "l_ip": [], "val_rel_l2": [], "val_mse": [],
               "lr": []}
    # phase-aware best (train_kan convention): artifact = stage-2 best when
    # it exists (it always does once stage 2 starts), else stage-1 best
    best = {"1": (float("inf"), -1, None), "2": (float("inf"), -1, None)}
    switch_epoch, stale, t_start = None, 0, time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        if args.mode == "twostage":
            stage = 1 if switch_epoch is None else 2
            # stage-2 warm-up: physics weights ramp 0 -> target (self-consistency
            # between psi and J is not learned yet at the stage boundary; a full
            # PDE weight at that point destroys the stage-1 psi fit)
            ramp = (0.0 if switch_epoch is None
                    else min(1.0, (epoch - switch_epoch)
                             / max(args.stage2_ramp_epochs, 1)))
        else:
            stage, ramp = 2, 1.0   # rhs 单阶段：无切换，物理权重全量
        model.train()
        ep_loss, n_batches = 0.0, 0
        ep_comps = {"l_psi": 0.0, "l_j": 0.0, "l_pde": 0.0, "l_ip": 0.0}
        for batch in train_loader:
            x, y_psi = batch["x"].to(device), batch["y_psi"].to(device)
            mask_b = batch["mask"].to(device)
            optimizer.zero_grad()
            pred = model(x)
            psi_z = pred[:, 0:1]
            psi_phys = denorm_psi(psi_z, stats)
            l_psi = F.mse_loss(psi_z, y_psi)
            l_j = l_pde = l_ip = torch.zeros((), device=device)
            if args.mode == "rhs":
                j_phys = batch["j_phys"].to(device)
                mask_i = erode_mask(mask_b, args.pde_mask_erode)[..., 1:-1, 1:-1]
                l_pde = pde_residual_rhs(psi_phys, j_phys, R_c, dR, dZ,
                                         mask_i, pde_scale)
                loss = l_psi + w_pde * l_pde
            else:
                j_z = pred[:, 1:2]
                l_j = masked_mse(j_z, batch["y_j"].to(device), mask_b)
                loss = l_psi + w_j * l_j
                if stage == 2:
                    j_phys = denorm_j(j_z, stats)
                    mask_i = erode_mask(mask_b, args.pde_mask_erode)[..., 1:-1, 1:-1]
                    l_pde = pde_residual_self(psi_phys, j_phys, R_c, dR, dZ,
                                              mask_i, pde_scale)
                    l_ip = ip_constraint(j_phys, mask_b, dA,
                                         batch["ip"].to(device), ip_scale)
                    loss = loss + w_pde * ramp * l_pde + w_ip * ramp * l_ip
            loss.backward()
            optimizer.step()
            ep_loss += loss.item() * len(x)
            ep_comps["l_psi"] += l_psi.item() * len(x)
            ep_comps["l_j"] += l_j.item() * len(x)
            ep_comps["l_pde"] += l_pde.item() * len(x)
            ep_comps["l_ip"] += l_ip.item() * len(x)
            n_batches += len(x)
        train_loss = ep_loss / n_batches
        comps = {k: v / n_batches for k, v in ep_comps.items()}

        val = validate(model, val_loader, device, stats, R_c, dR, dZ, dA,
                       args.pde_mask_erode)
        val_l2, val_mse = val["rel_l2"], val["mse"]
        scheduler.step(val_l2)
        lr = optimizer.param_groups[0]["lr"]

        # ---- stage switch (twostage, stage 1 only) ----
        if args.mode == "twostage" and switch_epoch is None:
            if val_l2 < args.stage1_threshold or epoch >= args.stage1_max_epochs:
                switch_epoch = epoch
                stale = 0
                print(f"  >> STAGE 2 begins at epoch {epoch} "
                      f"(val rel L2 {val_l2*100:.3f}%)")

        # ---- phase-aware best / early stop (stage 2 only) ----
        # rhs mode is single-stage: track under the stage-2 slot throughout
        key = "1" if (args.mode == "twostage" and switch_epoch is None) else "2"
        if val_l2 < best[key][0]:
            best[key] = (val_l2, epoch,
                         {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
            stale = 0
        elif switch_epoch is not None:
            stale += 1

        history["epoch"].append(epoch)
        history["stage"].append(stage)
        history["ramp"].append(ramp)
        history["train_loss"].append(train_loss)
        history["l_psi"].append(comps["l_psi"])
        history["l_j"].append(comps["l_j"])
        history["l_pde"].append(comps["l_pde"])
        history["l_ip"].append(comps["l_ip"])
        history["val_rel_l2"].append(val_l2)
        history["val_mse"].append(val_mse)
        history["lr"].append(lr)

        if epoch % 10 == 0 or (best[key][1] == epoch):
            print(f"  epoch {epoch:4d} s{stage} r{ramp:.2f} | train {train_loss:.4e} "
                  f"(psi {comps['l_psi']:.2e} j {comps['l_j']:.2e} "
                  f"pde {comps['l_pde']:.2e} ip {comps['l_ip']:.2e}) | "
                  f"val_rel_l2 {val_l2*100:.4f}% | lr {lr:.2e}")

        if switch_epoch is not None and stale >= args.patience:
            print(f"\n  early stop at epoch {epoch} "
                  f"(stage-2 no improvement for {args.patience} epochs)")
            break

    # ---- artifact: stage-2 best if it exists, else stage-1 best ----
    artifact_stage = "2" if best["2"][1] >= 0 else "1"
    best_val, best_epoch, best_state = best[artifact_stage]
    model.load_state_dict(best_state)
    print(f"\n  artifact stage {artifact_stage}: best val rel L2 {best_val*100:.4f}% "
          f"at epoch {best_epoch} | switch_epoch {switch_epoch}")
    print(f"  total time: {(time.perf_counter()-t_start)/60:.1f} min")

    torch.save({
        "model_state": best_state,
        "best_epoch": best_epoch,
        "best_val_rel_l2": best_val,
        "artifact_stage": artifact_stage,
        "switch_epoch": switch_epoch,
        "stage1_best": best["1"][0],
        "stage1_best_epoch": best["1"][1],
        "stage2_best": best["2"][0],
        "stage2_best_epoch": best["2"][1],
        "mode": args.mode,
        "n_params": n_params,
        "n_train": args.n_train,
        "seed": args.seed,
        "stats": stats,
        "args": vars(args),
    }, out_dir / "best.pt")
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    with open(out_dir / "args.json", "w") as f:
        json.dump(vars(args) | {"n_params": n_params, "device": str(device)}, f, indent=2)
    print(f"  saved -> {out_dir}/best.pt, history.json, args.json")


if __name__ == "__main__":
    main()
