"""Three-phase KAN training CLI (paper Sec. 2.1) for the 19-channel GS KAN.

Phases (epochs, paper Eq. 8-16):
  KAN-1  (t < 120):      L = L_psi + L_J                         (supervised)
  KAN-2  (120 <= t):     + lam_PDE*L_PDE + lam_Ip*L_Ip           (semi-sup.)
  KAN-3  (same window):  + lam(t)*L_reg,  lam(t): 0 (t<120),
                         0.001*(t-120)/120 (120<=t<240), 0.001 (240<=t<600)
  prune  (t = 600):      drop connections with magnitude < 1e-2 * max
  KAN-3' (600 < t):      back to Eq. 8 (no L_reg), fine-tuning with the
                         pruned connections frozen (gradients masked)

Losses (normalized outputs; physical for the physics terms):
  L_psi = MSE(psi_z, target_z),  L_J = MSE(J_z, target_z)
  L_PDE = mean over valid pts of [(Delta* psi_phys + mu0 R J_phys)/pde_scale]^2
          Delta* from the ANALYTIC B-spline derivatives (no autograd), in
          physical units via psi_std and the R/Z coordinate Jacobians.
          Valid pts: not coil cells (delta-like coil singularity), not the
          domain boundary. No explicit coil source term: the total-field
          residual Delta* psi_total + mu0 R J = 0 holds everywhere away from
          the conductors (lap*(psi_coils) ~ 0 there; see data_kan.py).
  L_Ip  = mean over batch of [(sum_k J_phys*weight - Ip)/ip_scale]^2
          weight = plasma-stratum integration weights (unbiased, data_kan.py)
  L_reg = mean|coeffs| (L1) + mean per-connection spline entropy
          H = -sum_b p_b log p_b, p_b = |c_b| / sum|c_b|

Outputs (aligned with train_dn_fno.py's checkpoint contract):
  best.pt = {model_state, best_epoch, best_val_rel_l2, n_params, n_train,
             seed, stats, prune_mask, prune_ratio, phase, history-tag}
  — the KAN-3 artifact is the post-prune best state (phase="pruned"); the
  pre-prune best is stored separately in best_pre_prune.pt (plan fix 2).
  history.json (per logged epoch: losses, lambda(t), val), args.json.

Usage (M1, mirrors exp011):
  PY="C:/Users/HP/.conda/envs/torch5060/python.exe"
  "$PY" -u -m gs_pino_kan_2608.train_kan \
    --train-data dn_fno_2608/data_v5/dn/train.npz,dn_fno_2608/data_v5/sn/train.npz \
    --val-data dn_fno_2608/data_v5/dn/val.npz,dn_fno_2608/data_v5/sn/val.npz \
    --n-train 500 --seed 1 --out-dir dn_fno_2608/kan/experiments/exp001_kan_v5/model_b19ch_kan_mix
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader  # noqa: F401  (FNO parity, unused here)

from gs_pino_dn_fno_2608.data_dn_fno import nested_train_indices
from gs_pino_kan_2608.data_kan import (MU0, PointKANDataset, compute_stats_kan,
                                       get_cfg, n_channels)
from gs_pino_kan_2608.model_kan import build_model

RMIN, RMAX = 0.1, 2.0     # physical domain (data_v5 MAST 65x65, FNO parity)
ZMIN, ZMAX = -2.0, 2.0
SR = 2.0 / (RMAX - RMIN)  # dRn/dR
SZ = 2.0 / (ZMAX - ZMIN)  # dZn/dZ


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def lam_t(t: int, reg_weight: float, b120: int = 120, b240: int = 240,
          b600: int = 600) -> float:
    """Paper Eq. 16: 0 | 0.001*(t-b120)/(b240-b120) | 0.001 | 0 (post-prune).

    Boundaries are the paper's 120/240/600 scaled linearly with --epochs
    (full 1000-epoch runs get the exact paper schedule)."""
    if t < b120:
        return 0.0
    if t < b240:
        return reg_weight * (t - b120) / (b240 - b120)
    if t < b600:
        return reg_weight
    return 0.0


def prune_phase(epoch: int, prune_epoch: int) -> bool:
    return epoch == prune_epoch


def entropy_reg(coeffs: torch.Tensor) -> torch.Tensor:
    """Mean per-connection spline-coefficient entropy (paper Eq. 14-15)."""
    p = coeffs.abs() / (coeffs.abs().sum(dim=-1, keepdim=True) + 1e-12)
    return -(p * (p + 1e-12).log()).sum(dim=-1).mean()


def compute_losses(model, b: dict, stats: dict, device: torch.device,
                   args) -> dict[str, torch.Tensor]:
    """Forward + all loss components for one batch (dict of tensors)."""
    x = b["x"].to(device)                                  # (B, k, in_dim)
    B, k, _ = x.shape
    xf = x.reshape(B * k, -1)
    out = model.forward_with_derivs(xf)
    psi_z, J_z = out["psi"][:, 0], out["psi"][:, 1]        # (B*k,)
    psi_phys = psi_z * stats["psi_std"] + stats["psi_mean"]
    J_phys = J_z * stats["j_std"] + stats["j_mean"]

    l_psi = torch.nn.functional.mse_loss(psi_z, b["y_psi"].reshape(-1).to(device))
    l_j = torch.nn.functional.mse_loss(J_z, b["y_j"].reshape(-1).to(device))

    # ---- L_PDE: Delta* psi_phys + mu0 R J_phys over valid points ----
    sR2 = SR * SR * stats["psi_std"]
    sZ2 = SZ * SZ * stats["psi_std"]
    R_phys = b["R_phys"].reshape(-1).to(device)
    lap = sR2 * out["d2psi_x"][:, 0, 0] \
        - (SR * stats["psi_std"] * out["dpsi_x"][:, 0, 0]) / (R_phys + 1e-12) \
        + sZ2 * out["d2psi_x"][:, 1, 0]
    if args.pde_coils_model == "plasma":
        # Plasma-field residual (user feedback 2026-08-19): subtract the known
        # coil curvature so LPDE constrains Delta* psi_plasma + mu0 R J = 0.
        # The old "total" form kept Delta* psi_coils in the residual — the coil
        # field is ~183% of |psi_total|, its curvature polluted the gradient
        # everywhere outside the excluded coil cells.
        dsc = b["dstar_coils"].reshape(-1).to(device)
        lap = lap - dsc
    res = lap + MU0 * R_phys * J_phys
    valid = b["pde_valid"].reshape(-1).to(device)
    scale = float(stats["pde_scale"])
    l_pde = ((res / scale).square() * valid).sum() / valid.sum().clamp(min=1)

    # ---- L_Ip: plasma-stratum current integral vs target Ip ----
    w = b["weight"].to(device)                             # (B, k)
    est = (J_phys.reshape(B, k) * w).sum(dim=1)            # (B,)
    ip = b["ip"].to(device)
    l_ip = (((est - ip) / float(stats["ip_scale"])) ** 2).mean()

    # ---- L_reg: L1 + entropy on spline coefficients (KAN-3 window) ----
    c0, c1 = model.layer0.coeffs, model.layer1.coeffs
    l_reg = c0.abs().mean() + c1.abs().mean() + 0.5 * (entropy_reg(c0) + entropy_reg(c1))

    return {"l_psi": l_psi, "l_j": l_j, "l_pde": l_pde, "l_ip": l_ip,
            "l_reg": l_reg}


def validate(model, val_ds: PointKANDataset, device, val_indices,
             stats, args) -> dict:
    """Full-grid rel L2 + component losses on a val subset (mean over rows)."""
    model.eval()
    rel2_sum = 0.0
    comps = {k: 0.0 for k in ("l_psi", "l_j", "l_pde", "l_ip")}
    n = 0
    with torch.no_grad():
        for i in val_indices:
            b = val_ds.full_grid(i)
            x = b["x"].to(device)
            out = model.forward_with_derivs(x)
            psi_z = out["psi"][:, 0]
            tgt = b["y_psi"].to(device)
            rel2_sum += float(torch.norm(psi_z - tgt) / (torch.norm(tgt) + 1e-12))
            J_phys = (out["psi"][:, 1] * stats["j_std"] + stats["j_mean"])
            est = (J_phys * b["weight"].to(device)).sum()
            comps["l_ip"] += float((((est - b["ip"].to(device)[0]) / stats["ip_scale"]) ** 2))
            comps["l_psi"] += float(torch.nn.functional.mse_loss(psi_z, tgt))
            comps["l_j"] += float(torch.nn.functional.mse_loss(
                out["psi"][:, 1], b["y_j"].to(device)))
            sR2 = SR * SR * stats["psi_std"]
            sZ2 = SZ * SZ * stats["psi_std"]
            R_phys = b["R_phys"].to(device)
            lap = sR2 * out["d2psi_x"][:, 0, 0] \
                - (SR * stats["psi_std"] * out["dpsi_x"][:, 0, 0]) / (R_phys + 1e-12) \
                + sZ2 * out["d2psi_x"][:, 1, 0]
            if args.pde_coils_model == "plasma":
                lap = lap - b["dstar_coils"].to(device)
            res = lap + MU0 * R_phys * J_phys
            valid = b["pde_valid"].to(device)
            comps["l_pde"] += float(((res / stats["pde_scale"]).square() * valid).sum()
                                    / valid.sum().clamp(min=1))
            n += 1
    model.train()
    return {"val_rel_l2": rel2_sum / max(n, 1),
            **{k: v / max(n, 1) for k, v in comps.items()}}


def main() -> None:
    ap = argparse.ArgumentParser(description="Three-phase KAN training (paper aps.75.20260331)")
    ap.add_argument("--train-data", required=True, help="comma-separated npz list (train pool)")
    ap.add_argument("--val-data", required=True, help="comma-separated npz list (val pool)")
    ap.add_argument("--n-train", type=int, default=500)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--points-per-sample", type=int, default=256)
    ap.add_argument("--batch-samples", type=int, default=32)
    ap.add_argument("--steps-per-epoch", type=int, default=100)
    ap.add_argument("--val-every", type=int, default=10)
    ap.add_argument("--val-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--milestones", type=str, default="300,500,700,900")
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--j-weight", type=float, default=1.0,
                    help="L_Jψ 权重（默认 1；=0 时第一阶段只预测 psi，J 不参与损失——"
                         "隔离 J 的分离面跳变对共享隐藏层的干扰）")
    ap.add_argument("--pde-weight", type=float, default=0.1)
    ap.add_argument("--pde-coils-model", choices=("plasma", "total"),
                    default="plasma",
                    help="LPDE 残差定义: plasma=Δ*ψ_total−Δ*ψ_coils+μ0RJ "
                         "(不含线圈场，正确物理，默认)；total=Δ*ψ_total+μ0RJ "
                         "(旧实现，线圈曲率残留污染残差)")
    ap.add_argument("--ip-weight", type=float, default=1.0)
    ap.add_argument("--reg-weight", type=float, default=0.001)
    ap.add_argument("--prune-threshold", type=float, default=1e-2)
    ap.add_argument("--prune-epoch", type=int, default=None,
                    help="default: int(600*epochs/1000) — paper's 600 scaled")
    ap.add_argument("--sampling", type=str, default="stratified",
                    choices=["stratified", "uniform"])
    ap.add_argument("--hidden", type=int, default=10,
                    help="hidden nodes (paper: 10; capacity boost for exp002)")
    ap.add_argument("--grid-size", type=int, default=2,
                    help="B-spline intervals per dim (paper: 2)")
    ap.add_argument("--degree", type=int, default=2,
                    help="B-spline degree (paper '3 阶' = quadratic, degree 2)")
    ap.add_argument("--device-config", type=str, default="mast",
                    choices=("mast", "testtokamak"),
                    help="设备参数化: mast=data_v5（11 线圈+config，19ch，65²）；"
                         "testtokamak=data_v2/data_v4（4 线圈，无 config，11ch，65²）")
    ap.add_argument("--target", type=str, default="total",
                    choices=("total", "plasma"),
                    help="psi 目标场: total=psi_total（默认，向后兼容）；"
                         "plasma=psi_plasma（光滑场，无线圈奇点——线圈场在评估时"
                         "用存储的 psi_coils/greens 精确加回，rel L2 照常在总场上度量）")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", type=str, default="auto")
    args = ap.parse_args()

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else ("cpu" if args.device == "auto" else args.device))
    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- data: train stats from the FULL concatenated pool, nested subset ----
    cfg = get_cfg(args.device_config)
    args.in_dim = n_channels(cfg)
    stats = compute_stats_kan(args.train_data, device=args.device_config,
                              target=args.target)
    full_ds = PointKANDataset(args.train_data, stats=stats,
                              device=args.device_config, target=args.target)
    n_full = full_ds.n_full
    train_idx = nested_train_indices(n_full, args.n_train, args.seed)
    train_ds = PointKANDataset(args.train_data, stats=stats,
                               indices=train_idx, sampling=args.sampling,
                               device=args.device_config, target=args.target)
    val_ds = PointKANDataset(args.val_data, stats=stats,
                             device=args.device_config, target=args.target)
    val_rng = np.random.default_rng(args.seed + 1)
    print(f"train rows {len(train_idx)} / {n_full}; val rows {len(val_ds)}; "
          f"device {device}; device-config {args.device_config}; "
          f"channels {args.in_dim}")
    print(f"stats: psi_std={stats['psi_std']:.4g} j_std={stats['j_std']:.3g} "
          f"pde_scale={stats['pde_scale']:.3g} ip_scale={stats['ip_scale']:.3g}")

    model = build_model(in_dim=args.in_dim, hidden=args.hidden,
                        grid_size=args.grid_size, degree=args.degree).to(device)
    n_params = model.count_params()
    milestones = [int(m) for m in args.milestones.split(",")]
    # paper schedule boundaries scaled with the run length (1000 epochs -> 120/240/600)
    scale = args.epochs / 1000.0
    b120, b240, b600 = int(120 * scale), int(240 * scale), int(600 * scale)
    if args.prune_epoch is None:
        args.prune_epoch = b600
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=milestones,
                                                 gamma=args.gamma)

    # ---- phase-aware best tracking (plan fix 2) ----
    best = {"pre": (1e9, None, 0), "post": (1e9, None, 0)}  # (val, state, epoch)
    history = []
    t0 = time.time()
    train_rng = np.random.default_rng(args.seed + 2)

    for epoch in range(1, args.epochs + 1):
        lam = lam_t(epoch, args.reg_weight, b120, b240, b600)
        model.train()
        acc = {k: 0.0 for k in ("l_psi", "l_j", "l_pde", "l_ip", "l_reg", "loss")}
        for step in range(args.steps_per_epoch):
            ids = train_rng.integers(0, len(train_idx), args.batch_samples)
            b = train_ds.sample_batch(train_idx[ids], args.points_per_sample)
            comps = compute_losses(model, b, stats, device, args)
            loss = comps["l_psi"] + args.j_weight * comps["l_j"] \
                + args.pde_weight * comps["l_pde"] + args.ip_weight * comps["l_ip"] \
                + lam * comps["l_reg"]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if getattr(model, "_prune_masks", None):
                model.apply_prune_grads()
            opt.step()
            for k in ("l_psi", "l_j", "l_pde", "l_ip", "l_reg"):
                acc[k] += float(comps[k].detach())
            acc["loss"] += float(loss.detach())
        sched.step()
        for k in acc:
            acc[k] /= args.steps_per_epoch

        if epoch % args.val_every == 0 or epoch in (1, args.prune_epoch, args.epochs):
            vids = val_rng.choice(len(val_ds), min(args.val_size, len(val_ds)),
                                  replace=False)
            v = validate(model, val_ds, device, vids, stats, args)
            # at the prune epoch the validation measures the PRE-prune weights
            phase = "pre" if epoch <= args.prune_epoch else "post"
            if v["val_rel_l2"] < best[phase][0]:
                best[phase] = (v["val_rel_l2"], {k: v.clone().cpu() for k, v in
                                                 model.state_dict().items()}, epoch)
            hist = {"epoch": epoch, "lr": sched.get_last_lr()[0], "lam_t": lam,
                    "phase": phase, **{k: round(vv, 6) for k, vv in acc.items()},
                    "val_rel_l2": round(v["val_rel_l2"], 6),
                    **{k: round(v[k], 6) for k in ("l_psi", "l_j", "l_pde", "l_ip")
                       if k in v}}
            history.append(hist)
            if epoch % (args.val_every * 5) == 0 or epoch in (1, args.prune_epoch, args.epochs):
                print(f"ep {epoch:4d} lam {lam:.5f} | loss {acc['loss']:.4f} "
                      f"psi {acc['l_psi']:.5f} j {acc['l_j']:.5f} "
                      f"pde {acc['l_pde']:.4f} ip {acc['l_ip']:.4f} "
                      f"reg {acc['l_reg']:.5f} | val {v['val_rel_l2']*100:.3f}% "
                      f"({phase} best {best[phase][0]*100:.3f}%) | "
                      f"{time.time()-t0:.0f}s")

        if epoch == args.prune_epoch:
            mags = model.connection_magnitudes()
            thresh = args.prune_threshold * max(mg.max() for mg in mags)
            masks = [mg < thresh for mg in mags]
            st = model.apply_prune_mask(masks)
            # Adam momentum (exp_avg/exp_avg_sq) would push the zeroed weights
            # back off zero on the next step (decay ~ beta1^epochs) — clear the
            # optimizer state of every pruned parameter so pruning stays frozen.
            pruned_params = {id(p) for p, _ in model._prune_masks}
            for p, stt in opt.state.items():
                if id(p) in pruned_params:
                    for v in stt.values():
                        if isinstance(v, torch.Tensor):
                            v.zero_()
            history.append({"event": "prune", "epoch": epoch,
                            "threshold": float(thresh.detach().cpu()),
                            **{f"pruned_{i}": int(m.sum()) for i, m in enumerate(masks)},
                            "prune_ratio": st["ratio"]})
            print(f"  PRUNE @ {epoch}: {st['pruned']}/{st['total']} connections "
                  f"({st['ratio']*100:.1f}%), threshold {thresh:.3e}")

    # ---- checkpoint (KAN-3 artifact = post-prune best; plan fix 2) ----
    v_post, st_post, ep_post = best["post"]
    v_pre, st_pre, ep_pre = best["pre"]
    if st_post is not None:
        best_state, best_epoch, best_val, phase = st_post, ep_post, v_post, "pruned"
    else:
        best_state, best_epoch, best_val, phase = st_pre, ep_pre, v_pre, "supervised"
    model.load_state_dict(best_state)
    prune_masks = [m.cpu().numpy() for m in model.prune_mask_tensors()]
    ratio = float(sum(m.sum() for m in prune_masks)) / sum(m.size for m in prune_masks) \
        if prune_masks else 0.0
    ckpt = {
        "model_state": best_state,  # torch tensors (FNO best.pt parity)
        "best_epoch": best_epoch,
        "best_val_rel_l2": best_val,
        "n_params": n_params,
        "n_train": args.n_train,
        "seed": args.seed,
        "stats": {k: (v.tolist() if hasattr(v, "tolist") else float(v))
                  for k, v in stats.items()},
        "prune_mask": [m.tolist() for m in prune_masks],
        "prune_ratio": ratio,
        "phase": phase,
        "arch": {"in_dim": args.in_dim, "hidden": args.hidden, "out_dim": 2,
                 "grid_size": args.grid_size, "degree": args.degree,
                 "base_fn": "silu", "device_config": args.device_config,
                 "target": args.target},
    }
    torch.save(ckpt, out_dir / "best.pt")
    if st_pre is not None:
        torch.save({"model_state": st_pre,
                    "best_epoch": ep_pre, "best_val_rel_l2": v_pre,
                    "phase": "supervised", "n_params": n_params,
                    "n_train": args.n_train, "seed": args.seed,
                    "stats": ckpt["stats"], "arch": ckpt["arch"]},
                   out_dir / "best_pre_prune.pt")
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    with open(out_dir / "args.json", "w") as f:
        json.dump(vars(args) | {"n_params": n_params, "device": str(device)},
                  f, indent=2)
    print(f"  done ({time.time()-t0:.0f}s): best {phase} rel L2 {best_val*100:.3f}% "
          f"@ ep {best_epoch} (pre-prune best {v_pre*100:.3f}% @ ep {ep_pre}); "
          f"prune ratio {ratio*100:.1f}%")
    print(f"  saved -> {out_dir}/best.pt, best_pre_prune.pt, history.json, args.json")


if __name__ == "__main__":
    main()
