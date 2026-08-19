"""Physics loss terms for physics-informed FNO training (exp101/102).

Formulas follow the repository convention (gs_pino.losses.py central
differences, KAN pde_scale/ip_scale normalization):
  - GS equation: Delta* psi = -mu0 R J_phi  <=>  Delta* psi + mu0 R J = 0
  - Delta* = d2/dR2 - (1/R) d/dR + d2/dZ2 (2nd-order central differences)
  - J_phi  = R p' + F F'/(mu0 R); the data RHS mu0 R J is stored per-sample
  - Ip = sum(mask * J) * dR * dZ

All PDE terms operate on PHYSICAL fields (Wb, A/m^2); the supervised MSE terms
stay in the z-score domain — the training loop denormalizes before calling
these.
"""
from __future__ import annotations

import torch

MU0 = 4.0 * torch.pi * 1e-7


def denorm_psi(z: torch.Tensor, stats: dict) -> torch.Tensor:
    """z-score -> physical Wb (psi_plasma statistics)."""
    return z * stats["psi_plasma_std"] + stats["psi_plasma_mean"]


def denorm_j(z: torch.Tensor, stats: dict) -> torch.Tensor:
    """z-score -> physical A/m^2 (J statistics)."""
    return z * stats["j_std"] + stats["j_mean"]


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
               eps: float = 1e-8) -> torch.Tensor:
    """Mask-weighted MSE (mirrors gs_pino.losses.masked_mse)."""
    return (((pred - target) ** 2) * mask).sum() / (mask.sum() + eps)


def lap_star_torch(psi: torch.Tensor, R_c: torch.Tensor, dR: float, dZ: float) -> torch.Tensor:
    """Delta* psi on the interior grid (B, 1, 63, 63).

    psi: (B, 1, 65, 65) physical Wb; R_c: (63, 63) physical R[m] interior grid.
    """
    d2r = (psi[..., 2:, 1:-1] - 2.0 * psi[..., 1:-1, 1:-1] + psi[..., :-2, 1:-1]) / (dR ** 2)
    dr = (psi[..., 2:, 1:-1] - psi[..., :-2, 1:-1]) / (2.0 * dR)
    d2z = (psi[..., 1:-1, 2:] - 2.0 * psi[..., 1:-1, 1:-1] + psi[..., 1:-1, :-2]) / (dZ ** 2)
    return d2r - dr / (R_c[None, None] + 1e-8) + d2z


def pde_residual_rhs(psi_phys: torch.Tensor, j_phys: torch.Tensor, R_c: torch.Tensor,
                     dR: float, dZ: float, mask_i: torch.Tensor, pde_scale: float) -> torch.Tensor:
    """Approach 1: residual from the DATASET RHS (mode `rhs`).

    res = Delta* psi_pred + mu0 R J_data, masked mean of (res / pde_scale)^2.
    j_phys: (B, 1, 65, 65) physical A/m^2 reconstructed from the npz fields.
    """
    lap = lap_star_torch(psi_phys, R_c, dR, dZ)
    res = lap + MU0 * R_c[None, None] * j_phys[..., 1:-1, 1:-1]
    return (((res / pde_scale) ** 2) * mask_i).sum() / (mask_i.sum() + 1e-8)


def pde_residual_self(psi_phys: torch.Tensor, j_pred_phys: torch.Tensor, R_c: torch.Tensor,
                      dR: float, dZ: float, mask_i: torch.Tensor, pde_scale: float) -> torch.Tensor:
    """Approach 2 stage 2: SELF-CONSISTENT residual with the network's J.

    res = Delta* psi_pred + mu0 R J_pred — no dataset RHS involved.
    """
    lap = lap_star_torch(psi_phys, R_c, dR, dZ)
    res = lap + MU0 * R_c[None, None] * j_pred_phys[..., 1:-1, 1:-1]
    return (((res / pde_scale) ** 2) * mask_i).sum() / (mask_i.sum() + 1e-8)


def ip_constraint(j_pred_phys: torch.Tensor, mask: torch.Tensor, dA: float,
                  ip_target: torch.Tensor, ip_scale: float) -> torch.Tensor:
    """Ip = sum(mask * J) * dR * dZ per sample; mean(((Ip_pred - Ip)/ip_scale)^2).

    j_pred_phys: (B, 1, 65, 65) physical A/m^2; ip_target: (B, 1) A.
    """
    ip_pred = (j_pred_phys * mask).sum(dim=(2, 3)) * dA          # (B, 1)
    return (((ip_pred - ip_target) / ip_scale) ** 2).mean()
