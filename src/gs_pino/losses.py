"""Loss functions for masked fixed-boundary GS surrogate training.

Includes the exact Grad-Shafranov PDE residual (replacing the old elliptic proxy)
and optional integral constraints for Ip and betap.
"""

from __future__ import annotations

import torch

MU0 = 4.0 * torch.pi * 1e-7


# ────────────────────────────────────────────────────────────────
#  基础损失
# ────────────────────────────────────────────────────────────────

def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Mean-squared error restricted to LCFS interior points."""
    return (((pred - target) ** 2) * mask).sum() / (mask.sum() + eps)


def boundary_band_loss(pred: torch.Tensor, sdf: torch.Tensor, width: float = 0.04) -> torch.Tensor:
    """Penalize non-zero normalized flux near the LCFS boundary band.

    psi_bar ~= 0 on LCFS, where sdf is approximately 0.
    """
    band = (sdf.abs() < width).float()
    return ((pred ** 2) * band).sum() / (band.sum() + 1e-8)


# ────────────────────────────────────────────────────────────────
#  真实的 Grad-Shafranov PDE 残差
# ────────────────────────────────────────────────────────────────

def gs_residual_loss(
    pred: torch.Tensor,
    *,
    R: torch.Tensor,
    Z: torch.Tensor,
    mask: torch.Tensor,
    L: torch.Tensor,
    Beta0: torch.Tensor,
    R0: torch.Tensor,
    alpha_m: torch.Tensor,
    alpha_n: torch.Tensor,
    psi_axis: torch.Tensor,
    psi_lcfs: torch.Tensor,
) -> torch.Tensor:
    """Exact Grad-Shafranov PDE residual for the Jeon (2015) profile.

    GS equation:  Δ*ψ + μ₀ R² p'(ψ) + FF'(ψ) = 0

    where Δ*ψ = ∂²ψ/∂R² - (1/R)∂ψ/∂R + ∂²ψ/∂Z²

    Profile (ConstrainBetapIp):
      p'(ψN)  = L · Beta0 / Raxis · (1 - ψN^{αm})^{αn}
      FF'(ψN) = μ₀ · L · (1 - Beta0) · Raxis · (1 - ψN^{αm})^{αn}

    Parameters
    ----------
    pred       : [B, 1, nr, nz]  predicted psi_bar (0 at LCFS, 1 at axis)
    R, Z       : [B, nr, nz]     coordinate grids
    mask       : [B, 1, nr, nz]  interior LCFS mask
    L, Beta0   : [B]             profile normalization constants
    R0         : [B]             major radius (= profile Raxis)
    alpha_m, alpha_n : [B]       current profile exponents
    psi_axis, psi_lcfs: [B]      boundary psi values

    Returns
    -------
    Scalar MSE of the GS residual over the interior mask.
    """
    B = pred.shape[0]

    # --- 网格间距 (均匀网格) ---
    # R在行方向变化，Z在列方向变化
    dR = (R[:, 1, 0] - R[:, 0, 0]).view(B, 1, 1, 1)
    dZ = (Z[:, 0, 1] - Z[:, 0, 0]).view(B, 1, 1, 1)

    # --- Δ* psi_bar (有限差分) ---
    # ∂²/∂R²
    d2r = (pred[:, :, 2:, 1:-1] - 2.0 * pred[:, :, 1:-1, 1:-1] + pred[:, :, :-2, 1:-1]) / (dR ** 2)
    # - (1/R) ∂/∂R
    dr = (pred[:, :, 2:, 1:-1] - pred[:, :, :-2, 1:-1]) / (2.0 * dR)
    R_c = R[:, 1:-1, 1:-1].unsqueeze(1)
    r_term = dr / (R_c + 1e-8)
    # ∂²/∂Z²
    d2z = (pred[:, :, 1:-1, 2:] - 2.0 * pred[:, :, 1:-1, 1:-1] + pred[:, :, 1:-1, :-2]) / (dZ ** 2)

    lap_psi_bar = d2r - r_term + d2z

    # --- 源项 ---
    # ψN = 1 - psi_bar (LCFS: psi_bar=0 → ψN=1; axis: psi_bar=1 → ψN=0)
    psiN_c = 1.0 - pred[:, :, 1:-1, 1:-1].clamp(0.0, 1.0)
    shape = (1.0 - psiN_c.clamp(0.0, 0.9999) ** alpha_m.view(B, 1, 1, 1)) ** alpha_n.view(B, 1, 1, 1)
    shape = shape.clamp(0.0, 1.0)

    # S = μ₀ · L · [R²·Beta0/Raxis + (1-Beta0)·Raxis] · shape
    S = MU0 * L.view(B, 1, 1, 1) * (
        Beta0.view(B, 1, 1, 1) * R_c ** 2 / R0.view(B, 1, 1, 1)
        + (1.0 - Beta0.view(B, 1, 1, 1)) * R0.view(B, 1, 1, 1)
    ) * shape

    # 归一化: 除以 dpsi = psi_axis - psi_lcfs
    dpsi = (psi_axis - psi_lcfs).view(B, 1, 1, 1).clamp_min(1e-30)

    residual = lap_psi_bar + S / dpsi

    m = mask[:, :, 1:-1, 1:-1]
    return ((residual ** 2) * m).sum() / (m.sum() + 1e-8)


# ────────────────────────────────────────────────────────────────
#  可选的 Ip 和 betap 积分约束
# ────────────────────────────────────────────────────────────────

def ip_constraint_loss(
    pred: torch.Tensor,
    *,
    R: torch.Tensor,
    Z: torch.Tensor,
    mask: torch.Tensor,
    L: torch.Tensor,
    Beta0: torch.Tensor,
    R0: torch.Tensor,
    alpha_m: torch.Tensor,
    alpha_n: torch.Tensor,
    Ip_target: torch.Tensor,
) -> torch.Tensor:
    """Ip integral constraint loss: (Ip_pred - Ip_target)².

    J_φ = L · [Beta0·R/Raxis + (1-Beta0)·Raxis/R] · (1 - ψN^{αm})^{αn}
    Ip  = ∫∫ J_φ dR dZ
    """
    B = pred.shape[0]
    # R在行方向变化，Z在列方向变化
    dR = (R[:, 1, 0] - R[:, 0, 0]).view(B, 1, 1)
    dZ = (Z[:, 0, 1] - Z[:, 0, 0]).view(B, 1, 1)

    psiN = 1.0 - pred.clamp(0.0, 1.0).squeeze(1)
    shape = (1.0 - psiN.clamp(0.0, 0.9999) ** alpha_m.view(B, 1, 1)) ** alpha_n.view(B, 1, 1)

    Jtor = L.view(B, 1, 1) * (
        Beta0.view(B, 1, 1) * R / R0.view(B, 1, 1)
        + (1.0 - Beta0.view(B, 1, 1)) * R0.view(B, 1, 1) / (R + 1e-8)
    ) * shape

    m = mask.squeeze(1)
    Ip_pred = (Jtor * m).sum(dim=(1, 2)) * dR.squeeze() * dZ.squeeze()

    return ((Ip_pred - Ip_target) ** 2 / (Ip_target ** 2 + 1e-6)).mean()


def betap_constraint_loss(
    pred: torch.Tensor,
    *,
    R: torch.Tensor,
    Z: torch.Tensor,
    mask: torch.Tensor,
    L: torch.Tensor,
    Beta0: torch.Tensor,
    R0: torch.Tensor,
    alpha_m: torch.Tensor,
    alpha_n: torch.Tensor,
    psi_axis: torch.Tensor,
    psi_lcfs: torch.Tensor,
    betap_target: torch.Tensor,
) -> torch.Tensor:
    """betap integral constraint loss: (betap_pred - betap_target)².

    βp = 2μ₀ ∫ p R dR dZ / ∫ Bpol² R dR dZ

    p(ψN) is obtained by numerical integration of p'(ψN).
    Bpol = sqrt(Br² + Bz²), where Br = -(1/R)∂ψ/∂Z, Bz = (1/R)∂ψ/∂R.
    """
    B = pred.shape[0]
    nr, nz = pred.shape[2], pred.shape[3]
    # R在行方向变化，Z在列方向变化
    dR = (R[:, 1, 0] - R[:, 0, 0]).view(B, 1, 1)
    dZ = (Z[:, 0, 1] - Z[:, 0, 0]).view(B, 1, 1)
    dpsi_val = (psi_axis - psi_lcfs).view(B, 1, 1).clamp_min(1e-30)

    psiN = 1.0 - pred.clamp(0.0, 1.0).squeeze(1)

    # --- 压强 p(ψN): 在 [0,psiN] 上积分 p' ---
    # p'(x) = L·Beta0/Raxis · (1 - x^{αm})^{αn}
    # p(ψN) = -dpsi · ∫_{ψN}^{1} p'(x) dx = dpsi · (L·Beta0/Raxis) · ∫_{0}^{1-ψN} (1-s^{αm})^{αn} ds
    # 直接对每个网格点数值积分（用 32 点梯形法则）
    n_int = 32
    x_int = torch.linspace(0, 1, n_int, device=pred.device).view(1, 1, 1, n_int)
    f_int = (1.0 - x_int.clamp(0.0, 0.9999) ** alpha_m.view(B, 1, 1, 1)) ** alpha_n.view(B, 1, 1, 1)
    dx = 1.0 / (n_int - 1)
    # 从 0 累计到 1
    I_cum = torch.cumsum(f_int * dx, dim=-1)                # [B, 1, 1, n_int]

    # 对每个网格点的 psiN 做最近邻插值
    psiN_1d = 1.0 - psiN.clamp(0.0, 1.0)                    # [B, nr, nz], range [0,1]
    idx = (psiN_1d * (n_int - 1)).long().clamp(0, n_int - 1) # [B, nr, nz]
    # gather: I_cum is [B, 1, 1, n_int], idx is [B, nr, nz]
    I_cum_s = I_cum.squeeze(2)                               # [B, 1, n_int]
    I_val = I_cum_s.gather(-1, idx.unsqueeze(1)).squeeze(1)  # [B, nr, nz]

    p = dpsi_val.squeeze() * L.view(B, 1, 1) * Beta0.view(B, 1, 1) / (R0.view(B, 1, 1) + 1e-8) * I_val
    p = p.clamp(min=0.0)

    # --- Bpol² ---
    pred_s = pred.squeeze(1)
    dpdr = torch.zeros_like(pred_s)
    dpdz = torch.zeros_like(pred_s)
    dpdr[:, 1:-1, :] = (pred_s[:, 2:, :] - pred_s[:, :-2, :]) / (2.0 * dR.squeeze())
    dpdz[:, :, 1:-1] = (pred_s[:, :, 2:] - pred_s[:, :, :-2]) / (2.0 * dZ.squeeze())

    dpsi_dr = dpsi_val.squeeze() * dpdr
    dpsi_dz = dpsi_val.squeeze() * dpdz

    Br = -dpsi_dz / (R + 1e-8)
    Bz = dpsi_dr / (R + 1e-8)
    Bpol2 = Br ** 2 + Bz ** 2

    m = mask.squeeze(1)
    num = 2.0 * MU0 * (p * R * m).sum(dim=(1, 2)) * dR.squeeze() * dZ.squeeze()
    den = (Bpol2 * R * m).sum(dim=(1, 2)) * dR.squeeze() * dZ.squeeze()

    betap_pred = num / (den + 1e-30)

    return ((betap_pred - betap_target) ** 2 / (betap_target ** 2 + 1e-6)).mean()


# ────────────────────────────────────────────────────────────────
#  磁轴约束损失
# ────────────────────────────────────────────────────────────────

def axis_constraint_loss(
    pred: torch.Tensor,
    *,
    R: torch.Tensor,
    Z: torch.Tensor,
    R_axis: torch.Tensor,
    Z_axis: torch.Tensor,
) -> torch.Tensor:
    """Penalize deviation of psi_bar at the magnetic axis from the target value 1.

    Uses bilinear interpolation on the uniform (R, Z) grid to evaluate
    psi_bar at the exact (R_axis, Z_axis) coordinate.

    Parameters
    ----------
    pred      : [B, 1, nr, nz]  predicted psi_bar
    R, Z      : [B, nr, nz]     coordinate grids
    R_axis    : [B]             magnetic axis R coordinate
    Z_axis    : [B]             magnetic axis Z coordinate

    Returns
    -------
    mean((psi_bar(R_axis, Z_axis) - 1.0) ** 2)
    """
    B, _, nr, nz = pred.shape
    pred_s = pred.squeeze(1)  # [B, nr, nz]

    # Grid spacing (uniform per sample)
    dR = (R[:, 1, 0] - R[:, 0, 0])  # [B]
    dZ = (Z[:, 0, 1] - Z[:, 0, 0])  # [B]

    # Find fractional indices
    i_frac = (R_axis - R[:, 0, 0]) / dR  # [B]
    j_frac = (Z_axis - Z[:, 0, 0]) / dZ  # [B]

    i0 = i_frac.floor().long().clamp(0, nr - 2)
    i1 = i0 + 1
    j0 = j_frac.floor().long().clamp(0, nz - 2)
    j1 = j0 + 1

    # Bilinear interpolation weights
    wi = (i_frac - i0.float()).unsqueeze(1).unsqueeze(2)  # [B, 1, 1]
    wj = (j_frac - j0.float()).unsqueeze(1).unsqueeze(2)  # [B, 1, 1]

    # Gather 4 corner values
    batch_idx = torch.arange(B, device=pred.device).unsqueeze(1).unsqueeze(2)  # [B, 1, 1]

    v00 = pred_s[batch_idx, i0.view(B, 1, 1), j0.view(B, 1, 1)]  # [B, 1, 1]
    v01 = pred_s[batch_idx, i0.view(B, 1, 1), j1.view(B, 1, 1)]
    v10 = pred_s[batch_idx, i1.view(B, 1, 1), j0.view(B, 1, 1)]
    v11 = pred_s[batch_idx, i1.view(B, 1, 1), j1.view(B, 1, 1)]

    # Interpolate: (1-wi)*(1-wj)*v00 + wi*(1-wj)*v10 + (1-wi)*wj*v01 + wi*wj*v11
    axis_pred = (
        (1.0 - wi) * (1.0 - wj) * v00
        + wi * (1.0 - wj) * v10
        + (1.0 - wi) * wj * v01
        + wi * wj * v11
    )  # [B, 1, 1]

    # 安全裁剪防止梯度爆炸
    axis_pred = axis_pred.clamp(-5.0, 5.0)
    return ((axis_pred - 1.0) ** 2).mean()
