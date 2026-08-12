"""Loss functions for free-boundary GS PINO surrogate training.

Following PlaNet-equil architecture:
- Network predicts psi_total directly
- MSE loss over entire domain
- PDE loss based on Grad-Shafranov operator with convolution kernels
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

MU0 = 4.0 * torch.pi * 1e-7

Gauss_kernel_3x3 = np.array(([1, 2, 1], [2, 4, 2], [1, 2, 1])) / 16
Gauss_kernel_5x5 = np.array(
    (
        [1, 4, 7, 4, 1],
        [4, 16, 26, 16, 4],
        [7, 26, 41, 26, 7],
        [4, 16, 26, 16, 4],
        [1, 4, 7, 4, 1],
    )
) / 273


def global_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return ((pred - target) ** 2).mean()


def plasma_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return (((pred - target) ** 2) * mask).sum() / (mask.sum() + eps)


def curvature_loss(pred: torch.Tensor, R: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
    B = pred.shape[0]
    dR = (R[:, 1, 0] - R[:, 0, 0]).view(B, 1, 1)
    dZ = (Z[:, 0, 1] - Z[:, 0, 0]).view(B, 1, 1)

    d2psi_dr2 = (pred[:, 2:, :] - 2 * pred[:, 1:-1, :] + pred[:, :-2, :]) / (dR ** 2)
    d2psi_dz2 = (pred[:, :, 2:] - 2 * pred[:, :, 1:-1] + pred[:, :, :-2]) / (dZ ** 2)

    laplace_psi = d2psi_dr2[:, :, 1:-1] + d2psi_dz2[:, 1:-1, :]

    return (laplace_psi ** 2).mean()


def compute_grad_shafranov_kernels(RR: torch.Tensor, ZZ: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    device = RR.device
    hr = RR[:, 1, 0] - RR[:, 0, 0]
    hz = ZZ[:, 0, 1] - ZZ[:, 0, 0]
    alpha = -2 * (hr**2 + hz**2)
    Laplace_kernel = torch.zeros(RR.shape[0], 3, 3, device=device)
    Df_dr_kernel = torch.zeros(RR.shape[0], 3, 3, device=device)
    
    for b in range(RR.shape[0]):
        alfa_b = alpha[b]
        Laplace_kernel[b] = torch.tensor([
            [0, hr[b]**2 / alfa_b, 0],
            [hz[b]**2 / alfa_b, 1, hz[b]**2 / alfa_b],
            [0, hr[b]**2 / alfa_b, 0]
        ])
        Df_dr_kernel[b] = (
            torch.tensor([[0, 0, 0], [+1, 0, -1], [0, 0, 0]], device=device)
            / (2 * hr[b] * alfa_b)
            * (hr[b]**2 * hz[b]**2)
        )
    
    return Laplace_kernel, Df_dr_kernel


def _compute_grad_shafranov_operator(
    pred: torch.Tensor,
    Laplace_kernel: torch.Tensor,
    Df_dr_kernel: torch.Tensor,
    RR: torch.Tensor,
    ZZ: torch.Tensor,
    Gauss_kernel: torch.Tensor,
    smooth: bool = True,
) -> torch.Tensor:
    batch_size = pred.shape[0]

    Lpsi = F.conv2d(
        pred[:, None, ...].permute(1, 0, 2, 3),
        weight=Laplace_kernel[:, None, ...],
        groups=batch_size,
    ).permute(1, 0, 2, 3)

    Dpsi_dr = -F.conv2d(
        pred[:, None, ...].permute(1, 0, 2, 3),
        weight=Df_dr_kernel[:, None, ...],
        groups=batch_size,
    ).permute(1, 0, 2, 3)
    Dpsi_dr = torch.div(Dpsi_dr, RR[:, None, 1:-1, 1:-1])

    lhs = Lpsi - Dpsi_dr

    hr = (RR[:, 1, 0] - RR[:, 0, 0])[:, None, None, None]
    hz = (ZZ[:, 0, 1] - ZZ[:, 0, 0])[:, None, None, None]
    alfa = -2 * (hr**2 + hz**2)
    beta = alfa / (hr**2 * hz**2)
    GS_ope = lhs * beta

    if smooth:
        GS_ope = F.conv2d(
            GS_ope,
            weight=Gauss_kernel[None, None, ...],
            groups=1,
            padding="same",
        )

    return GS_ope.squeeze(1)


class GSOperatorLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.register_buffer("Gauss_kernel", torch.tensor(Gauss_kernel_5x5, dtype=torch.float32))

    def forward(
        self,
        pred: torch.Tensor,
        rhs: torch.Tensor,
        Laplace_kernel: torch.Tensor,
        Df_dr_kernel: torch.Tensor,
        RR: torch.Tensor,
        ZZ: torch.Tensor,
    ) -> torch.Tensor:
        rhs_computed = _compute_grad_shafranov_operator(
            pred, Laplace_kernel, Df_dr_kernel, RR, ZZ, self.Gauss_kernel.to(pred.device)
        )
        return self.mse(rhs_computed, rhs)


class PlaNetLoss(nn.Module):
    # 默认权重与 train_freegs CLI 默认值保持一致（--scale-pde 0.01 --scale-axis 0.01 --scale-ip 0.001）
    def __init__(self, is_physics_informed: bool = True, scale_mse: float = 1.0, scale_pde: float = 0.01,
                 scale_axis: float = 0.01, scale_ip: float = 0.001, scale_curvature: float = 0.5):
        super().__init__()
        self.is_physics_informed = is_physics_informed
        self.loss_mse = nn.MSELoss()
        self.loss_pde = GSOperatorLoss()
        self.scale_mse = scale_mse
        self.scale_pde = scale_pde
        self.scale_axis = scale_axis
        self.scale_ip = scale_ip
        self.scale_curvature = scale_curvature
        self.log_dict = {}

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        rhs: torch.Tensor,
        Laplace_kernel: torch.Tensor,
        Df_dr_kernel: torch.Tensor,
        RR: torch.Tensor,
        ZZ: torch.Tensor,
        psi_coils: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        interior_mask: torch.Tensor | None = None,
        meta: dict | None = None,
        predict_plasma: bool = False,
    ) -> torch.Tensor:
        if interior_mask is not None:
            mse_loss = self.scale_mse * (((pred - target) ** 2) * interior_mask).mean()
        else:
            mse_loss = self.scale_mse * self.loss_mse(input=pred, target=target)
        self.log_dict["mse_loss"] = mse_loss.item()
        
        if not self.is_physics_informed:
            return mse_loss
        
        total_loss = mse_loss
        
        if self.scale_curvature > 0:
            curvature_l = self.scale_curvature * curvature_loss(pred, RR, ZZ)
            self.log_dict["curvature_loss"] = curvature_l.item()
            total_loss += curvature_l
        
        if predict_plasma:
            pred_plasma = pred
        elif psi_coils is not None:
            pred_plasma = pred - psi_coils
        else:
            pred_plasma = pred
        
        pde_loss = self.scale_pde * self.loss_pde(
            pred=pred_plasma,
            rhs=rhs,
            Laplace_kernel=Laplace_kernel,
            Df_dr_kernel=Df_dr_kernel,
            RR=RR,
            ZZ=ZZ,
        )
        self.log_dict["pde_loss"] = pde_loss.item()
        total_loss += pde_loss
        
        if meta is not None and self.scale_axis > 0:
            axis_loss = self.scale_axis * axis_constraint_loss(
                pred.unsqueeze(1),
                R=RR,
                Z=ZZ,
                psi_coils=psi_coils.unsqueeze(1) if psi_coils is not None else None,
                psi_axis=meta["psi_axis"],
                psi_bndry=meta["psi_bndry"],
                R_axis=meta["R_axis"],
                Z_axis=meta["Z_axis"],
                predict_plasma=predict_plasma,
            )
            self.log_dict["axis_loss"] = axis_loss.item()
            total_loss += axis_loss
        
        if meta is not None and self.scale_ip > 0 and mask is not None:
            ip_loss = self.scale_ip * ip_constraint_loss_freebnd(
                pred.unsqueeze(1),
                R=RR,
                Z=ZZ,
                mask=mask.unsqueeze(1),
                psi_coils=psi_coils.unsqueeze(1) if psi_coils is not None else None,
                L=meta["L"],
                Beta0=meta["Beta0"],
                R0=meta["R_axis"],
                alpha_m=meta["alpha_m"],
                alpha_n=meta["alpha_n"],
                psi_axis=meta["psi_axis"],
                psi_bndry=meta["psi_bndry"],
                Ip_target=meta["Ip"],
                predict_plasma=predict_plasma,
            )
            self.log_dict["ip_loss"] = ip_loss.item()
            total_loss += ip_loss
        
        return total_loss


def gs_residual_loss_freebnd(
    pred: torch.Tensor,
    *,
    R: torch.Tensor,
    Z: torch.Tensor,
    mask: torch.Tensor,
    psi_coils: torch.Tensor,
    L: torch.Tensor,
    Beta0: torch.Tensor,
    R0: torch.Tensor,
    alpha_m: torch.Tensor,
    alpha_n: torch.Tensor,
    psi_axis: torch.Tensor,
    psi_bndry: torch.Tensor,
) -> torch.Tensor:
    B = pred.shape[0]

    dR = (R[:, 1, 0] - R[:, 0, 0]).view(B, 1, 1, 1)
    dZ = (Z[:, 0, 1] - Z[:, 0, 0]).view(B, 1, 1, 1)

    psi_plasma = pred - psi_coils

    dpsi = (psi_axis - psi_bndry).view(B, 1, 1, 1).clamp_min(1e-30)
    psi_plasma_norm = (psi_plasma - psi_bndry.view(B, 1, 1, 1)) / dpsi

    d2r = (psi_plasma_norm[:, :, 2:, 1:-1] - 2.0 * psi_plasma_norm[:, :, 1:-1, 1:-1] + psi_plasma_norm[:, :, :-2, 1:-1]) / (dR ** 2)
    dr = (psi_plasma_norm[:, :, 2:, 1:-1] - psi_plasma_norm[:, :, :-2, 1:-1]) / (2.0 * dR)
    R_c = R[:, 1:-1, 1:-1].unsqueeze(1)
    r_term = dr / (R_c + 1e-8)
    d2z = (psi_plasma_norm[:, :, 1:-1, 2:] - 2.0 * psi_plasma_norm[:, :, 1:-1, 1:-1] + psi_plasma_norm[:, :, 1:-1, :-2]) / (dZ ** 2)

    lap_psi_plasma_norm = d2r - r_term + d2z

    psiN_c = psi_plasma_norm[:, :, 1:-1, 1:-1].clamp(0.0, 1.0)
    shape = (1.0 - psiN_c.clamp(0.0, 0.9999) ** alpha_m.view(B, 1, 1, 1)) ** alpha_n.view(B, 1, 1, 1)
    shape = shape.clamp(0.0, 1.0)

    S = MU0 * L.view(B, 1, 1, 1) * (
        Beta0.view(B, 1, 1, 1) * R_c ** 2 / R0.view(B, 1, 1, 1)
        + (1.0 - Beta0.view(B, 1, 1, 1)) * R0.view(B, 1, 1, 1)
    ) * shape

    residual = lap_psi_plasma_norm + S / dpsi

    m = mask[:, :, 1:-1, 1:-1]
    return ((residual ** 2) * m).sum() / (m.sum() + 1e-8)


def axis_constraint_loss(
    pred: torch.Tensor,
    *,
    R: torch.Tensor,
    Z: torch.Tensor,
    psi_coils: torch.Tensor,
    psi_axis: torch.Tensor,
    psi_bndry: torch.Tensor,
    R_axis: torch.Tensor,
    Z_axis: torch.Tensor,
    predict_plasma: bool = False,
) -> torch.Tensor:
    B, _, nr, nz = pred.shape

    if predict_plasma:
        pred_total = pred + psi_coils
    else:
        pred_total = pred
    dpsi = (psi_axis - psi_bndry).view(B, 1, 1, 1).clamp_min(1e-30)
    psi_norm = (pred_total - psi_bndry.view(B, 1, 1, 1)) / dpsi

    pred_s = psi_norm.squeeze(1)

    dR = (R[:, 1, 0] - R[:, 0, 0])
    dZ = (Z[:, 0, 1] - Z[:, 0, 0])

    i_frac = (R_axis - R[:, 0, 0]) / dR
    j_frac = (Z_axis - Z[:, 0, 0]) / dZ

    i0 = i_frac.floor().long().clamp(0, nr - 2)
    i1 = i0 + 1
    j0 = j_frac.floor().long().clamp(0, nz - 2)
    j1 = j0 + 1

    wi = (i_frac - i0.float()).unsqueeze(1).unsqueeze(2)
    wj = (j_frac - j0.float()).unsqueeze(1).unsqueeze(2)

    batch_idx = torch.arange(B, device=pred.device).unsqueeze(1).unsqueeze(2)

    v00 = pred_s[batch_idx, i0.view(B, 1, 1), j0.view(B, 1, 1)]
    v01 = pred_s[batch_idx, i0.view(B, 1, 1), j1.view(B, 1, 1)]
    v10 = pred_s[batch_idx, i1.view(B, 1, 1), j0.view(B, 1, 1)]
    v11 = pred_s[batch_idx, i1.view(B, 1, 1), j1.view(B, 1, 1)]

    axis_pred = (
        (1.0 - wi) * (1.0 - wj) * v00
        + wi * (1.0 - wj) * v10
        + (1.0 - wi) * wj * v01
        + wi * wj * v11
    )

    axis_pred = axis_pred.clamp(-5.0, 5.0)
    return ((axis_pred - 1.0) ** 2).mean()


def ip_constraint_loss_freebnd(
    pred: torch.Tensor,
    *,
    R: torch.Tensor,
    Z: torch.Tensor,
    mask: torch.Tensor,
    psi_coils: torch.Tensor,
    L: torch.Tensor,
    Beta0: torch.Tensor,
    R0: torch.Tensor,
    alpha_m: torch.Tensor,
    alpha_n: torch.Tensor,
    psi_axis: torch.Tensor,
    psi_bndry: torch.Tensor,
    Ip_target: torch.Tensor,
    predict_plasma: bool = False,
) -> torch.Tensor:
    B = pred.shape[0]
    dR = (R[:, 1, 0] - R[:, 0, 0]).view(B)
    dZ = (Z[:, 0, 1] - Z[:, 0, 0]).view(B)

    if predict_plasma:
        psi_plasma = pred
    else:
        psi_plasma = pred - psi_coils
    dpsi = (psi_axis - psi_bndry).view(B, 1, 1, 1).clamp_min(1e-30)
    psiN = ((psi_plasma - psi_bndry.view(B, 1, 1, 1)) / dpsi).clamp(0.0, 1.0).squeeze(1)

    shape = (1.0 - psiN.clamp(0.0, 0.9999) ** alpha_m.view(B, 1, 1)) ** alpha_n.view(B, 1, 1)

    Jtor = L.view(B, 1, 1) * (
        Beta0.view(B, 1, 1) * R / R0.view(B, 1, 1)
        + (1.0 - Beta0.view(B, 1, 1)) * R0.view(B, 1, 1) / (R + 1e-8)
    ) * shape

    m = mask.squeeze(1)
    Ip_pred = (Jtor * m).sum(dim=(1, 2)) * dR * dZ

    return ((Ip_pred - Ip_target) ** 2 / (Ip_target ** 2 + 1e-6)).mean()