from __future__ import annotations

import torch


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return (((pred - target) ** 2) * mask).sum() / (mask.sum() + eps)


def boundary_band_loss(pred: torch.Tensor, sdf: torch.Tensor, width: float = 0.04) -> torch.Tensor:
    band = (sdf.abs() < width).float()
    return ((pred ** 2) * band).sum() / (band.sum() + 1e-8)


def gs_residual_loss(pred: torch.Tensor, R_norm: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Finite-difference smoothness/PDE proxy for normalized psi.

    The full GS residual depends on the exact external solver's p' and FF'
    parameterization. This conservative proxy regularizes the elliptic operator
    only on interior points and can be replaced by the exact residual later.
    """
    drr = pred[:, :, 2:, 1:-1] - 2 * pred[:, :, 1:-1, 1:-1] + pred[:, :, :-2, 1:-1]
    dzz = pred[:, :, 1:-1, 2:] - 2 * pred[:, :, 1:-1, 1:-1] + pred[:, :, 1:-1, :-2]
    lap = drr + dzz
    m = mask[:, :, 1:-1, 1:-1]
    return ((lap ** 2) * m).sum() / (m.sum() + 1e-8)
