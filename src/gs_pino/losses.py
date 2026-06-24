"""Loss functions for masked fixed-boundary GS surrogate training."""

from __future__ import annotations

import torch


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Mean-squared error restricted to LCFS interior points."""
    # Multiplying by mask removes outside-LCFS pixels from both numerator and count.
    return (((pred - target) ** 2) * mask).sum() / (mask.sum() + eps)


def boundary_band_loss(pred: torch.Tensor, sdf: torch.Tensor, width: float = 0.04) -> torch.Tensor:
    """Penalize non-zero normalized flux near the LCFS boundary band."""
    # The fallback convention is psi_bar ~= 0 on LCFS, where sdf is approximately 0.
    band = (sdf.abs() < width).float()
    return ((pred**2) * band).sum() / (band.sum() + 1e-8)


def gs_residual_loss(pred: torch.Tensor, R_norm: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Finite-difference elliptic regularizer for normalized psi.

    The exact GS residual needs the real solver's `p'(psi)` and `FF'(psi)`
    conventions.  Until that adapter is wired in, this smoothness proxy keeps the
    predicted field elliptic-like on the interior mask.  `R_norm` is accepted now
    so the function signature can be upgraded to the exact operator later without
    touching the training loop.
    """
    # Second derivative along the R-index direction on the interior stencil.
    drr = pred[:, :, 2:, 1:-1] - 2 * pred[:, :, 1:-1, 1:-1] + pred[:, :, :-2, 1:-1]

    # Second derivative along the Z-index direction on the interior stencil.
    dzz = pred[:, :, 1:-1, 2:] - 2 * pred[:, :, 1:-1, 1:-1] + pred[:, :, 1:-1, :-2]

    # Laplacian proxy; replace with Delta* psi and source terms for exact PINO.
    lap = drr + dzz

    # Crop mask to match the finite-difference stencil shape.
    m = mask[:, :, 1:-1, 1:-1]
    return ((lap**2) * m).sum() / (m.sum() + 1e-8)
