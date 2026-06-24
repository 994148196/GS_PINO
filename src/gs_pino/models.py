"""Neural-operator model definitions used by the GS_PINO workflow."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class SpectralConv2d(nn.Module):
    """2-D Fourier convolution that keeps only a fixed number of low modes."""

    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()
        # Store dimensions and mode counts for use in the forward FFT path.
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        # Complex-valued weights for positive and negative frequency blocks.
        scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))

    def compl_mul2d(self, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Multiply Fourier coefficients by learned complex weights."""
        return torch.einsum("bixy,ioxy->boxy", x, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply FFT, learned low-mode multiplication, and inverse FFT."""
        batch, _, height, width = x.shape

        # Real FFT stores only non-negative frequencies in the last dimension.
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(batch, self.out_channels, height, width // 2 + 1, dtype=torch.cfloat, device=x.device)

        # Clamp mode counts so the same model can run on smaller smoke-test grids.
        m1 = min(self.modes1, height)
        m2 = min(self.modes2, width // 2 + 1)

        # Fill low positive and negative vertical-frequency modes.
        out_ft[:, :, :m1, :m2] = self.compl_mul2d(x_ft[:, :, :m1, :m2], self.weights1[:, :, :m1, :m2])
        out_ft[:, :, -m1:, :m2] = self.compl_mul2d(x_ft[:, :, -m1:, :m2], self.weights2[:, :, :m1, :m2])

        # Transform back to the physical grid.
        return torch.fft.irfft2(out_ft, s=(height, width))


class UNetBranch(nn.Module):
    """Small local branch that complements global spectral mixing."""

    def __init__(self, width: int):
        super().__init__()
        # Downsampled convolutions increase local receptive field cheaply.
        self.down = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1),
            nn.GELU(),
        )
        # Merge the original feature map and upsampled local features.
        self.up = nn.Sequential(
            nn.Conv2d(width * 2, width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a one-level U-Net correction branch."""
        y = F.avg_pool2d(x, 2, ceil_mode=True)
        y = self.down(y)
        y = F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return self.up(torch.cat([x, y], dim=1))


class UFNOBlock(nn.Module):
    """One U-FNO block: spectral path + pointwise path + U-Net path."""

    def __init__(self, width: int, modes1: int, modes2: int):
        super().__init__()
        self.spectral = SpectralConv2d(width, width, modes1, modes2)
        self.pointwise = nn.Conv2d(width, width, 1)
        self.unet = UNetBranch(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Combine global, local, and channel-wise updates."""
        return F.gelu(self.spectral(x) + self.pointwise(x) + self.unet(x))


class UFNO2d(nn.Module):
    """Masked-grid U-FNO for predicting normalized flux on a rectangular grid."""

    def __init__(self, in_channels: int, modes1: int = 16, modes2: int = 16, width: int = 32, layers: int = 4):
        super().__init__()
        # Lift raw coordinate/geometry/parameter channels into latent width.
        self.lift = nn.Conv2d(in_channels, width, 1)

        # Stack repeated U-FNO blocks to approximate the solution operator.
        self.blocks = nn.ModuleList([UFNOBlock(width, modes1, modes2) for _ in range(layers)])

        # Project the latent field back to one channel: normalized psi_bar.
        self.proj = nn.Sequential(nn.Conv2d(width, 128, 1), nn.GELU(), nn.Conv2d(128, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return `psi_bar` with shape `[batch, 1, nr, nz]`."""
        x = self.lift(x)
        for block in self.blocks:
            x = block(x)
        return self.proj(x)
