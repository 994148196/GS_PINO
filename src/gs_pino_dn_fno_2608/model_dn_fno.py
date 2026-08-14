"""FNO for the arXiv:2608.05555 double-null surrogate (paper-faithful assembly).

Paper architecture (Sec. II.C, Fig. 2):
  - lifting layer: 9 input channels -> hidden width 64
  - 4 Fourier layers, each: truncated spectral convolution
        K v(x) = F^{-1}( R(k) . F[v](x) )      (paper Eq. 6)
    with a single complex-valued learnable weight tensor R(k) per layer,
    retaining only the lowest nmodes=(16,16) Fourier modes,
    plus a pointwise (1x1 conv) mixing
  - final pointwise linear projection -> 1 output channel psi(R,Z)
  - no positional embeddings (R, Z passed as explicit input channels)
  - activation: GELU (neuraloperator default; not specified in the paper)

Note on param count: this assembly yields ~4.21M trainable params vs the
paper's reported 4,770,241 (the exact module assembly is not recoverable from
the text; the user accepted an approximate match). The repo's two-weight
SpectralConv2d (models.SpectralConv2d) can be swapped into FNOBlock for the
classic zongyi-li style variant (~8.4M params).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv2dR(nn.Module):
    """Truncated spectral convolution with a single complex weight R(k) (paper Eq. 6).

    Applies R(k) to the lowest (modes1, modes2) modes of the rfft2 spectrum and
    zeros the remaining modes; negative-H frequencies are left zero (classic FNO
    truncation), irfft2 reconstructs the real spatial field.
    """

    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = x.shape
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(
            batch, self.out_channels, height, width // 2 + 1,
            dtype=torch.cfloat, device=x.device,
        )
        m1 = min(self.modes1, height)
        m2 = min(self.modes2, width // 2 + 1)
        out_ft[:, :, :m1, :m2] = torch.einsum(
            "bixy,ioxy->boxy", x_ft[:, :, :m1, :m2], self.weights[:, :, :m1, :m2]
        )
        return torch.fft.irfft2(out_ft, s=(height, width))


class FNOBlock(nn.Module):
    """One Fourier layer: spectral conv + pointwise (1x1) mixing + GELU."""

    def __init__(self, width: int, modes1: int, modes2: int):
        super().__init__()
        self.spectral = SpectralConv2dR(width, width, modes1, modes2)
        self.pointwise = nn.Conv2d(width, width, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.spectral(x) + self.pointwise(x))


class FNO2d2608(nn.Module):
    """Geometry-conditioned FNO: 9-channel field input -> psi(R,Z)."""

    def __init__(self, in_channels: int = 9, out_channels: int = 1,
                 width: int = 64, modes1: int = 16, modes2: int = 16, layers: int = 4):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.width = width
        self.modes = (modes1, modes2)
        self.layers = layers

        self.lift = nn.Conv2d(in_channels, width, 1)
        self.blocks = nn.ModuleList([FNOBlock(width, modes1, modes2) for _ in range(layers)])
        self.proj = nn.Conv2d(width, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lift(x)
        for block in self.blocks:
            x = block(x)
        return self.proj(x)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(**overrides) -> FNO2d2608:
    """Build the paper FNO with optional overrides (e.g. width, modes, layers)."""
    kwargs = dict(in_channels=9, out_channels=1, width=64, modes1=16, modes2=16, layers=4)
    kwargs.update(overrides)
    return FNO2d2608(**kwargs)


if __name__ == "__main__":
    model = build_model()
    n = model.count_params()
    print(f"FNO2d2608: {n} trainable params (paper: 4,770,241, ratio {n/4770241:.3f})")
    x = torch.randn(2, 9, 65, 65)
    with torch.no_grad():
        y = model(x)
    print(f"input {tuple(x.shape)} -> output {tuple(y.shape)}")
    assert y.shape == (2, 1, 65, 65)
    print("smoke OK")
