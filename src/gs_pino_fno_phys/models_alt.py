"""Alternative backbones for the physics-informed two-stage pipeline (exp3xx).

All four architectures share the FNO2d2608 contract used by train_pino.py /
evaluate_pino.py:  forward(x: (B, C_in, H, W)) -> (B, C_out, H, W)  on the
18-channel (data_v5/dn) or 21-channel (data_v6) grid inputs, plus
count_params() (cfloat numel counts as 1, matching FNO2d2608's convention).

  unet     — plain-CNN U-Net, no normalization (parity with the FNO baseline)
  ufno     — Fourier U-Net: U-Net down/up path with FNOBlock (SpectralConv2dR
              + 1x1 + GELU) in the encoder levels and bottleneck, plain-conv
              decoder (per-level modes track the resolution: 16/16 -> 8/8 -> 4/4)
  fnokan   — FNO with the pointwise 1x1 conv replaced by a per-pixel KAN layer
              (KANConv1x1: B-spline + silu, tanh-gated into the spline domain;
              KANO-style argument: GS has R-dependent coefficients, so the
              fixed-shape activation FNO relies on may be a spectral bottleneck)
  deeponet — PI-DeepONet: two branch MLPs on the 16 z-scored scalars + one
              shared trunk MLP on the normalized (R, Z) grid, output = dot
              product per pixel (psi / J heads). No CNN head — the plasma
              field is smooth and the FD physics losses act on the field.

Backward compatibility: the registry delegates "fno2d2608" to the existing
model_dn_fno.build_model, and train/evaluate default to it, so exp101-203
checkpoints and launchers are unaffected.

Reused building blocks:
  SpectralConv2dR              gs_pino_dn_fno_2608.model_dn_fno
  _knots/_bspline_bases/_silu  gs_pino_kan_2608.model_kan (handwritten KAN)
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from gs_pino_dn_fno_2608.model_dn_fno import (
    FNOBlock,
    SpectralConv2dR,
    build_model as _build_fno2d2608,
)
from gs_pino_kan_2608.model_kan import _bspline_bases, _knots, _silu


def _bspline_bases_batched(x: torch.Tensor, grid_size: int, degree: int,
                           a: float = -1.0, b: float = 1.0) -> torch.Tensor:
    """All degree-p clamped uniform B-spline bases, Cox-de Boor recursion
    batched over the basis axis (no per-index Python loop):

        B_{i,r} = (x - t_i)/(t_{i+r} - t_i) * B_{i,r-1}
                + (t_{i+r+1} - x)/(t_{i+r+1} - t_{i+1}) * B_{i+1,r-1}

    _bspline_bases (model_kan) loops over i — O(grid+p) Python iterations on
    (..., grid+p)-wide temporaries, ~40 ms/call on the GPU hot shape — while
    here each degree is 2 elementwise ops + 1 padded shift on the same table.
    Semantics identical to _bspline_bases: same clamped knots, half-open
    degree-0 intervals, b-eps right endpoint, zero-denominator -> 0; the
    clamped-end tail bases are identically zero, so truncating the table to
    grid+p values and padding the shift with 0 loses nothing. Cross-checked in
    __main__.
    """
    p = degree
    h = (b - a) / grid_size
    nb = grid_size + p
    eps = 1e-6 * (b - a)            # float32-safe right endpoint (see model_kan)
    xc = torch.clamp(x, min=a, max=b - eps)
    n = torch.arange(nb + p + 1, device=x.device)
    t = torch.clamp(a + (n - p) * h, a, b)              # (nb+p+1,) knot vector
    b0 = ((xc[..., None] >= t[:-1]) & (xc[..., None] < t[1:])).to(x.dtype)
    B = b0                                # width G+2p at degree 0, -1 per degree
    for r in range(1, p + 1):
        w = nb + p - r                    # width at degree r (nb at the target)
        num1 = (xc[..., None] - t[:w]) * B[..., :w]          # B_{i,r-1}, i < w
        den1 = t[r:r + w] - t[:w]
        num2 = (t[r + 1:r + 1 + w] - xc[..., None]) * B[..., 1:]  # B_{i+1,r-1}
        den2 = t[r + 1:r + 1 + w] - t[1:1 + w]
        B = torch.where(den1 != 0, num1 / den1, torch.zeros_like(num1)) \
            + torch.where(den2 != 0, num2 / den2, torch.zeros_like(num2))
    return B


# ---------------------------------------------------------------------------
# shared conv blocks (U-Net / UFNO decoder)
# ---------------------------------------------------------------------------

class _DoubleConv(nn.Module):
    """Two 3x3 convs + GELU, same width in/out (no normalization)."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _UpBlock(nn.Module):
    """Bilinear up (explicit size=, never scale_factor — 65 is not a power of 2)
    + skip concat + DoubleConv."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = _DoubleConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear",
                          align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class _UpFNOBlock(nn.Module):
    """Spectral decoder path (exp305): same bilinear-up + skip concat as
    _UpBlock, but the second 3x3 conv is replaced by an FNOBlock — the
    upsampling path mixes in Fourier space too. The first 3x3 conv is kept
    (channel fusion of up + skip); FNOBlock is width-preserving."""

    def __init__(self, in_channels: int, out_channels: int,
                 modes1: int, modes2: int):
        super().__init__()
        self.fuse = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.spectral = FNOBlock(out_channels, modes1, modes2)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear",
                          align_corners=False)
        x = F.gelu(self.fuse(torch.cat([x, skip], dim=1)))
        return self.spectral(x)


# ---------------------------------------------------------------------------
# exp301 — U-Net (plain CNN)
# ---------------------------------------------------------------------------

class UNet2d2608(nn.Module):
    """Depth-4 U-Net, base width 24 (channels 24/48/96/192 + bottleneck 192).
    No BatchNorm/GroupNorm — matches the no-norm FNO baseline for a fair
    architecture comparison."""

    def __init__(self, in_channels: int = 18, out_channels: int = 2,
                 base_width: int = 24, depth: int = 4):
        super().__init__()
        widths = [base_width * (2 ** d) for d in range(depth)]
        self.lift = nn.Conv2d(in_channels, widths[0], 1)
        # enc level i: in = widths[i-1] (lift out), out = widths[i]
        self.enc = nn.ModuleList()
        prev = widths[0]
        for w in widths:
            self.enc.append(_DoubleConv(prev, w))
            prev = w
        self.bottleneck = _DoubleConv(widths[-1], widths[-1])
        # dec[d] input = up(below) + skip; below width: widths[d+1] (or bottleneck)
        self.dec = nn.ModuleList()
        for d in reversed(range(depth)):
            below = widths[d + 1] if d + 1 < depth else widths[-1]  # bottleneck width
            self.dec.append(_UpBlock(below + widths[d], widths[d]))
        self.proj = nn.Conv2d(widths[0], out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        x = self.lift(x)
        for i, enc in enumerate(self.enc):
            x = enc(x)
            skips.append(x)
            if i < len(self.enc) - 1:
                x = F.max_pool2d(x, 2)          # 65 -> 32 -> 16 -> 8 (floor ok)
        x = self.bottleneck(x)
        for d in range(len(self.dec)):
            x = self.dec[d](x, skips[len(self.dec) - 1 - d])
        return self.proj(x)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def _build_unet2d2608(**overrides) -> UNet2d2608:
    kwargs = dict(in_channels=18, out_channels=2, base_width=24, depth=4)
    kwargs.update(overrides)
    return UNet2d2608(**kwargs)


# ---------------------------------------------------------------------------
# exp302 — Fourier U-Net (spectral encoder levels + plain-conv decoder)
# ---------------------------------------------------------------------------

class UFNO2d2608(nn.Module):
    """U-Net skeleton where encoder levels and bottleneck use FNOBlock
    (SpectralConv2dR + 1x1 + GELU) with per-level modes tracking the
    resolution; decoder is the plain-conv path of the U-Net above."""

    def __init__(self, in_channels: int = 18, out_channels: int = 2,
                 widths: tuple = (64, 64, 32, 32),
                 modes: tuple = ((16, 16), (16, 16), (8, 8), (4, 4)),
                 bottleneck_modes: tuple = (4, 4),
                 bottleneck_layers: int = 1,
                 dec_modes: tuple | None = None):
        """bottleneck_layers > 1 stacks FNOBlocks at the bottom level;
        dec_modes != None swaps the plain-conv decoder for spectral _UpFNOBlocks
        (exp305 tuning; the default keeps the exp302 behavior exactly)."""
        super().__init__()
        assert len(widths) == len(modes) == 4
        self.lift = nn.Conv2d(in_channels, widths[0], 1)
        # level i: 1x1 channel projector (when the width changes) + FNOBlock
        self.enc = nn.ModuleList()
        prev = widths[0]
        for w, (m1, m2) in zip(widths, modes):
            trans = nn.Conv2d(prev, w, 1) if prev != w else nn.Identity()
            self.enc.append(nn.Sequential(trans, FNOBlock(w, m1, m2)))
            prev = w
        self.bottleneck = nn.Sequential(*[
            FNOBlock(widths[-1], *bottleneck_modes)
            for _ in range(bottleneck_layers)
        ])
        self.dec = nn.ModuleList()
        for d in reversed(range(4)):
            below = widths[d + 1] if d + 1 < 4 else widths[-1]
            if dec_modes is None:
                self.dec.append(_UpBlock(below + widths[d], widths[d]))
            else:
                self.dec.append(_UpFNOBlock(below + widths[d], widths[d],
                                            *dec_modes[len(self.dec)]))
        self.proj = nn.Conv2d(widths[0], out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        x = self.lift(x)
        for i, enc in enumerate(self.enc):
            x = enc(x)
            skips.append(x)
            if i < len(self.enc) - 1:
                x = F.max_pool2d(x, 2)
        x = self.bottleneck(x)
        for d in range(len(self.dec)):
            x = self.dec[d](x, skips[len(self.dec) - 1 - d])
        return self.proj(x)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def _build_ufno2d2608(**overrides) -> UFNO2d2608:
    kwargs = dict(in_channels=18, out_channels=2)
    kwargs.update(overrides)
    return UFNO2d2608(**kwargs)


def _build_ufno2d2608_tuned(**overrides) -> UFNO2d2608:
    """exp305: exp302 + 65x65-layer modes 16x16 -> 20x20 (the only layer below
    its min-guard ceiling: 32^2 cap is 17, 16^2 cap is 9, 8^2 cap is 5),
    bottleneck stacked 1 -> 2, and the plain-conv decoder swapped for spectral
    _UpFNOBlocks (modes per upsampled resolution). ~3.6M params — still below
    the FNO baseline's 4.21M."""
    kwargs = dict(in_channels=18, out_channels=2,
                  modes=((20, 20), (16, 16), (8, 8), (4, 4)),
                  bottleneck_layers=2,
                  dec_modes=((4, 4), (8, 8), (8, 8), (8, 8)))
    kwargs.update(overrides)
    return UFNO2d2608(**kwargs)


# ---------------------------------------------------------------------------
# exp303 — FNO with KAN pointwise mixing (KANConv1x1)
# ---------------------------------------------------------------------------

class KANConv1x1(nn.Module):
    """Per-pixel KAN layer: h = silu(x) @ Wb + B(x_gated) @ (Ws * C).

    Replaces the 1x1 conv of FNOBlock with the paper's KAN activation
    phi(x) = wb*silu(x) + ws*sum_b c_b*B_b(x) (model_kan.KANLayer) applied
    independently at every pixel. Fused-matmul form — never materializes the
    (N, in, out) activation tensor the KANLayer.forward_activations einsum
    would build. The spline branch gets tanh(x) as input: B-spline knots live
    on [-1,1] but post-spectral activations are unbounded; the silu base
    branch keeps the unbounded path. Deviations from FNOBlock (all documented
    in the exp303 README): the tanh gate, and basis values computed under
    no_grad (see forward) — gradients to the KAN parameters are unaffected.
    """

    def __init__(self, in_dim: int, out_dim: int, grid_size: int = 4,
                 degree: int = 2):
        super().__init__()
        self.in_dim, self.out_dim = in_dim, out_dim
        self.grid_size, self.degree = grid_size, degree
        self.n_coeffs = grid_size + degree          # clamped B-spline: G + p
        scale = 1.0 / np.sqrt(in_dim)
        self.base_w = nn.Parameter(torch.empty(out_dim, in_dim))
        self.spline_w = nn.Parameter(torch.empty(out_dim, in_dim))
        nn.init.uniform_(self.base_w, -scale, scale)
        nn.init.uniform_(self.spline_w, -scale, scale)
        # coefficients start at 0 -> phi(x) ~ wb*silu(x) at init (stable start,
        # same convention as model_kan.KANLayer)
        self.coeffs = nn.Parameter(torch.zeros(out_dim, in_dim, self.n_coeffs))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = x.shape
        xf = x.flatten(2).transpose(1, 2)                      # (B, N, C)
        # Basis values are computed under no_grad: gradients to the KAN
        # parameters flow through Bf as a constant (dL/d(coeffs) = Bf^T dL/dh),
        # only the gradient path x -> spline -> loss is dropped. Cost: the
        # recursion's elementwise ops saved ~12 (B,N,C,·) tensors for backward
        # per call (~2x the forward itself on the GPU hot shape); the dropped
        # path is tiny anyway — tanh gates saturate exactly where it matters.
        # Documented as a deliberate deviation in the exp303 README.
        with torch.no_grad():
            xn = torch.tanh(xf)                                # spline-domain gate
            bases = _bspline_bases_batched(xn, self.grid_size, self.degree)
        bf = bases.flatten(-2)                                 # (B, N, C*(G+p))
        eff = (self.coeffs * self.spline_w[..., None]).flatten(-2)  # (out, C*(G+p))
        h = _silu(xf) @ self.base_w.transpose(-1, -2) + bf @ eff.transpose(-1, -2)
        return h.transpose(1, 2).reshape(batch, self.out_dim, height, width)


class FNOKAN2d2608(nn.Module):
    """FNO2d2608 skeleton (lift + N blocks + proj) with KANConv1x1 mixing."""

    def __init__(self, in_channels: int = 18, out_channels: int = 2,
                 width: int = 64, modes1: int = 16, modes2: int = 16,
                 layers: int = 4, grid_size: int = 4, degree: int = 2):
        super().__init__()
        self.lift = nn.Conv2d(in_channels, width, 1)
        self.blocks = nn.ModuleList([
            _FNOKANBlock(width, modes1, modes2, grid_size, degree)
            for _ in range(layers)
        ])
        self.proj = nn.Conv2d(width, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lift(x)
        for block in self.blocks:
            x = block(x)
        return self.proj(x)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class _FNOKANBlock(nn.Module):
    """Spectral conv + KAN 1x1 mixing + GELU (FNOBlock with KAN pointwise)."""

    def __init__(self, width: int, modes1: int, modes2: int,
                 grid_size: int, degree: int):
        super().__init__()
        self.spectral = SpectralConv2dR(width, width, modes1, modes2)
        self.kan = KANConv1x1(width, width, grid_size, degree)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.spectral(x) + self.kan(x))


def _build_fnokan2d2608(**overrides) -> FNOKAN2d2608:
    kwargs = dict(in_channels=18, out_channels=2, width=64, modes1=16,
                  modes2=16, layers=4, grid_size=4, degree=2)
    kwargs.update(overrides)
    return FNOKAN2d2608(**kwargs)


# ---------------------------------------------------------------------------
# exp304 — PI-DeepONet (branch x2 on scalars + shared trunk on R,Z)
# ---------------------------------------------------------------------------

class _MLP(nn.Module):
    """Simple MLP with GELU on hidden layers (no norm / dropout)."""

    def __init__(self, in_dim: int, hidden: int, layers: int, out_dim: int):
        super().__init__()
        seq = [nn.Linear(in_dim, hidden), nn.GELU()]
        for _ in range(layers - 1):
            seq += [nn.Linear(hidden, hidden), nn.GELU()]
        seq.append(nn.Linear(hidden, out_dim))
        self.net = nn.Sequential(*seq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PIDeepONet2d(nn.Module):
    """Branch/trunk operator: psi(R,Z) = <branch_psi(theta), trunk(R,Z)>,
    j(R,Z) = <branch_j(theta), trunk(R,Z)>.

    theta = the 16 z-scored scalars (x channels 2:), extracted from the same
    input tensor (they are broadcast constants, so mean over the spatial dims
    recovers the scalar). trunk input = x channels 0:1 (already the normalized
    [-1,1] MAST grid). out_channels=1 (rhs mode) uses only the psi head.

    Init hygiene: the trunk's final linear weight is scaled x0.1 so the
    dot-product output starts small (avoids a large initial MSE).
    """

    def __init__(self, in_channels: int = 18, out_channels: int = 2,
                 p: int = 256, hidden: int = 256, layers: int = 3,
                 ff_l: int = 0):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.ff_l = ff_l
        branch_in = in_channels - 2
        self.branch_psi = _MLP(branch_in, hidden, layers, p)
        self.branch_j = _MLP(branch_in, hidden, layers, p) if out_channels > 1 else None
        # ff_l > 0: NeRF-style multiscale Fourier features on (R,Z) — the trunk
        # sees gamma(x) = [x, sin(2^k pi x), cos(2^k pi x)] for k < ff_l
        # (2(1+2*ff_l) inputs). Exp308 tuning: the trunk is a pointwise MLP, and
        # J is psi's second derivative, so high-frequency recovery benefits from
        # explicit frequency channels (the k=ff_l-1 term stays below the grid
        # Nyquist: 2^(ff_l-1)*pi <= pi / dx with dx = 2/64).
        trunk_in = 2 * (1 + 2 * ff_l) if ff_l > 0 else 2
        self.trunk = _MLP(trunk_in, hidden, layers, p)
        with torch.no_grad():
            self.trunk.net[-1].weight.mul_(0.1)
            self.trunk.net[-1].bias.mul_(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = x.shape
        br_in = x[:, 2:].mean(dim=(2, 3))                      # (B, branch_in)
        tr_in = x[:, 0:2].reshape(batch, 2, -1).transpose(1, 2)  # (B, N, 2)
        if self.ff_l > 0:
            feats = [tr_in]
            for k in range(self.ff_l):
                f = (2.0 ** k) * math.pi
                feats += [torch.sin(f * tr_in), torch.cos(f * tr_in)]
            tr_in = torch.cat(feats, dim=-1)                   # (B, N, 2(1+2L))
        t = self.trunk(tr_in)                                  # (B, N, p)
        psi = torch.einsum("bp,bnp->bn", self.branch_psi(br_in), t)
        if self.branch_j is not None:
            j = torch.einsum("bp,bnp->bn", self.branch_j(br_in), t)
            out = torch.stack([psi, j], dim=1)
        else:
            out = psi.unsqueeze(1)
        return out.reshape(batch, self.out_channels, height, width)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def _build_pideeponet2d(**overrides) -> PIDeepONet2d:
    kwargs = dict(in_channels=18, out_channels=2, p=256, hidden=256, layers=3)
    kwargs.update(overrides)
    return PIDeepONet2d(**kwargs)


def _build_pideeponet2d_wide(**overrides) -> PIDeepONet2d:
    """exp307: exp304 with branch/trunk width p=hidden=256 -> 384 (~1.35M
    params, 2.2x exp304) — J/Ip are psi's second-derivative fields and were
    2x weaker in exp304; the README listed this as the direct capacity fix."""
    kwargs = dict(in_channels=18, out_channels=2, p=384, hidden=384)
    kwargs.update(overrides)
    return PIDeepONet2d(**kwargs)


def _build_pideeponet2d_ff(**overrides) -> PIDeepONet2d:
    """exp308: deeponet_wide + NeRF-style multiscale Fourier features on the
    trunk input (ff_l=6, top frequency 2^5*pi = 32pi, below the grid Nyquist
    pi/dx = 32pi at dx = 2/64). The exp307 -> exp308 increment isolates the
    Fourier-feature contribution."""
    kwargs = dict(in_channels=18, out_channels=2, p=384, hidden=384, ff_l=6)
    kwargs.update(overrides)
    return PIDeepONet2d(**kwargs)


# ---------------------------------------------------------------------------
# exp311 — TKNO-lite: conv-stem transformer operator (TKNO minus KAN)
# ---------------------------------------------------------------------------

class _TransformerBlock(nn.Module):
    """Pre-LN transformer block: LayerNorm -> global self-attention (residual)
    -> LayerNorm -> MLP(GELU, 4x) (residual). No dropout — parity with the
    no-regularization FNO baseline of the series."""

    def __init__(self, d: int, heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(
            nn.Linear(d, int(d * mlp_ratio)), nn.GELU(),
            nn.Linear(int(d * mlp_ratio), d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        # need_weights=False -> nn.MultiheadAttention takes the fused SDPA
        # (flash / mem-efficient) fast path; the default need_weights=True
        # forces the slow math path that materializes the attention matrix
        x = x + self.attn(h, h, h, need_weights=False)[0]
        h = self.norm2(x)
        return x + self.mlp(h)


class TKNOlite2d2608(nn.Module):
    """Transformer operator for the GS problem (exp311): the TKNO of
    arXiv:2511.19114 (Transformer-KAN Neural Operator, ICML 2026 — the same
    Grad-Shafranov operator-learning task) stripped of its KAN pointwise path
    (exp303 showed per-pixel KAN gain ~0 on this data) and adapted to the
    65x65 grid.

    Global elliptic coupling is carried by self-attention instead of spectral
    convs (UFNO) or plain depth (UNet): a conv stem keeps the full 65^2 detail
    path, a 3x3/stride-2 patch embedding downsamples to 33^2 = 1089 tokens
    (4225 tokens would be quadratic-infeasible at batch 16), L blocks of global
    self-attention mix tokens, and the decoder up-samples back with a stem
    skip. Learned 2D positional embedding (the MAST 65x65 grid is fixed).
    """

    def __init__(self, in_channels: int = 18, out_channels: int = 2,
                 width: int = 128, heads: int = 8, layers: int = 4,
                 stem_channels: int = 64):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, stem_channels, 3, padding=1), nn.GELU())
        # 65x65 -> 32x32 = 1024 tokens (k=4 s=2 p=1: (65+2-4)/2+1 = 32). The
        # 1024 count is a multiple of 8, so fused flash attention applies
        # (33^2=1089 falls back to the slow mem-efficient path)
        self.patch_embed = nn.Conv2d(stem_channels, width, 4, stride=2, padding=1)
        self.pos_embed = nn.Parameter(torch.zeros(1, width, 32, 32))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.blocks = nn.ModuleList(
            [_TransformerBlock(width, heads) for _ in range(layers)])
        self.dec = _DoubleConv(stem_channels + width, stem_channels)
        self.proj = nn.Conv2d(stem_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stem = self.stem(x)                                # (B, 64, 65, 65)
        t = self.patch_embed(stem) + self.pos_embed        # (B, W, 32, 32)
        b, c, h, w = t.shape
        t = t.flatten(2).transpose(1, 2)                   # (B, 1024, W)
        for blk in self.blocks:
            t = blk(t)
        t = t.transpose(1, 2).reshape(b, c, h, w)          # (B, W, 32, 32)
        up = F.interpolate(t, size=stem.shape[-2:], mode="bilinear",
                           align_corners=False)
        return self.proj(self.dec(torch.cat([up, stem], dim=1)))

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def _build_tkno_lite(**overrides) -> TKNOlite2d2608:
    kwargs = dict(in_channels=18, out_channels=2, width=128, heads=8,
                  layers=4, stem_channels=64)
    kwargs.update(overrides)
    return TKNOlite2d2608(**kwargs)


# ---------------------------------------------------------------------------
# exp312 — POD-DeepONet: data-derived low-rank basis instead of a learned trunk
# ---------------------------------------------------------------------------

def pod_basis_from_snapshots(psi_snap: np.ndarray, j_snap: np.ndarray | None,
                             energy: float = 0.999, max_modes: int = 256):
    """Top-p POD basis of the centered z-domain training outputs (SVD of the
    N x npix snapshot matrix per channel). Returns a dict for PODDeepONet2d:

        psi0 / j0      mean field (npix,)
        phi_psi / phi_j  right singular vectors (p, npix)  (Vt[:p] rows)
        p_psi / p_j    retained mode counts
        energy_psi / energy_j  cumulative retained energy at p

    p = smallest count reaching `energy` (capped at max_modes)."""
    def _svd(X: np.ndarray, label: str) -> dict:
        mean = X.mean(axis=0)
        U, S, Vt = np.linalg.svd(X - mean, full_matrices=False)
        cum = np.cumsum(S ** 2) / np.sum(S ** 2)
        p = min(int(np.searchsorted(cum, energy)) + 1, max_modes, len(S))
        print(f"  POD[{label}]: n={len(X)} -> {p} modes "
              f"(energy {cum[p-1]*100:.4f}%, cap {max_modes})")
        return {"mean": mean, "modes": Vt[:p].copy(), "p": p,
                "energy": float(cum[p - 1])}
    out = {"n_train": len(psi_snap), "energy": energy, "max_modes": max_modes}
    a = _svd(psi_snap.reshape(len(psi_snap), -1).astype(np.float64), "psi")
    out.update({"psi0": a["mean"], "phi_psi": a["modes"], "p_psi": a["p"],
                "energy_psi": a["energy"]})
    if j_snap is not None:
        b = _svd(j_snap.reshape(len(j_snap), -1).astype(np.float64), "j")
        out.update({"j0": b["mean"], "phi_j": b["modes"], "p_j": b["p"],
                    "energy_j": b["energy"]})
    return out


class PODDeepONet2d(nn.Module):
    """DeepONet whose trunk is replaced by the data-derived POD basis (exp312):
    psi_z(R,Z) = psi0(R,Z) + sum_k b_k phi_k(R,Z), where {phi_k} are the top-p
    right singular vectors of the centered z-domain training outputs and the
    branch MLP predicts the coefficients from the 16 z-scored scalars (the
    same branch input as PIDeepONet2d). The basis is fixed (registered
    buffers, no gradients), so physics fine-tuning (stage 2) can only move the
    branch coefficients — the low-rank prior on the smooth elliptic solutions
    is the hypothesis under test (POD-DeepONet, Lu et al. 2022).

    pod_basis: dict from pod_basis_from_snapshots (or the one stored in a
    checkpoint under the "pod_basis" key). Required at construction — the
    buffer shapes cannot be known before the basis exists (train computes it
    from the N=500 train subset; evaluate reads it from best.pt)."""
    def __init__(self, in_channels: int = 18, out_channels: int = 2,
                 hidden: int = 256, layers: int = 3, pod_basis: dict | None = None):
        super().__init__()
        assert pod_basis is not None, "pod_deeponet requires pod_basis"
        self.in_channels, self.out_channels = in_channels, out_channels
        self.p_psi = int(pod_basis["p_psi"])
        self.p_j = int(pod_basis.get("p_j", 0))
        for key in ("psi0", "phi_psi"):
            self.register_buffer(key, torch.from_numpy(
                np.ascontiguousarray(pod_basis[key])).float())
        if out_channels > 1 and self.p_j > 0:
            for key in ("j0", "phi_j"):
                self.register_buffer(key, torch.from_numpy(
                    np.ascontiguousarray(pod_basis[key])).float())
        branch_in = in_channels - 2
        self.branch_psi = _MLP(branch_in, hidden, layers, self.p_psi)
        self.branch_j = (_MLP(branch_in, hidden, layers, self.p_j)
                         if out_channels > 1 and self.p_j > 0 else None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        br_in = x[:, 2:].mean(dim=(2, 3))                    # (B, branch_in)
        psi = self.psi0 + self.branch_psi(br_in) @ self.phi_psi  # (B, npix)
        out = [psi]
        if self.branch_j is not None:
            j = self.j0 + self.branch_j(br_in) @ self.phi_j
            out.append(j)
        h = w = int(round(psi.shape[1] ** 0.5))              # 65x65 fixed grid
        return torch.stack(out, dim=1).reshape(batch, len(out), h, w)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def _build_pod_deeponet(**overrides) -> PODDeepONet2d:
    kwargs = dict(in_channels=18, out_channels=2, hidden=256, layers=3,
                  pod_basis=None)
    kwargs.update(overrides)
    return PODDeepONet2d(**kwargs)


# ---------------------------------------------------------------------------
# registry / dispatcher (the only API train_pino / evaluate_pino need)
# ---------------------------------------------------------------------------

MODEL_REGISTRY = {
    "fno2d2608": _build_fno2d2608,      # exp102 baseline, unchanged
    "unet": _build_unet2d2608,
    "ufno": _build_ufno2d2608,
    "fnokan": _build_fnokan2d2608,
    "deeponet": _build_pideeponet2d,
    # exp3xx tuning variants (exp305/307/308): the tuned builds hard-code their
    # architecture kwargs so train/evaluate agree on both sides of a checkpoint
    "ufno_tuned": _build_ufno2d2608_tuned,
    "deeponet_wide": _build_pideeponet2d_wide,
    "deeponet_ff": _build_pideeponet2d_ff,
    # round-2 candidates (exp311/312): TKNO-lite (transformer operator) and
    # POD-DeepONet (low-rank basis). pod_deeponet needs pod_basis (train
    # computes it from the train subset; evaluate reads it from best.pt)
    "tkno_lite": _build_tkno_lite,
    "pod_deeponet": _build_pod_deeponet,
}


def build_model(name: str = "fno2d2608", **overrides) -> nn.Module:
    """Dispatch to MODEL_REGISTRY[name] with kwargs (in_channels, out_channels, ...)."""
    return MODEL_REGISTRY[name](**overrides)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    for name in MODEL_REGISTRY:
        kw = dict(in_channels=18, out_channels=2)
        if name == "pod_deeponet":
            # synthetic basis just for the shape check (the real basis comes
            # from the train data / checkpoint)
            kw["pod_basis"] = {
                "p_psi": 8, "p_j": 8,
                "psi0": rng.normal(size=4225).astype(np.float32),
                "phi_psi": rng.normal(size=(8, 4225)).astype(np.float32),
                "j0": rng.normal(size=4225).astype(np.float32),
                "phi_j": rng.normal(size=(8, 4225)).astype(np.float32),
            }
        m = build_model(name, **kw)
        n = m.count_params()
        x = torch.randn(2, 18, 65, 65)
        with torch.no_grad():
            y = m(x)
        assert y.shape == (2, 2, 65, 65), f"{name}: {tuple(y.shape)}"
        assert torch.isfinite(y).all(), f"{name}: non-finite output"
        print(f"{name}: {n} params | {tuple(x.shape)} -> {tuple(y.shape)} OK")

    # POD SVD self-check: snapshots from a known low-rank field must be
    # recovered by the basis up to the retained-energy error
    rng = np.random.default_rng(1)
    modes = rng.normal(size=(6, 4225))
    coeffs = rng.normal(size=(64, 6)) * 10.0
    psi = coeffs @ modes + 0.01 * rng.normal(size=(64, 4225))
    basis = pod_basis_from_snapshots(psi, None, energy=0.9999, max_modes=16)
    centered = psi[:8] - psi.mean(0)
    proj = centered @ basis["phi_psi"].T @ basis["phi_psi"]  # onto retained modes
    err = np.abs(centered - proj).max()
    print(f"POD SVD self-check: {basis['p_psi']} modes, "
          f"max projection err {err:.2e}")
    assert err < 0.1  # residual = the dropped additive noise (0.01 std)

    # spline-basis cross-check: batched recursion vs the per-index Cox-de Boor
    # table (_bspline_bases), including the [-1,1] boundary points
    torch.manual_seed(0)
    for grid, deg in [(4, 2), (2, 1), (4, 0)]:
        knots = _knots(grid, deg)
        xr = torch.rand(3, 5000, 64) * 2 - 1                     # interior
        xb = torch.tensor([-1.0, -0.999, 0.0, 0.999, 1.0])[None, :, None] \
            .expand(3, 5, 64).contiguous()                       # endpoints
        xs = torch.cat([xr, xb], dim=1)
        ref = _bspline_bases(xs, knots, deg)[deg]                # (…, grid+deg)
        got = _bspline_bases_batched(xs, grid, deg)
        assert got.shape == ref.shape, f"{grid}/{deg}: {tuple(got.shape)} vs {tuple(ref.shape)}"
        err = (got - ref).abs().max().item()
        print(f"bases cross-check grid={grid} deg={deg}: max abs err {err:.2e}")
        assert err < 1e-4

    # timing: fwd+bwd on the KANConv1x1 hot shape (B=16, N=4225, C=64)
    import time
    xc = torch.randn(16, 4225, 64, requires_grad=True)
    for label, fn in [("Cox-de Boor table", lambda: _bspline_bases(xc, _knots(4, 2), 2)[2]),
                      ("batched recursion", lambda: _bspline_bases_batched(xc, 4, 2))]:
        ts = []
        for _ in range(3):
            t0 = time.perf_counter()
            fn().sum().backward(retain_graph=True)
            ts.append(time.perf_counter() - t0)
        print(f"{label}: fwd+bwd {min(ts) * 1e3:.1f} ms")
    print("all backbones smoke OK")
