"""Handwritten B-spline KAN for the free-boundary Grad-Shafranov equation.

Ref: "基于 Kolmogorov-Arnold Networks 的 Grad-Shafranov 方程自由边界问题求解方法"
     (Acta Phys. Sin. 75(2026)150504, DOI 10.7498/aps.75.20260331).

Paper architecture (Table 1):
  - point-wise regression: (physical params, R, Z) -> (psi, J_psi)
  - 1 hidden layer x 10 nodes; input 14 (here 19 on data_v5: R, Z + 5 params
    + 11 MAST coil currents + config), output 2
  - B-spline: 2 grid intervals, 3rd order (degree 2, quadratic) on [-1, 1],
    base function silu; activation (paper Eq. 3):
        phi(x) = wb * b(x) + ws * sum_b c_b * B_b(x)
  - spline coefficient count: Table 4 reports 960 trainable params for KAN-2
    -> 160 connections x 6 params/connection = (wb + ws + 4 coefficients);
    with 2 intervals + degree 2 a clamped B-spline has G + p = 4 basis
    functions, consistent.

Everything is plain PyTorch (no pykan / efficient_kan) so the multi-stage
training (supervised -> semi-supervised -> pruning, paper Sec. 2.1), the
symbolic regression (Sec. 2.2) and the linear extrapolation can later read
the weights directly.

Derivatives: analytic. For the PDE residual loss we need d2 psi / dR2 and
d2 psi / dZ2 per training step; B-spline derivatives come from the
Cox-de Boor recurrence (a degree-p basis differentiated = difference of
degree-(p-1) bases), so no autograd graph is required.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# B-spline basis (clamped uniform knots on [-1, 1], vectorized Cox-de Boor)
# ---------------------------------------------------------------------------

def _knots(grid_size: int, degree: int, a: float = -1.0, b: float = 1.0) -> np.ndarray:
    """Clamped uniform knot vector: degree+1 repeats at the ends.

    G intervals + degree p -> 2*(p+1) + G - 1 knots -> G + p basis functions.
    """
    interior = np.linspace(a, b, grid_size + 1)[1:-1]
    return np.concatenate([np.full(degree + 1, a), interior, np.full(degree + 1, b)])


def _bspline_bases(x: torch.Tensor, knots: np.ndarray, degree: int) -> list[torch.Tensor]:
    """Basis values at ALL degrees 0..degree (Cox-de Boor recursion).

    Returns bases[p] = B_{i,p}(x) with shape (B, n_basis(degree=degree)).
    Zero denominators (repeated knots) -> 0 by convention.
    """
    k = knots
    n_int = len(k) - degree - 1  # basis count at the target degree
    # Clamp the right endpoint: at x = b exactly, the Cox-de Boor recursion
    # gives 0 (the last degree-0 interval is half-open), but by convention the
    # final basis function equals 1 there (partition of unity on [a, b]).
    # x = b - eps -> B_{n-1,p} -> 1 up to O(eps), indistinguishable at float
    # precision of the learned weights.
    # FIX (2026-08-19): eps must be representable in float32. 1e-10*span
    # rounds to 0.0 in float32, so x = b was NOT clamped and the basis
    # vanished on the whole boundary ring (|R| or |Z| = 1.0) — the spline
    # contribution was exactly 0 there. 1e-6*span is float32-safe and still
    # negligible vs the learned weights.
    eps = 1e-6 * (k[-1] - k[0])
    xc = torch.clamp(x, min=k[0], max=k[-1] - eps)
    b0 = torch.zeros(x.shape + (len(k) - 1,), dtype=x.dtype, device=x.device)
    for i in range(len(k) - 1):
        # degree-0 basis: 1 on [k_i, k_{i+1})
        b0[..., i] = ((xc >= k[i]) & (xc < k[i + 1])).to(x.dtype)
    out = [b0]
    for p in range(1, degree + 1):
        bp = torch.zeros(x.shape + (len(k) - p - 1,), dtype=x.dtype, device=x.device)
        bp_prev = out[-1]
        for i in range(len(k) - p - 1):
            d1 = (k[i + p] - k[i])
            d2 = (k[i + p + 1] - k[i + 1])
            t1 = (x - k[i]) / d1 if d1 > 0 else torch.zeros_like(x)
            t2 = (k[i + p + 1] - x) / d2 if d2 > 0 else torch.zeros_like(x)
            bp[..., i] = t1 * bp_prev[..., i] + t2 * bp_prev[..., i + 1]
        out.append(bp)
    return out


def _basis_and_derivs(x: torch.Tensor, knots: np.ndarray, degree: int,
                      n_deriv: int = 2) -> tuple[torch.Tensor, ...]:
    """Basis and analytic derivatives up to n_deriv (each (B, n_basis)).

    Recurrence: B'_{i,p} = p * [B_{i,p-1}/(t_{i+p}-t_i)
                                - B_{i+1,p-1}/(t_{i+p+1}-t_{i+1})]
    applied iteratively on the degree table (B^(r)_{i,p} combines B^(r-1) of
    degree p-1, so each derivative order reuses the previous order's table).
    """
    k = knots
    table = _bspline_bases(x, knots, degree)          # table[p]: (B, len(k)-p-1)
    derivs: list[torch.Tensor] = []
    cur = table
    for r in range(1, n_deriv + 1):
        nxt = [None] * (degree + 1)
        nxt[0] = torch.zeros_like(cur[0])             # deriv of degree-0 bases = 0
        for p in range(1, degree + 1):
            src = cur[p - 1]                          # (B, len(k)-p), degree p-1
            n_b = len(k) - p - 1                      # basis count at degree p
            dp = torch.zeros(x.shape + (n_b,), dtype=x.dtype, device=x.device)
            for i in range(n_b):
                d1 = k[i + p] - k[i]
                d2 = k[i + p + 1] - k[i + 1]
                a1 = 1.0 / d1 if d1 > 0 else 0.0
                a2 = 1.0 / d2 if d2 > 0 else 0.0
                dp[..., i] = p * (a1 * src[..., i] - a2 * src[..., i + 1])
            nxt[p] = dp
        derivs.append(nxt[degree])
        cur = nxt
    return table[degree], *derivs


# ---------------------------------------------------------------------------
# KAN layer
# ---------------------------------------------------------------------------

def _silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


def _silu_d1(x: torch.Tensor) -> torch.Tensor:
    s = torch.sigmoid(x)
    return s * (1.0 + x * (1.0 - s))


def _silu_d2(x: torch.Tensor) -> torch.Tensor:
    s = torch.sigmoid(x)
    return s * (1.0 - s) * (2.0 + x * (1.0 - 2.0 * s))


class KANLayer(nn.Module):
    """One KAN layer: h_j = sum_i phi_{j,i}(x_i), phi = wb*b + ws*sum c_b B_b.

    Parameters (paper Eq. 3):
      base_w   (out, in)        — wb, silu base-function weights
      spline_w (out, in)        — ws, spline-scale weights
      coeffs   (out, in, N_b)   — B-spline basis coefficients c_b

    forward_activations(x) also returns the per-connection activation values
    and their analytic 1st/2nd derivatives wrt x_i (the chain-rule ingredients
    the GS PDE-residual loss needs).
    """

    def __init__(self, in_dim: int, out_dim: int, grid_size: int = 2,
                 degree: int = 2, base_fn: str = "silu", seed: int | None = None):
        super().__init__()
        assert base_fn == "silu", "paper fixes the base function to silu"
        self.in_dim, self.out_dim = in_dim, out_dim
        self.grid_size, self.degree = grid_size, degree
        self.n_coeffs = grid_size + degree          # clamped B-spline: G + p
        self.knots = _knots(grid_size, degree)      # shared by all connections

        if seed is not None:
            torch.manual_seed(seed)
        # kaiming-uniform-style scale for both weight tensors
        scale = 1.0 / np.sqrt(in_dim)
        self.base_w = nn.Parameter(torch.empty(out_dim, in_dim))
        self.spline_w = nn.Parameter(torch.empty(out_dim, in_dim))
        nn.init.uniform_(self.base_w, -scale, scale)
        nn.init.uniform_(self.spline_w, -scale, scale)
        # coefficients start at 0 -> phi(x) ~ wb*silu(x) at init (spline grows)
        self.coeffs = nn.Parameter(torch.zeros(out_dim, in_dim, self.n_coeffs))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_activations(x)[0]

    def forward_activations(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Returns (h, acts, d1, d2): layer output, per-connection activation
        phi_{o,i}(x_i) and its 1st/2nd derivatives wrt x_i.

        Shapes: h (B, out); acts/d1/d2 (B, in, out).
        """
        b0, b1, b2 = _silu(x), _silu_d1(x), _silu_d2(x)          # (B, in)
        B0, B1, B2 = _basis_and_derivs(x, self.knots, self.degree)
        # spline part: sum_b (ws*c_b) * B_b(x_i)  -> (B, in, out)
        eff = self.coeffs * self.spline_w[..., None]
        spl = torch.einsum("bin,oin->bio", B0, eff)
        spl_d1 = torch.einsum("bin,oin->bio", B1, eff)
        spl_d2 = torch.einsum("bin,oin->bio", B2, eff)
        # base part
        base = torch.einsum("bi,oi->bio", b0, self.base_w)
        base_d1 = torch.einsum("bi,oi->bio", b1, self.base_w)
        base_d2 = torch.einsum("bi,oi->bio", b2, self.base_w)
        acts, d1, d2 = base + spl, base_d1 + spl_d1, base_d2 + spl_d2
        h = acts.sum(dim=1)                                     # (B, out)
        return h, acts, d1, d2

    def connection_magnitudes(self) -> torch.Tensor:
        """|ws| * sum_b |c_b| per connection (pruning criterion, pykan-style)."""
        return self.spline_w.abs() * self.coeffs.abs().sum(dim=-1)


class BSplineKAN(nn.Module):
    """Two-layer point-wise KAN (paper Table 1): (19,) -> (10,) -> (2,).

    Output channel 0 = psi (normalized), channel 1 = J_psi (normalized).
    forward_with_derivs provides d(psi)/dx and d2(psi)/dx2 for the R and Z
    input channels (analytic chain rule, no autograd).
    """

    def __init__(self, in_dim: int = 19, hidden: int = 10, out_dim: int = 2,
                 grid_size: int = 2, degree: int = 2, base_fn: str = "silu"):
        super().__init__()
        self.in_dim, self.hidden, self.out_dim = in_dim, hidden, out_dim
        self.layer0 = KANLayer(in_dim, hidden, grid_size, degree, base_fn)
        self.layer1 = KANLayer(hidden, out_dim, grid_size, degree, base_fn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, in_dim) in [-1, 1] -> (B, out_dim)."""
        h, _, _, _ = self.layer0.forward_activations(x)
        return self.layer1.forward(h)

    def forward_activations(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Full activation record for both layers (symbolic regression input)."""
        h0, a0, d0, dd0 = self.layer0.forward_activations(x)
        h1, a1, d1, dd1 = self.layer1.forward_activations(h0)
        return h0, a0, d0, dd0, h1, a1, d1, dd1

    def forward_with_derivs(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """psi + dpsi/dx_m + d2psi/dx_m^2 for every input channel m (chain rule).

        Chain rule (output k):
          d psi_k/dx_m = sum_j phi1'_k,j(h_j) * phi0'_j,m(x_m)
          d2psi_k/dx_m2 = sum_j [phi1''_k,j * (phi0'_j,m)^2 + phi1'_k,j * phi0''_j,m]
        Returns dict: psi (B,2), dpsi_x (B,in,2), d2psi_x (B,in,2).
        """
        h0, _, d0, dd0, h1, _, d1, dd1 = self.forward_activations(x)
        dpsi = torch.einsum("bjk,bmj->bmk", d1, d0)              # (B, in, out)
        d2psi = torch.einsum("bjk,bmj->bmk", dd1, d0 * d0) \
            + torch.einsum("bjk,bmj->bmk", d1, dd0)
        return {"psi": h1, "dpsi_x": dpsi, "d2psi_x": d2psi}

    def connection_magnitudes(self) -> list[torch.Tensor]:
        """Per-layer (out, in) connection magnitudes (pruning criterion)."""
        return [self.layer0.connection_magnitudes(),
                self.layer1.connection_magnitudes()]

    def apply_prune_mask(self, masks: list[torch.Tensor]) -> dict:
        """Zero pruned connections and register per-parameter gradient masks.

        masks: per-layer (out, in) bool (True = prune). Zeroed weights stay at
        zero because apply_prune_grads() (called after loss.backward() in the
        fine-tuning phase) blanks their gradients — requires_grad cannot be set
        per element. Returns {"pruned": n, "total": n, "ratio": r}.
        """
        self._prune_masks = []  # [(param, broadcastable bool mask)]
        stats = {"pruned": 0, "total": 0}
        for layer, mask in zip((self.layer0, self.layer1), masks):
            m = mask.to(layer.base_w.device)
            n = int(m.sum())
            stats["pruned"] += n
            stats["total"] += m.numel()
            for p in (layer.base_w, layer.spline_w):
                p.data.masked_fill_(m, 0.0)
                self._prune_masks.append((p, m))
            layer.coeffs.data.masked_fill_(m.unsqueeze(-1), 0.0)
            self._prune_masks.append((layer.coeffs, m.unsqueeze(-1)))
        stats["ratio"] = stats["pruned"] / max(stats["total"], 1)
        return stats

    def apply_prune_grads(self) -> None:
        """Zero gradients of pruned parameters (call after loss.backward())."""
        for p, m in getattr(self, "_prune_masks", []):
            if p.grad is not None:
                p.grad.masked_fill_(m, 0.0)

    def prune_mask_tensors(self) -> list[torch.Tensor]:
        """Current per-layer (out, in) prune masks (persisted in best.pt)."""
        if not getattr(self, "_prune_masks", None):
            return [torch.zeros_like(layer.base_w, dtype=torch.bool)
                    for layer in (self.layer0, self.layer1)]
        out = []
        for layer in (self.layer0, self.layer1):
            mask = None
            for p, m in self._prune_masks:
                if p is layer.base_w:
                    mask = m
            out.append(mask)
        return out

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(**overrides) -> BSplineKAN:
    """Build the paper KAN with optional overrides (in_dim, hidden, ...)."""
    kwargs = dict(in_dim=19, hidden=10, out_dim=2, grid_size=2, degree=2,
                  base_fn="silu")
    kwargs.update(overrides)
    return BSplineKAN(**kwargs)


# ---------------------------------------------------------------------------
# smoke self-tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from scipy.interpolate import BSpline as SciBSpline

    # 1) basis values vs scipy BSpline
    np.random.seed(0)
    x_np = np.linspace(-1.0, 1.0, 1001)
    x = torch.tensor(x_np, dtype=torch.float64)
    model0 = BSplineKAN(in_dim=3, hidden=4, out_dim=2, degree=2).double()
    knots = model0.layer0.knots
    t = np.asarray(knots, dtype=float)
    # scipy expects (t, c, k) with k = degree and c of len(t)-k-1 = n_basis
    for kk in range(model0.layer0.n_coeffs):
        c = np.zeros(model0.layer0.n_coeffs)
        c[kk] = 1.0
        sb = SciBSpline(t, c, 2)
        ours = _bspline_bases(x, knots, 2)[2][:, kk].numpy()
        ref = sb(x_np)
        assert np.allclose(ours, ref, atol=1e-9), f"basis {kk} mismatch"
    print(f"  basis vs scipy BSpline: OK ({model0.layer0.n_coeffs} bases)")

    # 2) analytic 1st/2nd derivatives vs central finite differences
    y = torch.linspace(-0.95, 0.95, 251, dtype=torch.float64).reshape(-1, 1)
    y = y.repeat(1, 3)
    _, _, d1, d2 = model0.layer0.forward_activations(y)
    h = 1e-5
    yp, ym = y + h, y - h
    num1 = (model0.layer0(yp) - model0.layer0(ym)) / (2 * h)
    num2 = (model0.layer0(yp) - 2 * model0.layer0(y) + model0.layer0(ym)) / h**2
    # d1/d2 are per-connection; sum over inputs = derivative of the output
    assert np.allclose(d1.detach().sum(1).numpy(), num1.detach().numpy(),
                       rtol=1e-6, atol=1e-6), "first derivative mismatch"
    assert np.allclose(d2.detach().sum(1).numpy(), num2.detach().numpy(),
                       rtol=1e-4, atol=1e-4), "second derivative mismatch"
    print("  analytic d1/d2 vs central differences: OK")

    # 3) forward_with_derivs vs torch.autograd on a random input
    xg = torch.randn(7, 3, dtype=torch.float64)
    xg.requires_grad_(True)
    psi = model0(xg)[:, 0]
    (dR_ref,) = torch.autograd.grad(psi, xg, grad_outputs=torch.ones_like(psi),
                                    create_graph=True)
    d2R_ref = torch.autograd.grad(dR_ref[:, 0].sum(), xg, create_graph=True)[0][:, 0]
    with torch.no_grad():
        out = model0.forward_with_derivs(xg.detach())
    assert np.allclose(out["dpsi_x"][:, 0, 0].numpy(), dR_ref[:, 0].detach().numpy(),
                       atol=1e-8)
    assert np.allclose(out["d2psi_x"][:, 0, 0].numpy(), d2R_ref.detach().numpy(),
                       atol=1e-6)
    print("  forward_with_derivs vs autograd: OK")

    # 4) prune mask: connection removal + gradient freezing
    m = build_model()
    xp = torch.randn(5, 19)
    with torch.no_grad():
        y_before = m(xp)
    masks = [torch.zeros_like(mg, dtype=torch.bool) for mg in m.connection_magnitudes()]
    masks[0][:3, :] = True  # prune 3 outgoing connections of layer0
    st = m.apply_prune_mask(masks)
    with torch.no_grad():
        y_after = m(xp)
    # pruned connections carried nonzero weight -> output must change...
    assert not torch.allclose(y_before, y_after), "pruning removed nothing"
    # ...and re-applying the same mask must be idempotent (forward unchanged)
    m.apply_prune_mask(masks)
    with torch.no_grad():
        y_again = m(xp)
    assert torch.allclose(y_after, y_again), "prune mask not idempotent"
    assert st["pruned"] == 3 * 19 and st["ratio"] == 3 * 19 / (19 * 10 + 10 * 2)
    # gradient freezing: pruned weights get zero grads after apply_prune_grads
    m.zero_grad()
    loss = m(xp).square().mean()
    loss.backward()
    m.apply_prune_grads()
    assert torch.all(m.layer0.base_w.grad[:3] == 0)
    assert torch.all(m.layer0.coeffs.grad[:3] == 0)
    assert m.layer0.base_w.grad[3:].abs().sum() > 0
    # prune_mask_tensors round-trip
    got = m.prune_mask_tensors()
    assert len(got) == 2 and torch.equal(got[0], masks[0]) and torch.equal(got[1], masks[1])
    print(f"  prune mask: OK (removed {st['pruned']}/{st['total']} connections, "
          f"ratio {st['ratio']:.4f})")

    # 5) param count sanity (paper: 14ch -> 960; here 19ch -> 1260)
    m19 = build_model(in_dim=19)
    n = m19.count_params()
    print(f"  BSplineKAN 19->10->2: {n} trainable params "
          f"(paper 14ch: 960; expected 210 conns x 6 = 1260)")
    assert n == (19 * 10 + 10 * 2) * 6, n
    print("smoke OK")
