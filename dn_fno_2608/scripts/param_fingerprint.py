"""Param-count fingerprint search for the paper's FNO: target exactly 4,770,241.

Two families are searched:
  A) neuraloperator 2.0 FNO (the locally installed library) across
     implementation/rank/separable settings.
  B) classic hand-written FNO2d assembly (paper's described structure):
     lifting(9->W, 1x1 conv) -> L x [SpectralConv2d(W,W,M1,M2) + pointwise] ->
     projection(W->1, 1x1 conv), with variants:
       - spectral weights complex (x2) or real (x1)
       - full modes (M1,M2) or rfft-half (M1, M2//2+1)
       - pointwise: none / 1x1 conv / MLP(W->128->W)
       - optional BatchNorm per layer
"""
from __future__ import annotations

import itertools

TARGET = 4_770_241


def family_b_count(w, m1, m2, layers, complex_x, rfft_half, pointwise, bn, lift_bias=True, proj_bias=True):
    if rfft_half:
        m2 = m2 // 2 + 1
    n = 9 * w + (w if lift_bias else 0)                      # lifting
    per_layer = complex_x * w * w * m1 * m2                  # spectral conv
    if pointwise == "conv":
        per_layer += w * w + w
    elif pointwise == "mlp":
        per_layer += w * 128 + 128 + 128 * w + w
    if bn:
        per_layer += 2 * w
    n += layers * per_layer
    n += w + (1 if proj_bias else 0)                         # projection
    return n


def main():
    # ---- family A: neuraloperator 2.0 FNO ----
    import torch
    from neuralop.models import FNO

    print("== family A: neuraloperator 2.0 FNO ==")
    found = []
    for impl, rank, sep, mlp in itertools.product(
            ["factorized", "reconstructed"], [0.25, 0.5, 1.0], [False, True], [True, False]):
        try:
            m = FNO(n_modes=(16, 16), in_channels=9, out_channels=1, hidden_channels=64,
                    n_layers=4, positional_embedding=None, non_linearity="gelu",
                    implementation=impl, rank=rank, separable=sep, use_channel_mlp=mlp)
            n = sum(p.numel() for p in m.parameters() if p.requires_grad)
            mark = "  <<< MATCH" if n == TARGET else ""
            print(f"  impl={impl}, rank={rank}, sep={sep}, mlp={mlp}: {n} (diff {n-TARGET}){mark}")
            if n == TARGET:
                found.append(("A", impl, rank, sep, mlp))
        except Exception as e:
            print(f"  impl={impl}, rank={rank}, sep={sep}, mlp={mlp}: ERROR {type(e).__name__}")

    # ---- family B: classic hand-written FNO2d ----
    print("\n== family B: classic FNO2d assembly ==")
    best = []
    for w in range(24, 81):
        for m1 in range(8, 21):
            for m2 in range(8, 21):
                for layers in (2, 3, 4, 5, 6):
                    for cx in (1, 2):
                        for rfft in (False, True):
                            for pt in ("none", "conv", "mlp"):
                                for bn in (False, True):
                                    n = family_b_count(w, m1, m2, layers, cx, rfft, pt, bn)
                                    if n == TARGET:
                                        print(f"  MATCH: W={w}, M=({m1},{m2}), L={layers}, "
                                              f"cx={cx}, rfft={rfft}, pt={pt}, bn={bn}")
                                        found.append(("B", w, m1, m2, layers, cx, rfft, pt, bn))
                                    best.append((abs(n - TARGET), n, w, m1, m2, layers, cx, rfft, pt, bn))
    best.sort()
    print("\n  closest 10 configs:")
    for d, n, w, m1, m2, L, cx, rfft, pt, bn in best[:10]:
        print(f"    W={w}, M=({m1},{m2}), L={L}, cx={cx}, rfft={rfft}, pt={pt}, bn={bn}: "
              f"{n} (diff {d})")

    print(f"\n{'='*60}\nEXACT MATCHES: {len(found)}")
    for f in found:
        print(" ", f)


if __name__ == "__main__":
    main()
