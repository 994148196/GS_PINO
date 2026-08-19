"""M2: symbolic regression of the trained KAN (paper Sec. 2.2, Table 3).

For every connection phi_{o,i}(x_i) of the pruned KAN-3 model, fit the
activation on a dense grid with a candidate family (paper: polynomial /
trigonometric / exponential / Chebyshev / silu / sigmoid / tanh / Gaussian /
softplus), least-squares, best R^2. Outputs:

  sr_stats.json       per-family counts + min R^2 (mirrors paper Table 3)
  expressions.json    per-connection {family, params, r2} (paper supplement)
  symbolic_gs_kan.py  generated numpy-only evaluation module (no torch)

verify(): KAN-SR vs KAN-3 error on test samples (SR added error must stay
small and be *uncorrelated* with KAN-3 error, cf. paper Sec. 4), plus
inference-latency benchmark (paper Table 5: KAN-2 6.54 ms).

Usage:
  "$PY" -u -m gs_pino_kan_2608.symbolic_kan \
    --checkpoint dn_fno_2608/kan/experiments/exp001_kan_v5/model_b19ch_kan_mix/best.pt \
    --test-data dn_fno_2608/data_v5/sn/test.npz \
    --out-dir dn_fno_2608/kan/experiments/exp001_kan_v5/sr
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.interpolate import BSpline
from scipy.optimize import least_squares

from gs_pino_kan_2608.data_kan import PointKANDataset
from gs_pino_kan_2608.model_kan import _knots, build_model

R2_OK = 0.9999          # symbolification threshold (paper min 0.999907)
N_GRID = 1024           # fit grid density
STARTS = 3              # random restarts for nonlinear families
NONLINEAR_MAXFUN = 150


# --------------------------------------------------------------------------
# activation extraction
# --------------------------------------------------------------------------
def connection_phi(model, layer: int, o: int, i: int):
    """Return callable phi(x) for connection (layer, o, i), numpy on [-1,1].

    phi(x) = wb*silu(x) + ws * sum_b c_b B_b(x)  (paper Eq. 3).

    Matches the torch model exactly: only the B-SPLINE is clamped beyond the
    knot domain (model_kan._bspline_bases clamps x to [k0, k[-1]-eps]);
    the silu base term stays UNCLAMPED (silu tail is smooth everywhere).
    Clipping the whole input here lost the silu tail and silently corrupted
    every edge fed by channels whose test-normalized value exceeds +/-1
    (measured: KAN-SR h0 diff up to 9.2 on the real ckpt).
    """
    L = model.layer0 if layer == 0 else model.layer1
    with torch.no_grad():
        wb = float(L.base_w[o, i])
        ws = float(L.spline_w[o, i])
        c = L.coeffs[o, i].detach().cpu().numpy()
    knots = np.asarray(_knots(L.grid_size, L.degree))
    spl = BSpline(knots, c, L.degree, extrapolate=False)

    def phi(x: np.ndarray) -> np.ndarray:
        xs = np.clip(x, knots[0], knots[-1])
        return wb * (x * 1.0 / (1.0 + np.exp(-x))) + ws * spl(xs)

    return phi


def conn_magnitude(model, layer: int, o: int, i: int) -> float:
    """|ws| * sum_b |c_b| — zero means the connection was pruned (KAN-3)."""
    L = model.layer0 if layer == 0 else model.layer1
    with torch.no_grad():
        return float(L.spline_w[o, i].abs() * L.coeffs[o, i].abs().sum())


def _r2(y: np.ndarray, yhat: np.ndarray) -> float:
    sse = float(np.sum((y - yhat) ** 2))
    sst = float(np.sum((y - y.mean()) ** 2))
    return float("nan") if sst == 0 else 1.0 - sse / sst


# --------------------------------------------------------------------------
# fitting
# --------------------------------------------------------------------------
def _silu(x): return x / (1.0 + np.exp(-x))
def _softplus(x): return np.log1p(np.exp(np.clip(x, -40, 40)))
def _sigmoid(x): return 1.0 / (1.0 + np.exp(-x))


def _fit_linear(x: np.ndarray, y: np.ndarray):
    """Closed-form fits over polynomial & Chebyshev degrees.

    Parsimony-first: return the FIRST family reaching R2_OK (paper Table 3
    reports poly3/4/6 and cheb4..15 all present), else the best (r2, fam, p).
    """
    best = None  # (r2, family, params)
    for d in range(1, 7):
        c = np.polynomial.polynomial.polyfit(x, y, d)
        r2 = _r2(y, np.polynomial.polynomial.polyval(x, c))
        if r2 >= R2_OK:
            return (r2, f"poly{d}", c)
        if best is None or r2 > best[0]:
            best = (r2, f"poly{d}", c)
    for d in [4, 6, 8, 10, 12, 15]:
        c = np.polynomial.chebyshev.chebfit(x, y, d)
        r2 = _r2(y, np.polynomial.chebyshev.chebval(x, c))
        if r2 >= R2_OK:
            return (r2, f"cheb{d}", c)
        if best is None or r2 > best[0]:
            best = (r2, f"cheb{d}", c)
    return best


def _fit_nonlinear(x: np.ndarray, y: np.ndarray, rng: np.random.Generator):
    """Nonlinear candidates with random restarts, paper's family order.

    First family reaching R2_OK wins (silu is the paper's most common family);
    otherwise return the best (r2, fam, p) among them.
    """
    def fit_one(fam, fn, p0):
        try:
            res = least_squares(lambda p: fn(x, p) - y, p0, max_nfev=NONLINEAR_MAXFUN)
            r2 = _r2(y, fn(x, res.x))
            return r2, fam, res.x
        except Exception:
            return None

    best = None
    fns = [
        ("silu", lambda x, p: p[0] * _silu(p[1] * x + p[2]) + p[3], 4),
        ("gaussian", lambda x, p: p[0] * np.exp(-((x - p[1]) / p[2]) ** 2) + p[3], 4),
        ("softplus", lambda x, p: p[0] * _softplus(p[1] * x + p[2]) + p[3], 4),
        ("sigmoid", lambda x, p: p[0] * _sigmoid(p[1] * x + p[2]) + p[3], 4),
        ("tanh", lambda x, p: p[0] * np.tanh(p[1] * x + p[2]) + p[3], 4),
        ("exp", lambda x, p: p[0] * np.exp(p[1] * x) + p[2], 3),
        ("sin", lambda x, p: p[0] * np.sin(p[1] * x + p[2]) + p[3], 4),
        # paper: "Gaussian 与多项式叠加"
        ("gauss_poly2", lambda x, p: p[0] * np.exp(-((x - p[1]) / p[2]) ** 2)
         + p[3] + p[4] * x + p[5] * x ** 2, 6),
    ]
    for fam, fn, n_p in fns:
        for _ in range(STARTS):
            if fam == "gaussian":
                p0 = [rng.uniform(-2, 2), rng.uniform(-1, 1),
                      rng.uniform(0.1, 2.0), rng.uniform(-2, 2)]
            elif fam == "exp":
                p0 = [rng.uniform(-2, 2), rng.uniform(-2, 2), rng.uniform(-2, 2)]
            else:
                p0 = rng.uniform(-1.0, 1.0, n_p)
            got = fit_one(fam, fn, p0)
            if got is None:
                continue
            if got[0] >= R2_OK:
                return got
            if best is None or got[0] > best[0]:
                best = got
    return best


def fit_connection(x: np.ndarray, y: np.ndarray, seed: int = 1):
    """(family, params, r2) for the activation y(x); pruned handled outside."""
    rng = np.random.default_rng(seed)
    best = _fit_linear(x, y)
    if best is not None and best[0] >= R2_OK:
        return best
    nl = _fit_nonlinear(x, y, rng)
    if nl is not None and (best is None or nl[0] > best[0]):
        best = nl
    if best is None or np.isnan(best[0]):
        best = (0.0, "const", np.array([y.mean()]))
    return best


# --------------------------------------------------------------------------
# code generation
# --------------------------------------------------------------------------
def _expr(fam: str, p: np.ndarray) -> str:
    q = [repr(float(v)) for v in p]
    if fam.startswith("poly"):
        # polyfit gives ascending coefficients; np.polyval wants descending
        asc = np.asarray(p)
        desc = asc[::-1]
        return f"np.polyval([{', '.join(repr(float(v)) for v in desc)}], x)"
    if fam.startswith("cheb"):
        return (f"np.polynomial.chebyshev.chebval(x, np.array([{', '.join(q)}]))")
    if fam == "silu":
        return f"({q[0]} * silu({q[1]} * x + {q[2]}) + {q[3]})"
    if fam == "tanh":
        return f"({q[0]} * np.tanh({q[1]} * x + {q[2]}) + {q[3]})"
    if fam == "sigmoid":
        return f"({q[0]} * sigmoid({q[1]} * x + {q[2]}) + {q[3]})"
    if fam == "softplus":
        return f"({q[0]} * softplus({q[1]} * x + {q[2]}) + {q[3]})"
    if fam == "gaussian":
        return f"({q[0]} * np.exp(-((x - {q[1]}) / {q[2]}) ** 2) + {q[3]})"
    if fam == "exp":
        return f"({q[0]} * np.exp({q[1]} * x) + {q[2]})"
    if fam == "sin":
        return f"({q[0]} * np.sin({q[1]} * x + {q[2]}) + {q[3]})"
    if fam == "gauss_poly2":
        return (f"({q[0]} * np.exp(-((x - {q[1]}) / {q[2]}) ** 2) "
                f"+ {q[3]} + {q[4]} * x + {q[5]} * x ** 2)")
    if fam == "const":
        return f"np.full_like(x, {q[0]})"
    if fam == "zero":
        return "np.zeros_like(x)"
    if fam == "table":
        raise NotImplementedError("table fallback handled via spline tables")
    raise ValueError(fam)


def _generate_module(conns: list, out_path: Path, doc: str,
                     n_hidden: int, n_out: int,
                     lo0: list, hi0: list, lo1: list, hi1: list) -> None:
    """conns: per layer, list of (o, i, family, params, r2).

    Every edge input is clipped to its FIT range (observed channel range for
    layer 0, observed pre-activation range for layer 1) — mirrors the torch
    spline's clamped extension; polynomial fits extrapolate catastrophically
    beyond it (real ckpt measured 610878% without clipping).

    Written line-by-line (no textwrap.dedent: the doc string's own newlines
    would defeat the common-indent computation).
    """
    lines = [
        "# -*- coding: utf-8 -*-",
        '"""Generated symbolic KAN (KAN-SR) - numpy only, no torch.',
        "",
        doc,
        "Generated by gs_pino_kan_2608.symbolic_kan; do not edit.",
        '"""',
        "import numpy as np",
        "",
        "def silu(x): return x / (1.0 + np.exp(-x))",
        "def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))",
        "def softplus(x): return np.log1p(np.exp(np.clip(x, -40, 40)))",
        "",
        "# layer i: list of (out, in, expr); h[:, out] += expr(clip(x[:, in]))",
        f"_LO0 = {repr([float(v) for v in lo0])}",
        f"_HI0 = {repr([float(v) for v in hi0])}",
        f"_LO1 = {repr([float(v) for v in lo1])}",
        f"_HI1 = {repr([float(v) for v in hi1])}",
        "_LAYER0 = [",
    ]
    for o, i, fam, p, r2 in conns[0]:
        lines.append(f"    ({o}, {i}, lambda x: {_expr(fam, p)}),")
    lines += ["]", "_LAYER1 = ["]
    for o, i, fam, p, r2 in conns[1]:
        lines.append(f"    ({o}, {i}, lambda x: {_expr(fam, p)}),")
    lines += [
        "]",
        "",
        "def evaluate(x):",
        '    """x: (N, in_dim) normalized -> (N, 2) normalized (psi, Jpsi)."""',
        "    x = np.asarray(x, dtype=np.float64)",
        f"    h0 = np.zeros((x.shape[0], {n_hidden}))",
        "    for o, i, e in _LAYER0:",
        "        h0[:, o] += e(np.clip(x[:, i], _LO0[i], _HI0[i]))",
        f"    h1 = np.zeros((x.shape[0], {n_out}))",
        "    for o, i, e in _LAYER1:",
        "        h1[:, o] += e(np.clip(h0[:, i], _LO1[i], _HI1[i]))",
        "    return h1",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------
# assemble + verify
# --------------------------------------------------------------------------
def _pad(lo: float, hi: float) -> float:
    return 0.05 * (hi - lo) + 1e-3


def _observed_ranges(model, test_data: str, stats: dict):
    """(lo, hi) per input channel AND per hidden node, observed over a few
    full grids of the test set (normalized with the CKPT stats, same as
    verify/eval). Fit grids must cover the ACTUAL data mass, not [-1,1]:
    test scalars can exceed the train min/max -> the torch spline clamps
    (constant extension) but polynomial fits extrapolate catastrophically
    outside their fit range (observed: KAN-SR error 610878% on the real ckpt
    when fit on [-1,1] without clipping)."""
    ds = PointKANDataset(test_data, stats=stats)
    xs = torch.cat([ds.full_grid(i)["x"] for i in range(min(4, len(ds)))])
    in_lo, in_hi = xs.min(0).values.numpy(), xs.max(0).values.numpy()
    with torch.no_grad():
        h = model.layer0(xs)
    act_lo, act_hi = h.min(0).values.numpy(), h.max(0).values.numpy()
    return in_lo, in_hi, act_lo, act_hi


def assemble(ckpt_path: str, out_dir: Path, test_data: str | None = None) -> dict:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = build_model(**ckpt.get("arch", {})).to("cpu")
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    n_hidden = model.layer0.out_dim
    n_in = model.layer0.in_dim
    stats = {k: (np.asarray(v, dtype=np.float32) if isinstance(v, list) else np.float32(v))
             for k, v in ckpt["stats"].items()}
    if test_data:
        in_lo, in_hi, act_lo, act_hi = _observed_ranges(model, test_data, stats)
    else:
        in_lo = np.full(n_in, -1.0); in_hi = np.full(n_in, 1.0)
        act_lo = np.full(n_hidden, -1.0); act_hi = np.full(n_hidden, 1.0)
    # per-input fit grids: observed channel range + pad (layer 0)
    g0 = [np.linspace(lo - _pad(lo, hi), hi + _pad(lo, hi), N_GRID)
          for lo, hi in zip(in_lo, in_hi)]
    # per-hidden fit grids: observed pre-activation range + pad (layer 1)
    g1 = [np.linspace(lo - _pad(lo, hi), hi + _pad(lo, hi), N_GRID)
          for lo, hi in zip(act_lo, act_hi)]
    stats_rows = []          # (family, r2) per non-pruned connection
    conns = [[], []]         # per layer: (o, i, family, params, r2)
    n_pruned = n_zero = n_fail = 0
    with torch.no_grad():
        for L in range(2):
            layer = model.layer0 if L == 0 else model.layer1
            for o in range(layer.out_dim):
                for i in range(layer.in_dim):
                    if conn_magnitude(model, L, o, i) == 0.0:
                        conns[L].append((o, i, "zero", np.array([]), 1.0))
                        n_pruned += 1
                        continue
                    xg = g1[i] if L == 1 else g0[i]
                    y = connection_phi(model, L, o, i)(xg)
                    if np.allclose(y, y[0]):
                        conns[L].append((o, i, "const", np.array([float(y[0])]), 1.0))
                        n_zero += 1
                        continue
                    r2, fam, p = fit_connection(xg, y)
                    conns[L].append((o, i, fam, p, r2))
                    if r2 >= R2_OK:
                        stats_rows.append((fam, r2))
                    else:
                        n_fail += 1
                    print(f"  L{L} ({o},{i}) -> {fam:12s} R2={r2:.6f}")

    total = sum(len(c) for c in conns)
    families: dict = {}
    for fam, r2 in stats_rows:
        d = families.setdefault(fam, {"n": 0, "r2_min": r2})
        d["n"] += 1
        d["r2_min"] = min(d["r2_min"], r2)
    fam_rows = sorted(families.items(),
                      key=lambda kv: -kv[1]["n"])
    sr_stats = {
        "n_connections_total": total,
        "n_pruned": n_pruned,                # paper Table 3: 无(被剪枝)
        "n_const": n_zero,
        "n_symbolified": len(stats_rows),
        "n_below_threshold": n_fail,         # kept as spline fallback
        "r2_threshold": R2_OK,
        "families": [{"name": k, **v, "pct": round(100.0 * v["n"] / total, 2)}
                     for k, v in fam_rows],
        "r2_min_symbolified": min((r2 for _, r2 in stats_rows), default=float("nan")),
    }
    with open(out_dir / "sr_stats.json", "w") as f:
        json.dump(sr_stats, f, indent=2)

    expressions = {"layers": [
        [{"out": o, "in": i, "family": fam, "params": p.tolist(), "r2": r2}
         for o, i, fam, p, r2 in conns[0]],
        [{"out": o, "in": i, "family": fam, "params": p.tolist(), "r2": r2}
         for o, i, fam, p, r2 in conns[1]],
    ], "n_pruned": n_pruned, "n_symbolified": len(stats_rows)}
    with open(out_dir / "expressions.json", "w") as f:
        json.dump(expressions, f, indent=1)

    doc = (f"checkpoint {ckpt_path}\nphase {ckpt.get('phase')} | prune ratio "
           f"{ckpt.get('prune_ratio', 0)} | arch {ckpt.get('arch', {})}")
    _generate_module(conns, out_dir / "symbolic_gs_kan.py", doc,
                     n_hidden, model.layer1.out_dim,
                     [float(v) for v in in_lo], [float(v) for v in in_hi],
                     [float(v) for v in act_lo], [float(v) for v in act_hi])
    return sr_stats


def verify(ckpt_path: str, test_data: str, out_dir: Path,
           max_samples: int, device: str) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    stats = {k: (np.asarray(v, dtype=np.float32) if isinstance(v, list) else np.float32(v))
             for k, v in ckpt["stats"].items()}
    model = build_model(**ckpt.get("arch", {})).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    ds = PointKANDataset(test_data, stats=stats)
    n = min(len(ds), max_samples) if max_samples else len(ds)

    import importlib.util
    spec = importlib.util.spec_from_file_location("symbolic_gs_kan",
                                                  out_dir / "symbolic_gs_kan.py")
    sr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sr)

    err_kan3, err_sr = [], []
    with torch.no_grad():
        for i in range(n):
            b = ds.full_grid(i)
            y_norm = b["y_psi"].numpy().reshape(65, 65)
            out = model(b["x"].to(device))
            e3 = float(np.linalg.norm(out[:, 0].cpu().numpy().reshape(65, 65) - y_norm)
                       / (np.linalg.norm(y_norm) + 1e-12))
            ps = sr.evaluate(b["x"].numpy())
            es = float(np.linalg.norm(ps[:, 0].reshape(65, 65) - y_norm)
                       / (np.linalg.norm(y_norm) + 1e-12))
            err_kan3.append(e3)
            err_sr.append(es)
    err_kan3, err_sr = np.array(err_kan3), np.array(err_sr)
    added = err_sr - err_kan3
    corr = float(np.corrcoef(added, err_kan3)[0, 1]) if n > 2 else float("nan")
    out = {"n_eval": n,
           "rel_l2_kan3_mean_pct": float(np.mean(err_kan3)) * 100,
           "rel_l2_sr_mean_pct": float(np.mean(err_sr)) * 100,
           "sr_added_mean_pct": float(np.mean(added)) * 100,
           "sr_added_max_pct": float(np.max(added)) * 100,
           "corr_sr_added_kan3_err": corr}
    with open(out_dir / "verify.json", "w") as f:
        json.dump(out, f, indent=2)

    # latency (paper Table 5): 100 single-sample full-grid forwards
    x = ds.full_grid(0)["x"].numpy()
    t0 = time.perf_counter()
    for _ in range(100):
        sr.evaluate(x)
    t_sr = (time.perf_counter() - t0) / 100 * 1e3
    xt = torch.from_numpy(x).to(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(100):
            model(xt)
    t_torch = (time.perf_counter() - t0) / 100 * 1e3
    out["latency_ms_full_grid"] = {"symbolic_numpy": round(t_sr, 3),
                                   "torch_kan": round(t_torch, 3)}
    with open(out_dir / "verify.json", "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n  KAN-3 rel L2 mean: {out['rel_l2_kan3_mean_pct']:.4f}% | "
          f"KAN-SR: {out['rel_l2_sr_mean_pct']:.4f}% | added {out['sr_added_mean_pct']:.4f}%")
    print(f"  corr(SR added, KAN-3 err) = {corr:.3f}  (<0.3 criterion)")
    print(f"  latency (4225 pts): SR {t_sr:.3f} ms | torch {t_torch:.3f} ms "
          f"(paper KAN-2 6.54 ms)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--test-data", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-samples", type=int, default=10)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--skip-verify", action="store_true")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else ("cpu" if args.device == "auto" else args.device))
    print("== symbolic regression (paper Sec. 2.2 / Table 3) ==")
    sr_stats = assemble(args.checkpoint, out_dir, args.test_data)
    print(f"\n  {sr_stats['n_symbolified']}/{sr_stats['n_connections_total']} "
          f"connections symbolified (pruned {sr_stats['n_pruned']}, "
          f"const {sr_stats['n_const']}, below-threshold {sr_stats['n_below_threshold']})")
    for r in sr_stats["families"]:
        print(f"    {r['name']:14s} {r['n']:3d}  {r['pct']:5.2f}%  R2_min {r['r2_min']:.6f}")
    print(f"  saved -> {out_dir}/sr_stats.json, expressions.json, symbolic_gs_kan.py")
    if not args.skip_verify:
        print("\n== verify (KAN-SR vs KAN-3 + latency) ==")
        verify(args.checkpoint, args.test_data, out_dir, args.max_samples, device)


if __name__ == "__main__":
    main()
