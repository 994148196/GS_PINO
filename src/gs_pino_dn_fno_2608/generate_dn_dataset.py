"""Dataset generation CLI reproducing arXiv:2608.05555 (double-null FNO surrogate).

Solves double-null free-boundary GS equilibria with FREEGS (TestTokamak geometry)
following the paper's data-generation config:
  - domain R in [0.1, 2.0], Z in [-2.0, 2.0], 65x65 grid, freeBoundaryHagenow
  - profiles: ConstrainPaxisIp(paxis, Ip, fvac) (default shapes alpha_m=1, alpha_n=2)
  - params: paxis ~ U[200, 3000] Pa, Ip ~ U[5e4, 4e5] A, fvac ~ U[0.5, 3.0]
  - X-points: reference (1.1, +-0.6) m with symmetric jitter |dR|,|dZ| <= 0.02 m
  - constraints: double X-points + isoflux to fixed outboard midplane (1.5, 0.0),
    gamma=1e-12; Picard solve rtol=1e-3, maxits=50
  - acceptance: converged AND |Ip_sol - Ip_tgt|/Ip_tgt <= 10% AND >= 2 X-points

Per-sample saved fields (float32; full set kept for the future PINO stage):
  psi_total, psi_plasma, psi_plasma_norm, psi_coils, mask,
  greens (4 coils), coil_currents (4), dpdpsi, FdFdpsi,
  params [Ip, paxis, fvac], x_coords [R_lo, Z_lo, R_up, Z_up],
  axes [R_axis, Z_axis, psi_bndry, psi_axis], L, Beta0, solve_time

Generation is chunked (default 500 samples/chunk): each chunk is a standalone
.npz in <out-dir>/<split>/chunk_XXX.npz and acts as a resume checkpoint (existing
chunks are skipped on re-run).

Usage:
  python -m gs_pino_dn_fno_2608.generate_dn_dataset --split train --n-samples 5000 --seed 123 \
      --out-dir dn_fno_2608/data --chunk-size 500 --n-jobs 24
  python -m gs_pino_dn_fno_2608.generate_dn_dataset --split train --out-dir dn_fno_2608/data --merge
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from tqdm import trange

try:
    import freegs
    from freegs import boundary, control, critical, jtor
    HAS_FREEGS = True
except ImportError:
    HAS_FREEGS = False

# ---- paper settings (arXiv:2608.05555) ----
RMIN, RMAX = 0.1, 2.0
ZMIN, ZMAX = -2.0, 2.0
NX, NY = 65, 65
XPT_R, XPT_Z = 1.1, 0.6          # reference X-point targets (up/down symmetric)
XPT_JITTER = 0.02                 # paper Eq. 4: |dR|, |dZ| <= 0.02 m
ISOFLUX_REF = (1.5, 0.0)          # fixed outboard midplane reference (gauge anchor)
GAMMA = 1e-12                     # Tikhonov regularization (paper)
RTOL, MAXITS = 1e-3, 50           # Picard tolerance / max iterations (paper)
IP_TOL = 0.10                     # acceptance: |Ip_sol - Ip_tgt| / Ip_tgt <= 10%
MIN_XPTS = 2                      # acceptance: find_critical finds >= 2 X-points

PARAM_RANGES = {
    "paxis": (200.0, 3000.0),     # Pa
    "Ip": (5e4, 4e5),             # A
    "fvac": (0.5, 3.0),
}

COIL_NAMES = ["P1L", "P1U", "P2L", "P2U"]

# scalar fields stacked per-sample on merge
STACKED_KEYS = [
    "psi_total", "psi_plasma", "psi_plasma_norm", "psi_coils", "mask",
    "dpdpsi", "FdFdpsi", "greens", "coil_currents", "params", "x_coords",
    "axes", "L", "Beta0", "solve_time",
]
SHARED_KEYS = ["R", "Z"]  # stored once per chunk, not stacked


def sample_params(rng: np.random.Generator) -> dict[str, float]:
    """Sample one equilibrium parameter vector (paper Eq. 2-4 ranges)."""
    return {name: float(rng.uniform(*PARAM_RANGES[name])) for name in PARAM_RANGES}


def sample_xpoints(rng: np.random.Generator):
    """Symmetric X-point jitter preserving exact up-down symmetry (paper Eq. 4)."""
    dR = float(rng.uniform(-XPT_JITTER, XPT_JITTER))
    dZ = float(rng.uniform(-XPT_JITTER, XPT_JITTER))
    lo = (XPT_R + dR, -(XPT_Z + dZ))
    up = (XPT_R + dR, +(XPT_Z + dZ))
    return lo, up


def _solve_one(args: tuple) -> dict | None:
    """Solve a single DN equilibrium. Module-level for joblib on Windows."""
    params, i_seed = args
    t0 = time.perf_counter()

    paxis, Ip, fvac = params["paxis"], params["Ip"], params["fvac"]
    rng = np.random.default_rng(i_seed)
    lo, up = sample_xpoints(rng)

    try:
        tokamak = freegs.machine.TestTokamak()

        eq = freegs.Equilibrium(
            tokamak=tokamak,
            Rmin=RMIN, Rmax=RMAX, Zmin=ZMIN, Zmax=ZMAX,
            nx=NX, ny=NY,
            boundary=boundary.freeBoundaryHagenow,
        )

        # Profile shapes fixed by the FREEGS setup (default alpha_m=1, alpha_n=2);
        # only scalar amplitudes vary (paper: "shapes fixed by the FREEGS setup").
        profiles = jtor.ConstrainPaxisIp(eq, paxis=paxis, Ip=Ip, fvac=fvac)

        constrain = control.constrain(
            xpoints=[lo, up],
            isoflux=[(*lo, *ISOFLUX_REF), (*up, *ISOFLUX_REF)],
            gamma=GAMMA,
        )

        freegs.solve(eq, profiles, constrain, rtol=RTOL, maxits=MAXITS, show=False)

        # ---- acceptance checks ----
        if eq.psi_axis is None or eq.psi_bndry is None:
            return None
        ip_err = abs(eq.plasmaCurrent() - Ip) / Ip
        if ip_err > IP_TOL:
            return None
        opt, xpt = critical.find_critical(eq.R, eq.Z, eq.psi())
        if len(xpt) < MIN_XPTS:
            return None

        # ---- extract fields ----
        psi_total = eq.psi()
        psi_plasma = eq.plasma_psi
        psi_coils = eq.tokamak.psi(eq.R, eq.Z)
        psi_plasma_norm = (psi_plasma - eq.psi_bndry) / (eq.psi_axis - eq.psi_bndry + 1e-30)

        mask = critical.core_mask(eq.R, eq.Z, psi_total, opt, xpt)

        # GS RHS ingredients for residual diagnostics / future PINO stage:
        # RHS = -mu0 R^2 dp/dpsi - F dF/dpsi, with p'/ff' evaluated on psi_norm
        psi_norm = (psi_total - eq.psi_axis) / (eq.psi_bndry - eq.psi_axis + 1e-30)
        dpdpsi = profiles.pprime(psi_norm)
        fdFdpsi = profiles.ffprime(psi_norm)

        greens, coil_currents = [], []
        for label, coil in tokamak.coils:
            if coil.control:
                greens.append(coil.createPsiGreens(eq.R, eq.Z).astype(np.float32))
                coil_currents.append(float(coil.current))

        R_axis = opt[0][0] if opt else 1.0
        Z_axis = opt[0][1] if opt else 0.0
        # paper: boundary flux is the average of the two X-point fluxes
        psi_bndry = 0.5 * (xpt[0][2] + xpt[1][2])

        return {
            "psi_total": psi_total.astype(np.float32),
            "psi_plasma": psi_plasma.astype(np.float32),
            "psi_plasma_norm": psi_plasma_norm.astype(np.float32),
            "psi_coils": psi_coils.astype(np.float32),
            "mask": mask.astype(np.float32),
            "dpdpsi": dpdpsi.astype(np.float32),
            "FdFdpsi": fdFdpsi.astype(np.float32),
            "greens": np.stack(greens),
            "coil_currents": np.array(coil_currents, dtype=np.float32),
            "params": np.array([Ip, paxis, fvac], dtype=np.float32),
            "x_coords": np.array([*lo, *up], dtype=np.float32),
            "axes": np.array([R_axis, Z_axis, psi_bndry, eq.psi_axis], dtype=np.float32),
            "L": np.array([profiles.L], dtype=np.float32),
            "Beta0": np.array([profiles.Beta0], dtype=np.float32),
            "solve_time": np.array([time.perf_counter() - t0], dtype=np.float32),
        }
    except Exception:
        return None


def _solve_with_retry(args: tuple, max_retries: int = 5, rng: np.random.Generator | None = None) -> dict | None:
    """Solve with param resampling on rejection (paper: 100% acceptance expected)."""
    params, i_seed = args
    for attempt in range(max_retries):
        result = _solve_one((params, i_seed + attempt * 1_000_000))
        if result is not None:
            return result
        if rng is not None:
            params = sample_params(rng)
    return None


def save_chunk(chunk_dir: Path, chunk_idx: int, results: list[dict]) -> Path:
    """Save one chunk as a standalone .npz (resume checkpoint)."""
    arrays = {key: np.stack([r[key] for r in results]) for key in STACKED_KEYS}
    arrays["R"] = R_GLOBAL
    arrays["Z"] = Z_GLOBAL
    chunk_dir.mkdir(parents=True, exist_ok=True)
    path = chunk_dir / f"chunk_{chunk_idx:03d}.npz"
    np.savez(path, **arrays)
    return path


# shared grid arrays (set in generate())
R_GLOBAL = None
Z_GLOBAL = None


def generate(out_dir: str, split: str, n_samples: int, seed: int,
             chunk_size: int, n_jobs: int) -> None:
    """Generate one split in resumable chunks."""
    global R_GLOBAL, Z_GLOBAL

    if not HAS_FREEGS:
        raise RuntimeError("freegs not found. Install freegs first.")

    out_dir = Path(out_dir)
    chunk_dir = out_dir / split
    chunk_dir.mkdir(parents=True, exist_ok=True)

    # shared grid (identical across samples)
    eq_tmp = freegs.Equilibrium(
        tokamak=freegs.machine.TestTokamak(),
        Rmin=RMIN, Rmax=RMAX, Zmin=ZMIN, Zmax=ZMAX, nx=NX, ny=NY,
        boundary=boundary.freeBoundaryHagenow,
    )
    R_GLOBAL = eq_tmp.R.astype(np.float32)
    Z_GLOBAL = eq_tmp.Z.astype(np.float32)

    rng = np.random.default_rng(seed)
    n_chunks = int(np.ceil(n_samples / chunk_size))

    print(f"\n{'='*70}")
    print(f"  DN dataset generation | split={split} | n={n_samples} | seed={seed}")
    print(f"  grid {NX}x{NY}, R [{RMIN},{RMAX}], Z [{ZMIN},{ZMAX}], chunk_size={chunk_size}")
    print(f"  X-pts (1.1,+-0.6)+-0.02 m, isoflux->{ISOFLUX_REF}, gamma={GAMMA}, maxits={MAXITS}")
    print(f"{'='*70}")

    from joblib import Parallel, delayed

    total_accepted, total_solves = 0, 0
    for ci in trange(n_chunks, desc=f"{split} chunks"):
        chunk_path = chunk_dir / f"chunk_{ci:03d}.npz"
        if chunk_path.exists():
            print(f"  chunk {ci:03d} exists, skipping (resume)")
            with np.load(chunk_path) as d:
                total_accepted += d["psi_total"].shape[0]
            continue

        i0 = ci * chunk_size
        idx = range(i0, min(i0 + chunk_size, n_samples))
        # pre-sample deterministic params for this chunk
        chunk_params = [sample_params(rng) for _ in idx]

        t0 = time.perf_counter()
        results = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(_solve_with_retry)((p, seed * 100_000 + i))
            for i, p in zip(idx, chunk_params)
        )
        dt = time.perf_counter() - t0

        valid = [r for r in results if r is not None]
        total_accepted += len(valid)
        total_solves += len(results)
        save_chunk(chunk_dir, ci, valid)
        print(f"  chunk {ci:03d}: {len(valid)}/{len(results)} accepted, "
              f"{dt:.1f}s ({dt/max(len(results),1):.2f}s/solve)")

    print(f"\n  {split}: {total_accepted}/{total_solves} accepted")
    print(f"  chunks saved to {chunk_dir}")


def merge(out_dir: str, split: str) -> None:
    """Merge all chunks of a split into a single .npz."""
    chunk_dir = Path(out_dir) / split
    chunks = sorted(chunk_dir.glob("chunk_*.npz"))
    if not chunks:
        raise FileNotFoundError(f"no chunks found in {chunk_dir}")

    merged, total = {}, 0
    for path in chunks:
        with np.load(path) as d:
            for key in STACKED_KEYS:
                if key not in d:
                    continue
                merged.setdefault(key, []).append(d[key])
            merged.setdefault("R", d["R"])
            merged.setdefault("Z", d["Z"])
            total += d["psi_total"].shape[0]

    out_path = Path(out_dir) / f"{split}.npz"
    arrays = {key: np.concatenate(v) for key, v in merged.items() if isinstance(v, list)}
    arrays["R"] = merged["R"]
    arrays["Z"] = merged["Z"]
    np.savez(out_path, **arrays)

    print(f"\n  merged {len(chunks)} chunks -> {out_path} ({total} samples)")
    for key in sorted(arrays):
        print(f"    {key}: {arrays[key].shape} {arrays[key].dtype}")

    # parameter-range sanity print
    p = arrays["params"]
    print("\n  Parameter statistics:")
    print(f"    Ip:    [{p[:,0].min():.2e}, {p[:,0].max():.2e}] A   (paper [5e4, 4e5])")
    print(f"    paxis: [{p[:,1].min():.1f}, {p[:,1].max():.1f}] Pa (paper [200, 3000])")
    print(f"    fvac:  [{p[:,2].min():.2f}, {p[:,2].max():.2f}]     (paper [0.5, 3.0])")
    x = arrays["x_coords"]
    print(f"    X-pt R: [{x[:,0].min():.3f}, {x[:,0].max():.3f}] m (paper 1.1 +- 0.02)")


def main() -> None:
    parser = argparse.ArgumentParser(description="DN dataset generation (arXiv:2608.05555)")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--n-samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--out-dir", default="dn_fno_2608/data")
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--merge", action="store_true",
                        help="merge existing chunks of --split into a single npz")
    args = parser.parse_args()

    if args.merge:
        merge(args.out_dir, args.split)
    else:
        generate(args.out_dir, args.split, args.n_samples, args.seed,
                 args.chunk_size, args.n_jobs)


if __name__ == "__main__":
    main()
