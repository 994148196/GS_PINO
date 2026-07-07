"""Dataset generation CLI for free-boundary GS surrogate experiments using freegs.

Input:  PF coil currents + plasma profile parameters (Ip, betap, alpha_m, alpha_n, fvac)
Output: psi_total (total poloidal flux) over entire computational domain

The network predicts psi_total directly. During inference, psi_coils is computed from
coil currents and Green functions: psi_coils = sum(I_k * G_k(R,Z)). The mask is only
used during training for loss weighting and PDE residual computation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tqdm import trange

try:
    import freegs
    from freegs import jtor, control, boundary
    HAS_FREEGS = True
except ImportError:
    HAS_FREEGS = False


COIL_NAMES = ["P1L", "P1U", "P2L", "P2U"]


def sample_params(rng: np.random.Generator) -> dict[str, float]:
    """Sample one equilibrium parameter vector for free-boundary experiments."""
    ranges = {
        "Ip": (1.5e5, 2.5e5),
        "paxis": (800.0, 1500.0),
        "alpha_m": (1.0, 2.0),
        "alpha_n": (1.5, 2.5),
        "fvac": (1.8, 2.2),
    }
    return {name: float(rng.uniform(*ranges[name])) for name in ranges}


def sample_xpoints(rng: np.random.Generator):
    """Sample X-point locations within reasonable range."""
    r1 = rng.uniform(1.0, 1.3)
    z1 = rng.uniform(0.5, 0.7)
    return [(r1, -z1), (r1, z1)]


def compute_grad_shafranov_rhs(psi_plasma: np.ndarray, R: np.ndarray, Z: np.ndarray) -> np.ndarray:
    dR = R[1, 0] - R[0, 0]
    dZ = Z[0, 1] - Z[0, 0]
    
    psi = psi_plasma
    
    d2r = (psi[2:, 1:-1] - 2.0 * psi[1:-1, 1:-1] + psi[:-2, 1:-1]) / (dR ** 2)
    dr = (psi[2:, 1:-1] - psi[:-2, 1:-1]) / (2.0 * dR)
    R_c = R[1:-1, 1:-1]
    r_term = dr / (R_c + 1e-8)
    d2z = (psi[1:-1, 2:] - 2.0 * psi[1:-1, 1:-1] + psi[1:-1, :-2]) / (dZ ** 2)
    
    lap_psi = d2r - r_term + d2z
    return lap_psi


def _solve_one(args: tuple) -> dict | None:
    """Solve a single free-boundary equilibrium. Used by parallel workers."""
    params, nx, ny, seed = args
    rng = np.random.default_rng(seed)

    try:
        tokamak = freegs.machine.TestTokamak()

        eq = freegs.Equilibrium(
            tokamak=tokamak,
            Rmin=0.1,
            Rmax=2.0,
            Zmin=-1.0,
            Zmax=1.0,
            nx=nx,
            ny=ny,
            boundary=boundary.freeBoundaryHagenow,
        )

        profiles = jtor.ConstrainPaxisIp(
            eq,
            paxis=params["paxis"],
            Ip=params["Ip"],
            fvac=params["fvac"],
            alpha_m=params["alpha_m"],
            alpha_n=params["alpha_n"],
            Raxis=1.0,
        )

        xpoints = sample_xpoints(rng)
        constrain = control.constrain(xpoints=xpoints)

        freegs.solve(eq, profiles, constrain, show=False)

        if eq.psi_axis is None or eq.psi_bndry is None:
            return None

        psi_total = eq.psi()
        psi_plasma = eq.plasma_psi
        psi_coils = eq.tokamak.psi(eq.R, eq.Z)

        rtol = 1e-3
        if not np.allclose(psi_total, psi_plasma + psi_coils, rtol=rtol):
            return None

        rhs = compute_grad_shafranov_rhs(psi_plasma, eq.R, eq.Z)

        greens = []
        coil_currents = []
        for label, coil in tokamak.coils:
            if coil.control:
                greens.append(coil.createPsiGreens(eq.R, eq.Z).astype(np.float32))
                coil_currents.append(float(coil.current))

        coil_currents = np.array(coil_currents, dtype=np.float32)
        greens = np.stack(greens)

        psi_plasma_norm = (psi_plasma - eq.psi_bndry) / (eq.psi_axis - eq.psi_bndry + 1e-30)

        opt, xpt = freegs.critical.find_critical(eq.R, eq.Z, psi_total)
        R_axis = opt[0][0] if opt else 1.0
        Z_axis = opt[0][1] if opt else 0.0

        mask = freegs.critical.core_mask(eq.R, eq.Z, psi_total, opt, xpt) if xpt else np.ones_like(psi_total)

        if np.isnan(psi_plasma_norm).any() or np.isinf(psi_plasma_norm).any():
            return None

        return {
            "R": eq.R.astype(np.float32),
            "Z": eq.Z.astype(np.float32),
            "psi_total": psi_total.astype(np.float32),
            "psi_plasma": psi_plasma.astype(np.float32),
            "psi_plasma_norm": psi_plasma_norm.astype(np.float32),
            "psi_coils": psi_coils.astype(np.float32),
            "rhs": rhs.astype(np.float32),
            "greens": greens,
            "coil_currents": coil_currents,
            "mask": mask.astype(np.float32),
            "params": np.array([params["Ip"], params["paxis"], params["alpha_m"], params["alpha_n"], params["fvac"]], dtype=np.float32),
            "psi_axis": float(eq.psi_axis),
            "psi_bndry": float(eq.psi_bndry),
            "R_axis": float(R_axis),
            "Z_axis": float(Z_axis),
            "L": getattr(profiles, "L", 0.0),
            "Beta0": getattr(profiles, "Beta0", 0.0),
        }
    except Exception as e:
        if seed == 0:
            print(f"Debug: Exception for seed {seed}: {e}")
        return None


def _solve_with_retry(args: tuple, max_retries: int = 3) -> dict | None:
    """Solve with retry logic for robustness with large-scale generation."""
    params, nx, ny, seed = args
    for attempt in range(max_retries):
        result = _solve_one((params, nx, ny, seed + attempt * 10000))
        if result is not None:
            return result
    return None


def generate(out: str, n_samples: int, nx: int, ny: int, seed: int, n_jobs: int = -1) -> None:
    """Generate a compressed `.npz` dataset for free-boundary GS PINO."""

    if not HAS_FREEGS:
        print("Error: freegs not found. Please install freegs first.")
        return

    print(f"\n{'='*60}")
    print(f"  Free-boundary GS Dataset Generation")
    print(f"{'='*60}")
    print(f"  Using freegs (version: {getattr(freegs, '__version__', 'unknown')})")
    print(f"  Target samples: {n_samples}")
    print(f"  Grid size: {nx}x{ny}")
    print(f"  Seed: {seed}")
    print(f"  Parallel workers: {n_jobs if n_jobs > 0 else 'all'}")
    print(f"  Output: {out}")
    print(f"{'='*60}")

    rng = np.random.default_rng(seed)

    all_params = [sample_params(rng) for _ in range(n_samples)]

    from joblib import Parallel, delayed
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(_solve_with_retry)((p, nx, ny, seed + i))
        for i, p in enumerate(all_params)
    )

    valid = [r for r in results if r is not None]
    n_failed = n_samples - len(valid)
    
    print(f"\n{'='*60}")
    print(f"  Generation complete")
    print(f"{'='*60}")
    print(f"  Valid samples: {len(valid)}/{n_samples}")
    print(f"  Failed solves: {n_failed} ({100*n_failed/n_samples:.1f}%)")
    
    if len(valid) == 0:
        print("  Error: No valid samples generated.")
        return

    arrays = {
        key: [r[key] for r in valid]
        for key in ["R", "Z", "psi_total", "psi_plasma", "psi_plasma_norm", "psi_coils", "rhs", "greens", "coil_currents", "mask", "params"]
    }

    axes = [[r["R_axis"], r["Z_axis"], r["psi_bndry"], r["psi_axis"]] for r in valid]
    L_vals = [r["L"] for r in valid]
    Beta0_vals = [r["Beta0"] for r in valid]

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        **{key: np.stack(value) for key, value in arrays.items()},
        axes=np.array(axes, dtype=np.float32),
        L=np.array(L_vals, dtype=np.float32),
        Beta0=np.array(Beta0_vals, dtype=np.float32),
        coil_names=np.array(COIL_NAMES),
    )
    
    params_array = np.array([r["params"] for r in valid])
    print(f"\n  Parameter statistics:")
    print(f"    Ip: [{params_array[:,0].min():.1e}, {params_array[:,0].max():.1e}] A")
    print(f"    paxis: [{params_array[:,1].min():.1f}, {params_array[:,1].max():.1f}] Pa")
    print(f"    alpha_m: [{params_array[:,2].min():.2f}, {params_array[:,2].max():.2f}]")
    print(f"    alpha_n: [{params_array[:,3].min():.2f}, {params_array[:,3].max():.2f}]")
    print(f"    fvac: [{params_array[:,4].min():.2f}, {params_array[:,4].max():.2f}]")
    print(f"\n  Dataset saved to {out} ({len(valid)} samples).")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/gs_free_boundary.npz", help="Output .npz dataset path.")
    parser.add_argument("--n-samples", type=int, default=64, help="Number of equilibrium cases to generate.")
    parser.add_argument("--nx", type=int, default=64, help="Number of R grid points per case (must be power of 2).")
    parser.add_argument("--ny", type=int, default=64, help="Number of Z grid points per case (must be power of 2).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for parameter sampling.")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Number of parallel workers (-1 = all cores).")
    args = parser.parse_args()
    generate(args.out, args.n_samples, args.nx, args.ny, args.seed, args.n_jobs)


if __name__ == "__main__":
    main()