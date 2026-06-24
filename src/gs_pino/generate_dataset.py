"""Dataset generation CLI for fixed-boundary GS surrogate experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tqdm import trange

from .geometry import PARAM_NAMES, geometry_channels
from .solvers import AnalyticFixedBoundarySolver


def sample_params(rng: np.random.Generator) -> dict[str, float]:
    """Sample one equilibrium parameter vector from broad development ranges."""
    # These ranges should be narrowed/expanded to match the real GS_solver study.
    ranges = {
        "R0": (1.5, 1.9),
        "a": (0.35, 0.55),
        "kappa": (1.2, 1.9),
        "delta": (-0.45, 0.45),
        "Ip": (8.0e5, 1.6e6),
        "betap": (0.2, 1.0),
        "alpha_m": (1.0, 4.0),
        "alpha_n": (1.0, 4.0),
    }
    return {name: float(rng.uniform(*ranges[name])) for name in PARAM_NAMES}


def make_case_grid(R0: float, a: float, kappa: float, delta: float, nr: int, nz: int) -> tuple[np.ndarray, np.ndarray]:
    """Build a case-specific rectangular grid that encloses the shaped LCFS."""
    # Triangularity changes radial extent, so include `abs(delta)` in the margin.
    r_extent = a * (1.25 + abs(delta))
    z_extent = kappa * a * 1.25

    # Use float32 because the generated archive feeds directly into PyTorch.
    r = np.linspace(R0 - r_extent, R0 + r_extent, nr, dtype=np.float32)
    z = np.linspace(-z_extent, z_extent, nz, dtype=np.float32)
    return np.meshgrid(r, z, indexing="ij")


def generate(out: str, n_samples: int, nr: int, nz: int, seed: int) -> None:
    """Generate a compressed `.npz` dataset with fields documented in PROJECT.md."""
    rng = np.random.default_rng(seed)

    # Swap this object for a real GS_solver adapter when the external solver is available.
    solver = AnalyticFixedBoundarySolver()

    # Lists are accumulated sample-by-sample and stacked once at the end.
    arrays: dict[str, list[np.ndarray]] = {key: [] for key in ["R", "Z", "params", "psi", "psi_bar", "mask", "sdf", "rho", "theta"]}
    axes: list[list[float]] = []

    for _ in trange(n_samples, desc="Generating"):
        # 1. Draw one eight-parameter equilibrium setting.
        params = sample_params(rng)

        # 2. Build a rectangular grid that contains the non-rectangular LCFS.
        R, Z = make_case_grid(params["R0"], params["a"], params["kappa"], params["delta"], nr, nz)

        # 3. Compute geometry channels and solve/approximate psi on that grid.
        geom = geometry_channels(R, Z, params)
        sol = solver.solve(R, Z, params)

        # 4. Store coordinates and ordered scalar parameters.
        arrays["R"].append(R)
        arrays["Z"].append(Z)
        arrays["params"].append(np.array([params[name] for name in PARAM_NAMES], dtype=np.float32))

        # 5. Store physical and normalized flux fields.
        arrays["psi"].append(sol["psi"])
        arrays["psi_bar"].append(sol["psi_bar"])

        # 6. Store LCFS-aware geometry channels.
        for key in ["mask", "sdf", "rho", "theta"]:
            arrays[key].append(geom[key])

        # 7. Store axis and normalization metadata for later diagnostics.
        axes.append([sol["R_axis"], sol["Z_axis"], sol["psi_lcfs"], sol["psi_axis"]])

    # Create parent directory and write everything into one portable archive.
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        **{key: np.stack(value) for key, value in arrays.items()},
        axes=np.array(axes, dtype=np.float32),
        param_names=np.array(PARAM_NAMES),
    )


def main() -> None:
    """Parse command-line arguments and generate a dataset."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/gs_fixed_boundary.npz", help="Output .npz dataset path.")
    parser.add_argument("--n-samples", type=int, default=64, help="Number of equilibrium cases to generate.")
    parser.add_argument("--nr", type=int, default=64, help="Number of R grid points per case.")
    parser.add_argument("--nz", type=int, default=64, help="Number of Z grid points per case.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for parameter sampling.")
    args = parser.parse_args()
    generate(args.out, args.n_samples, args.nr, args.nz, args.seed)


if __name__ == "__main__":
    main()
